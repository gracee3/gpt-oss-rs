//! HTTP server setup, AppState, router construction, and graceful shutdown.
//!
//! Device policy selects the CUDA engine, native batch-one GPT-OSS CPU worker,
//! or an explicitly requested test-only mock executor.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use gpt_oss_core::prelude::RequestId;
use gpt_oss_engine::config::EngineConfig;
use gpt_oss_engine::AsyncLLMEngine;
use gpt_oss_engine::{CpuWorker, ExecutorAdapter, ExecutorConfig};
use gpt_oss_tokenizer::Tokenizer;

use crate::routes;
use crate::runtime_policy::{validate_gpt_oss_runtime, RuntimeBackendPath, RuntimeDecision};

// ------------------------------------------------------------------
// Engine trait object for unified API
// ------------------------------------------------------------------

/// Trait abstracting over AsyncLLMEngine and AsyncGpuLLMEngine so the
/// AppState can hold either one.
#[async_trait::async_trait]
pub trait InferenceEngine: Send + Sync {
    async fn generate(
        &self,
        prompt: String,
        params: gpt_oss_core::prelude::SamplingParams,
    ) -> gpt_oss_core::prelude::Result<(
        RequestId,
        tokio_stream::wrappers::ReceiverStream<gpt_oss_core::prelude::RequestOutput>,
    )>;
}

#[async_trait::async_trait]
impl InferenceEngine for AsyncLLMEngine {
    async fn generate(
        &self,
        prompt: String,
        params: gpt_oss_core::prelude::SamplingParams,
    ) -> gpt_oss_core::prelude::Result<(
        RequestId,
        tokio_stream::wrappers::ReceiverStream<gpt_oss_core::prelude::RequestOutput>,
    )> {
        self.generate(prompt, params).await
    }
}

#[cfg(feature = "cuda")]
#[async_trait::async_trait]
impl InferenceEngine for gpt_oss_engine::AsyncGpuLLMEngine {
    async fn generate(
        &self,
        prompt: String,
        params: gpt_oss_core::prelude::SamplingParams,
    ) -> gpt_oss_core::prelude::Result<(
        RequestId,
        tokio_stream::wrappers::ReceiverStream<gpt_oss_core::prelude::RequestOutput>,
    )> {
        self.generate(prompt, params).await
    }
}

/// Shared application state available to all route handlers.
pub struct AppState {
    pub engine: Arc<dyn InferenceEngine>,
    pub model_name: String,
    pub runtime_decision: RuntimeDecision,
    pub tokenizer: Arc<RwLock<Tokenizer>>,
    /// Batch job store (None if batch API is not enabled).
    pub batch_store: Option<crate::routes::batch::SharedBatchStore>,
    /// Stored response objects for Responses API follow-up turns and retrieval.
    pub response_store: crate::routes::responses::SharedResponseStore,
    next_id: AtomicU64,
}

impl AppState {
    pub fn new(
        engine: Arc<dyn InferenceEngine>,
        model_name: String,
        runtime_decision: RuntimeDecision,
        tokenizer: Tokenizer,
    ) -> Self {
        Self {
            engine,
            model_name,
            runtime_decision,
            tokenizer: Arc::new(RwLock::new(tokenizer)),
            batch_store: Some(crate::routes::batch::create_batch_store(None)),
            response_store: Arc::new(RwLock::new(HashMap::new())),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn next_request_id(&self) -> RequestId {
        RequestId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }
}

/// Build the axum router with all API routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(
            "/v1/completions",
            post(routes::completions::create_completion),
        )
        .route(
            "/v1/chat/completions",
            post(routes::chat::create_chat_completion),
        )
        .route("/v1/responses", post(routes::responses::create_response))
        .route(
            "/v1/responses/:response_id",
            get(routes::responses::get_response),
        )
        .route(
            "/v1/responses/:response_id/input_items",
            get(routes::responses::list_response_input_items),
        )
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/batches", post(routes::batch::create_batch))
        .route("/v1/batches/:batch_id", get(routes::batch::get_batch))
        .route(
            "/v1/batches/:batch_id/output",
            get(routes::batch::get_batch_output),
        )
        .route(
            "/v1/batches/:batch_id/cancel",
            post(routes::batch::cancel_batch),
        )
        .route(
            "/v1/chat/completions/tools",
            post(routes::chat::create_chat_completion),
        )
        .route("/tools", post(routes::chat::create_chat_completion))
        .route("/health", get(routes::health::health_check))
        .route("/metrics", get(metrics_placeholder))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn metrics_placeholder() -> &'static str {
    "# gpt-oss-rs metrics endpoint\n"
}

#[derive(Debug, Clone, Copy)]
struct CudaRuntimeInfo {
    available: bool,
    primary_gpu_total_memory: Option<usize>,
}

fn detect_cuda_runtime() -> CudaRuntimeInfo {
    #[cfg(feature = "cuda")]
    {
        let devices = gpt_oss_gpu::prelude::list_devices();
        if devices.is_empty() {
            info!("cuda feature enabled but no CUDA devices found");
            return CudaRuntimeInfo {
                available: false,
                primary_gpu_total_memory: None,
            };
        }
        for dev in &devices {
            info!(
                id = dev.id, name = %dev.name,
                memory_gb = dev.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                "CUDA device available"
            );
        }
        CudaRuntimeInfo {
            available: true,
            primary_gpu_total_memory: devices.first().map(|d| d.total_memory),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        CudaRuntimeInfo {
            available: false,
            primary_gpu_total_memory: None,
        }
    }
}

pub async fn serve(config: EngineConfig) -> gpt_oss_core::prelude::Result<()> {
    gpt_oss_engine::config::validate(&config)
        .map_err(gpt_oss_core::prelude::LLMError::ConfigError)?;
    let model_name = config.model.model_path.clone();
    let tokenizer_path = config
        .model
        .tokenizer_path
        .clone()
        .unwrap_or_else(|| config.model.model_path.clone());

    info!(model = %model_name, "initializing engine");

    // Automatic GPT-OSS startup is CPU-first and must not initialize the CUDA
    // driver. Discovery remains available for explicit CUDA and `info`.
    let cuda_runtime = if config.device.device == "cuda" {
        detect_cuda_runtime()
    } else {
        CudaRuntimeInfo {
            available: false,
            primary_gpu_total_memory: None,
        }
    };
    let allow_long_context_override = std::env::var("GPT_OSS_RS_ALLOW_LONG_CONTEXT")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);

    let runtime_decision = validate_gpt_oss_runtime(
        &config.model.model_path,
        config.runtime_mode,
        &config.device.device,
        cuda_runtime.available,
        config.model.max_model_len,
        config.parallel.tensor_parallel_size,
        config.parallel.pipeline_parallel_size,
        config.scheduler.max_num_seqs,
        cuda_runtime.primary_gpu_total_memory,
        allow_long_context_override,
    )
    .map_err(gpt_oss_core::prelude::LLMError::ConfigError)?;
    info!(runtime = %runtime_decision.summary(), "resolved runtime path");

    let (engine, tokenizer): (Arc<dyn InferenceEngine>, Tokenizer) =
        match runtime_decision.backend_path {
            RuntimeBackendPath::Cuda => {
                info!("GPU-backed runtime selected, creating AsyncGpuLLMEngine");
                let tokenizer = Tokenizer::from_pretrained(&tokenizer_path)?;
                (create_gpu_engine(config).await?, tokenizer)
            }
            RuntimeBackendPath::Cpu => {
                info!("native CPU runtime selected, creating CpuWorker");
                create_cpu_engine(config).await?
            }
            RuntimeBackendPath::Mock => {
                info!("explicit mock runtime selected, creating AsyncLLMEngine");
                let tokenizer = Tokenizer::from_pretrained(&tokenizer_path)?;
                (
                    Arc::new(create_mock_engine(config, &tokenizer_path)?),
                    tokenizer,
                )
            }
        };

    let state = Arc::new(AppState::new(
        engine,
        model_name,
        runtime_decision,
        tokenizer,
    ));
    let app = build_router(state);

    let host = std::env::var("VLLM_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port = std::env::var("VLLM_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(8000);
    let addr = format!("{host}:{port}");
    info!(addr = %addr, "starting API server");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(gpt_oss_core::prelude::LLMError::IoError)?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(gpt_oss_core::prelude::LLMError::IoError)?;

    info!("server shut down gracefully");
    Ok(())
}

/// Create the real GPU engine using AsyncGpuLLMEngine.
#[cfg(feature = "cuda")]
async fn create_gpu_engine(
    config: EngineConfig,
) -> gpt_oss_core::prelude::Result<Arc<dyn InferenceEngine>> {
    let engine = gpt_oss_engine::AsyncGpuLLMEngine::new(config).await?;
    Ok(Arc::new(engine))
}

#[cfg(not(feature = "cuda"))]
async fn create_gpu_engine(
    _config: EngineConfig,
) -> gpt_oss_core::prelude::Result<Arc<dyn InferenceEngine>> {
    Err(gpt_oss_core::prelude::LLMError::GpuError(
        "CUDA not available".into(),
    ))
}

fn create_mock_engine(
    config: EngineConfig,
    tokenizer_path: &str,
) -> gpt_oss_core::prelude::Result<AsyncLLMEngine> {
    let tokenizer = Tokenizer::from_pretrained(tokenizer_path)?;

    let executor_config = ExecutorConfig {
        num_gpus: config.parallel.tensor_parallel_size,
        model_name: config.model.model_path.clone(),
        block_size: config.cache.block_size,
        gpu_memory_utilization: config.cache.gpu_memory_utilization,
        tensor_parallel_size: config.parallel.tensor_parallel_size,
        pipeline_parallel_size: config.parallel.pipeline_parallel_size,
    };
    let rt = tokio::runtime::Handle::current();
    let executor = ExecutorAdapter::from_config(executor_config, rt).map_err(|e| {
        gpt_oss_core::prelude::LLMError::ConfigError(format!("failed to create executor: {}", e))
    })?;

    let scheduler = Box::new(PlaceholderScheduler::new());

    let engine = AsyncLLMEngine::new(config, Box::new(executor), scheduler, tokenizer)?;
    Ok(engine)
}

async fn create_cpu_engine(
    mut config: EngineConfig,
) -> gpt_oss_core::prelude::Result<(Arc<dyn InferenceEngine>, Tokenizer)> {
    if !matches!(
        config.model.dtype,
        gpt_oss_core::types::Dtype::Auto | gpt_oss_core::types::Dtype::BFloat16
    ) {
        return Err(gpt_oss_core::prelude::LLMError::ConfigError(format!(
            "GPT-OSS CPU serving requires --dtype auto or bfloat16, got {}",
            config.model.dtype
        )));
    }
    let model = config.model.model_path.clone();
    let tokenizer_override = config.model.tokenizer_path.clone();
    let repack_cache = config.device.cpu_repack_cache.clone();
    let kernel_path = config
        .device
        .cpu_kernel
        .parse::<gpt_oss_cpu_kernels::KernelPath>()
        .map_err(|error| gpt_oss_core::prelude::LLMError::ConfigError(error.to_string()))?;
    let matmul_backend = config
        .device
        .cpu_matmul_backend
        .parse::<gpt_oss_cpu_kernels::Mxfp4MatmulBackend>()
        .map_err(|error| gpt_oss_core::prelude::LLMError::ConfigError(error.to_string()))?;
    let threads = config.device.cpu_threads;
    let context_cap = config.model.max_model_len;

    let (worker, snapshot) = tokio::task::spawn_blocking(move || {
        let model_path = std::path::Path::new(&model);
        let snapshot = if model_path.is_dir() {
            model_path.to_path_buf()
        } else {
            gpt_oss_engine::model_fetch::fetch_snapshot(
                &gpt_oss_engine::model_fetch::FetchOptions {
                    model,
                    revision: "main".into(),
                    cache_dir: None,
                },
            )?
            .snapshot_dir
        };
        let worker = CpuWorker::load(
            &snapshot,
            &repack_cache,
            kernel_path,
            matmul_backend,
            threads,
            context_cap,
        )?;
        Ok::<_, gpt_oss_core::prelude::LLMError>((worker, snapshot))
    })
    .await
    .map_err(|error| {
        gpt_oss_core::prelude::LLMError::ModelError(format!(
            "CPU model initialization task failed: {error}"
        ))
    })??;

    let tokenizer_source = match tokenizer_override {
        Some(path) => path,
        None => snapshot
            .to_str()
            .ok_or_else(|| {
                gpt_oss_core::prelude::LLMError::TokenizerError(
                    "CPU snapshot path is not valid UTF-8".into(),
                )
            })?
            .to_string(),
    };
    let engine_tokenizer = Tokenizer::from_pretrained(&tokenizer_source)?;
    let app_tokenizer = Tokenizer::from_pretrained(&tokenizer_source)?;
    let scheduler = Box::new(SingleRequestScheduler::new());
    config.device.device = "cpu".into();
    let engine = AsyncLLMEngine::new(config, Box::new(worker), scheduler, engine_tokenizer)?;
    Ok((Arc::new(engine), app_tokenizer))
}

struct PlaceholderScheduler {
    groups: Vec<gpt_oss_engine::sequence::SequenceGroup>,
}

/// FIFO scheduler that exposes only the front request to the batch-one CPU
/// executor. The engine removes that group when its output state finishes.
struct SingleRequestScheduler {
    groups: Vec<gpt_oss_engine::sequence::SequenceGroup>,
}

impl SingleRequestScheduler {
    fn new() -> Self {
        Self { groups: Vec::new() }
    }
}

impl gpt_oss_engine::Scheduler for SingleRequestScheduler {
    fn add_seq_group(&mut self, seq_group: gpt_oss_engine::sequence::SequenceGroup) {
        self.groups.push(seq_group);
    }

    fn abort_seq_group(&mut self, request_id: &RequestId) {
        self.groups.retain(|group| group.request_id != *request_id);
    }

    fn schedule(&mut self) -> gpt_oss_engine::SchedulerOutputs {
        let scheduled_seq_groups = self.groups.first().cloned().into_iter().collect::<Vec<_>>();
        let num_batched_tokens = scheduled_seq_groups
            .first()
            .map(|group| {
                group
                    .get_seqs()
                    .iter()
                    .map(|sequence| sequence.num_new_tokens().max(1))
                    .sum()
            })
            .unwrap_or(0);
        gpt_oss_engine::SchedulerOutputs {
            scheduled_seq_groups,
            num_batched_tokens,
            preempted: false,
        }
    }

    fn has_unfinished_seqs(&self) -> bool {
        !self.groups.is_empty()
    }

    fn get_num_unfinished_seq_groups(&self) -> usize {
        self.groups.len()
    }
}

impl PlaceholderScheduler {
    fn new() -> Self {
        Self { groups: Vec::new() }
    }
}

impl gpt_oss_engine::Scheduler for PlaceholderScheduler {
    fn add_seq_group(&mut self, seq_group: gpt_oss_engine::sequence::SequenceGroup) {
        self.groups.push(seq_group);
    }

    fn abort_seq_group(&mut self, request_id: &RequestId) {
        self.groups.retain(|g| g.request_id != *request_id);
    }

    fn schedule(&mut self) -> gpt_oss_engine::SchedulerOutputs {
        let groups = self.groups.clone();
        self.groups.retain(|g| !g.is_finished());
        let num_tokens = groups
            .iter()
            .flat_map(|g| g.get_seqs())
            .map(|s| s.num_new_tokens().max(1))
            .sum();
        gpt_oss_engine::SchedulerOutputs {
            scheduled_seq_groups: groups,
            num_batched_tokens: num_tokens,
            preempted: false,
        }
    }

    fn has_unfinished_seqs(&self) -> bool {
        !self.groups.is_empty()
    }
    fn get_num_unfinished_seq_groups(&self) -> usize {
        self.groups.len()
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => { info!("received Ctrl+C, shutting down"); }
        _ = terminate => { info!("received SIGTERM, shutting down"); }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use gpt_oss_core::prelude::{SamplingParams, SequenceId};
    use gpt_oss_engine::{Scheduler, Sequence, SequenceGroup};

    use super::*;

    fn group(request_id: u64) -> SequenceGroup {
        SequenceGroup::new(
            RequestId(request_id),
            vec![Sequence::new(SequenceId(request_id), vec![1, 2])],
            SamplingParams::default(),
            Instant::now(),
            "prompt".into(),
        )
    }

    #[test]
    fn single_request_scheduler_is_fifo_and_never_batches() {
        let mut scheduler = SingleRequestScheduler::new();
        scheduler.add_seq_group(group(1));
        scheduler.add_seq_group(group(2));
        let first = scheduler.schedule();
        assert_eq!(first.scheduled_seq_groups.len(), 1);
        assert_eq!(first.scheduled_seq_groups[0].request_id, RequestId(1));
        scheduler.abort_seq_group(&RequestId(1));
        let second = scheduler.schedule();
        assert_eq!(second.scheduled_seq_groups[0].request_id, RequestId(2));
    }

    #[tokio::test]
    async fn cpu_engine_rejects_non_bf16_request_before_loading() {
        let mut config = EngineConfig::default();
        config.model.model_path = "openai/gpt-oss-20b".into();
        config.model.dtype = gpt_oss_core::types::Dtype::Float16;
        let error = match create_cpu_engine(config).await {
            Ok(_) => panic!("float16 CPU request unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("auto or bfloat16"));
    }
}
