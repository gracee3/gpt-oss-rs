//! HTTP server setup, AppState, router construction, and graceful shutdown.
//!
//! Device policy selects the CUDA engine, native batched GPT-OSS CPU runtime,
//! or an explicitly requested test-only mock executor.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::DefaultBodyLimit;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use metrics_exporter_prometheus::PrometheusHandle;
use tokio::sync::RwLock;
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use gpt_oss_core::prelude::{CompletionOutput, FinishReason, RequestId, RequestOutput};
use gpt_oss_engine::config::EngineConfig;
use gpt_oss_engine::AsyncLLMEngine;
use gpt_oss_engine::{
    AsyncCpuBatchEngine, CommittedEvent, CpuBatchEngine, CpuExpertProjection, CpuModel,
    CpuTopology, ExecutorAdapter, ExecutorConfig, FailurePhase, ManagedRequest, ServiceLifecycle,
    ServiceState, StableFailure, StableFailureCode,
};
use gpt_oss_evidence::{
    DiagnosticConfig, DiagnosticMode, DiagnosticRecord, DiagnosticSink, EffectiveRuntimeSnapshot,
};
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
    ) -> Result<InferenceStream, StableFailure>;

    async fn shutdown(&self) -> Result<(), StableFailure> {
        Ok(())
    }

    fn begin_shutdown(&self) -> Result<(), StableFailure> {
        Ok(())
    }
}

pub enum InferenceStream {
    Cpu(CpuEventStream),
    Cumulative(CumulativeOutputStream),
}

pub struct CpuEventStream {
    request: ManagedRequest,
    prompt_tokens: usize,
    completion_tokens: usize,
    pending_finish: Option<(u32, FinishReason)>,
}

pub struct CumulativeOutputStream {
    request_id: RequestId,
    inner: tokio_stream::wrappers::ReceiverStream<RequestOutput>,
    cursors: HashMap<usize, (String, Vec<u32>, usize, f32)>,
    metadata_published: bool,
}

impl InferenceStream {
    pub fn from_cumulative(
        request_id: RequestId,
        inner: tokio_stream::wrappers::ReceiverStream<RequestOutput>,
    ) -> Self {
        Self::Cumulative(CumulativeOutputStream {
            request_id,
            inner,
            cursors: HashMap::new(),
            metadata_published: false,
        })
    }

    pub fn request_id(&self) -> RequestId {
        match self {
            Self::Cpu(stream) => stream.request.request_id,
            Self::Cumulative(stream) => stream.request_id,
        }
    }

    pub async fn recv(&mut self) -> Result<Option<RequestOutput>, StableFailure> {
        match self {
            Self::Cpu(stream) => stream.recv().await,
            Self::Cumulative(stream) => stream.recv().await,
        }
    }
}

impl CpuEventStream {
    async fn recv(&mut self) -> Result<Option<RequestOutput>, StableFailure> {
        loop {
            match self.request.events.recv().await {
                Some(CommittedEvent::Delta {
                    choice,
                    text,
                    token_ids,
                    logprobs,
                }) => {
                    return Ok(Some(RequestOutput {
                        request_id: self.request.request_id,
                        prompt: String::new(),
                        prompt_token_ids: Vec::new(),
                        prompt_logprobs: None,
                        outputs: vec![CompletionOutput {
                            index: choice as usize,
                            text,
                            token_ids,
                            cumulative_logprob: 0.0,
                            logprobs,
                            finish_reason: None,
                        }],
                        finished: false,
                    }));
                }
                Some(CommittedEvent::Usage {
                    committed_prompt,
                    committed_completion,
                }) => {
                    self.prompt_tokens = committed_prompt as usize;
                    self.completion_tokens = committed_completion as usize;
                }
                Some(CommittedEvent::Finish { choice, reason }) => {
                    if reason == FinishReason::Abort {
                        return Err(StableFailure::new(
                            StableFailureCode::ClientCancelled,
                            FailurePhase::Delivery,
                            false,
                            "aborted generation is not a model finish",
                        ));
                    }
                    self.pending_finish = Some((choice, reason));
                }
                Some(CommittedEvent::Error { failure }) => return Err(failure),
                Some(CommittedEvent::Done) => {
                    let (choice, reason) = self.pending_finish.take().ok_or_else(|| {
                        StableFailure::new(
                            StableFailureCode::SerializationFailed,
                            FailurePhase::Delivery,
                            false,
                            "native CPU delivery committed Done before Finish",
                        )
                    })?;
                    return Ok(Some(RequestOutput {
                        request_id: self.request.request_id,
                        prompt: String::new(),
                        prompt_token_ids: vec![0; self.prompt_tokens],
                        prompt_logprobs: None,
                        outputs: vec![CompletionOutput {
                            index: choice as usize,
                            text: String::new(),
                            token_ids: Vec::new(),
                            cumulative_logprob: 0.0,
                            logprobs: None,
                            finish_reason: Some(reason),
                        }],
                        finished: true,
                    }));
                }
                None => {
                    let status = self.request.lifecycle.status();
                    return Err(status.failure.unwrap_or_else(|| {
                        StableFailure::new(
                            StableFailureCode::OwnerStopped,
                            FailurePhase::Delivery,
                            false,
                            "native CPU owner closed delivery before Done",
                        )
                    }));
                }
            }
        }
    }
}

impl CumulativeOutputStream {
    async fn recv(&mut self) -> Result<Option<RequestOutput>, StableFailure> {
        let Some(output) = self.inner.next().await else {
            return Ok(None);
        };
        if output
            .outputs
            .iter()
            .any(|choice| choice.finish_reason == Some(FinishReason::Abort))
        {
            return Err(StableFailure::new(
                StableFailureCode::ClientCancelled,
                FailurePhase::Delivery,
                false,
                "aborted generation is not a model finish",
            ));
        }
        let mut deltas = Vec::with_capacity(output.outputs.len());
        for choice in &output.outputs {
            let cursor =
                self.cursors
                    .entry(choice.index)
                    .or_insert((String::new(), Vec::new(), 0, 0.0));
            if !choice.text.starts_with(&cursor.0)
                || !choice.token_ids.starts_with(&cursor.1)
                || cursor.2 > choice.logprobs.as_ref().map_or(0, Vec::len)
            {
                return Err(StableFailure::new(
                    StableFailureCode::ExecutionFailed,
                    FailurePhase::Commit,
                    false,
                    "backend output retracted committed content",
                ));
            }
            let logprobs = choice
                .logprobs
                .as_ref()
                .map(|values| values[cursor.2..].to_vec());
            deltas.push(CompletionOutput {
                index: choice.index,
                text: choice.text[cursor.0.len()..].to_string(),
                token_ids: choice.token_ids[cursor.1.len()..].to_vec(),
                cumulative_logprob: choice.cumulative_logprob - cursor.3,
                logprobs,
                finish_reason: choice.finish_reason,
            });
            *cursor = (
                choice.text.clone(),
                choice.token_ids.clone(),
                choice.logprobs.as_ref().map_or(0, Vec::len),
                choice.cumulative_logprob,
            );
        }
        let delta = RequestOutput {
            request_id: output.request_id,
            prompt: if self.metadata_published {
                String::new()
            } else {
                output.prompt.clone()
            },
            prompt_token_ids: if self.metadata_published {
                Vec::new()
            } else {
                output.prompt_token_ids.clone()
            },
            prompt_logprobs: (!self.metadata_published)
                .then(|| output.prompt_logprobs.clone())
                .flatten(),
            outputs: deltas,
            finished: output.finished,
        };
        self.metadata_published = true;
        Ok(Some(delta))
    }
}

#[derive(Debug, Default)]
pub struct RequestOutputAccumulator {
    output: Option<RequestOutput>,
}

pub fn ensure_non_streaming_size<T: serde::Serialize>(
    value: &T,
    limit: usize,
) -> Result<(), crate::error::ApiError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        crate::error::ApiError::from(StableFailure::new(
            StableFailureCode::SerializationFailed,
            FailurePhase::Delivery,
            false,
            error.to_string(),
        ))
    })?;
    if bytes.len() > limit {
        return Err(crate::error::ApiError::from(StableFailure::new(
            StableFailureCode::OverloadedDelivery,
            FailurePhase::Delivery,
            true,
            "non-streaming response exceeds configured byte limit",
        )));
    }
    Ok(())
}

#[derive(Clone)]
pub struct BoundedSseSender {
    inner: tokio::sync::mpsc::Sender<Result<String, std::convert::Infallible>>,
    max_event_bytes: usize,
}

impl BoundedSseSender {
    pub fn new(
        inner: tokio::sync::mpsc::Sender<Result<String, std::convert::Infallible>>,
        max_event_bytes: usize,
    ) -> Self {
        Self {
            inner,
            max_event_bytes,
        }
    }

    pub async fn send(&self, event: Result<String, std::convert::Infallible>) -> Result<(), ()> {
        let Ok(event) = event;
        if event.len() > self.max_event_bytes {
            let failure = StableFailure::new(
                StableFailureCode::OverloadedDelivery,
                FailurePhase::Delivery,
                true,
                "serialized stream event exceeds configured byte limit",
            );
            let terminal = crate::types::streaming::format_sse_failure(&failure);
            if terminal.len() <= self.max_event_bytes {
                let _ = self.inner.send(Ok(terminal)).await;
            }
            return Err(());
        }
        self.inner.send(Ok(event)).await.map_err(|_| ())
    }
}

impl RequestOutputAccumulator {
    pub fn push(&mut self, delta: RequestOutput) -> Result<(), StableFailure> {
        let output = self.output.get_or_insert_with(|| RequestOutput {
            request_id: delta.request_id,
            prompt: String::new(),
            prompt_token_ids: Vec::new(),
            prompt_logprobs: None,
            outputs: Vec::new(),
            finished: false,
        });
        if output.request_id != delta.request_id {
            return Err(StableFailure::new(
                StableFailureCode::ExecutionFailed,
                FailurePhase::Delivery,
                false,
                "request stream mixed request IDs",
            ));
        }
        if !delta.prompt.is_empty() {
            output.prompt.push_str(&delta.prompt);
        }
        if !delta.prompt_token_ids.is_empty() {
            output.prompt_token_ids = delta.prompt_token_ids;
        }
        if delta.prompt_logprobs.is_some() {
            output.prompt_logprobs = delta.prompt_logprobs;
        }
        for choice_delta in delta.outputs {
            if output.outputs.len() <= choice_delta.index {
                output
                    .outputs
                    .resize_with(choice_delta.index + 1, || CompletionOutput {
                        index: 0,
                        text: String::new(),
                        token_ids: Vec::new(),
                        cumulative_logprob: 0.0,
                        logprobs: None,
                        finish_reason: None,
                    });
                for (index, choice) in output.outputs.iter_mut().enumerate() {
                    choice.index = index;
                }
            }
            let choice = &mut output.outputs[choice_delta.index];
            choice.text.push_str(&choice_delta.text);
            choice.token_ids.extend(choice_delta.token_ids);
            choice.cumulative_logprob += choice_delta.cumulative_logprob;
            if let Some(values) = choice_delta.logprobs {
                choice.logprobs.get_or_insert_with(Vec::new).extend(values);
            }
            if choice_delta.finish_reason.is_some() {
                choice.finish_reason = choice_delta.finish_reason;
            }
        }
        output.finished |= delta.finished;
        Ok(())
    }

    pub fn finish(self) -> Option<RequestOutput> {
        self.output
    }
}

#[async_trait::async_trait]
impl InferenceEngine for AsyncLLMEngine {
    async fn generate(
        &self,
        prompt: String,
        params: gpt_oss_core::prelude::SamplingParams,
    ) -> Result<InferenceStream, StableFailure> {
        let (request_id, inner) = self
            .generate(prompt, params)
            .await
            .map_err(stable_engine_error)?;
        Ok(InferenceStream::from_cumulative(request_id, inner))
    }
}

#[async_trait::async_trait]
impl InferenceEngine for AsyncCpuBatchEngine {
    async fn generate(
        &self,
        prompt: String,
        params: gpt_oss_core::prelude::SamplingParams,
    ) -> Result<InferenceStream, StableFailure> {
        let request = AsyncCpuBatchEngine::generate(self, prompt, params).await?;
        Ok(InferenceStream::Cpu(CpuEventStream {
            request,
            prompt_tokens: 0,
            completion_tokens: 0,
            pending_finish: None,
        }))
    }

    async fn shutdown(&self) -> Result<(), StableFailure> {
        AsyncCpuBatchEngine::shutdown(self).await
    }

    fn begin_shutdown(&self) -> Result<(), StableFailure> {
        AsyncCpuBatchEngine::begin_shutdown(self)
    }
}

#[cfg(feature = "cuda")]
#[async_trait::async_trait]
impl InferenceEngine for gpt_oss_engine::AsyncGpuLLMEngine {
    async fn generate(
        &self,
        prompt: String,
        params: gpt_oss_core::prelude::SamplingParams,
    ) -> Result<InferenceStream, StableFailure> {
        let (request_id, inner) = self
            .generate(prompt, params)
            .await
            .map_err(stable_engine_error)?;
        Ok(InferenceStream::from_cumulative(request_id, inner))
    }
}

fn stable_engine_error(error: gpt_oss_core::prelude::LLMError) -> StableFailure {
    StableFailure::new(
        StableFailureCode::ExecutionFailed,
        FailurePhase::Execution,
        false,
        error.to_string(),
    )
}

/// Shared application state available to all route handlers.
pub struct AppState {
    engine: Arc<RwLock<Option<Arc<dyn InferenceEngine>>>>,
    pub model_name: String,
    pub runtime_decision: RuntimeDecision,
    pub tokenizer: Arc<RwLock<Option<Tokenizer>>>,
    /// Batch job store (None if batch API is not enabled).
    pub batch_store: Option<crate::routes::batch::SharedBatchStore>,
    /// Stored response objects for Responses API follow-up turns and retrieval.
    pub response_store: crate::routes::responses::SharedResponseStore,
    pub lifecycle: ServiceLifecycle,
    pub metrics: Option<PrometheusHandle>,
    pub limits: ServiceLimits,
    next_id: AtomicU64,
}

impl AppState {
    pub fn new(
        engine: Arc<dyn InferenceEngine>,
        model_name: String,
        runtime_decision: RuntimeDecision,
        tokenizer: Tokenizer,
    ) -> Self {
        let lifecycle = ServiceLifecycle::starting(model_name.clone());
        lifecycle
            .mark_ready("0".repeat(64))
            .expect("static test runtime hash is valid");
        Self::with_service(
            engine,
            model_name,
            runtime_decision,
            tokenizer,
            lifecycle,
            None,
            ServiceLimits::for_max_num_seqs(1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_service(
        engine: Arc<dyn InferenceEngine>,
        model_name: String,
        runtime_decision: RuntimeDecision,
        tokenizer: Tokenizer,
        lifecycle: ServiceLifecycle,
        metrics: Option<PrometheusHandle>,
        limits: ServiceLimits,
    ) -> Self {
        Self {
            engine: Arc::new(RwLock::new(Some(engine))),
            model_name,
            runtime_decision,
            tokenizer: Arc::new(RwLock::new(Some(tokenizer))),
            batch_store: None,
            response_store: Arc::new(RwLock::new(
                crate::routes::responses::BoundedResponseStore::new(
                    limits.response_store_bytes,
                    limits.response_store_entries,
                    limits.max_store_entry_bytes,
                ),
            )),
            lifecycle,
            metrics,
            limits,
            next_id: AtomicU64::new(1),
        }
    }

    pub fn next_request_id(&self) -> RequestId {
        RequestId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    fn starting(
        model_name: String,
        runtime_decision: RuntimeDecision,
        lifecycle: ServiceLifecycle,
        metrics: Option<PrometheusHandle>,
        limits: ServiceLimits,
    ) -> Self {
        Self {
            engine: Arc::new(RwLock::new(None)),
            model_name,
            runtime_decision,
            tokenizer: Arc::new(RwLock::new(None)),
            batch_store: None,
            response_store: Arc::new(RwLock::new(
                crate::routes::responses::BoundedResponseStore::new(
                    limits.response_store_bytes,
                    limits.response_store_entries,
                    limits.max_store_entry_bytes,
                ),
            )),
            lifecycle,
            metrics,
            limits,
            next_id: AtomicU64::new(1),
        }
    }

    pub async fn engine(&self) -> Result<Arc<dyn InferenceEngine>, StableFailure> {
        let status = self.lifecycle.status();
        if status.state != ServiceState::Ready {
            return Err(StableFailure::unavailable(status.state));
        }
        self.installed_engine().await.ok_or_else(|| {
            StableFailure::new(
                StableFailureCode::NotReady,
                FailurePhase::Admission,
                true,
                "inference engine is not installed",
            )
        })
    }

    async fn installed_engine(&self) -> Option<Arc<dyn InferenceEngine>> {
        self.engine.read().await.clone()
    }

    async fn install_runtime(
        &self,
        engine: Arc<dyn InferenceEngine>,
        tokenizer: Tokenizer,
    ) -> Result<(), StableFailure> {
        let mut engine_slot = self.engine.write().await;
        let mut tokenizer_slot = self.tokenizer.write().await;
        if engine_slot.is_some() || tokenizer_slot.is_some() {
            return Err(StableFailure::new(
                StableFailureCode::InvalidRequest,
                FailurePhase::Startup,
                false,
                "service runtime was already installed",
            ));
        }
        *engine_slot = Some(engine);
        *tokenizer_slot = Some(tokenizer);
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct EvidenceConfig {
    pub directory: Option<PathBuf>,
    pub diagnostics: DiagnosticConfig,
}

impl Default for EvidenceConfig {
    fn default() -> Self {
        Self {
            directory: None,
            diagnostics: DiagnosticConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ServiceLimits {
    pub request_body_bytes: usize,
    pub max_non_streaming_bytes: usize,
    pub max_stream_event_bytes: usize,
    pub per_request_delivery_bytes: usize,
    pub terminal_control_allowance: usize,
    pub global_delivery_bytes: usize,
    pub max_admitted_requests: usize,
    pub response_store_bytes: usize,
    pub response_store_entries: usize,
    pub max_store_entry_bytes: usize,
    pub max_logprobs: usize,
    pub drain_deadline: Duration,
    pub cpu_request_budget_bytes: Option<u128>,
}

impl ServiceLimits {
    pub fn for_max_num_seqs(max_num_seqs: usize) -> Self {
        const MIB: usize = 1024 * 1024;
        Self {
            request_body_bytes: 2 * MIB,
            max_non_streaming_bytes: 8 * MIB,
            max_stream_event_bytes: 256 * 1024,
            per_request_delivery_bytes: MIB,
            terminal_control_allowance: 16 * 1024,
            global_delivery_bytes: max_num_seqs.saturating_mul(MIB),
            max_admitted_requests: max_num_seqs,
            response_store_bytes: 64 * MIB,
            response_store_entries: 64,
            max_store_entry_bytes: 8 * MIB,
            max_logprobs: 20,
            drain_deadline: Duration::from_secs(30),
            cpu_request_budget_bytes: None,
        }
    }

    pub fn delivery_limits(self) -> gpt_oss_engine::DeliveryLimits {
        gpt_oss_engine::DeliveryLimits {
            per_request_queued_bytes: self.per_request_delivery_bytes,
            terminal_control_allowance: self.terminal_control_allowance,
            global_queued_bytes: self.global_delivery_bytes,
            max_event_bytes: self.max_stream_event_bytes,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_address: SocketAddr,
    pub served_model_name: Option<String>,
    pub limits: ServiceLimits,
    pub evidence: EvidenceConfig,
    pub engine: EngineConfig,
}

/// Build the axum router with all API routes.
pub fn build_router(state: Arc<AppState>) -> Router {
    let body_limit = state.limits.request_body_bytes;
    let metrics_enabled = state.metrics.is_some();
    let mut router = Router::new()
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
        .route(
            "/v1/chat/completions/tools",
            post(routes::chat::create_chat_completion),
        )
        .route("/tools", post(routes::chat::create_chat_completion))
        .route("/health", get(routes::health::health_check))
        .route("/ready", get(routes::health::ready_check))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .layer(DefaultBodyLimit::max(body_limit))
        .layer(axum::middleware::from_fn(typed_body_limit_response))
        .fallback(api_not_found);
    if metrics_enabled {
        router = router.route("/metrics", get(metrics_handler));
    }
    router.with_state(state)
}

async fn metrics_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> impl IntoResponse {
    match &state.metrics {
        Some(handle) => (
            axum::http::StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            handle.render(),
        )
            .into_response(),
        None => axum::http::StatusCode::NOT_FOUND.into_response(),
    }
}

async fn api_not_found() -> impl IntoResponse {
    crate::error::ApiError::NotFound("route not found".into())
}

async fn typed_body_limit_response(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let response = next.run(request).await;
    if response.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
        crate::error::ApiError::from(StableFailure::new(
            StableFailureCode::BodyTooLarge,
            FailurePhase::Envelope,
            false,
            "request body exceeds configured byte limit",
        ))
        .into_response()
    } else {
        response
    }
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

pub async fn serve(server_config: ServerConfig) -> gpt_oss_core::prelude::Result<()> {
    let config = server_config.engine;
    gpt_oss_engine::config::validate(&config)
        .map_err(gpt_oss_core::prelude::LLMError::ConfigError)?;
    let model_name = resolve_served_model_id(
        &config.model.model_path,
        server_config.served_model_name.as_deref(),
    )?;
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

    let metrics = if config.telemetry.enabled {
        let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .install_recorder()
            .map_err(|error| {
                gpt_oss_core::prelude::LLMError::ConfigError(format!(
                    "failed to install Prometheus recorder: {error}"
                ))
            })?;
        gpt_oss_engine::telemetry::metrics::register_descriptions();
        Some(handle)
    } else {
        None
    };
    let backend_class = match runtime_decision.backend_path {
        RuntimeBackendPath::Cpu => gpt_oss_engine::telemetry::metrics::BackendClass::Cpu,
        RuntimeBackendPath::Cuda => gpt_oss_engine::telemetry::metrics::BackendClass::Cuda,
        RuntimeBackendPath::Mock => gpt_oss_engine::telemetry::metrics::BackendClass::Mock,
    };
    gpt_oss_engine::telemetry::metrics::record_dispatch(
        backend_class,
        gpt_oss_engine::telemetry::metrics::DispatchResult::Selected,
        gpt_oss_engine::telemetry::metrics::ReasonCode::None,
    );

    let lifecycle = ServiceLifecycle::starting(model_name.clone());
    let limits = server_config.limits;
    let state = Arc::new(AppState::starting(
        model_name.clone(),
        runtime_decision.clone(),
        lifecycle.clone(),
        metrics,
        limits,
    ));
    let app = build_router(state.clone());
    let addr = server_config.bind_address;
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(gpt_oss_core::prelude::LLMError::IoError)?;
    info!(addr = %addr, "API listener bound; service is starting");

    let (startup_stop_tx, startup_stop_rx) = tokio::sync::watch::channel(false);
    let shutdown_state = state.clone();
    let server_task = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(service_shutdown(shutdown_state, startup_stop_rx))
            .await
    });

    let initialized: gpt_oss_core::prelude::Result<(Arc<dyn InferenceEngine>, Tokenizer)> =
        match runtime_decision.backend_path {
            RuntimeBackendPath::Cuda => {
                info!("GPU-backed runtime selected, creating AsyncGpuLLMEngine");
                match Tokenizer::from_pretrained(&tokenizer_path) {
                    Ok(tokenizer) => create_gpu_engine(config.clone())
                        .await
                        .map(|engine| (engine, tokenizer)),
                    Err(error) => Err(error),
                }
            }
            RuntimeBackendPath::Cpu => {
                info!("native CPU runtime selected, creating batched CPU engine");
                create_cpu_engine(config.clone(), lifecycle.clone(), limits).await
            }
            RuntimeBackendPath::Mock => {
                info!("explicit mock runtime selected, creating AsyncLLMEngine");
                match Tokenizer::from_pretrained(&tokenizer_path) {
                    Ok(tokenizer) => create_mock_engine(config.clone(), &tokenizer_path)
                        .map(|engine| (Arc::new(engine) as Arc<dyn InferenceEngine>, tokenizer)),
                    Err(error) => Err(error),
                }
            }
        };
    let (engine, tokenizer) = match initialized {
        Ok(runtime) => runtime,
        Err(error) => {
            let failure = StableFailure::new(
                StableFailureCode::EngineFailed,
                FailurePhase::Startup,
                false,
                error.to_string(),
            );
            let _ = lifecycle.mark_failed(failure);
            let _ = startup_stop_tx.send(true);
            let _ = server_task.await;
            return Err(error);
        }
    };
    let finalize_startup = async {
        state
            .install_runtime(engine, tokenizer)
            .await
            .map_err(|failure| gpt_oss_core::prelude::LLMError::ConfigError(failure.to_string()))?;

        let mut snapshot = EffectiveRuntimeSnapshot::default();
        snapshot
            .requested
            .insert("device".into(), serde_json::json!(config.device.device));
        snapshot.requested.insert(
            "context".into(),
            serde_json::json!(config.model.max_model_len),
        );
        snapshot.requested.insert(
            "concurrency".into(),
            serde_json::json!(config.scheduler.max_num_seqs),
        );
        snapshot.effective.insert(
            "backend".into(),
            serde_json::json!(runtime_decision.backend_path.as_str()),
        );
        snapshot.effective.insert(
            "context".into(),
            serde_json::json!(config.model.max_model_len),
        );
        snapshot.effective.insert(
            "service_limits".into(),
            serde_json::json!({
                "request_body_bytes": limits.request_body_bytes,
                "max_non_streaming_bytes": limits.max_non_streaming_bytes,
                "max_stream_event_bytes": limits.max_stream_event_bytes,
                "per_request_delivery_bytes": limits.per_request_delivery_bytes,
                "terminal_control_allowance": limits.terminal_control_allowance,
                "global_delivery_bytes": limits.global_delivery_bytes,
                "max_admitted_requests": limits.max_admitted_requests,
                "response_store_bytes": limits.response_store_bytes,
                "response_store_entries": limits.response_store_entries,
                "max_store_entry_bytes": limits.max_store_entry_bytes,
                "max_logprobs": limits.max_logprobs,
                "drain_deadline_ms": limits.drain_deadline.as_millis(),
                "cpu_request_budget_bytes": limits.cpu_request_budget_bytes,
            }),
        );
        snapshot.effective.insert(
            "diagnostics".into(),
            serde_json::json!({
                "mode": server_config.evidence.diagnostics.mode,
                "byte_cap": server_config.evidence.diagnostics.byte_cap,
                "boundary": server_config.evidence.diagnostics.boundary.as_deref(),
            }),
        );
        snapshot
            .identity
            .insert("served_model_id".into(), serde_json::json!(model_name));
        snapshot.capability.insert(
            "architecture".into(),
            serde_json::json!(std::env::consts::ARCH),
        );
        snapshot
            .capability
            .insert("os".into(), serde_json::json!(std::env::consts::OS));
        match gpt_oss_engine::SmapsRollup::sample_self() {
            Ok(sample) => {
                snapshot.capability.insert(
                    "linux_smaps_rollup_bytes".into(),
                    serde_json::to_value(sample.fields_bytes).map_err(|error| {
                        gpt_oss_core::prelude::LLMError::SerializationError(error.to_string())
                    })?,
                );
            }
            Err(_) => snapshot.omissions.push("smaps_rollup_unavailable".into()),
        }
        snapshot
            .omissions
            .push("allocator_introspection_unavailable".into());
        let snapshot_hash = snapshot.sha256().map_err(|error| {
            gpt_oss_core::prelude::LLMError::SerializationError(error.to_string())
        })?;
        if let Some(directory) = &server_config.evidence.directory {
            snapshot
                .write_atomic(directory.join("effective-runtime.json"))
                .map_err(|error| {
                    gpt_oss_core::prelude::LLMError::IoError(std::io::Error::other(error))
                })?;
        }
        if server_config.evidence.diagnostics.mode != DiagnosticMode::Off {
            let mut sink = DiagnosticSink::open(
                &server_config.evidence.diagnostics,
                &format!("service-{}.jsonl", std::process::id()),
                true,
            )
            .map_err(|error| {
                gpt_oss_core::prelude::LLMError::IoError(std::io::Error::other(error))
            })?;
            let mut record = DiagnosticRecord::new("runtime_ready", 0);
            record.fields.insert(
                "runtime_snapshot_sha256".into(),
                snapshot_hash.clone().into(),
            );
            record
                .fields
                .insert("served_model_id".into(), model_name.clone().into());
            record.fields.insert(
                "backend".into(),
                runtime_decision.backend_path.as_str().into(),
            );
            let _ = sink.write(&record).map_err(|error| {
                gpt_oss_core::prelude::LLMError::IoError(std::io::Error::other(error))
            })?;
            sink.flush().map_err(|error| {
                gpt_oss_core::prelude::LLMError::IoError(std::io::Error::other(error))
            })?;
        }
        lifecycle
            .mark_ready(snapshot_hash)
            .map_err(|failure| gpt_oss_core::prelude::LLMError::ConfigError(failure.to_string()))
    }
    .await;
    if let Err(error) = finalize_startup {
        let _ = lifecycle.mark_failed(StableFailure::new(
            StableFailureCode::EngineFailed,
            FailurePhase::Startup,
            false,
            error.to_string(),
        ));
        let _ = startup_stop_tx.send(true);
        let _ = server_task.await;
        if let Some(engine) = state.installed_engine().await {
            let _ = engine.shutdown().await;
        }
        return Err(error);
    }

    info!(addr = %addr, "API service is ready");
    let serve_result = server_task
        .await
        .map_err(|error| {
            gpt_oss_core::prelude::LLMError::SchedulerError(format!(
                "HTTP server task failed: {error}"
            ))
        })?
        .map_err(gpt_oss_core::prelude::LLMError::IoError);

    if lifecycle.status().state == ServiceState::Ready {
        let _ = lifecycle.begin_draining();
    }
    if let Some(engine) = state.installed_engine().await {
        engine.shutdown().await.map_err(|failure| {
            gpt_oss_core::prelude::LLMError::SchedulerError(failure.to_string())
        })?;
    }
    if lifecycle.status().state == ServiceState::Draining {
        let _ = lifecycle.mark_stopped();
    }
    serve_result?;

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
    lifecycle: ServiceLifecycle,
    limits: ServiceLimits,
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
    let topology = CpuTopology::observe(threads);
    info!(
        topology = %topology,
        allowed_cpus = ?topology.allowed_cpus,
        allowed_memory_nodes = ?topology.allowed_memory_nodes,
        "observed CPU topology without applying placement policy"
    );

    let (cpu_model, snapshot) = tokio::task::spawn_blocking(move || {
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
        let cpu_model = CpuModel::load_with_matmul_backend(
            &snapshot,
            &repack_cache,
            kernel_path,
            threads,
            CpuExpertProjection::default(),
            matmul_backend,
        )?;
        Ok::<_, gpt_oss_core::prelude::LLMError>((cpu_model, snapshot))
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
    let dispatch = cpu_model.kernel_dispatch_plan();
    info!(
        kernel = %cpu_model.kernel_path(),
        dispatch = %dispatch,
        mxfp4_gemv = %dispatch.mxfp4_gemv(),
        mxfp4_weight_layout = %cpu_model.mxfp4_weight_layout(),
        mxfp4_matmul_backend = %cpu_model.matmul_backend(),
        amx_runtime = ?cpu_model.amx_runtime_status().map(|status| status.to_string()),
        context_cap,
        "loaded native CPU model"
    );
    config.device.device = "cpu".into();
    let engine = CpuBatchEngine::new_with_request_budget(
        config,
        cpu_model,
        engine_tokenizer,
        limits.cpu_request_budget_bytes,
    )?;
    let managed = AsyncCpuBatchEngine::with_service(
        engine,
        lifecycle,
        limits.delivery_limits(),
        limits.drain_deadline,
    )
    .map_err(|failure| gpt_oss_core::prelude::LLMError::ConfigError(failure.to_string()))?;
    Ok((Arc::new(managed), app_tokenizer))
}

pub fn resolve_served_model_id(
    source_model: &str,
    explicit: Option<&str>,
) -> gpt_oss_core::prelude::Result<String> {
    if let Some(alias) = explicit {
        if alias.trim().is_empty() || looks_like_local_path(alias) {
            return Err(gpt_oss_core::prelude::LLMError::ConfigError(
                "served model name must be a non-empty public alias".into(),
            ));
        }
        return Ok(alias.to_string());
    }
    let source = Path::new(source_model);
    if source.is_dir() {
        let manifest = source.join(gpt_oss_engine::model_fetch::FETCH_MANIFEST_FILENAME);
        if manifest.is_file() {
            let parsed: gpt_oss_engine::model_fetch::SnapshotManifest =
                serde_json::from_slice(&std::fs::read(&manifest)?).map_err(|error| {
                    gpt_oss_core::prelude::LLMError::ConfigError(format!(
                        "invalid fetch manifest {}: {error}",
                        manifest.display()
                    ))
                })?;
            if parsed.model.trim().is_empty() || looks_like_local_path(&parsed.model) {
                return Err(gpt_oss_core::prelude::LLMError::ConfigError(format!(
                    "fetch manifest {} does not contain a public model ID",
                    manifest.display()
                )));
            }
            return Ok(parsed.model);
        }
    }
    if !looks_like_local_path(source_model) {
        return Ok(source_model.to_string());
    }
    Err(gpt_oss_core::prelude::LLMError::ConfigError(
        "a local model path requires --served-model-name unless it has a fetch manifest".into(),
    ))
}

fn looks_like_local_path(value: &str) -> bool {
    let path = Path::new(value);
    path.is_absolute()
        || path.exists()
        || value == "."
        || value == ".."
        || value.starts_with("./")
        || value.starts_with("../")
        || value.starts_with("~/")
        || value.contains('\\')
}

struct PlaceholderScheduler {
    groups: Vec<gpt_oss_engine::sequence::SequenceGroup>,
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

async fn service_shutdown(
    state: Arc<AppState>,
    mut startup_stop: tokio::sync::watch::Receiver<bool>,
) {
    let internal_stop = async {
        while !*startup_stop.borrow() {
            if startup_stop.changed().await.is_err() {
                std::future::pending::<()>().await;
            }
        }
    };
    let external = tokio::select! {
        _ = shutdown_signal() => true,
        _ = internal_stop => false,
    };
    if !external {
        return;
    }

    if let Some(engine) = state.installed_engine().await {
        if let Err(failure) = engine.begin_shutdown() {
            let _ = state.lifecycle.mark_failed(failure);
            return;
        }
        if state.lifecycle.status().state == ServiceState::Ready {
            let _ = state.lifecycle.begin_draining();
        }
    } else {
        let _ = state.lifecycle.mark_failed(StableFailure::new(
            StableFailureCode::Shutdown,
            FailurePhase::Shutdown,
            false,
            "shutdown requested during service startup",
        ));
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
    use super::*;

    #[tokio::test]
    async fn cpu_stream_consumes_committed_done_before_reporting_finished() {
        let lifecycle = ServiceLifecycle::starting("public-model");
        lifecycle.mark_ready("0".repeat(64)).unwrap();
        let (cancel_tx, mut cancel_rx) = tokio::sync::mpsc::unbounded_channel();
        let request_id = RequestId(7);
        let lease = gpt_oss_engine::RequestLease::new(request_id, cancel_tx);
        let limits = gpt_oss_engine::DeliveryLimits::default();
        let global = gpt_oss_engine::GlobalDeliveryBudget::new(limits.global_queued_bytes);
        let (publisher, events) =
            gpt_oss_engine::delivery_session(limits, global.clone(), lease.clone()).unwrap();
        publisher
            .try_publish(CommittedEvent::Usage {
                committed_prompt: 3,
                committed_completion: 1,
            })
            .unwrap();
        publisher
            .try_publish(CommittedEvent::Finish {
                choice: 0,
                reason: FinishReason::Length,
            })
            .unwrap();
        publisher.try_publish(CommittedEvent::Done).unwrap();
        drop(publisher);

        let mut stream = CpuEventStream {
            request: ManagedRequest {
                request_id,
                events,
                lease,
                lifecycle,
            },
            prompt_tokens: 0,
            completion_tokens: 0,
            pending_finish: None,
        };
        let output = stream.recv().await.unwrap().unwrap();
        assert!(output.finished);
        assert_eq!(output.prompt_token_ids.len(), 3);
        assert_eq!(output.outputs[0].finish_reason, Some(FinishReason::Length));
        drop(stream);
        assert_eq!(global.queued_bytes(), 0);
        assert!(cancel_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn bounded_sse_sender_replaces_oversized_payload_and_stops_producer() {
        let (raw_tx, mut rx) = tokio::sync::mpsc::channel(1);
        let tx = BoundedSseSender::new(raw_tx, 256);
        assert!(tx.send(Ok("x".repeat(257))).await.is_err());
        let terminal = rx.recv().await.unwrap().unwrap();
        assert!(terminal.len() <= 256);
        assert!(terminal.contains("overloaded_delivery"));
        assert!(!terminal.contains(&"x".repeat(257)));
    }

    #[tokio::test]
    async fn starting_state_serves_liveness_but_rejects_readiness_and_admission() {
        let lifecycle = ServiceLifecycle::starting("public-model");
        let state = Arc::new(AppState::starting(
            "public-model".into(),
            RuntimeDecision {
                runtime_mode: gpt_oss_engine::RuntimeMode::Experimental,
                backend_path: RuntimeBackendPath::Cpu,
                reason: "test".into(),
            },
            lifecycle,
            None,
            ServiceLimits::for_max_num_seqs(1),
        ));
        let live = crate::routes::health::health_check().await.into_response();
        let ready = crate::routes::health::ready_check(axum::extract::State(state.clone()))
            .await
            .into_response();
        assert_eq!(live.status(), axum::http::StatusCode::OK);
        assert_eq!(ready.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            state.engine().await.err().expect("starting rejects").code,
            StableFailureCode::NotReady
        );
    }

    #[test]
    fn served_identity_rejects_path_like_values() {
        assert_eq!(
            resolve_served_model_id("openai/gpt-oss-20b", None).unwrap(),
            "openai/gpt-oss-20b"
        );
        assert!(resolve_served_model_id("./missing-local-snapshot", None).is_err());
        assert!(resolve_served_model_id("openai/gpt-oss-20b", Some("/tmp/model")).is_err());
        assert_eq!(
            resolve_served_model_id("./missing-local-snapshot", Some("public-alias")).unwrap(),
            "public-alias"
        );
    }

    #[tokio::test]
    async fn cpu_engine_rejects_non_bf16_request_before_loading() {
        let mut config = EngineConfig::default();
        config.model.model_path = "openai/gpt-oss-20b".into();
        config.model.dtype = gpt_oss_core::types::Dtype::Float16;
        let lifecycle = ServiceLifecycle::starting("test-model");
        let error =
            match create_cpu_engine(config, lifecycle, ServiceLimits::for_max_num_seqs(1)).await {
                Ok(_) => panic!("float16 CPU request unexpectedly succeeded"),
                Err(error) => error,
            };
        assert!(error.to_string().contains("auto or bfloat16"));
    }
}
