//! gpt-oss-rs: High-performance LLM inference server in Rust
//!
//! Usage: gpt-oss-rs serve --model <model_path> [options]
//!
//! Compatible with OpenAI API at http://localhost:8000/v1/

use clap::{Parser, Subcommand, ValueEnum};
use gpt_oss_core::types::Dtype;
use gpt_oss_engine::RuntimeMode;
use std::path::PathBuf;
use tracing::info;

use gpt_oss_server::runtime_policy::{
    is_gpt_oss_model, GPT_OSS_CONSUMER_GPU_MEMORY_UTILIZATION, GPT_OSS_CONSUMER_MAX_MODEL_LEN,
};

#[derive(Parser)]
#[command(name = "gpt-oss-rs", about = "High-performance LLM inference server")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the inference server
    Serve {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        #[arg(long, default_value = "auto")]
        dtype: Dtype,
        #[arg(long)]
        max_model_len: Option<usize>,
        #[arg(long)]
        gpu_memory_utilization: Option<f32>,
        #[arg(long, default_value_t = 1)]
        tensor_parallel_size: usize,
        #[arg(long)]
        max_num_seqs: Option<usize>,
        #[arg(long)]
        max_num_batched_tokens: Option<usize>,
        #[arg(long)]
        max_prefill_chunk: Option<usize>,
        #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
        device: DeviceChoice,
        #[arg(long, value_enum, default_value_t = CpuKernelChoice::Auto)]
        cpu_kernel: CpuKernelChoice,
        #[arg(long, value_enum, default_value_t = CpuMatmulBackendChoice::Auto)]
        cpu_matmul_backend: CpuMatmulBackendChoice,
        #[arg(long)]
        cpu_threads: Option<usize>,
        #[arg(long)]
        cpu_repack_cache: Option<PathBuf>,
        #[arg(long, default_value_t = 128)]
        xe_max_resident_mib: usize,
        #[arg(long, value_enum, default_value_t = RuntimeMode::Experimental)]
        runtime_mode: RuntimeMode,
        #[arg(long, value_enum, default_value_t = ServeProfile::Auto)]
        profile: ServeProfile,
        #[arg(long)]
        tokenizer: Option<String>,
        /// Stable public model ID. Required for local paths without a fetch manifest.
        #[arg(long)]
        served_model_name: Option<String>,
        #[arg(long, default_value = "info")]
        log_level: String,
        #[arg(long)]
        disable_telemetry: bool,
        #[arg(long, default_value_t = 2)]
        request_body_limit_mib: usize,
        #[arg(long, default_value_t = 8)]
        non_streaming_limit_mib: usize,
        #[arg(long, default_value_t = 256)]
        stream_event_limit_kib: usize,
        #[arg(long, default_value_t = 1)]
        delivery_limit_mib: usize,
        #[arg(long)]
        global_delivery_limit_mib: Option<usize>,
        #[arg(long, default_value_t = 64)]
        response_store_limit_mib: usize,
        #[arg(long, default_value_t = 64)]
        response_store_max_entries: usize,
        #[arg(long, default_value_t = 8)]
        response_store_entry_limit_mib: usize,
        #[arg(long, default_value_t = 20)]
        max_logprobs: usize,
        #[arg(long, default_value_t = 30)]
        drain_deadline_seconds: u64,
        #[arg(long)]
        cpu_request_budget_mib: Option<u128>,
        #[arg(long)]
        evidence_dir: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = DiagnosticModeChoice::Off)]
        diagnostic_mode: DiagnosticModeChoice,
        #[arg(long)]
        diagnostic_cap_mib: Option<u64>,
        #[arg(long)]
        diagnostic_boundary: Option<String>,
        #[arg(long)]
        diagnostic_acknowledge: bool,
    },
    /// Show system info (GPU, memory, etc.)
    Info {
        #[arg(long, value_enum, default_value_t = InfoFormat::Text)]
        format: InfoFormat,
        #[arg(long, value_enum, default_value_t = CpuKernelChoice::Auto)]
        cpu_kernel: CpuKernelChoice,
        #[arg(long, value_enum, default_value_t = CpuMatmulBackendChoice::Auto)]
        cpu_matmul_backend: CpuMatmulBackendChoice,
    },
    /// Run benchmarks
    Benchmark {
        #[arg(long)]
        model: String,
        #[arg(long, default_value_t = 100)]
        num_prompts: usize,
        #[arg(long, default_value_t = 128)]
        input_len: usize,
        #[arg(long, default_value_t = 128)]
        output_len: usize,
    },
    /// Download a model snapshot without loading it.
    Fetch {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long)]
        cache_dir: Option<std::path::PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ServeProfile {
    Auto,
    Generic,
    GptOss3090,
    GptOssCpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DeviceChoice {
    Auto,
    Cpu,
    Xe,
    Cuda,
    Mock,
}

impl DeviceChoice {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Xe => "xe",
            Self::Cuda => "cuda",
            Self::Mock => "mock",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CpuKernelChoice {
    Auto,
    Scalar,
    Avx2,
    Avx512Vnni,
}

impl CpuKernelChoice {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512Vnni => "avx512-vnni",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CpuMatmulBackendChoice {
    Auto,
    Scalar,
    Avx2,
    Avx512Vnni,
    AmxInt8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum InfoFormat {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum DiagnosticModeChoice {
    Off,
    Metadata,
    Summary,
    Tensor,
}

impl From<DiagnosticModeChoice> for gpt_oss_evidence::DiagnosticMode {
    fn from(value: DiagnosticModeChoice) -> Self {
        match value {
            DiagnosticModeChoice::Off => Self::Off,
            DiagnosticModeChoice::Metadata => Self::Metadata,
            DiagnosticModeChoice::Summary => Self::Summary,
            DiagnosticModeChoice::Tensor => Self::Tensor,
        }
    }
}

impl CpuMatmulBackendChoice {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
            Self::Avx512Vnni => "avx512-vnni",
            Self::AmxInt8 => "amx-int8",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ResolvedServeProfile {
    profile: ServeProfile,
    max_model_len: usize,
    gpu_memory_utilization: f32,
    max_num_seqs: usize,
    max_num_batched_tokens: usize,
    max_prefill_chunk: usize,
}

fn init_tracing(log_level: &str) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();
}

#[allow(dead_code)]
fn detect_gpu_and_log() -> bool {
    let devices = gpt_oss_gpu::prelude::list_devices();
    if devices.is_empty() {
        info!("no CUDA devices detected");
        return false;
    } else {
        for dev in &devices {
            info!(
                id = dev.id,
                name = %dev.name,
                compute = %format!("{}.{}", dev.compute_capability.0, dev.compute_capability.1),
                memory_gb = dev.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                "detected GPU device"
            );
        }
    }
    true
}

fn resolve_serve_profile(
    model: &str,
    requested_profile: ServeProfile,
    cpu_selected: bool,
    max_model_len: Option<usize>,
    gpu_memory_utilization: Option<f32>,
    max_num_seqs: Option<usize>,
    max_num_batched_tokens: Option<usize>,
    max_prefill_chunk: Option<usize>,
) -> ResolvedServeProfile {
    let profile = match requested_profile {
        ServeProfile::Auto if is_gpt_oss_model(model) && cpu_selected => ServeProfile::GptOssCpu,
        ServeProfile::Auto if is_gpt_oss_model(model) => ServeProfile::GptOss3090,
        ServeProfile::Auto => ServeProfile::Generic,
        profile => profile,
    };

    let (default_max_model_len, default_gpu_memory_utilization, default_max_num_seqs) =
        match profile {
            ServeProfile::Auto | ServeProfile::Generic => (2048, 0.90, 256),
            ServeProfile::GptOss3090 => (
                GPT_OSS_CONSUMER_MAX_MODEL_LEN,
                GPT_OSS_CONSUMER_GPU_MEMORY_UTILIZATION,
                256,
            ),
            ServeProfile::GptOssCpu => (GPT_OSS_CONSUMER_MAX_MODEL_LEN, 0.90, 1),
        };

    ResolvedServeProfile {
        profile,
        max_model_len: max_model_len.unwrap_or(default_max_model_len),
        gpu_memory_utilization: gpu_memory_utilization.unwrap_or(default_gpu_memory_utilization),
        max_num_seqs: max_num_seqs.unwrap_or(default_max_num_seqs),
        max_num_batched_tokens: max_num_batched_tokens.unwrap_or(2048),
        max_prefill_chunk: max_prefill_chunk.unwrap_or(0),
    }
}

fn default_cpu_repack_cache() -> PathBuf {
    if let Some(path) = std::env::var_os("GPT_OSS_RS_CACHE") {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(path).join("gpt-oss-rs");
    }
    if let Some(path) = std::env::var_os("HOME") {
        return PathBuf::from(path).join(".cache/gpt-oss-rs");
    }
    PathBuf::from(".cache/gpt-oss-rs")
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if matches!(&cli.command, Commands::Serve { .. }) {
        gpt_oss_tokenizer::initialize_harmony_encoding()
            .map_err(|error| anyhow::anyhow!("failed to initialize Harmony: {error}"))?;
    }

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run(cli))
}

async fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Commands::Serve {
            model,
            host,
            port,
            dtype,
            max_model_len,
            gpu_memory_utilization,
            tensor_parallel_size,
            max_num_seqs,
            max_num_batched_tokens,
            max_prefill_chunk,
            device,
            cpu_kernel,
            cpu_matmul_backend,
            cpu_threads,
            cpu_repack_cache,
            xe_max_resident_mib,
            runtime_mode,
            profile,
            tokenizer,
            served_model_name,
            log_level,
            disable_telemetry,
            request_body_limit_mib,
            non_streaming_limit_mib,
            stream_event_limit_kib,
            delivery_limit_mib,
            global_delivery_limit_mib,
            response_store_limit_mib,
            response_store_max_entries,
            response_store_entry_limit_mib,
            max_logprobs,
            drain_deadline_seconds,
            cpu_request_budget_mib,
            evidence_dir,
            diagnostic_mode,
            diagnostic_cap_mib,
            diagnostic_boundary,
            diagnostic_acknowledge,
        } => {
            init_tracing(&log_level);
            info!("gpt-oss-rs v0.1.0");

            let cpu_selected = matches!(
                device,
                DeviceChoice::Auto | DeviceChoice::Cpu | DeviceChoice::Xe
            ) && is_gpt_oss_model(&model);
            let resolved_profile = resolve_serve_profile(
                &model,
                profile,
                cpu_selected,
                max_model_len,
                gpu_memory_utilization,
                max_num_seqs,
                max_num_batched_tokens,
                max_prefill_chunk,
            );
            if matches!(
                resolved_profile.profile,
                ServeProfile::GptOss3090 | ServeProfile::GptOssCpu
            ) && !is_gpt_oss_model(&model)
            {
                tracing::warn!(
                    model = %model,
                    "GPT-OSS profile selected for a non-GPT-OSS model"
                );
            }
            let cpu_threads = cpu_threads.unwrap_or_else(|| num_cpus::get_physical().max(1));
            let cpu_repack_cache = cpu_repack_cache.unwrap_or_else(default_cpu_repack_cache);

            // Build EngineConfig from CLI args
            let config = {
                use gpt_oss_engine::config::*;
                EngineConfig::builder()
                    .model({
                        let mut m = ModelConfigImpl::builder()
                            .model_path(&model)
                            .dtype(dtype)
                            .max_model_len(resolved_profile.max_model_len);
                        if let Some(ref tok) = tokenizer {
                            m = m.tokenizer_path(tok);
                        }
                        m.build()
                    })
                    .cache(
                        CacheConfigImpl::builder()
                            .gpu_memory_utilization(resolved_profile.gpu_memory_utilization)
                            .build(),
                    )
                    .runtime_mode(runtime_mode)
                    .scheduler(
                        SchedulerConfigImpl::builder()
                            .max_num_seqs(resolved_profile.max_num_seqs)
                            .max_num_batched_tokens(resolved_profile.max_num_batched_tokens)
                            .max_prefill_chunk(resolved_profile.max_prefill_chunk)
                            .build(),
                    )
                    .runtime_mode(runtime_mode)
                    .parallel(
                        ParallelConfigImpl::builder()
                            .tensor_parallel_size(tensor_parallel_size)
                            .build(),
                    )
                    .device(
                        DeviceConfig::builder()
                            .device(device.as_str())
                            .cpu_kernel(cpu_kernel.as_str())
                            .cpu_matmul_backend(cpu_matmul_backend.as_str())
                            .cpu_threads(cpu_threads)
                            .cpu_repack_cache(cpu_repack_cache.clone())
                            .xe_max_resident_mib(xe_max_resident_mib)
                            .build(),
                    )
                    .telemetry(
                        TelemetryConfig::builder()
                            .enabled(!disable_telemetry)
                            .log_level(&log_level)
                            .build(),
                    )
                    .build()
            };

            info!(
                model = %model,
                host = %host,
                port = port,
                dtype = %dtype,
                runtime_mode = ?runtime_mode,
                profile = ?resolved_profile.profile,
                max_model_len = resolved_profile.max_model_len,
                gpu_memory_utilization = resolved_profile.gpu_memory_utilization,
                max_num_seqs = resolved_profile.max_num_seqs,
                max_num_batched_tokens = resolved_profile.max_num_batched_tokens,
                max_prefill_chunk = resolved_profile.max_prefill_chunk,
                tp_size = tensor_parallel_size,
                requested_device = device.as_str(),
                cpu_kernel = cpu_kernel.as_str(),
                cpu_matmul_backend = cpu_matmul_backend.as_str(),
                cpu_threads,
                cpu_repack_cache = %cpu_repack_cache.display(),
                xe_max_resident_mib,
                "starting server"
            );

            let mib = 1024usize * 1024;
            let checked_mib = |value: usize, name: &str| -> anyhow::Result<usize> {
                value
                    .checked_mul(mib)
                    .ok_or_else(|| anyhow::anyhow!("{name} overflows bytes"))
            };
            let mut limits =
                gpt_oss_server::ServiceLimits::for_max_num_seqs(resolved_profile.max_num_seqs);
            limits.request_body_bytes = checked_mib(request_body_limit_mib, "request body limit")?;
            limits.max_non_streaming_bytes =
                checked_mib(non_streaming_limit_mib, "non-streaming limit")?;
            limits.max_stream_event_bytes = stream_event_limit_kib
                .checked_mul(1024)
                .ok_or_else(|| anyhow::anyhow!("stream event limit overflows bytes"))?;
            limits.per_request_delivery_bytes = checked_mib(delivery_limit_mib, "delivery limit")?;
            limits.global_delivery_bytes = checked_mib(
                global_delivery_limit_mib.unwrap_or(
                    resolved_profile
                        .max_num_seqs
                        .checked_mul(delivery_limit_mib)
                        .ok_or_else(|| anyhow::anyhow!("global delivery limit overflows MiB"))?,
                ),
                "global delivery limit",
            )?;
            limits.response_store_bytes =
                checked_mib(response_store_limit_mib, "response store limit")?;
            limits.response_store_entries = response_store_max_entries;
            limits.max_store_entry_bytes =
                checked_mib(response_store_entry_limit_mib, "response store entry limit")?;
            limits.max_logprobs = max_logprobs;
            limits.drain_deadline = std::time::Duration::from_secs(drain_deadline_seconds);
            limits.cpu_request_budget_bytes = match cpu_request_budget_mib {
                Some(value) => Some(
                    value
                        .checked_mul(mib as u128)
                        .ok_or_else(|| anyhow::anyhow!("CPU request budget overflows bytes"))?,
                ),
                None => None,
            };

            let diagnostics = gpt_oss_evidence::DiagnosticConfig {
                mode: diagnostic_mode.into(),
                directory: evidence_dir.as_ref().map(|dir| dir.join("diagnostics")),
                byte_cap: diagnostic_cap_mib
                    .unwrap_or(0)
                    .checked_mul(mib as u64)
                    .ok_or_else(|| anyhow::anyhow!("diagnostic cap overflows bytes"))?,
                boundary: diagnostic_boundary,
                acknowledge_sensitive_payload: diagnostic_acknowledge,
            };
            diagnostics
                .validate(true)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            let bind_address = format!("{host}:{port}")
                .parse()
                .map_err(|error| anyhow::anyhow!("invalid bind address {host}:{port}: {error}"))?;
            gpt_oss_server::serve(gpt_oss_server::ServerConfig {
                bind_address,
                served_model_name,
                limits,
                evidence: gpt_oss_server::EvidenceConfig {
                    directory: evidence_dir,
                    diagnostics,
                },
                engine: config,
            })
            .await?;
        }
        Commands::Info {
            format,
            cpu_kernel,
            cpu_matmul_backend,
        } => {
            let requested_kernel: gpt_oss_cpu_kernels::KernelPath = cpu_kernel.as_str().parse()?;
            let requested_matmul: gpt_oss_cpu_kernels::Mxfp4MatmulBackend =
                cpu_matmul_backend.as_str().parse()?;
            let kernels = gpt_oss_cpu_kernels::Kernels::new(requested_kernel)?;
            let identity = gpt_oss_cpu_kernels::CpuHardwareIdentity::detect();
            let features = gpt_oss_cpu_kernels::CpuFeatures::detect();
            let plan = kernels.dispatch_plan();
            let report = serde_json::json!({
                "schema": "gpt-oss-rs.cpu-info/v1",
                "platform": { "os": std::env::consts::OS, "arch": std::env::consts::ARCH },
                "identity": {
                    "vendor": identity.vendor, "family": identity.family,
                    "model": identity.model, "stepping": identity.stepping,
                    "logical_cpus": identity.logical_cpus, "osxsave": identity.osxsave,
                    "xcr0": identity.xcr0, "hardware_profile_key": identity.profile_key()
                },
                "legal_capabilities": {
                    "avx2": features.avx2, "fma": features.fma,
                    "avx_vnni": features.avx_vnni, "avx512_f": features.avx512_f,
                    "avx512_bw": features.avx512_bw, "avx512_vl": features.avx512_vl,
                    "avx512_vnni": features.avx512_vnni, "avx512_bf16": features.avx512_bf16,
                    "amx_tile": features.amx_tile, "amx_int8": features.amx_int8
                },
                "requested": { "cpu_kernel": requested_kernel.to_string(), "cpu_matmul_backend": requested_matmul.to_string() },
                "resolved": {
                    "cpu_kernel": kernels.path().to_string(), "bf16_matvec": plan.bf16_matvec().to_string(),
                    "quantize_q8": plan.quantize_q8().to_string(), "mxfp4_q8_dot": plan.mxfp4_q8_dot().to_string(),
                    "mxfp4_gemv": plan.mxfp4_gemv().to_string(), "mxfp4_weight_layout": plan.mxfp4_weight_layout().to_string(),
                    "mxfp4_matrix": if requested_matmul == gpt_oss_cpu_kernels::Mxfp4MatmulBackend::Auto { "scalar-multi-row".to_string() } else { requested_matmul.to_string() },
                    "rms_norm": plan.rms_norm().to_string()
                },
                "matrix_crossover_regions": []
            });
            match format {
                InfoFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
                InfoFormat::Text => {
                    println!(
                        "CPU: {} family {} model {} stepping {}",
                        identity.vendor, identity.family, identity.model, identity.stepping
                    );
                    println!("hardware profile: {}", identity.profile_key());
                    println!("XCR0: 0x{:x} (OSXSAVE={})", identity.xcr0, identity.osxsave);
                    println!(
                        "requested: kernel={}, matrix={}",
                        requested_kernel, requested_matmul
                    );
                    println!(
                        "resolved: {} matrix={}",
                        plan,
                        report["resolved"]["mxfp4_matrix"]
                            .as_str()
                            .unwrap_or("unknown")
                    );
                    println!("matrix crossover regions: none (Auto is scalar for M>1)");
                }
            }
        }
        Commands::Benchmark {
            model,
            num_prompts,
            input_len,
            output_len,
        } => {
            init_tracing("info");
            info!(
                model = %model,
                num_prompts = num_prompts,
                input_len = input_len,
                output_len = output_len,
                "running benchmark"
            );
        }
        Commands::Fetch {
            model,
            revision,
            cache_dir,
        } => {
            let result = gpt_oss_engine::model_fetch::fetch_snapshot(
                &gpt_oss_engine::model_fetch::FetchOptions {
                    model,
                    revision,
                    cache_dir,
                },
            )?;
            println!("snapshot: {}", result.snapshot_dir.display());
            println!("revision: {}", result.manifest.resolved_revision);
            println!("manifest: {}", result.manifest_path.display());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_cuda_profile_uses_gpt_oss_3090_defaults() {
        let resolved = resolve_serve_profile(
            "openai/gpt-oss-20b",
            ServeProfile::Auto,
            false,
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(resolved.profile, ServeProfile::GptOss3090);
        assert_eq!(resolved.max_model_len, GPT_OSS_CONSUMER_MAX_MODEL_LEN);
        assert_eq!(resolved.max_num_seqs, 256);
        assert_eq!(
            resolved.gpu_memory_utilization,
            GPT_OSS_CONSUMER_GPU_MEMORY_UTILIZATION
        );
    }

    #[test]
    fn auto_profile_keeps_generic_defaults_for_non_gpt_oss_names() {
        let resolved = resolve_serve_profile(
            "/models/local-checkpoint",
            ServeProfile::Auto,
            false,
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(resolved.profile, ServeProfile::Generic);
        assert_eq!(resolved.max_model_len, 2048);
        assert_eq!(resolved.gpu_memory_utilization, 0.90);
    }

    #[test]
    fn explicit_values_override_profile_defaults() {
        let resolved = resolve_serve_profile(
            "openai/gpt-oss-20b",
            ServeProfile::GptOss3090,
            false,
            Some(4096),
            Some(0.82),
            Some(8),
            Some(1024),
            Some(64),
        );
        assert_eq!(resolved.profile, ServeProfile::GptOss3090);
        assert_eq!(resolved.max_model_len, 4096);
        assert_eq!(resolved.gpu_memory_utilization, 0.82);
        assert_eq!(resolved.max_num_seqs, 8);
        assert_eq!(resolved.max_num_batched_tokens, 1024);
        assert_eq!(resolved.max_prefill_chunk, 64);
    }

    #[test]
    fn auto_profile_uses_batch_one_cpu_defaults() {
        let resolved = resolve_serve_profile(
            "openai/gpt-oss-20b",
            ServeProfile::Auto,
            true,
            None,
            None,
            None,
            None,
            None,
        );
        assert_eq!(resolved.profile, ServeProfile::GptOssCpu);
        assert_eq!(resolved.max_model_len, 8192);
        assert_eq!(resolved.max_num_seqs, 1);
        assert_eq!(resolved.max_num_batched_tokens, 2048);
        assert_eq!(resolved.max_prefill_chunk, 0);
    }

    #[test]
    fn device_and_kernel_cli_values_are_stable() {
        assert_eq!(DeviceChoice::Auto.as_str(), "auto");
        assert_eq!(DeviceChoice::Xe.as_str(), "xe");
        assert_eq!(DeviceChoice::Mock.as_str(), "mock");
        assert_eq!(CpuKernelChoice::Avx512Vnni.as_str(), "avx512-vnni");
    }
}
