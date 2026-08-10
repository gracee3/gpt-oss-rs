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
        #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
        device: DeviceChoice,
        #[arg(long, value_enum, default_value_t = CpuKernelChoice::Auto)]
        cpu_kernel: CpuKernelChoice,
        #[arg(long)]
        cpu_threads: Option<usize>,
        #[arg(long)]
        cpu_repack_cache: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = RuntimeMode::Experimental)]
        runtime_mode: RuntimeMode,
        #[arg(long, value_enum, default_value_t = ServeProfile::Auto)]
        profile: ServeProfile,
        #[arg(long)]
        tokenizer: Option<String>,
        #[arg(long, default_value = "info")]
        log_level: String,
        #[arg(long)]
        disable_telemetry: bool,
    },
    /// Show system info (GPU, memory, etc.)
    Info,
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
    Cuda,
    Mock,
}

impl DeviceChoice {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
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

#[derive(Debug, Clone, Copy, PartialEq)]
struct ResolvedServeProfile {
    profile: ServeProfile,
    max_model_len: usize,
    gpu_memory_utilization: f32,
    max_num_seqs: usize,
}

fn init_tracing(log_level: &str) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();
}

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
            device,
            cpu_kernel,
            cpu_threads,
            cpu_repack_cache,
            runtime_mode,
            profile,
            tokenizer,
            log_level,
            disable_telemetry,
        } => {
            init_tracing(&log_level);
            info!("gpt-oss-rs v0.1.0");

            let gpu_available = detect_gpu_and_log();

            let cpu_selected = device == DeviceChoice::Cpu
                || (device == DeviceChoice::Auto && !gpu_available && is_gpt_oss_model(&model));
            let resolved_profile = resolve_serve_profile(
                &model,
                profile,
                cpu_selected,
                max_model_len,
                gpu_memory_utilization,
                max_num_seqs,
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
                            .cpu_threads(cpu_threads)
                            .cpu_repack_cache(cpu_repack_cache.clone())
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
                tp_size = tensor_parallel_size,
                requested_device = device.as_str(),
                cpu_kernel = cpu_kernel.as_str(),
                cpu_threads,
                cpu_repack_cache = %cpu_repack_cache.display(),
                "starting server"
            );

            // Pass host/port to the server via env vars so gpt_oss_server::serve
            // can pick them up without changing its public signature.
            std::env::set_var("VLLM_HOST", &host);
            std::env::set_var("VLLM_PORT", port.to_string());

            gpt_oss_server::serve(config).await?;
        }
        Commands::Info => {
            init_tracing("info");
            info!("gpt-oss-rs system info");

            detect_gpu_and_log();

            info!(platform = %std::env::consts::OS, arch = %std::env::consts::ARCH, "system");
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
    fn auto_profile_uses_gpt_oss_3090_defaults() {
        let resolved = resolve_serve_profile(
            "openai/gpt-oss-20b",
            ServeProfile::Auto,
            false,
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
        );
        assert_eq!(resolved.profile, ServeProfile::GptOss3090);
        assert_eq!(resolved.max_model_len, 4096);
        assert_eq!(resolved.gpu_memory_utilization, 0.82);
        assert_eq!(resolved.max_num_seqs, 8);
    }

    #[test]
    fn auto_profile_uses_batch_one_cpu_defaults_without_cuda() {
        let resolved = resolve_serve_profile(
            "openai/gpt-oss-20b",
            ServeProfile::Auto,
            true,
            None,
            None,
            None,
        );
        assert_eq!(resolved.profile, ServeProfile::GptOssCpu);
        assert_eq!(resolved.max_model_len, 8192);
        assert_eq!(resolved.max_num_seqs, 1);
    }

    #[test]
    fn device_and_kernel_cli_values_are_stable() {
        assert_eq!(DeviceChoice::Auto.as_str(), "auto");
        assert_eq!(DeviceChoice::Mock.as_str(), "mock");
        assert_eq!(CpuKernelChoice::Avx512Vnni.as_str(), "avx512-vnni");
    }
}
