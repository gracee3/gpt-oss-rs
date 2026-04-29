//! gpt-oss-rs: High-performance LLM inference server in Rust
//!
//! Usage: gpt-oss-rs serve --model <model_path> [options]
//!
//! Compatible with OpenAI API at http://localhost:8000/v1/

use clap::{Parser, Subcommand, ValueEnum};
use gpt_oss_core::types::Dtype;
use gpt_oss_engine::RuntimeMode;
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
        #[arg(long, default_value_t = 256)]
        max_num_seqs: usize,
        #[arg(long, value_enum, default_value_t = RuntimeMode::Experimental)]
        runtime_mode: RuntimeMode,
        #[arg(long, value_enum, default_value_t = ServeProfile::Auto)]
        profile: ServeProfile,
        #[arg(long, default_value = "single")]
        device_map: String,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ServeProfile {
    Auto,
    Generic,
    GptOss3090,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ResolvedServeProfile {
    profile: ServeProfile,
    max_model_len: usize,
    gpu_memory_utilization: f32,
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
        info!("no GPU devices detected, using mock backend");
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

fn resolve_device_string(
    model: &str,
    runtime_mode: RuntimeMode,
    gpu_available: bool,
) -> anyhow::Result<&'static str> {
    if runtime_mode == RuntimeMode::Trusted && is_gpt_oss_model(model) && !gpu_available {
        return Err(anyhow::anyhow!(
            "trusted GPT-OSS mode requires a CUDA-capable GPU; falling back to mock is not allowed"
        ));
    }

    if gpu_available {
        Ok("cuda")
    } else {
        Ok("cpu")
    }
}

fn resolve_serve_profile(
    model: &str,
    requested_profile: ServeProfile,
    max_model_len: Option<usize>,
    gpu_memory_utilization: Option<f32>,
) -> ResolvedServeProfile {
    let profile = match requested_profile {
        ServeProfile::Auto if is_gpt_oss_model(model) => ServeProfile::GptOss3090,
        ServeProfile::Auto => ServeProfile::Generic,
        profile => profile,
    };

    let (default_max_model_len, default_gpu_memory_utilization) = match profile {
        ServeProfile::Auto | ServeProfile::Generic => (2048, 0.90),
        ServeProfile::GptOss3090 => (
            GPT_OSS_CONSUMER_MAX_MODEL_LEN,
            GPT_OSS_CONSUMER_GPU_MEMORY_UTILIZATION,
        ),
    };

    ResolvedServeProfile {
        profile,
        max_model_len: max_model_len.unwrap_or(default_max_model_len),
        gpu_memory_utilization: gpu_memory_utilization.unwrap_or(default_gpu_memory_utilization),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
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
            runtime_mode,
            profile,
            device_map,
            tokenizer,
            log_level,
            disable_telemetry,
        } => {
            init_tracing(&log_level);
            info!("gpt-oss-rs v0.1.0");

            let gpu_available = detect_gpu_and_log();

            let resolved_profile =
                resolve_serve_profile(&model, profile, max_model_len, gpu_memory_utilization);
            if resolved_profile.profile == ServeProfile::GptOss3090 && !is_gpt_oss_model(&model) {
                tracing::warn!(
                    model = %model,
                    "gpt-oss-3090 profile selected for a non-GPT-OSS model"
                );
            }
            let device = resolve_device_string(&model, runtime_mode, gpu_available)?;

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
                            .max_num_seqs(max_num_seqs)
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
                            .device(device)
                            .device_map(device_map.clone())
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
                tp_size = tensor_parallel_size,
                device_map = %device_map,
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
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_profile_uses_gpt_oss_3090_defaults() {
        let resolved = resolve_serve_profile("openai/gpt-oss-20b", ServeProfile::Auto, None, None);
        assert_eq!(resolved.profile, ServeProfile::GptOss3090);
        assert_eq!(resolved.max_model_len, GPT_OSS_CONSUMER_MAX_MODEL_LEN);
        assert_eq!(
            resolved.gpu_memory_utilization,
            GPT_OSS_CONSUMER_GPU_MEMORY_UTILIZATION
        );
    }

    #[test]
    fn auto_profile_keeps_generic_defaults_for_non_gpt_oss_names() {
        let resolved =
            resolve_serve_profile("/models/local-checkpoint", ServeProfile::Auto, None, None);
        assert_eq!(resolved.profile, ServeProfile::Generic);
        assert_eq!(resolved.max_model_len, 2048);
        assert_eq!(resolved.gpu_memory_utilization, 0.90);
    }

    #[test]
    fn explicit_values_override_profile_defaults() {
        let resolved = resolve_serve_profile(
            "openai/gpt-oss-20b",
            ServeProfile::GptOss3090,
            Some(4096),
            Some(0.82),
        );
        assert_eq!(resolved.profile, ServeProfile::GptOss3090);
        assert_eq!(resolved.max_model_len, 4096);
        assert_eq!(resolved.gpu_memory_utilization, 0.82);
    }

    #[test]
    fn trusted_gpt_oss_rejects_mock_fallback_without_gpu() {
        let err = resolve_device_string("openai/gpt-oss-20b", RuntimeMode::Trusted, false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("falling back to mock is not allowed"));
    }

    #[test]
    fn experimental_non_gpu_uses_cpu() {
        let device =
            resolve_device_string("/models/local-checkpoint", RuntimeMode::Experimental, false)
                .unwrap();
        assert_eq!(device, "cpu");
    }
}
