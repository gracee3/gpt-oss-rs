use std::process::Command;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use gpt_oss_config::{
    CacheConfigImpl, DeviceConfig, EngineConfig, ModelConfigImpl, ParallelConfigImpl,
    SchedulerConfigImpl,
};
use gpt_oss_core::prelude::{Dtype, RequestId, RequestOutput, SamplingParams};
use gpt_oss_engine::GpuLLMEngine;
use serde::{Deserialize, Serialize};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser, Clone)]
#[command(about = "Live CUDA parity harness for single-rank vs tensor-parallel runs")]
struct Cli {
    #[arg(long)]
    model: String,

    #[arg(long, default_value = "Explain tensor parallelism in one sentence.")]
    prompt: String,

    #[arg(long, default_value_t = 16)]
    max_tokens: usize,

    #[arg(long, default_value = "float16")]
    dtype: Dtype,

    #[arg(long, default_value_t = 2048)]
    max_model_len: usize,

    #[arg(long, default_value_t = 0.85)]
    gpu_memory_utilization: f32,

    #[arg(long, default_value_t = 1)]
    single_device_id: usize,

    #[arg(long, value_delimiter = ',', default_values_t = vec![0usize, 1usize])]
    tp_device_ids: Vec<usize>,

    #[arg(long, default_value_t = 2)]
    tp_size: usize,

    #[arg(long, default_value = "info")]
    log_level: String,

    #[arg(long, hide = true)]
    child_run: bool,

    #[arg(long, hide = true)]
    child_label: Option<String>,

    #[arg(long, hide = true)]
    child_visible_devices: Option<String>,

    #[arg(long, hide = true)]
    child_tp_size: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RunSummary {
    label: String,
    visible_devices: String,
    tp_size: usize,
    elapsed_ms: u128,
    request: RequestOutput,
}

fn init_tracing(log_level: &str) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}

fn build_config(cli: &Cli, tp_size: usize) -> EngineConfig {
    EngineConfig::builder()
        .model(
            ModelConfigImpl::builder()
                .model_path(&cli.model)
                .dtype(cli.dtype)
                .max_model_len(cli.max_model_len)
                .build(),
        )
        .cache(
            CacheConfigImpl::builder()
                .gpu_memory_utilization(cli.gpu_memory_utilization)
                .build(),
        )
        .scheduler(
            SchedulerConfigImpl::builder()
                .max_num_seqs(1)
                .max_num_batched_tokens(cli.max_model_len.min(2048))
                .build(),
        )
        .parallel(
            ParallelConfigImpl::builder()
                .tensor_parallel_size(tp_size)
                .build(),
        )
        .device(DeviceConfig::builder().device("cuda").build())
        .build()
}

fn run_decode(
    cli: &Cli,
    label: String,
    visible_devices: String,
    tp_size: usize,
) -> Result<RunSummary> {
    let config = build_config(cli, tp_size);
    let started = Instant::now();
    let mut engine = GpuLLMEngine::new(config).context("failed to construct GPU engine")?;

    let params = SamplingParams {
        temperature: 0.0,
        max_tokens: cli.max_tokens,
        seed: Some(0),
        ..SamplingParams::default()
    };
    engine
        .add_request(RequestId(1), cli.prompt.clone(), params)
        .context("failed to add request")?;

    let mut outputs = engine.run().context("engine run failed")?;
    if outputs.len() != 1 {
        bail!("expected exactly one request output, got {}", outputs.len());
    }

    Ok(RunSummary {
        label,
        visible_devices,
        tp_size,
        elapsed_ms: started.elapsed().as_millis(),
        request: outputs.remove(0),
    })
}

fn run_child(cli: &Cli) -> Result<()> {
    let label = cli
        .child_label
        .clone()
        .context("missing --child-label for child run")?;
    let visible_devices = cli
        .child_visible_devices
        .clone()
        .or_else(|| std::env::var("CUDA_VISIBLE_DEVICES").ok())
        .unwrap_or_else(|| "<unset>".into());
    let tp_size = cli
        .child_tp_size
        .context("missing --child-tp-size for child run")?;

    let summary = run_decode(cli, label, visible_devices, tp_size)?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

fn spawn_child(
    cli: &Cli,
    label: &str,
    visible_devices: &str,
    tp_size: usize,
) -> Result<RunSummary> {
    let current_exe = std::env::current_exe().context("failed to resolve current executable")?;
    let output = Command::new(current_exe)
        .env("CUDA_VISIBLE_DEVICES", visible_devices)
        .arg("--child-run")
        .arg("--child-label")
        .arg(label)
        .arg("--child-visible-devices")
        .arg(visible_devices)
        .arg("--child-tp-size")
        .arg(tp_size.to_string())
        .arg("--model")
        .arg(&cli.model)
        .arg("--prompt")
        .arg(&cli.prompt)
        .arg("--max-tokens")
        .arg(cli.max_tokens.to_string())
        .arg("--dtype")
        .arg(cli.dtype.to_string())
        .arg("--max-model-len")
        .arg(cli.max_model_len.to_string())
        .arg("--gpu-memory-utilization")
        .arg(cli.gpu_memory_utilization.to_string())
        .arg("--log-level")
        .arg(&cli.log_level)
        .output()
        .with_context(|| format!("failed to spawn child run for {label}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "{label} child run failed with status {}.\nstdout:\n{}\nstderr:\n{}",
            output.status,
            stdout,
            stderr
        );
    }

    let stdout = String::from_utf8(output.stdout).context("child stdout was not utf-8")?;
    let summary = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .context("child did not emit JSON summary")?;
    Ok(serde_json::from_str(summary).context("failed to parse child JSON summary")?)
}

fn compare_runs(single: &RunSummary, tp: &RunSummary) -> Result<()> {
    let single_output = single
        .request
        .outputs
        .first()
        .context("single-rank run produced no completion output")?;
    let tp_output = tp
        .request
        .outputs
        .first()
        .context("tp run produced no completion output")?;

    if single_output.token_ids != tp_output.token_ids {
        bail!(
            "token mismatch between single-rank and TP runs\nsingle: {:?}\ntp: {:?}",
            single_output.token_ids,
            tp_output.token_ids
        );
    }

    if single_output.finish_reason != tp_output.finish_reason {
        bail!(
            "finish-reason mismatch between single-rank and TP runs\nsingle: {:?}\ntp: {:?}",
            single_output.finish_reason,
            tp_output.finish_reason
        );
    }

    if single_output.text != tp_output.text {
        bail!(
            "decoded-text mismatch between single-rank and TP runs\nsingle: {:?}\ntp: {:?}",
            single_output.text,
            tp_output.text
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level);

    if cli.child_run {
        return run_child(&cli);
    }

    if cli.tp_size != cli.tp_device_ids.len() {
        bail!(
            "--tp-size={} must match the number of --tp-device-ids entries ({})",
            cli.tp_size,
            cli.tp_device_ids.len()
        );
    }

    let single_visible_devices = cli.single_device_id.to_string();
    let tp_visible_devices = cli
        .tp_device_ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",");

    info!(
        model = %cli.model,
        prompt = %cli.prompt,
        single_visible_devices = %single_visible_devices,
        tp_visible_devices = %tp_visible_devices,
        tp_size = cli.tp_size,
        "starting live CUDA parity harness"
    );

    let single = spawn_child(&cli, "single", &single_visible_devices, 1)?;
    let tp = spawn_child(&cli, "tp", &tp_visible_devices, cli.tp_size)?;
    compare_runs(&single, &tp)?;

    println!("single-rank and TP outputs match");
    println!("{}", serde_json::to_string_pretty(&single)?);
    println!("{}", serde_json::to_string_pretty(&tp)?);
    Ok(())
}
