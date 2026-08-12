use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use gpt_oss_core::prelude::{FinishReason, RequestId};
use gpt_oss_engine::telemetry::metrics::{
    self as service_metrics, BackendClass, Phase, ReasonCode, ResultClass, TokenClass,
};
use gpt_oss_engine::{
    delivery_session, CommittedEvent, DeliveryLimits, GlobalDeliveryBudget, RequestLease,
};
use gpt_oss_evidence::{
    ArtifactRef, EvidenceStatus, ModelEvidence, RunManifestV1, SourceProvenance, WorkloadEvidence,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const THROUGHPUT_BUDGET_PERCENT: f64 = 1.0;
const P99_LATENCY_BUDGET_PERCENT: f64 = 2.0;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ChildMode {
    Baseline,
    Instrumented,
}

#[derive(Debug, Parser)]
#[command(about = "Paired model-free CPU service instrumentation overhead gate")]
struct Cli {
    /// Atomic A/B capture destination. A .manifest.json sidecar is also written.
    #[arg(long, required_unless_present = "child_mode")]
    output: Option<PathBuf>,

    #[arg(long, default_value_t = 5)]
    trials: usize,

    #[arg(long, default_value_t = 500)]
    iterations: usize,

    #[arg(long, default_value_t = 50)]
    warmup: usize,

    /// Deterministic arithmetic operations standing in for a small execute slice.
    #[arg(long, default_value_t = 1_000_000)]
    work_units: usize,

    #[arg(long, value_enum, hide = true)]
    child_mode: Option<ChildMode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrialResult {
    throughput_per_second: f64,
    median_latency_ns: u64,
    p99_latency_ns: u64,
    checksum: u64,
    metrics_bytes: usize,
}

#[derive(Debug, Serialize)]
struct Aggregate {
    median_throughput_per_second: f64,
    median_p99_latency_ns: f64,
    trials: Vec<TrialResult>,
}

#[derive(Debug, Serialize)]
struct OverheadCapture {
    schema: &'static str,
    status: EvidenceStatus,
    workload: Workload,
    thresholds: Thresholds,
    baseline: Aggregate,
    instrumented: Aggregate,
    throughput_regression_percent: f64,
    p99_latency_regression_percent: f64,
}

#[derive(Debug, Serialize)]
struct Workload {
    trials: usize,
    iterations_per_trial: usize,
    warmup_per_trial: usize,
    work_units_per_request: usize,
    delivery_events_per_request: usize,
}

#[derive(Debug, Serialize)]
struct Thresholds {
    max_median_throughput_regression_percent: f64,
    max_p99_latency_regression_percent: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(mode) = cli.child_mode {
        let result = run_child(mode, cli.iterations, cli.warmup, cli.work_units)?;
        println!("{}", serde_json::to_string(&result)?);
        return Ok(());
    }
    run_parent(cli)
}

fn run_parent(cli: Cli) -> Result<()> {
    if cli.trials < 3 || cli.iterations == 0 || cli.work_units == 0 {
        bail!("A/B gate requires at least three trials and positive iteration/work limits");
    }
    let executable = std::env::current_exe()?;
    let mut baseline = Vec::with_capacity(cli.trials);
    let mut instrumented = Vec::with_capacity(cli.trials);
    for trial in 0..cli.trials {
        let modes = if trial % 2 == 0 {
            [ChildMode::Baseline, ChildMode::Instrumented]
        } else {
            [ChildMode::Instrumented, ChildMode::Baseline]
        };
        for mode in modes {
            let result = spawn_trial(
                &executable,
                mode,
                cli.iterations,
                cli.warmup,
                cli.work_units,
            )?;
            match mode {
                ChildMode::Baseline => baseline.push(result),
                ChildMode::Instrumented => instrumented.push(result),
            }
        }
    }
    if baseline
        .iter()
        .zip(&instrumented)
        .any(|(left, right)| left.checksum != right.checksum)
    {
        bail!("paired A/B workloads produced different checksums");
    }
    let baseline = aggregate(baseline);
    let instrumented = aggregate(instrumented);
    let throughput_regression_percent = regression_percent(
        baseline.median_throughput_per_second,
        instrumented.median_throughput_per_second,
    );
    let p99_latency_regression_percent = increase_percent(
        baseline.median_p99_latency_ns,
        instrumented.median_p99_latency_ns,
    );
    let passed = throughput_regression_percent < THROUGHPUT_BUDGET_PERCENT
        && p99_latency_regression_percent < P99_LATENCY_BUDGET_PERCENT;
    let status = if passed {
        EvidenceStatus::Pass
    } else {
        EvidenceStatus::Fail
    };
    let capture = OverheadCapture {
        schema: "gpt-oss-rs.cpu-service-overhead/v1",
        status,
        workload: Workload {
            trials: cli.trials,
            iterations_per_trial: cli.iterations,
            warmup_per_trial: cli.warmup,
            work_units_per_request: cli.work_units,
            delivery_events_per_request: 4,
        },
        thresholds: Thresholds {
            max_median_throughput_regression_percent: THROUGHPUT_BUDGET_PERCENT,
            max_p99_latency_regression_percent: P99_LATENCY_BUDGET_PERCENT,
        },
        baseline,
        instrumented,
        throughput_regression_percent,
        p99_latency_regression_percent,
    };
    let encoded = serde_json::to_vec_pretty(&capture)?;
    let output = cli.output.context("--output is required in parent mode")?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    gpt_oss_evidence::atomic_write(&output, &encoded)?;
    write_sidecar(&output, status, &encoded)?;
    println!("{}", String::from_utf8(encoded)?);
    if !passed {
        bail!(
            "instrumentation overhead exceeded its budget: throughput {:.3}%, p99 {:.3}%",
            throughput_regression_percent,
            p99_latency_regression_percent
        );
    }
    Ok(())
}

fn spawn_trial(
    executable: &Path,
    mode: ChildMode,
    iterations: usize,
    warmup: usize,
    work_units: usize,
) -> Result<TrialResult> {
    let mode = match mode {
        ChildMode::Baseline => "baseline",
        ChildMode::Instrumented => "instrumented",
    };
    let output = Command::new(executable)
        .args([
            "--child-mode",
            mode,
            "--iterations",
            &iterations.to_string(),
            "--warmup",
            &warmup.to_string(),
            "--work-units",
            &work_units.to_string(),
        ])
        .output()?;
    if !output.status.success() {
        bail!(
            "{mode} child failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    serde_json::from_slice(&output.stdout).context("child emitted invalid trial JSON")
}

fn run_child(
    mode: ChildMode,
    iterations: usize,
    warmup: usize,
    work_units: usize,
) -> Result<TrialResult> {
    let metrics = if matches!(mode, ChildMode::Instrumented) {
        let handle = metrics_exporter_prometheus::PrometheusBuilder::new().install_recorder()?;
        gpt_oss_engine::telemetry::metrics::register_descriptions();
        Some(handle)
    } else {
        None
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        for request in 0..warmup {
            let _ = one_request(request as u64, work_units).await?;
        }
        let started = Instant::now();
        let mut latencies = Vec::with_capacity(iterations);
        let mut checksum = 0u64;
        for request in 0..iterations {
            let request_started = Instant::now();
            checksum ^= one_request(request as u64, work_units).await?;
            latencies.push(request_started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        }
        let elapsed = started.elapsed().as_secs_f64();
        latencies.sort_unstable();
        let median_latency_ns = percentile(&latencies, 0.50);
        let p99_latency_ns = percentile(&latencies, 0.99);
        let metrics_bytes = metrics.as_ref().map_or(0, |handle| handle.render().len());
        Ok(TrialResult {
            throughput_per_second: iterations as f64 / elapsed,
            median_latency_ns,
            p99_latency_ns,
            checksum,
            metrics_bytes,
        })
    })
}

async fn one_request(request: u64, work_units: usize) -> Result<u64> {
    service_metrics::record_admission(BackendClass::Cpu, ResultClass::Accepted, ReasonCode::None);
    service_metrics::adjust_current_requests(BackendClass::Cpu, 1.0);
    service_metrics::record_phase_duration(
        BackendClass::Cpu,
        Phase::Tokenization,
        ResultClass::Completed,
        Duration::from_micros(10),
    );
    service_metrics::record_tokens(BackendClass::Cpu, TokenClass::Prompt, 8);
    let execute_started = Instant::now();
    let checksum = deterministic_execute(request, work_units);
    service_metrics::record_phase_duration(
        BackendClass::Cpu,
        Phase::Execute,
        ResultClass::Completed,
        execute_started.elapsed(),
    );
    service_metrics::record_phase_duration(
        BackendClass::Cpu,
        Phase::Commit,
        ResultClass::Completed,
        Duration::from_micros(2),
    );
    service_metrics::record_tokens(BackendClass::Cpu, TokenClass::Committed, 1);
    let limits = DeliveryLimits::default();
    let global = GlobalDeliveryBudget::new(limits.global_queued_bytes);
    let (cancel_tx, mut cancel_rx) = tokio::sync::mpsc::unbounded_channel();
    let lease = RequestLease::new(RequestId(request + 1), cancel_tx);
    let (publisher, mut receiver) = delivery_session(limits, global.clone(), lease)?;
    publisher.try_publish(CommittedEvent::Usage {
        committed_prompt: 8,
        committed_completion: 1,
    })?;
    publisher.try_publish(CommittedEvent::Delta {
        choice: 0,
        text: "x".into(),
        token_ids: vec![7],
        logprobs: None,
    })?;
    publisher.try_publish(CommittedEvent::Finish {
        choice: 0,
        reason: FinishReason::Length,
    })?;
    publisher.try_publish(CommittedEvent::Done)?;
    drop(publisher);
    let mut events = 0;
    while receiver.recv().await.is_some() {
        events += 1;
    }
    if events != 4 || global.queued_bytes() != 0 || cancel_rx.try_recv().is_ok() {
        bail!("delivery fixture violated terminal cleanup invariants");
    }
    service_metrics::adjust_current_requests(BackendClass::Cpu, -1.0);
    service_metrics::record_terminal(BackendClass::Cpu, ResultClass::Completed, ReasonCode::None);
    Ok(checksum)
}

fn deterministic_execute(seed: u64, work_units: usize) -> u64 {
    let mut value = seed ^ 0x9e37_79b9_7f4a_7c15;
    for _ in 0..work_units {
        value ^= value.rotate_left(13);
        value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value ^= value >> 17;
        black_box(value);
    }
    value
}

fn percentile(sorted: &[u64], quantile: f64) -> u64 {
    let index = ((sorted.len() as f64 * quantile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len().saturating_sub(1));
    sorted[index]
}

fn aggregate(trials: Vec<TrialResult>) -> Aggregate {
    let mut throughput = trials
        .iter()
        .map(|trial| trial.throughput_per_second)
        .collect::<Vec<_>>();
    throughput.sort_by(f64::total_cmp);
    let mut p99 = trials
        .iter()
        .map(|trial| trial.p99_latency_ns as f64)
        .collect::<Vec<_>>();
    p99.sort_by(f64::total_cmp);
    Aggregate {
        median_throughput_per_second: throughput[throughput.len() / 2],
        median_p99_latency_ns: p99[p99.len() / 2],
        trials,
    }
}

fn regression_percent(reference: f64, candidate: f64) -> f64 {
    ((reference - candidate) / reference * 100.0).max(0.0)
}

fn increase_percent(reference: f64, candidate: f64) -> f64 {
    ((candidate - reference) / reference * 100.0).max(0.0)
}

fn write_sidecar(output: &Path, status: EvidenceStatus, raw_bytes: &[u8]) -> Result<()> {
    let artifact = ArtifactRef::from_path("raw-output", output)?;
    if artifact.sha256 != sha256(raw_bytes) {
        bail!("written A/B output does not match its in-memory capture");
    }
    let mut evidence =
        RunManifestV1::new("cpu-service-overhead", "instrumentation-overhead", status);
    evidence.source = local_source_provenance();
    evidence.model = ModelEvidence {
        id: "model-free".into(),
        revision: "delivery-fixture-v1".into(),
        ..ModelEvidence::default()
    };
    evidence.command.argv_redacted = std::env::args().collect();
    evidence.workload = WorkloadEvidence {
        id: "paired-bounded-delivery".into(),
        prompt_sha256: None,
        seed: 0,
        repetitions: 1,
    };
    evidence.artifacts.push(artifact);
    evidence.limitations.push(
        "model-free paired fixture covers bounded delivery and Prometheus recording, not model kernels"
            .into(),
    );
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .context("output has no UTF-8 file name")?;
    evidence.write_atomic(output.with_file_name(format!("{file_name}.manifest.json")))?;
    Ok(())
}

fn local_source_provenance() -> SourceProvenance {
    let command_output = |program: &str, arguments: &[&str]| {
        Command::new(program)
            .args(arguments)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|value| value.trim().to_string())
    };
    SourceProvenance {
        repository_commit: command_output("git", &["rev-parse", "HEAD"])
            .filter(|value| value.len() == 40)
            .unwrap_or_default(),
        dirty: Command::new("git")
            .args(["status", "--porcelain"])
            .output()
            .ok()
            .is_some_and(|output| output.status.success() && !output.stdout.is_empty()),
        branch_role: "candidate".into(),
        cargo_lock_sha256: std::fs::read("Cargo.lock")
            .ok()
            .map(|bytes| sha256(&bytes))
            .unwrap_or_default(),
        toolchain: command_output("rustc", &["--version"]).unwrap_or_else(|| "unknown".into()),
        profile: "release".into(),
        features: Vec::new(),
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_and_regressions_are_stable() {
        assert_eq!(percentile(&[1, 2, 3, 4, 5], 0.5), 3);
        assert_eq!(percentile(&[1, 2, 3, 4, 5], 0.99), 5);
        assert_eq!(regression_percent(100.0, 99.0), 1.0);
        assert_eq!(regression_percent(100.0, 101.0), 0.0);
        assert_eq!(increase_percent(100.0, 101.0), 1.0);
        assert_eq!(increase_percent(100.0, 99.0), 0.0);
    }
}
