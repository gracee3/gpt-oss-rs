use std::ffi::{c_int, c_ulong};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicI32, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use gpt_oss_bench::construction_memory_policy::{
    CONSTRUCTION_MEMORY_EVENT_SCHEMA, MAX_CONSTRUCTION_MEMORY_EVENTS,
    MAX_CONSTRUCTION_MEMORY_EVENT_BYTES, MAX_CONSTRUCTION_MEMORY_JOURNAL_BYTES,
};
use gpt_oss_bench::h8_watchdog::{
    analyze_preflight, evaluate_runtime_guard, read_host_snapshot, sha256_bytes, sha256_file,
    GuardViolation, HostSnapshot, PreflightAnalysis, RuntimeGuardLimits, ENV_EXECUTABLE_SHA256,
    ENV_PARENT_PID, ENV_PREFLIGHT_SHA256, ENV_RUN_ID, ENV_SCHEMA, ENV_SWAP_BASELINE,
    MIN_MEM_AVAILABLE_BYTES, MIN_PREFLIGHT_DURATION_MS, PREFLIGHT_SCHEMA, WATCHDOG_SCHEMA,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAX_PREFLIGHT_DURATION_MS: u64 = 10 * 60 * 1000;
const MAX_RUN_DURATION_MS: u64 = 12 * 60 * 60 * 1000;
const MAX_RETAINED_RUNTIME_SAMPLES: usize = 4_096;
const MAX_COMMAND_ARGUMENTS: usize = 128;
const SIGNAL_POLL_MS: u64 = 100;
const PR_SET_PDEATHSIG: c_int = 1;
const SIGHUP: c_int = 1;
const SIGINT: c_int = 2;
const SIGKILL: c_int = 9;
const SIGTERM: c_int = 15;
const SIGNAL_ERROR: usize = usize::MAX;

static ACTIVE_CHILD_PGID: AtomicI32 = AtomicI32::new(0);

unsafe extern "C" {
    fn prctl(option: c_int, arg2: c_ulong, arg3: c_ulong, arg4: c_ulong, arg5: c_ulong) -> c_int;
    fn getppid() -> c_int;
    fn kill(pid: c_int, signal: c_int) -> c_int;
    fn signal(signal: c_int, handler: extern "C" fn(c_int)) -> usize;
    fn _exit(status: c_int) -> !;
}

extern "C" fn termination_signal_handler(received: c_int) {
    let pgid = ACTIVE_CHILD_PGID.load(Ordering::SeqCst);
    if pgid > 0 {
        // SAFETY: POSIX kill is async-signal-safe. A negative PID targets only
        // the separately created child process group.
        unsafe {
            kill(-pgid, SIGKILL);
        }
    }
    // SAFETY: `_exit` is async-signal-safe and avoids running ordinary Rust
    // destructors from a signal handler. PR_SET_PDEATHSIG independently kills
    // the direct H8 child if the process-group signal did not reach it.
    unsafe {
        _exit(128_i32.saturating_add(received));
    }
}

#[derive(Debug, Parser)]
#[command(about = "Read-only preflight and fail-closed supervisor for an authorized H8 run")]
struct Cli {
    #[command(subcommand)]
    action: Action,
}

#[derive(Debug, Subcommand)]
enum Action {
    /// Observe only; never starts heterogeneous construction.
    Preflight(PreflightArgs),
    /// Start one explicitly authorized H8 child after a passing fresh preflight.
    Run(RunArgs),
    #[cfg(debug_assertions)]
    #[command(hide = true)]
    LifecycleProbe(LifecycleProbeArgs),
}

#[cfg(debug_assertions)]
#[derive(Debug, Args)]
struct LifecycleProbeArgs {
    #[arg(long)]
    ready_file: PathBuf,
}

#[derive(Debug, Args)]
struct PreflightArgs {
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 120)]
    duration_seconds: u64,
    #[arg(long, default_value_t = 30)]
    sample_interval_seconds: u64,
}

#[derive(Debug, Args)]
struct RunArgs {
    #[arg(long)]
    preflight: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 250)]
    poll_interval_ms: u64,
    #[arg(long, default_value_t = 10_000)]
    retain_interval_ms: u64,
    #[arg(long, default_value_t = 30)]
    interrupt_grace_seconds: u64,
    #[arg(long, default_value_t = 21_600)]
    max_run_seconds: u64,
    #[arg(long, default_value_t = 900)]
    max_preflight_age_seconds: u64,
    #[arg(last = true, required = true, num_args = 1..)]
    command: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PreflightEvidence {
    schema: String,
    captured_start_unix_ms: u128,
    captured_end_unix_ms: u128,
    repository_head: String,
    watchdog_executable_sha256: String,
    command: Vec<String>,
    privileged_host_controls_changed: bool,
    samples: Vec<HostSnapshot>,
    sample_chain_sha256: String,
    gpus_before: Vec<GpuSnapshot>,
    gpus_after: Vec<GpuSnapshot>,
    observation_errors: Vec<String>,
    analysis: PreflightAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GpuSnapshot {
    pci_bus_id: String,
    memory_used_mib: u64,
    memory_free_mib: u64,
}

#[derive(Debug, Serialize)]
struct WatchdogRunEvidence {
    schema: &'static str,
    captured_start_unix_ms: u128,
    captured_end_unix_ms: u128,
    repository_head: String,
    watchdog_executable_sha256: String,
    run_id_sha256: String,
    preflight_path: String,
    preflight_sha256: String,
    preflight_age_ms_at_admission: u128,
    command: Vec<String>,
    child_executable_sha256: String,
    child_evidence_path: String,
    child_evidence_sha256: Option<String>,
    poll_interval_ms: u64,
    retain_interval_ms: u64,
    max_run_ms: u64,
    swap_baseline_bytes: u64,
    samples_observed: u64,
    sample_chain_sha256: String,
    retained_samples: Vec<HostSnapshot>,
    minimum_mem_available_bytes: u64,
    maximum_swap_used_bytes: u64,
    maximum_target_tree_vm_swap_bytes: u64,
    violation: Option<GuardViolation>,
    observation_error: Option<String>,
    signal_actions: Vec<SignalAction>,
    child_status: Option<ChildStatus>,
    gpus_before: Vec<GpuSnapshot>,
    gpus_after: Vec<GpuSnapshot>,
    protected_nvme_safe_throughout: bool,
    privileged_host_controls_changed: bool,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct SignalAction {
    signal: &'static str,
    process_group_only: bool,
    command_succeeded: bool,
    group_empty_after: bool,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChildStatus {
    success: bool,
    code: Option<i32>,
    unix_signal: Option<i32>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.action {
        Action::Preflight(args) => run_preflight(args),
        Action::Run(args) => run_supervised(args),
        #[cfg(debug_assertions)]
        Action::LifecycleProbe(args) => run_lifecycle_probe(args),
    }
}

#[cfg(debug_assertions)]
fn run_lifecycle_probe(args: LifecycleProbeArgs) -> Result<()> {
    install_termination_handlers()?;
    let expected_parent = c_int::try_from(std::process::id()).context("probe PID exceeds i32")?;
    let mut command = Command::new("/bin/sleep");
    command
        .arg("600")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0);
    configure_parent_death(&mut command, expected_parent);
    let child = ChildGroupGuard::spawn(command)?;
    fs::write(&args.ready_file, format!("{}\n", child.id()))?;
    loop {
        thread::sleep(Duration::from_secs(60));
    }
}

fn run_preflight(args: PreflightArgs) -> Result<()> {
    let duration_ms = args
        .duration_seconds
        .checked_mul(1_000)
        .context("preflight duration overflows")?;
    let interval_ms = args
        .sample_interval_seconds
        .checked_mul(1_000)
        .context("preflight interval overflows")?;
    validate_preflight_bounds(duration_ms, interval_ms)?;
    ensure_new_output(&args.output)?;

    let executable_sha256 = sha256_file(Path::new("/proc/self/exe"))?;
    let started_unix_ms = now_unix_ms();
    let gpus_before = gpu_snapshot()?;
    let mut samples = Vec::with_capacity(
        usize::try_from(duration_ms / interval_ms)
            .unwrap_or(0)
            .saturating_add(2),
    );
    let mut observation_errors = Vec::new();

    match read_host_snapshot(None, 0) {
        Ok(sample) => samples.push(sample),
        Err(error) => observation_errors.push(format!("host observation failed: {error:#}")),
    }
    // The stability window begins at the first retained sample, not at process
    // startup or the preceding GPU query. Every later elapsed value is relative
    // to this exact evidence boundary.
    let window_started = Instant::now();
    while observation_errors.is_empty() {
        let elapsed_ms = elapsed_millis_u64(window_started)?;
        if elapsed_ms >= duration_ms {
            match read_host_snapshot(None, elapsed_ms) {
                Ok(sample) => samples.push(sample),
                Err(error) => {
                    observation_errors.push(format!("host observation failed: {error:#}"));
                }
            }
            break;
        }
        let remaining = duration_ms.saturating_sub(elapsed_ms);
        thread::sleep(Duration::from_millis(interval_ms.min(remaining)));
        let elapsed_ms = elapsed_millis_u64(window_started)?;
        match read_host_snapshot(None, elapsed_ms) {
            Ok(sample) => samples.push(sample),
            Err(error) => {
                observation_errors.push(format!("host observation failed: {error:#}"));
            }
        }
        if elapsed_ms >= duration_ms || !observation_errors.is_empty() {
            break;
        }
    }

    let mut analysis = analyze_preflight(&samples, duration_ms);
    if !observation_errors.is_empty() {
        analysis.passed = false;
        analysis.failures.extend(observation_errors.clone());
    }
    let evidence = PreflightEvidence {
        schema: PREFLIGHT_SCHEMA.into(),
        captured_start_unix_ms: started_unix_ms,
        captured_end_unix_ms: now_unix_ms(),
        repository_head: command_text("git", &["rev-parse", "HEAD"]),
        watchdog_executable_sha256: executable_sha256,
        command: std::env::args().collect(),
        privileged_host_controls_changed: false,
        sample_chain_sha256: sample_chain_sha256(&samples)?,
        samples,
        gpus_before,
        gpus_after: gpu_snapshot()?,
        observation_errors,
        analysis,
    };
    write_json_new(&args.output, &evidence)?;
    if !evidence.analysis.passed {
        bail!(
            "H8 preflight failed; immutable evidence written to {}: {:?}",
            args.output.display(),
            evidence.analysis.failures
        );
    }
    Ok(())
}

fn run_supervised(args: RunArgs) -> Result<()> {
    validate_run_bounds(&args)?;
    ensure_new_output(&args.output)?;
    let preflight_bytes = fs::read(&args.preflight)?;
    let preflight: PreflightEvidence = serde_json::from_slice(&preflight_bytes)?;
    validate_preflight_evidence(&preflight, args.max_preflight_age_seconds)?;
    let preflight_sha256 = sha256_bytes(&preflight_bytes);
    let baseline = preflight
        .samples
        .last()
        .context("passing preflight has no final sample")?;
    let current = read_host_snapshot(None, 0)?;
    validate_fresh_admission(baseline, &current)?;
    let child_spec = validate_child_command(&args.command, &args.preflight, &args.output)?;

    let watchdog_executable_sha256 = sha256_file(Path::new("/proc/self/exe"))?;
    if watchdog_executable_sha256 != preflight.watchdog_executable_sha256 {
        bail!("preflight was produced by a different watchdog executable");
    }
    let started_unix_ms = now_unix_ms();
    let run_id_sha256 = sha256_bytes(
        format!(
            "{preflight_sha256}\n{watchdog_executable_sha256}\n{started_unix_ms}\n{}",
            args.command.join("\0")
        )
        .as_bytes(),
    );
    let max_run_ms = args
        .max_run_seconds
        .checked_mul(1_000)
        .context("maximum run duration overflows")?;
    let gpus_before = gpu_snapshot()?;
    let repository_head = command_text("git", &["rev-parse", "HEAD"]);
    if repository_head == "unknown" || repository_head != preflight.repository_head {
        bail!("repository HEAD changed after the reviewed H8 preflight");
    }
    let preflight_age_ms_at_admission =
        started_unix_ms.saturating_sub(preflight.captured_end_unix_ms);

    install_termination_handlers()?;
    let executable_fd_path = format!("/proc/self/fd/{}", child_spec.executable_file.as_raw_fd());
    let expected_parent =
        c_int::try_from(std::process::id()).context("watchdog PID exceeds i32")?;
    let mut command = Command::new(executable_fd_path);
    command
        .args(&child_spec.arguments)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .process_group(0)
        .env(ENV_SCHEMA, WATCHDOG_SCHEMA)
        .env(ENV_PARENT_PID, std::process::id().to_string())
        .env(ENV_RUN_ID, &run_id_sha256)
        .env(ENV_PREFLIGHT_SHA256, &preflight_sha256)
        .env(ENV_EXECUTABLE_SHA256, &watchdog_executable_sha256)
        .env(ENV_SWAP_BASELINE, baseline.swap_used_bytes.to_string());
    configure_parent_death(&mut command, expected_parent);
    let mut child = ChildGroupGuard::spawn(command)?;
    let spawned_executable_sha256 =
        sha256_file(&PathBuf::from(format!("/proc/{}/exe", child.id())))?;
    if spawned_executable_sha256 != child_spec.executable_sha256 {
        bail!("spawned H8 executable identity differs from the opened inode");
    }
    let started = Instant::now();
    let limits = RuntimeGuardLimits {
        swap_baseline_bytes: baseline.swap_used_bytes,
        min_mem_available_bytes: MIN_MEM_AVAILABLE_BYTES,
    };
    let mut retained_samples = Vec::with_capacity(
        usize::try_from(max_run_ms / args.retain_interval_ms)
            .unwrap_or(MAX_RETAINED_RUNTIME_SAMPLES)
            .saturating_add(2)
            .min(MAX_RETAINED_RUNTIME_SAMPLES),
    );
    let mut samples_observed = 0_u64;
    let mut sample_hasher = Sha256::new();
    let mut next_retain_ms = 0_u64;
    let mut minimum_mem_available_bytes = u64::MAX;
    let mut maximum_swap_used_bytes = 0_u64;
    let mut maximum_target_tree_vm_swap_bytes = 0_u64;
    let mut protected_nvme_safe_throughout = true;
    let mut violation = None;
    let mut observation_error = None;

    loop {
        let elapsed_ms = elapsed_millis_u64(started)?;
        let sample = match read_host_snapshot(Some(child.id()), elapsed_ms) {
            Ok(sample) => sample,
            Err(error) => {
                observation_error = Some(format!("host observation failed closed: {error:#}"));
                break;
            }
        };
        samples_observed = samples_observed.saturating_add(1);
        hash_sample(&mut sample_hasher, &sample)?;
        minimum_mem_available_bytes = minimum_mem_available_bytes.min(sample.mem_available_bytes);
        maximum_swap_used_bytes = maximum_swap_used_bytes.max(sample.swap_used_bytes);
        maximum_target_tree_vm_swap_bytes =
            maximum_target_tree_vm_swap_bytes.max(sample.target_tree_vm_swap_bytes);
        protected_nvme_safe_throughout &=
            sample.protected_nvme_read_only && !sample.protected_nvme_mounted;
        if elapsed_ms >= next_retain_ms {
            if retained_samples.len() >= MAX_RETAINED_RUNTIME_SAMPLES {
                observation_error = Some("bounded retained-sample capacity exhausted".into());
                break;
            }
            retained_samples.push(sample.clone());
            next_retain_ms = next_retain_ms.saturating_add(args.retain_interval_ms);
        }
        if let Err(failure) = evaluate_runtime_guard(&sample, &limits) {
            if retained_samples.last() != Some(&sample) {
                retained_samples.push(sample);
            }
            violation = Some(failure);
            break;
        }
        if elapsed_ms > max_run_ms {
            violation = Some(GuardViolation {
                reasons: vec!["guarded H8 child exceeded the bounded run duration".into()],
            });
            break;
        }
        if child.poll_reaped()? {
            break;
        }
        thread::sleep(Duration::from_millis(args.poll_interval_ms));
    }

    let mut signal_actions = Vec::new();
    if !child.poll_reaped()? {
        terminate_process_group(
            &mut child,
            Duration::from_secs(args.interrupt_grace_seconds),
            &mut signal_actions,
        )?;
    } else {
        if child.finish_reaped_group(&mut signal_actions)? {
            merge_violation(
                &mut violation,
                GuardViolation {
                    reasons: vec![
                        "child leader exited while another guarded process-group member remained"
                            .into(),
                    ],
                },
            );
        }
    }

    let final_elapsed_ms = elapsed_millis_u64(started)?;
    match read_host_snapshot(None, final_elapsed_ms) {
        Ok(final_sample) => {
            samples_observed = samples_observed.saturating_add(1);
            hash_sample(&mut sample_hasher, &final_sample)?;
            minimum_mem_available_bytes =
                minimum_mem_available_bytes.min(final_sample.mem_available_bytes);
            maximum_swap_used_bytes = maximum_swap_used_bytes.max(final_sample.swap_used_bytes);
            maximum_target_tree_vm_swap_bytes =
                maximum_target_tree_vm_swap_bytes.max(final_sample.target_tree_vm_swap_bytes);
            protected_nvme_safe_throughout &=
                final_sample.protected_nvme_read_only && !final_sample.protected_nvme_mounted;
            if let Err(final_violation) = evaluate_runtime_guard(&final_sample, &limits) {
                merge_violation(&mut violation, final_violation);
            }
            if final_sample.active_h8_process_found {
                merge_violation(
                    &mut violation,
                    GuardViolation {
                        reasons: vec!["an H8 process remained after guarded child exit".into()],
                    },
                );
            }
            retained_samples.push(final_sample);
        }
        Err(error) => {
            observation_error = Some(format!(
                "mandatory post-exit host observation failed closed: {error:#}"
            ));
        }
    }

    let child_status_record = child.status().map(status_record);
    let child_succeeded = child.status().is_some_and(ExitStatus::success);
    let child_evidence_sha256 = if child_spec.evidence_output.is_file() {
        match sha256_file(&child_spec.evidence_output) {
            Ok(hash) => match validate_child_evidence(
                &child_spec.evidence_output,
                &child_spec.memory_events,
                &child_spec.executable_sha256,
                &repository_head,
                &run_id_sha256,
                &preflight_sha256,
            ) {
                Ok(()) => Some(hash),
                Err(error) => {
                    observation_error =
                        Some(format!("child evidence identity failed closed: {error:#}"));
                    None
                }
            },
            Err(error) => {
                observation_error = Some(format!("child evidence hash failed closed: {error:#}"));
                None
            }
        }
    } else {
        None
    };
    let gpus_after = match gpu_snapshot() {
        Ok(gpus) => gpus,
        Err(error) => {
            observation_error = Some(format!(
                "post-exit GPU observation failed closed: {error:#}"
            ));
            Vec::new()
        }
    };
    if minimum_mem_available_bytes == u64::MAX {
        minimum_mem_available_bytes = 0;
    }
    let passed = violation.is_none()
        && observation_error.is_none()
        && child_succeeded
        && child_evidence_sha256.is_some()
        && maximum_swap_used_bytes <= baseline.swap_used_bytes
        && maximum_target_tree_vm_swap_bytes == 0
        && protected_nvme_safe_throughout;
    let evidence = WatchdogRunEvidence {
        schema: WATCHDOG_SCHEMA,
        captured_start_unix_ms: started_unix_ms,
        captured_end_unix_ms: now_unix_ms(),
        repository_head,
        watchdog_executable_sha256,
        run_id_sha256,
        preflight_path: args.preflight.to_string_lossy().into_owned(),
        preflight_sha256,
        preflight_age_ms_at_admission,
        command: args.command,
        child_executable_sha256: child_spec.executable_sha256,
        child_evidence_path: child_spec.evidence_output.to_string_lossy().into_owned(),
        child_evidence_sha256,
        poll_interval_ms: args.poll_interval_ms,
        retain_interval_ms: args.retain_interval_ms,
        max_run_ms,
        swap_baseline_bytes: baseline.swap_used_bytes,
        samples_observed,
        sample_chain_sha256: format!("{:x}", sample_hasher.finalize()),
        retained_samples,
        minimum_mem_available_bytes,
        maximum_swap_used_bytes,
        maximum_target_tree_vm_swap_bytes,
        violation,
        observation_error,
        signal_actions,
        child_status: child_status_record,
        gpus_before,
        gpus_after,
        protected_nvme_safe_throughout,
        privileged_host_controls_changed: false,
        passed,
    };
    write_json_new(&args.output, &evidence)?;
    if !passed {
        bail!(
            "guarded H8 run did not pass; immutable watchdog evidence written to {}",
            args.output.display()
        );
    }
    Ok(())
}

#[derive(Debug)]
struct ChildSpec {
    executable_file: File,
    executable_sha256: String,
    arguments: Vec<String>,
    evidence_output: PathBuf,
    memory_events: PathBuf,
}

fn validate_child_command(
    command: &[String],
    preflight: &Path,
    output: &Path,
) -> Result<ChildSpec> {
    if command.is_empty() || command.len() > MAX_COMMAND_ARGUMENTS {
        bail!("guarded command has an invalid argument count");
    }
    let executable = fs::canonicalize(&command[0])?;
    if executable.file_name().and_then(|name| name.to_str()) != Some("heterogeneous_construct") {
        bail!("watchdog may launch only the heterogeneous_construct binary");
    }
    let arguments = command[1..].to_vec();
    if single_argument_value(&arguments, "--mode")?.as_deref() != Some("h8") {
        bail!("watchdog child must request exactly one --mode h8");
    }
    let evidence_output =
        argument_path(&arguments, "--output")?.context("guarded H8 child lacks --output")?;
    let memory_events = argument_path(&arguments, "--memory-events")?
        .context("guarded H8 child lacks --memory-events")?;
    if evidence_output.exists() {
        bail!("guarded H8 child output already exists");
    }
    if memory_events.exists() {
        bail!("guarded H8 child memory-event directory already exists");
    }
    if !memory_events.parent().is_some_and(Path::is_dir) {
        bail!("guarded H8 child memory-event parent is not a directory");
    }
    if paths_same_lexically(&evidence_output, preflight)
        || paths_same_lexically(&evidence_output, output)
        || paths_same_lexically(&memory_events, &evidence_output)
        || paths_same_lexically(&memory_events, preflight)
        || paths_same_lexically(&memory_events, output)
        || paths_same_lexically(preflight, output)
    {
        bail!("watchdog, preflight, and child evidence paths must be distinct");
    }
    let parent = evidence_output
        .parent()
        .context("guarded H8 child output has no parent")?;
    if parent.join("placement-120b-h8.json").exists() {
        bail!("guarded H8 evidence directory already contains a placement artifact");
    }
    let mut executable_file = File::open(&executable)?;
    if !executable_file.metadata()?.is_file() {
        bail!("guarded H8 executable is not a regular file");
    }
    let executable_sha256 = sha256_open_file(&mut executable_file)?;
    Ok(ChildSpec {
        executable_file,
        executable_sha256,
        arguments,
        evidence_output,
        memory_events,
    })
}

fn single_argument_value(arguments: &[String], flag: &str) -> Result<Option<String>> {
    let mut found = None;
    let prefix = format!("{flag}=");
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        let value = if argument == flag {
            index = index.saturating_add(1);
            Some(
                arguments
                    .get(index)
                    .with_context(|| format!("{flag} lacks a value"))?
                    .clone(),
            )
        } else {
            argument.strip_prefix(&prefix).map(str::to_owned)
        };
        if let Some(value) = value {
            if found.replace(value).is_some() {
                bail!("{flag} appears more than once");
            }
        }
        index = index.saturating_add(1);
    }
    Ok(found)
}

fn argument_path(arguments: &[String], flag: &str) -> Result<Option<PathBuf>> {
    Ok(single_argument_value(arguments, flag)?.map(PathBuf::from))
}

fn paths_same_lexically(first: &Path, second: &Path) -> bool {
    absolute_lexical(first) == absolute_lexical(second)
}

fn absolute_lexical(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn validate_preflight_bounds(duration_ms: u64, interval_ms: u64) -> Result<()> {
    if !(MIN_PREFLIGHT_DURATION_MS..=MAX_PREFLIGHT_DURATION_MS).contains(&duration_ms) {
        bail!("preflight duration is outside the reviewed bounds");
    }
    if interval_ms == 0 || interval_ms > duration_ms / 3 {
        bail!("preflight interval cannot produce four boundary samples");
    }
    Ok(())
}

fn validate_run_bounds(args: &RunArgs) -> Result<()> {
    let max_run_ms = args
        .max_run_seconds
        .checked_mul(1_000)
        .context("maximum run duration overflows")?;
    if !(100..=5_000).contains(&args.poll_interval_ms) {
        bail!("watchdog polling interval is outside 100..=5000 ms");
    }
    if args.retain_interval_ms < args.poll_interval_ms || args.retain_interval_ms > 60_000 {
        bail!("retained-sample interval is outside the reviewed bounds");
    }
    if max_run_ms == 0 || max_run_ms > MAX_RUN_DURATION_MS {
        bail!("maximum H8 duration exceeds the 12-hour hard bound");
    }
    let retained = max_run_ms / args.retain_interval_ms + 2;
    if retained > MAX_RETAINED_RUNTIME_SAMPLES as u64 {
        bail!("configured H8 run can exceed the bounded retained-sample capacity");
    }
    if args.interrupt_grace_seconds == 0 || args.interrupt_grace_seconds > 120 {
        bail!("interrupt grace is outside 1..=120 seconds");
    }
    if args.max_preflight_age_seconds == 0 || args.max_preflight_age_seconds > 3_600 {
        bail!("preflight maximum age is outside 1..=3600 seconds");
    }
    Ok(())
}

fn validate_preflight_evidence(evidence: &PreflightEvidence, max_age_seconds: u64) -> Result<()> {
    if evidence.schema != PREFLIGHT_SCHEMA {
        bail!("H8 run requires a passing v1 preflight");
    }
    if evidence.privileged_host_controls_changed || !evidence.observation_errors.is_empty() {
        bail!("H8 preflight used host controls or retained observation errors");
    }
    if evidence.samples.is_empty()
        || sample_chain_sha256(&evidence.samples)? != evidence.sample_chain_sha256
    {
        bail!("preflight sample chain identity is invalid");
    }
    let recomputed = analyze_preflight(&evidence.samples, evidence.analysis.required_duration_ms);
    if !recomputed.passed || recomputed != evidence.analysis {
        bail!("serialized H8 preflight analysis does not match its recorded samples");
    }
    let max_age_ms = u128::from(max_age_seconds)
        .checked_mul(1_000)
        .context("preflight maximum age overflows")?;
    if now_unix_ms().saturating_sub(evidence.captured_end_unix_ms) > max_age_ms {
        bail!("passing H8 preflight is stale");
    }
    Ok(())
}

fn validate_fresh_admission(baseline: &HostSnapshot, current: &HostSnapshot) -> Result<()> {
    if baseline.swap_free_bytes != current.swap_free_bytes
        || baseline.swap_used_bytes != current.swap_used_bytes
        || baseline.swap_cached_bytes != current.swap_cached_bytes
    {
        bail!("swap counters changed after the reviewed stability window");
    }
    let limits = RuntimeGuardLimits {
        swap_baseline_bytes: baseline.swap_used_bytes,
        min_mem_available_bytes: MIN_MEM_AVAILABLE_BYTES,
    };
    evaluate_runtime_guard(current, &limits)
        .map_err(|violation| anyhow::anyhow!("fresh H8 admission failed: {violation:?}"))?;
    if current.active_h8_process_found {
        bail!("an H8 construction process is already active");
    }
    Ok(())
}

fn validate_child_evidence(
    path: &Path,
    memory_events: &Path,
    executable_sha256: &str,
    repository_head: &str,
    run_id_sha256: &str,
    preflight_sha256: &str,
) -> Result<()> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let string = |pointer: &str| {
        value
            .pointer(pointer)
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("child evidence lacks {pointer}"))
    };
    if string("/schema")? != "gpt-oss-rs.heterogeneous-construction/v5"
        || string("/mode")? != "h8"
        || string("/repository_head")? != repository_head
        || string("/executable_sha256")? != executable_sha256
        || string("/h8_campaign/watchdog/schema")? != WATCHDOG_SCHEMA
        || string("/h8_campaign/watchdog/run_id_sha256")? != run_id_sha256
        || string("/h8_campaign/watchdog/preflight_sha256")? != preflight_sha256
        || value.pointer("/h8_campaign").is_none()
        || value
            .pointer("/passed")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || value
            .pointer("/h8_campaign/watchdog/direct_parent_validated")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || value
            .pointer("/h8_campaign/watchdog/parent_executable_validated")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
    {
        bail!("child evidence does not match the admitted H8 run identity");
    }
    validate_construction_memory_journal(
        &value,
        memory_events,
        executable_sha256,
        repository_head,
    )?;
    Ok(())
}

fn validate_construction_memory_journal(
    evidence: &serde_json::Value,
    root: &Path,
    executable_sha256: &str,
    repository_head: &str,
) -> Result<()> {
    let summary = evidence
        .pointer("/construction_memory_events")
        .context("child evidence lacks construction memory journal summary")?;
    let number = |field: &str| {
        summary
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .with_context(|| format!("construction memory summary lacks {field}"))
    };
    if summary.get("schema").and_then(serde_json::Value::as_str)
        != Some(CONSTRUCTION_MEMORY_EVENT_SCHEMA)
        || summary
            .get("persisted")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || number("max_events")? != MAX_CONSTRUCTION_MEMORY_EVENTS as u64
        || number("max_event_bytes")? != MAX_CONSTRUCTION_MEMORY_EVENT_BYTES as u64
        || number("max_total_bytes")? != MAX_CONSTRUCTION_MEMORY_JOURNAL_BYTES as u64
    {
        bail!("construction memory journal policy does not match the reviewed bounds");
    }
    let entries = summary
        .get("entries")
        .and_then(serde_json::Value::as_array)
        .context("construction memory summary lacks entries")?;
    let event_count = usize::try_from(number("event_count")?)?;
    if event_count == 0
        || event_count > MAX_CONSTRUCTION_MEMORY_EVENTS
        || entries.len() != event_count
    {
        bail!("construction memory journal event count is invalid");
    }
    let metadata = fs::symlink_metadata(root)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        bail!("construction memory journal root is not a real directory");
    }
    let expected_checkpoint_revision = evidence
        .pointer("/checkpoint_120b/revision")
        .and_then(serde_json::Value::as_str)
        .context("child evidence lacks the 120B revision")?;
    let expected_checkpoint_metadata = evidence
        .pointer("/checkpoint_120b/metadata_sha256")
        .and_then(serde_json::Value::as_str)
        .context("child evidence lacks the 120B metadata identity")?;
    let expected_checkpoint_mapping = evidence
        .pointer("/checkpoint_120b/mapping_sha256")
        .and_then(serde_json::Value::as_str)
        .context("child evidence lacks the 120B mapping identity")?;
    let expected_placement = evidence
        .pointer("/placement_120b/manifest_sha256")
        .and_then(serde_json::Value::as_str)
        .context("child evidence lacks the 120B placement identity")?;
    let mut observed_total = 0_u64;
    for (expected_sequence, entry) in entries.iter().enumerate() {
        let sequence = entry
            .get("sequence")
            .and_then(serde_json::Value::as_u64)
            .context("construction memory entry lacks sequence")?;
        if sequence != expected_sequence as u64 {
            bail!("construction memory event sequence is not contiguous");
        }
        let filename = entry
            .get("filename")
            .and_then(serde_json::Value::as_str)
            .context("construction memory entry lacks filename")?;
        let mut components = Path::new(filename).components();
        if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
            bail!("construction memory event filename is not a bounded leaf");
        }
        let event_path = root.join(filename);
        let event_metadata = fs::symlink_metadata(&event_path)?;
        if !event_metadata.is_file() || event_metadata.file_type().is_symlink() {
            bail!("construction memory event is not a regular file");
        }
        let expected_bytes = entry
            .get("bytes")
            .and_then(serde_json::Value::as_u64)
            .context("construction memory entry lacks byte count")?;
        let expected_sha256 = entry
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .context("construction memory entry lacks SHA-256")?;
        let event_bytes = fs::read(&event_path)?;
        if expected_bytes == 0
            || expected_bytes > MAX_CONSTRUCTION_MEMORY_EVENT_BYTES as u64
            || event_metadata.len() != expected_bytes
            || event_bytes.len() as u64 != expected_bytes
            || sha256_bytes(&event_bytes) != expected_sha256
        {
            bail!("construction memory event bytes or identity differ");
        }
        let event: serde_json::Value = serde_json::from_slice(&event_bytes)?;
        let diagnostic_fields_present = [
            "/captured_unix_ms",
            "/elapsed_ms",
            "/memory/process_status/vm_rss_bytes",
            "/memory/process_status/vm_swap_bytes",
            "/memory/smaps_rollup/pss_anon_bytes",
            "/memory/smaps_rollup/pss_file_bytes",
            "/memory/global_meminfo/swap_used_bytes",
            "/memory/vmstat/pswpin_pages",
            "/memory/vmstat/pswpout_pages",
            "/memory/current_cgroup/memory_current_bytes",
            "/memory/current_cgroup/memory_swap_current_bytes",
            "/memory/residency/global_page_cache_estimate_bytes",
        ]
        .into_iter()
        .all(|pointer| {
            event
                .pointer(pointer)
                .and_then(serde_json::Value::as_u64)
                .is_some()
        }) && matches!(
            event.get("phase").and_then(serde_json::Value::as_str),
            Some("before_checkpoint_open" | "after_checkpoint_open" | "stage" | "post_drop")
        ) && event
            .get("gpus")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|gpus| gpus.len() == 2);
        if event.get("schema").and_then(serde_json::Value::as_str)
            != Some(CONSTRUCTION_MEMORY_EVENT_SCHEMA)
            || event.get("sequence").and_then(serde_json::Value::as_u64) != Some(sequence)
            || event
                .pointer("/identity/repository_head")
                .and_then(serde_json::Value::as_str)
                != Some(repository_head)
            || event
                .pointer("/identity/executable_sha256")
                .and_then(serde_json::Value::as_str)
                != Some(executable_sha256)
            || event
                .pointer("/identity/checkpoint_class")
                .and_then(serde_json::Value::as_str)
                != Some("120b_h8")
            || event
                .pointer("/identity/checkpoint_revision")
                .and_then(serde_json::Value::as_str)
                != Some(expected_checkpoint_revision)
            || event
                .pointer("/identity/checkpoint_metadata_sha256")
                .and_then(serde_json::Value::as_str)
                != Some(expected_checkpoint_metadata)
            || event
                .pointer("/identity/checkpoint_mapping_sha256")
                .and_then(serde_json::Value::as_str)
                != Some(expected_checkpoint_mapping)
            || event
                .pointer("/identity/placement_manifest_sha256")
                .and_then(serde_json::Value::as_str)
                != Some(expected_placement)
            || !diagnostic_fields_present
        {
            bail!("construction memory event identity differs from the child run");
        }
        observed_total = observed_total
            .checked_add(expected_bytes)
            .context("construction memory journal total overflows")?;
    }
    let directory_entries = fs::read_dir(root)?.collect::<std::io::Result<Vec<_>>>()?;
    if directory_entries.len() != entries.len()
        || observed_total != number("encoded_event_bytes")?
        || observed_total != number("persisted_bytes")?
        || observed_total > MAX_CONSTRUCTION_MEMORY_JOURNAL_BYTES as u64
    {
        bail!("construction memory journal summary differs from its immutable files");
    }
    Ok(())
}

struct ChildGroupGuard {
    child: Option<Child>,
    status: Option<ExitStatus>,
    pgid: c_int,
    group_disarmed: bool,
}

impl ChildGroupGuard {
    fn spawn(mut command: Command) -> Result<Self> {
        if ACTIVE_CHILD_PGID.load(Ordering::SeqCst) != 0 {
            bail!("watchdog already owns a child process group");
        }
        let mut child = command
            .spawn()
            .context("failed to start guarded H8 child")?;
        let pgid = match c_int::try_from(child.id()) {
            Ok(pgid) => pgid,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error).context("H8 child PID exceeds i32");
            }
        };
        if ACTIVE_CHILD_PGID
            .compare_exchange(0, pgid, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            let _ = signal_process_group(pgid, SIGKILL);
            let _ = child.kill();
            let _ = child.wait();
            bail!("watchdog child ownership changed during spawn");
        }
        Ok(Self {
            child: Some(child),
            status: None,
            pgid,
            group_disarmed: false,
        })
    }

    fn id(&self) -> u32 {
        u32::try_from(self.pgid).expect("positive child process group")
    }

    fn poll_reaped(&mut self) -> Result<bool> {
        if self.status.is_some() {
            return Ok(true);
        }
        let status = self
            .child
            .as_mut()
            .context("active child status lacks owned process")?
            .try_wait()?;
        if let Some(status) = status {
            self.status = Some(status);
            self.child.take();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn status(&self) -> Option<&ExitStatus> {
        self.status.as_ref()
    }

    fn group_exists(&self) -> Result<bool> {
        process_group_exists(self.pgid)
    }

    fn finish_reaped_group(&mut self, actions: &mut Vec<SignalAction>) -> Result<bool> {
        if !self.poll_reaped()? {
            bail!("cannot finish a live child as a reaped group");
        }
        let had_remaining_members = self.group_exists()?;
        if had_remaining_members {
            let result = signal_process_group(self.pgid, SIGKILL);
            let error = result.as_ref().err().map(ToString::to_string);
            wait_for_group_empty(self.pgid);
            actions.push(SignalAction {
                signal: "SIGKILL-after-leader-exit",
                process_group_only: true,
                command_succeeded: result.is_ok(),
                group_empty_after: true,
                error,
            });
        }
        self.disarm_group()?;
        Ok(had_remaining_members)
    }

    fn disarm_group(&mut self) -> Result<()> {
        if self.group_exists()? {
            bail!("cannot disarm a nonempty H8 process group");
        }
        ACTIVE_CHILD_PGID
            .compare_exchange(self.pgid, 0, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|actual| {
                anyhow::anyhow!(
                    "active child process group identity changed: expected={} actual={actual}",
                    self.pgid
                )
            })?;
        self.group_disarmed = true;
        Ok(())
    }
}

impl Drop for ChildGroupGuard {
    fn drop(&mut self) {
        if self.group_disarmed {
            return;
        }
        let _ = signal_process_group(self.pgid, SIGKILL);
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        wait_for_group_empty(self.pgid);
        let _ =
            ACTIVE_CHILD_PGID.compare_exchange(self.pgid, 0, Ordering::SeqCst, Ordering::SeqCst);
        self.group_disarmed = true;
    }
}

fn terminate_process_group(
    child: &mut ChildGroupGuard,
    interrupt_grace: Duration,
    actions: &mut Vec<SignalAction>,
) -> Result<()> {
    let interrupt = signal_process_group(child.pgid, SIGINT);
    let mut interrupt_record = SignalAction {
        signal: "SIGINT",
        process_group_only: true,
        command_succeeded: interrupt.is_ok(),
        group_empty_after: false,
        error: interrupt.err().map(|error| error.to_string()),
    };
    let deadline = Instant::now() + interrupt_grace;
    while Instant::now() < deadline {
        child.poll_reaped()?;
        if !child.group_exists()? {
            interrupt_record.group_empty_after = true;
            actions.push(interrupt_record);
            child.disarm_group()?;
            return Ok(());
        }
        thread::sleep(Duration::from_millis(SIGNAL_POLL_MS));
    }
    actions.push(interrupt_record);

    let kill = signal_process_group(child.pgid, SIGKILL);
    let mut kill_record = SignalAction {
        signal: "SIGKILL",
        process_group_only: true,
        command_succeeded: kill.is_ok(),
        group_empty_after: false,
        error: kill.err().map(|error| error.to_string()),
    };
    if let Some(process) = child.child.as_mut() {
        let _ = process.kill();
        let status = process.wait()?;
        child.status = Some(status);
        child.child.take();
    }
    wait_for_group_empty(child.pgid);
    kill_record.group_empty_after = true;
    actions.push(kill_record);
    child.disarm_group()?;
    Ok(())
}

fn signal_process_group(pgid: c_int, requested_signal: c_int) -> Result<()> {
    // SAFETY: pgid is the positive process-group leader created by this
    // watchdog. Negating it confines the signal to that group.
    if unsafe { kill(-pgid, requested_signal) } != 0 {
        return Err(std::io::Error::last_os_error()).context("process-group signal failed");
    }
    Ok(())
}

fn process_group_exists(pgid: c_int) -> Result<bool> {
    // SAFETY: signal 0 performs permission/existence checking only.
    if unsafe { kill(-pgid, 0) } == 0 {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(3) {
        Ok(false)
    } else {
        Err(error).context("process-group existence check failed")
    }
}

fn wait_for_group_empty(pgid: c_int) {
    while process_group_exists(pgid).unwrap_or(true) {
        let _ = signal_process_group(pgid, SIGKILL);
        thread::sleep(Duration::from_millis(SIGNAL_POLL_MS));
    }
}

fn install_termination_handlers() -> Result<()> {
    for requested_signal in [SIGHUP, SIGINT, SIGTERM] {
        // SAFETY: installs one static async-signal-safe handler before the
        // guarded child is spawned. No handler state is stack-borrowed.
        if unsafe { signal(requested_signal, termination_signal_handler) } == SIGNAL_ERROR {
            return Err(std::io::Error::last_os_error())
                .context("failed to install watchdog termination handler");
        }
    }
    Ok(())
}

fn configure_parent_death(command: &mut Command, expected_parent: c_int) {
    // SAFETY: `pre_exec` runs after fork and before exec. The closure calls only
    // async-signal-safe Linux syscalls, owns its captured integer, and returns
    // an I/O error rather than touching shared Rust state.
    unsafe {
        command.pre_exec(move || {
            if prctl(PR_SET_PDEATHSIG, SIGKILL as c_ulong, 0, 0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if getppid() != expected_parent {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "watchdog parent changed before H8 exec",
                ));
            }
            Ok(())
        });
    }
}

fn merge_violation(destination: &mut Option<GuardViolation>, mut additional: GuardViolation) {
    if let Some(existing) = destination {
        existing.reasons.append(&mut additional.reasons);
    } else {
        *destination = Some(additional);
    }
}

fn status_record(status: &ExitStatus) -> ChildStatus {
    use std::os::unix::process::ExitStatusExt;
    ChildStatus {
        success: status.success(),
        code: status.code(),
        unix_signal: status.signal(),
    }
}

fn gpu_snapshot() -> Result<Vec<GpuSnapshot>> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=pci.bus_id,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !output.status.success() {
        bail!("nvidia-smi memory query failed");
    }
    String::from_utf8(output.stdout)?
        .lines()
        .map(|line| {
            let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
            if fields.len() != 3 {
                bail!("unexpected nvidia-smi memory row");
            }
            Ok(GpuSnapshot {
                pci_bus_id: fields[0].to_ascii_lowercase(),
                memory_used_mib: fields[1].parse()?,
                memory_free_mib: fields[2].parse()?,
            })
        })
        .collect()
}

fn sha256_open_file(file: &mut File) -> Result<String> {
    file.seek(SeekFrom::Start(0))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    file.seek(SeekFrom::Start(0))?;
    Ok(format!("{:x}", digest.finalize()))
}

fn sample_chain_sha256(samples: &[HostSnapshot]) -> Result<String> {
    let mut digest = Sha256::new();
    for sample in samples {
        hash_sample(&mut digest, sample)?;
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn hash_sample(digest: &mut Sha256, sample: &HostSnapshot) -> Result<()> {
    digest.update(serde_json::to_vec(sample)?);
    digest.update(b"\n");
    Ok(())
}

fn ensure_new_output(path: &Path) -> Result<()> {
    if path.exists() {
        bail!(
            "immutable evidence output already exists: {}",
            path.display()
        );
    }
    path.parent().context("evidence output has no parent")?;
    Ok(())
}

fn write_json_new(path: &Path, value: &impl Serialize) -> Result<()> {
    ensure_new_output(path)?;
    let parent = path.parent().context("evidence output has no parent")?;
    fs::create_dir_all(parent)?;
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("h8-watchdog"),
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    match fs::hard_link(&temporary, path) {
        Ok(()) => {
            fs::remove_file(&temporary)?;
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(error).context("failed to publish immutable watchdog evidence")
        }
    }
}

fn command_text(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|text| text.trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn elapsed_millis_u64(started: Instant) -> Result<u64> {
    u64::try_from(started.elapsed().as_millis()).context("elapsed time exceeds u64")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static PROCESS_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn passing_preflight() -> PreflightEvidence {
        let sample = HostSnapshot {
            elapsed_ms: 0,
            mem_available_bytes: MIN_MEM_AVAILABLE_BYTES,
            swap_total_bytes: 100,
            swap_free_bytes: 90,
            swap_used_bytes: 10,
            swap_cached_bytes: 2,
            pressure: Default::default(),
            target_tree_vm_swap_bytes: 0,
            attributed_process_vm_swap_bytes: 10,
            attribution: Vec::new(),
            proc_scan_complete: true,
            active_h8_process_found: false,
            cgroups: Vec::new(),
            swappiness: 60,
            protected_nvme_read_only: true,
            protected_nvme_mounted: false,
        };
        let mut samples = vec![sample.clone(); 4];
        for (index, sample) in samples.iter_mut().enumerate() {
            sample.elapsed_ms = u64::try_from(index).unwrap() * 40_000;
        }
        let analysis = analyze_preflight(&samples, MIN_PREFLIGHT_DURATION_MS);
        PreflightEvidence {
            schema: PREFLIGHT_SCHEMA.into(),
            captured_start_unix_ms: now_unix_ms(),
            captured_end_unix_ms: now_unix_ms(),
            repository_head: "test".into(),
            watchdog_executable_sha256: "0".repeat(64),
            command: vec!["test".into()],
            privileged_host_controls_changed: false,
            sample_chain_sha256: sample_chain_sha256(&samples).unwrap(),
            samples,
            gpus_before: Vec::new(),
            gpus_after: Vec::new(),
            observation_errors: Vec::new(),
            analysis,
        }
    }

    #[test]
    fn preflight_bounds_require_two_minutes_and_four_samples() {
        assert!(validate_preflight_bounds(120_000, 40_000).is_ok());
        assert!(validate_preflight_bounds(119_999, 30_000).is_err());
        assert!(validate_preflight_bounds(120_000, 40_001).is_err());
    }

    #[test]
    fn preflight_identity_detects_sample_tampering() {
        let mut evidence = passing_preflight();
        assert!(validate_preflight_evidence(&evidence, 60).is_ok());
        evidence.samples[1].swap_free_bytes -= 1;
        assert!(validate_preflight_evidence(&evidence, 60).is_err());

        let mut forged = passing_preflight();
        forged.samples[1].pressure.some_avg10_millionths = 1;
        forged.sample_chain_sha256 = sample_chain_sha256(&forged.samples).unwrap();
        assert!(forged.analysis.passed);
        assert!(validate_preflight_evidence(&forged, 60).is_err());

        let mut errored = passing_preflight();
        errored
            .observation_errors
            .push("synthetic read error".into());
        assert!(validate_preflight_evidence(&errored, 60).is_err());
    }

    #[test]
    fn fresh_admission_requires_exact_swap_counters() {
        let evidence = passing_preflight();
        let baseline = evidence.samples.last().unwrap();
        assert!(validate_fresh_admission(baseline, baseline).is_ok());
        let mut changed = baseline.clone();
        changed.swap_cached_bytes += 1;
        assert!(validate_fresh_admission(baseline, &changed).is_err());
    }

    #[test]
    fn command_validation_rejects_non_h8_and_existing_outputs() {
        let arguments: Vec<String> = vec!["--mode".into(), "h8".into()];
        assert_eq!(
            single_argument_value(&arguments, "--mode").unwrap(),
            Some("h8".into())
        );
        let duplicate_mode = vec!["--mode".into(), "h8".into(), "--mode=warm".into()];
        assert!(single_argument_value(&duplicate_mode, "--mode").is_err());
        let duplicate = vec!["--output".into(), "one".into(), "--output=two".into()];
        assert!(argument_path(&duplicate, "--output").is_err());
    }

    #[test]
    fn guarded_h8_command_requires_a_new_memory_event_directory() {
        let root = std::env::temp_dir().join(format!(
            "gpt-oss-h8-memory-command-{}-{}",
            std::process::id(),
            now_unix_ms()
        ));
        fs::create_dir(&root).unwrap();
        let executable = root.join("heterogeneous_construct");
        fs::copy("/bin/true", &executable).unwrap();
        let preflight = root.join("preflight.json");
        let watchdog_output = root.join("watchdog.json");
        let child_output = root.join("child.json");
        let memory_events = root.join("memory-events");
        let mut command = vec![
            executable.to_string_lossy().into_owned(),
            "--mode".into(),
            "h8".into(),
            "--output".into(),
            child_output.to_string_lossy().into_owned(),
        ];
        assert!(validate_child_command(&command, &preflight, &watchdog_output).is_err());
        command.extend([
            "--memory-events".into(),
            memory_events.to_string_lossy().into_owned(),
        ]);
        let spec = validate_child_command(&command, &preflight, &watchdog_output).unwrap();
        assert_eq!(spec.memory_events, memory_events);
        drop(spec);
        fs::create_dir(&memory_events).unwrap();
        assert!(validate_child_command(&command, &preflight, &watchdog_output).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn run_bounds_keep_polling_and_evidence_bounded() {
        let args = RunArgs {
            preflight: "preflight.json".into(),
            output: "run.json".into(),
            poll_interval_ms: 250,
            retain_interval_ms: 10_000,
            interrupt_grace_seconds: 30,
            max_run_seconds: 21_600,
            max_preflight_age_seconds: 900,
            command: vec!["heterogeneous_construct".into()],
        };
        assert!(validate_run_bounds(&args).is_ok());
        let mut invalid = args;
        invalid.retain_interval_ms = 1;
        assert!(validate_run_bounds(&invalid).is_err());
    }

    #[test]
    fn child_evidence_must_bind_executable_source_and_watchdog() {
        let root = std::env::temp_dir().join(format!(
            "gpt-oss-h8-child-evidence-{}-{}",
            std::process::id(),
            now_unix_ms()
        ));
        let events = root.join("memory-events");
        fs::create_dir_all(&events).unwrap();
        let path = root.join("child.json");
        let hash = "a".repeat(64);
        let event = serde_json::json!({
            "schema": CONSTRUCTION_MEMORY_EVENT_SCHEMA,
            "sequence": 0,
            "captured_unix_ms": 1,
            "elapsed_ms": 0,
            "phase": "before_checkpoint_open",
            "memory": {
                "process_status": {"vm_rss_bytes": 1, "vm_swap_bytes": 0},
                "smaps_rollup": {"pss_anon_bytes": 1, "pss_file_bytes": 1},
                "global_meminfo": {"swap_used_bytes": 0},
                "vmstat": {"pswpin_pages": 0, "pswpout_pages": 0},
                "current_cgroup": {"memory_current_bytes": 1, "memory_swap_current_bytes": 0},
                "residency": {"global_page_cache_estimate_bytes": 1}
            },
            "gpus": [{}, {}],
            "identity": {
                "repository_head": "head",
                "executable_sha256": hash,
                "checkpoint_class": "120b_h8",
                "checkpoint_revision": "revision",
                "checkpoint_metadata_sha256": "metadata",
                "checkpoint_mapping_sha256": "mapping",
                "placement_manifest_sha256": "placement",
            }
        });
        let mut event_bytes = serde_json::to_vec_pretty(&event).unwrap();
        event_bytes.push(b'\n');
        let event_filename = "000-h8_cold-identity.json";
        fs::write(events.join(event_filename), &event_bytes).unwrap();
        let value = serde_json::json!({
            "schema": "gpt-oss-rs.heterogeneous-construction/v5",
            "mode": "h8",
            "repository_head": "head",
            "executable_sha256": hash,
            "passed": true,
            "checkpoint_120b": {
                "revision": "revision",
                "metadata_sha256": "metadata",
                "mapping_sha256": "mapping"
            },
            "placement_120b": {
                "manifest_sha256": "placement"
            },
            "construction_memory_events": {
                "schema": CONSTRUCTION_MEMORY_EVENT_SCHEMA,
                "persisted": true,
                "event_count": 1,
                "encoded_event_bytes": event_bytes.len(),
                "persisted_bytes": event_bytes.len(),
                "max_events": MAX_CONSTRUCTION_MEMORY_EVENTS,
                "max_event_bytes": MAX_CONSTRUCTION_MEMORY_EVENT_BYTES,
                "max_total_bytes": MAX_CONSTRUCTION_MEMORY_JOURNAL_BYTES,
                "entries": [{
                    "sequence": 0,
                    "filename": event_filename,
                    "sha256": sha256_bytes(&event_bytes),
                    "bytes": event_bytes.len()
                }]
            },
            "h8_campaign": {
                "watchdog": {
                    "schema": WATCHDOG_SCHEMA,
                    "run_id_sha256": "run",
                    "preflight_sha256": "preflight",
                    "direct_parent_validated": true,
                    "parent_executable_validated": true
                }
            }
        });
        fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(validate_child_evidence(&path, &events, &hash, "head", "run", "preflight").is_ok());
        fs::write(events.join(event_filename), b"{}\n").unwrap();
        assert!(
            validate_child_evidence(&path, &events, &hash, "head", "run", "preflight").is_err()
        );
        fs::write(events.join(event_filename), &event_bytes).unwrap();

        let mut changed = value;
        changed["h8_campaign"]["watchdog"]["run_id_sha256"] = "wrong".into();
        fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(
            validate_child_evidence(&path, &events, &hash, "head", "run", "preflight").is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn opened_executable_inode_identity_survives_path_replacement() {
        let root =
            std::env::temp_dir().join(format!("gpt-oss-h8-open-inode-{}", std::process::id()));
        let path = root.join("heterogeneous_construct");
        fs::create_dir_all(&root).unwrap();
        fs::write(&path, b"first").unwrap();
        let mut opened = File::open(&path).unwrap();
        let opened_hash = sha256_open_file(&mut opened).unwrap();
        fs::remove_file(&path).unwrap();
        fs::write(&path, b"second").unwrap();
        assert_eq!(opened_hash, sha256_bytes(b"first"));
        assert_ne!(opened_hash, sha256_file(&path).unwrap());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn process_group_interrupt_is_scoped_and_bounded() {
        let _lock = PROCESS_TEST_LOCK.lock().unwrap();
        let mut command = Command::new("/bin/sh");
        command
            .args(["-c", "trap 'exit 0' INT; while :; do sleep 0.05; done"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0);
        let mut child = ChildGroupGuard::spawn(command).unwrap();
        thread::sleep(Duration::from_millis(50));
        let mut actions = Vec::new();
        terminate_process_group(&mut child, Duration::from_secs(2), &mut actions).unwrap();
        assert!(child.status().unwrap().success());
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].signal, "SIGINT");
        assert!(actions[0].process_group_only);
        assert!(actions[0].command_succeeded);
        assert!(actions[0].group_empty_after);
    }

    #[test]
    fn ignored_interrupt_escalates_to_process_group_kill() {
        let _lock = PROCESS_TEST_LOCK.lock().unwrap();
        let mut command = Command::new("/bin/sh");
        command
            .args(["-c", "trap '' INT; while :; do sleep 1; done"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0);
        let mut child = ChildGroupGuard::spawn(command).unwrap();
        thread::sleep(Duration::from_millis(50));
        let mut actions = Vec::new();
        terminate_process_group(&mut child, Duration::from_millis(100), &mut actions).unwrap();
        assert!(!child.status().unwrap().success());
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0].signal, "SIGINT");
        assert!(!actions[0].group_empty_after);
        assert_eq!(actions[1].signal, "SIGKILL");
        assert!(actions[1].command_succeeded);
        assert!(actions[1].group_empty_after);
    }

    #[test]
    fn child_guard_drop_kills_and_reaps_the_owned_group() {
        let _lock = PROCESS_TEST_LOCK.lock().unwrap();
        let pgid;
        {
            let mut command = Command::new("/bin/sh");
            command
                .args(["-c", "while :; do sleep 1; done"])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .process_group(0);
            let child = ChildGroupGuard::spawn(command).unwrap();
            pgid = child.pgid;
            assert!(process_group_exists(pgid).unwrap());
        }
        assert!(!process_group_exists(pgid).unwrap());
        assert_eq!(ACTIVE_CHILD_PGID.load(Ordering::SeqCst), 0);
    }
}
