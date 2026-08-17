use std::collections::BTreeMap;
use std::ffi::{c_int, c_ulong};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicI32, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use gpt_oss_bench::h8_watchdog::{
    analyze_r2_preflight, evaluate_r2_runtime_guard, read_host_snapshot, sha256_bytes, sha256_file,
    GuardViolation, HostSnapshot, PreflightAnalysis, RuntimeGuardLimits, MIN_MEM_AVAILABLE_BYTES,
    MIN_PREFLIGHT_DURATION_MS,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const PREFLIGHT_SCHEMA: &str = "gpt-oss-rs.retained-20b-comparison-preflight/v2";
const RUN_SCHEMA: &str = "gpt-oss-rs.retained-20b-comparison-run/v2";
const R2_POLICY_SHA256: &str = "f269a4c984bbfa0d2a18c037b42ded2c81330094b18c6fc8dc668b7ad81bb90f";
const PLACEMENT_SHA256: &str = "cd72f92fb9d72be23efae72053db5c166108d853cf4cdddfa4f5dc688904a0fe";
const MAPPING_FILE_SHA256: &str =
    "82e7f703954480b735795173f518723806cc8645056518fa252a352545d067f0";
const ORIGINAL_CPU_TRACE_SHA256: &str =
    "1be8a61eeeec92e7af980a6f190788f0829048015c5f54fe7fa6ce334618db39";
const RUN_PREFIX: &str = "gpt-oss-rs-het-r4-";
const DISK_RESERVE_BYTES: u64 = 64 * 1024 * 1024 * 1024;
const CLEAN_FILE_ALLOWANCE_BYTES: u64 = 11_488_417_896;
const DIRTY_WRITEBACK_ALLOWANCE_BYTES: u64 = 944_377_216;
const POST_EXIT_DRIFT_BYTES: u64 = 64 * 1024 * 1024;
const GPU_CLEANUP_TOLERANCE_MIB: u64 = 16;
const GPU_ADMISSION_FREE_MIB: u64 = 4 * 1024 + 64;
const MAX_RUN_SECONDS: u64 = 2 * 60 * 60;
const SETTLE_SECONDS: u64 = 30;
const SETTLE_SAMPLES: usize = 5;
const EXPECTED_TOKENS: [u64; 8] = [200_005, 35_644, 200_008, 976, 1_825, 5_003, 25, 392];

const PR_SET_PDEATHSIG: c_int = 1;
const SIGINT: c_int = 2;
const SIGKILL: c_int = 9;
const SIGTERM: c_int = 15;

static ACTIVE_CHILD_PGID: AtomicI32 = AtomicI32::new(0);

unsafe extern "C" {
    fn prctl(option: c_int, arg2: c_ulong, arg3: c_ulong, arg4: c_ulong, arg5: c_ulong) -> c_int;
    fn getppid() -> c_int;
    fn kill(pid: c_int, signal: c_int) -> c_int;
    fn signal(signal: c_int, handler: extern "C" fn(c_int)) -> usize;
    fn _exit(status: c_int) -> !;
}

extern "C" fn termination_handler(received: c_int) {
    let pgid = ACTIVE_CHILD_PGID.load(Ordering::SeqCst);
    if pgid > 0 {
        // SAFETY: POSIX `kill` is async-signal-safe and the negative PID is the
        // separately created comparison-child process group only.
        unsafe { kill(-pgid, SIGKILL) };
    }
    // SAFETY: `_exit` is async-signal-safe and avoids Rust destruction in the
    // signal handler.
    unsafe { _exit(128_i32.saturating_add(received)) }
}

#[derive(Debug, Parser)]
#[command(about = "R2-bound retained-20B constructor comparison supervisor")]
struct Cli {
    #[command(subcommand)]
    action: Action,
}

#[derive(Debug, Subcommand)]
enum Action {
    /// Observe the host only; never opens a model or starts a child.
    Preflight(PreflightArgs),
    /// Run the fixed eight-cell 20B matrix after a fresh passing preflight.
    Run(Box<RunArgs>),
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
    #[arg(long)]
    run_root: PathBuf,
    #[arg(long)]
    construct_executable: PathBuf,
    #[arg(long)]
    control_executable: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    native_model: PathBuf,
    #[arg(long)]
    mapping: PathBuf,
    #[arg(long)]
    placement: PathBuf,
    #[arg(long)]
    retained_trace: PathBuf,
    #[arg(long, default_value_t = 250)]
    poll_interval_ms: u64,
    #[arg(long, default_value_t = 10_000)]
    retain_interval_ms: u64,
    #[arg(long, default_value_t = 900)]
    max_preflight_age_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PreflightEvidence {
    schema: String,
    captured_start_unix_ms: u128,
    captured_end_unix_ms: u128,
    repository_head: String,
    supervisor_sha256: String,
    command: Vec<String>,
    r2_policy_sha256: String,
    samples: Vec<HostSnapshot>,
    sample_chain_sha256: String,
    cgroup_baseline: CgroupSnapshot,
    gpus_before: Vec<GpuSnapshot>,
    gpus_after: Vec<GpuSnapshot>,
    observation_errors: Vec<String>,
    analysis: PreflightAnalysis,
    privileged_host_controls_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GpuSnapshot {
    pci_bus_id: String,
    memory_used_mib: u64,
    memory_free_mib: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CgroupSnapshot {
    memory_current_bytes: u64,
    anon_bytes: u64,
    file_bytes: u64,
    file_mapped_bytes: u64,
    file_dirty_bytes: u64,
    file_writeback_bytes: u64,
    swap_current_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct RetainedObservation {
    host: HostSnapshot,
    cgroup: CgroupSnapshot,
}

#[derive(Debug, Serialize)]
struct RunEvidence {
    schema: &'static str,
    captured_start_unix_ms: u128,
    captured_end_unix_ms: u128,
    repository_head: String,
    supervisor_sha256: String,
    construct_sha256: String,
    control_sha256: String,
    preflight_sha256: String,
    r2_policy_sha256: &'static str,
    model_root: String,
    native_model_root: String,
    mapping_sha256: String,
    placement_sha256: String,
    retained_trace_sha256: String,
    run_root: String,
    protected_nvme_kernel_name: String,
    swap_baseline_bytes: u64,
    cgroup_baseline: CgroupSnapshot,
    cells: Vec<CellEvidence>,
    cache_comparison: Option<CacheComparison>,
    gpus_before: Vec<GpuSnapshot>,
    gpus_after: Vec<GpuSnapshot>,
    violation: Option<GuardViolation>,
    observation_error: Option<String>,
    privileged_host_controls_changed: bool,
    h8_or_120b_started: bool,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct CellEvidence {
    name: String,
    constructor: String,
    load_class: String,
    command: Vec<String>,
    output_path: String,
    output_sha256: Option<String>,
    memory_events_path: Option<String>,
    captured_start_unix_ms: u128,
    captured_end_unix_ms: u128,
    child_status: Option<ChildStatus>,
    samples_observed: u64,
    retained_samples: Vec<RetainedObservation>,
    settle_samples: Vec<RetainedObservation>,
    minimum_mem_available_bytes: u64,
    maximum_swap_used_bytes: u64,
    maximum_target_tree_swap_bytes: u64,
    maximum_clean_file_delta_bytes: u64,
    maximum_dirty_writeback_delta_bytes: u64,
    maximum_post_exit_current_drift_bytes: u64,
    swap_free_byte_stable: bool,
    swap_cached_byte_stable: bool,
    violation: Option<GuardViolation>,
    observation_error: Option<String>,
    output_validated: bool,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct ChildStatus {
    success: bool,
    code: Option<i32>,
    signal: Option<i32>,
}

#[derive(Debug, Serialize)]
struct CacheComparison {
    monolithic_files: BTreeMap<String, String>,
    capacity_one_files: BTreeMap<String, String>,
    exact_file_and_payload_identity: bool,
}

#[derive(Debug)]
struct CellSpec {
    name: String,
    constructor: &'static str,
    load_class: &'static str,
    executable: PathBuf,
    arguments: Vec<String>,
    output: PathBuf,
    memory_events: Option<PathBuf>,
}

fn main() -> Result<()> {
    match Cli::parse().action {
        Action::Preflight(args) => run_preflight(args),
        Action::Run(args) => run_matrix(*args),
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
    if duration_ms < MIN_PREFLIGHT_DURATION_MS || interval_ms == 0 || interval_ms > duration_ms / 3
    {
        bail!("preflight bounds cannot retain four samples across 120 seconds");
    }
    ensure_new_output(&args.output)?;
    validate_run_root(
        args.output
            .parent()
            .context("preflight output has no run-root parent")?,
    )?;

    let started_unix_ms = now_unix_ms();
    let gpus_before = gpu_snapshot()?;
    let mut samples = Vec::new();
    let mut observation_errors = Vec::new();
    match read_host_snapshot(None, 0) {
        Ok(sample) => samples.push(sample),
        Err(error) => observation_errors.push(format!("initial host observation: {error:#}")),
    }
    let started = Instant::now();
    while observation_errors.is_empty() {
        let elapsed = elapsed_ms(started)?;
        if elapsed >= duration_ms {
            match read_host_snapshot(None, elapsed) {
                Ok(sample) => samples.push(sample),
                Err(error) => observation_errors.push(format!("final host observation: {error:#}")),
            }
            break;
        }
        thread::sleep(Duration::from_millis(
            interval_ms.min(duration_ms - elapsed),
        ));
        let elapsed = elapsed_ms(started)?;
        match read_host_snapshot(None, elapsed) {
            Ok(sample) => samples.push(sample),
            Err(error) => observation_errors.push(format!("host observation: {error:#}")),
        }
    }
    let mut analysis = analyze_r2_preflight(&samples, duration_ms);
    analysis
        .failures
        .extend(gpu_admission_failures(&gpus_before, "initial"));
    let gpus_after = gpu_snapshot()?;
    analysis
        .failures
        .extend(gpu_admission_failures(&gpus_after, "final"));
    if gpu_identities(&gpus_before) != gpu_identities(&gpus_after) {
        analysis
            .failures
            .push("GPU PCI identities changed during preflight".into());
    }
    if !observation_errors.is_empty() {
        analysis.failures.extend(observation_errors.clone());
    }
    analysis.passed = analysis.failures.is_empty();
    let evidence = PreflightEvidence {
        schema: PREFLIGHT_SCHEMA.into(),
        captured_start_unix_ms: started_unix_ms,
        captured_end_unix_ms: now_unix_ms(),
        repository_head: git_text(&["rev-parse", "HEAD"])?,
        supervisor_sha256: sha256_file(Path::new("/proc/self/exe"))?,
        command: std::env::args().collect(),
        r2_policy_sha256: R2_POLICY_SHA256.into(),
        sample_chain_sha256: sample_chain_sha256(&samples)?,
        samples,
        cgroup_baseline: cgroup_snapshot()?,
        gpus_before,
        gpus_after,
        observation_errors,
        analysis,
        privileged_host_controls_changed: false,
    };
    write_json_new(&args.output, &evidence)?;
    if !evidence.analysis.passed {
        bail!(
            "R4 preflight failed; immutable evidence is at {}",
            args.output.display()
        );
    }
    Ok(())
}

fn run_matrix(args: RunArgs) -> Result<()> {
    validate_run_args(&args)?;
    ensure_new_output(&args.output)?;
    let preflight_bytes = fs::read(&args.preflight)?;
    let preflight: PreflightEvidence = serde_json::from_slice(&preflight_bytes)?;
    validate_preflight(&preflight, args.max_preflight_age_seconds)?;
    let current_head = git_text(&["rev-parse", "HEAD"])?;
    if current_head != preflight.repository_head
        || !git_text(&["status", "--porcelain"])?.is_empty()
    {
        bail!("R4 run requires the unchanged clean preflight commit");
    }
    let supervisor_sha256 = sha256_file(Path::new("/proc/self/exe"))?;
    if supervisor_sha256 != preflight.supervisor_sha256 {
        bail!("preflight was produced by a different supervisor executable");
    }
    let construct_sha256 = sha256_file(&args.construct_executable)?;
    let control_sha256 = sha256_file(&args.control_executable)?;
    let baseline = preflight
        .samples
        .last()
        .context("preflight has no samples")?;
    let current = read_host_snapshot(None, 0)?;
    if current.protected_nvme_kernel_name != baseline.protected_nvme_kernel_name {
        bail!("protected NVMe identity changed after the R4 preflight");
    }
    let limits = RuntimeGuardLimits {
        swap_baseline_bytes: baseline.swap_used_bytes,
        min_mem_available_bytes: MIN_MEM_AVAILABLE_BYTES,
    };
    evaluate_r2_runtime_guard(&current, &limits)
        .map_err(|violation| anyhow::anyhow!("fresh R4 admission failed: {violation:?}"))?;
    if available_bytes(&args.run_root)? < DISK_RESERVE_BYTES {
        bail!("R4 run root does not retain the frozen 64-GiB filesystem reserve");
    }

    install_signal_handlers()?;
    let gpus_before = gpu_snapshot()?;
    let gpu_failures = gpu_admission_failures(&gpus_before, "run admission");
    if !gpu_failures.is_empty()
        || gpu_identities(&gpus_before) != gpu_identities(&preflight.gpus_after)
    {
        bail!("fresh R4 GPU admission failed: {gpu_failures:?}");
    }
    let started_unix_ms = now_unix_ms();
    let specs = cell_specs(&args)?;
    let mut cells = Vec::with_capacity(specs.len());
    let mut violation = None;
    let mut observation_error = None;

    for spec in specs {
        match run_cell(
            &spec,
            &limits,
            &preflight.cgroup_baseline,
            args.poll_interval_ms,
            args.retain_interval_ms,
            &baseline.protected_nvme_kernel_name,
        ) {
            Ok(cell) => {
                let passed = cell.passed;
                cells.push(cell);
                if !passed {
                    violation = Some(GuardViolation {
                        reasons: vec!["comparison cell failed its retained gates".into()],
                    });
                    break;
                }
            }
            Err(error) => {
                observation_error = Some(format!("comparison cell failed closed: {error:#}"));
                break;
            }
        }
    }

    let cache_comparison = if cells.len() == 8 && violation.is_none() && observation_error.is_none()
    {
        Some(compare_caches(
            &args.run_root.join("monolithic-cache"),
            &args.run_root.join("capacity-one-cache"),
        )?)
    } else {
        None
    };
    if cache_comparison
        .as_ref()
        .is_some_and(|comparison| !comparison.exact_file_and_payload_identity)
    {
        violation = Some(GuardViolation {
            reasons: vec!["constructor caches differ in immutable record identity".into()],
        });
    }
    let gpus_after = gpu_snapshot()?;
    if !gpu_cleanup_within(&gpus_before, &gpus_after) {
        violation = Some(GuardViolation {
            reasons: vec!["post-matrix GPU cleanup exceeded 16 MiB".into()],
        });
    }
    let passed = cells.len() == 8
        && cells.iter().all(|cell| cell.passed)
        && cache_comparison
            .as_ref()
            .is_some_and(|comparison| comparison.exact_file_and_payload_identity)
        && violation.is_none()
        && observation_error.is_none();
    let evidence = RunEvidence {
        schema: RUN_SCHEMA,
        captured_start_unix_ms: started_unix_ms,
        captured_end_unix_ms: now_unix_ms(),
        repository_head: current_head,
        supervisor_sha256,
        construct_sha256,
        control_sha256,
        preflight_sha256: sha256_bytes(&preflight_bytes),
        r2_policy_sha256: R2_POLICY_SHA256,
        model_root: args.model.to_string_lossy().into_owned(),
        native_model_root: args.native_model.to_string_lossy().into_owned(),
        mapping_sha256: sha256_file(&args.mapping)?,
        placement_sha256: sha256_file(&args.placement)?,
        retained_trace_sha256: sha256_file(&args.retained_trace)?,
        run_root: args.run_root.to_string_lossy().into_owned(),
        protected_nvme_kernel_name: baseline.protected_nvme_kernel_name.clone(),
        swap_baseline_bytes: baseline.swap_used_bytes,
        cgroup_baseline: preflight.cgroup_baseline,
        cells,
        cache_comparison,
        gpus_before,
        gpus_after,
        violation,
        observation_error,
        privileged_host_controls_changed: false,
        h8_or_120b_started: false,
        passed,
    };
    write_json_new(&args.output, &evidence)?;
    if !passed {
        bail!(
            "R4 comparison failed; immutable evidence is at {}",
            args.output.display()
        );
    }
    Ok(())
}

fn cell_specs(args: &RunArgs) -> Result<Vec<CellSpec>> {
    let mono_cache = args.run_root.join("monolithic-cache");
    let capacity_cache = args.run_root.join("capacity-one-cache");
    let mut specs = Vec::with_capacity(8);
    for (load_class, mode) in [("cold", "cold"), ("warm", "warm"), ("repeat", "warm")] {
        for (constructor, cache) in [
            ("monolithic-control", &mono_cache),
            ("capacity-one", &capacity_cache),
        ] {
            let name = format!("{load_class}-{constructor}");
            let output = args.run_root.join(format!("{name}.json"));
            let memory_events = args.run_root.join(format!("{name}-memory"));
            let mut arguments = vec![
                "--mode".into(),
                mode.into(),
                "--constructor".into(),
                constructor.into(),
                "--model-20b".into(),
                args.native_model.to_string_lossy().into_owned(),
                "--mapping-20b".into(),
                args.mapping.to_string_lossy().into_owned(),
                "--cache-root".into(),
                cache.to_string_lossy().into_owned(),
                "--output".into(),
                output.to_string_lossy().into_owned(),
                "--memory-events".into(),
                memory_events.to_string_lossy().into_owned(),
            ];
            if constructor == "capacity-one" {
                arguments.extend([
                    "--capacity-one-placement".into(),
                    args.placement.to_string_lossy().into_owned(),
                ]);
            }
            specs.push(CellSpec {
                name,
                constructor,
                load_class,
                executable: args.construct_executable.clone(),
                arguments,
                output,
                memory_events: Some(memory_events),
            });
        }
    }
    for (constructor, cache) in [
        ("monolithic-control", mono_cache),
        ("capacity-one", capacity_cache),
    ] {
        let name = format!("h7-{constructor}");
        let output = args.run_root.join(format!("{name}.json"));
        specs.push(CellSpec {
            name,
            constructor,
            load_class: "h7-repeat-two",
            executable: args.control_executable.clone(),
            arguments: vec![
                "--constructor".into(),
                constructor.into(),
                "--model".into(),
                args.model.to_string_lossy().into_owned(),
                "--native-model".into(),
                args.native_model.to_string_lossy().into_owned(),
                "--owner-cache".into(),
                cache.to_string_lossy().into_owned(),
                "--placement".into(),
                args.placement.to_string_lossy().into_owned(),
                "--retained-trace".into(),
                args.retained_trace.to_string_lossy().into_owned(),
                "--output".into(),
                output.to_string_lossy().into_owned(),
                "--max-new-tokens".into(),
                "8".into(),
                "--repeat".into(),
                "2".into(),
            ],
            output,
            memory_events: None,
        });
    }
    Ok(specs)
}

fn run_cell(
    spec: &CellSpec,
    limits: &RuntimeGuardLimits,
    cgroup_baseline: &CgroupSnapshot,
    poll_interval_ms: u64,
    retain_interval_ms: u64,
    protected_nvme_kernel_name: &str,
) -> Result<CellEvidence> {
    if spec.output.exists()
        || spec
            .memory_events
            .as_ref()
            .is_some_and(|path| path.exists())
    {
        bail!("cell output already exists: {}", spec.name);
    }
    let command_record = std::iter::once(spec.executable.to_string_lossy().into_owned())
        .chain(spec.arguments.iter().cloned())
        .collect::<Vec<_>>();
    let started_unix_ms = now_unix_ms();
    let started = Instant::now();
    let expected_parent =
        c_int::try_from(std::process::id()).context("supervisor PID overflows")?;
    let mut command = Command::new(&spec.executable);
    command
        .args(&spec.arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .process_group(0);
    configure_parent_death(&mut command, expected_parent);
    let mut child = command.spawn().context("spawn comparison child")?;
    let pgid = c_int::try_from(child.id()).context("child PID overflows")?;
    ACTIVE_CHILD_PGID.store(pgid, Ordering::SeqCst);

    let mut retained_samples = Vec::new();
    let mut samples_observed = 0_u64;
    let mut next_retain_ms = 0_u64;
    let mut minimum_mem_available_bytes = u64::MAX;
    let mut maximum_swap_used_bytes = 0_u64;
    let mut maximum_target_tree_swap_bytes = 0_u64;
    let mut maximum_clean_file_delta_bytes = 0_u64;
    let mut maximum_dirty_writeback_delta_bytes = 0_u64;
    let mut violation = None;
    let mut observation_error = None;
    let mut status = None;

    loop {
        let elapsed = elapsed_ms(started)?;
        let sample = match read_host_snapshot(Some(child.id()), elapsed) {
            Ok(sample) => sample,
            Err(error) => {
                observation_error = Some(format!("host observation failed: {error:#}"));
                break;
            }
        };
        samples_observed = samples_observed.saturating_add(1);
        update_sample_extrema(
            &sample,
            &mut minimum_mem_available_bytes,
            &mut maximum_swap_used_bytes,
            &mut maximum_target_tree_swap_bytes,
        );
        let cgroup = match cgroup_snapshot() {
            Ok(cgroup) => cgroup,
            Err(error) => {
                observation_error = Some(format!("cgroup observation failed: {error:#}"));
                break;
            }
        };
        update_cgroup_extrema(
            &cgroup,
            cgroup_baseline,
            &mut maximum_clean_file_delta_bytes,
            &mut maximum_dirty_writeback_delta_bytes,
        );
        if let Err(error) = enforce_cgroup(&cgroup, cgroup_baseline, false) {
            violation = Some(GuardViolation {
                reasons: vec![error.to_string()],
            });
        }
        if elapsed >= next_retain_ms {
            retained_samples.push(RetainedObservation {
                host: sample.clone(),
                cgroup,
            });
            next_retain_ms = next_retain_ms.saturating_add(retain_interval_ms);
        }
        if let Err(guard) = evaluate_r2_runtime_guard(&sample, limits) {
            violation = Some(guard);
        }
        if sample.protected_nvme_kernel_name != protected_nvme_kernel_name {
            violation = Some(GuardViolation {
                reasons: vec!["protected NVMe identity changed during comparison cell".into()],
            });
        }
        if elapsed > MAX_RUN_SECONDS * 1_000 {
            violation = Some(GuardViolation {
                reasons: vec!["cell exceeded two-hour bound".into()],
            });
        }
        if violation.is_some() || observation_error.is_some() {
            break;
        }
        if let Some(exit) = child.try_wait()? {
            status = Some(exit);
            break;
        }
        thread::sleep(Duration::from_millis(poll_interval_ms));
    }
    if status.is_none() {
        terminate_child_group(&mut child, pgid)?;
        status = child.try_wait()?;
    }
    ACTIVE_CHILD_PGID.store(0, Ordering::SeqCst);

    let mut settle_samples = Vec::with_capacity(SETTLE_SAMPLES);
    thread::sleep(Duration::from_secs(SETTLE_SECONDS));
    let mut maximum_post_exit_current_drift_bytes = 0_u64;
    for index in 0..SETTLE_SAMPLES {
        if index != 0 {
            thread::sleep(Duration::from_secs(1));
        }
        let sample = read_host_snapshot(None, elapsed_ms(started)?)?;
        if let Err(guard) = evaluate_r2_runtime_guard(&sample, limits) {
            violation = Some(guard);
        }
        if sample.protected_nvme_kernel_name != protected_nvme_kernel_name {
            violation = Some(GuardViolation {
                reasons: vec!["protected NVMe identity changed during post-exit settle".into()],
            });
        }
        let cgroup = cgroup_snapshot()?;
        maximum_post_exit_current_drift_bytes = maximum_post_exit_current_drift_bytes.max(
            cgroup
                .memory_current_bytes
                .saturating_sub(cgroup_baseline.memory_current_bytes),
        );
        if let Err(error) = enforce_cgroup(&cgroup, cgroup_baseline, true) {
            violation = Some(GuardViolation {
                reasons: vec![error.to_string()],
            });
        }
        settle_samples.push(RetainedObservation {
            host: sample,
            cgroup,
        });
    }

    let output_sha256 = spec
        .output
        .is_file()
        .then(|| sha256_file(&spec.output))
        .transpose()?;
    let output_validated = if output_sha256.is_some() {
        validate_cell_output(spec, &fs::read(&spec.output)?, protected_nvme_kernel_name).is_ok()
    } else {
        false
    };
    let status_record = status.map(status_record);
    let child_succeeded = status_record.as_ref().is_some_and(|record| record.success);
    let first_swap_free = retained_samples
        .first()
        .map(|sample| sample.host.swap_free_bytes);
    let first_swap_cached = retained_samples
        .first()
        .map(|sample| sample.host.swap_cached_bytes);
    let swap_free_byte_stable = retained_samples
        .iter()
        .all(|sample| Some(sample.host.swap_free_bytes) == first_swap_free);
    let swap_cached_byte_stable = retained_samples
        .iter()
        .all(|sample| Some(sample.host.swap_cached_bytes) == first_swap_cached);
    if minimum_mem_available_bytes == u64::MAX {
        minimum_mem_available_bytes = 0;
    }
    let passed = child_succeeded
        && output_validated
        && violation.is_none()
        && observation_error.is_none()
        && maximum_swap_used_bytes <= limits.swap_baseline_bytes
        && maximum_target_tree_swap_bytes == 0;
    Ok(CellEvidence {
        name: spec.name.clone(),
        constructor: spec.constructor.into(),
        load_class: spec.load_class.into(),
        command: command_record,
        output_path: spec.output.to_string_lossy().into_owned(),
        output_sha256,
        memory_events_path: spec
            .memory_events
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned()),
        captured_start_unix_ms: started_unix_ms,
        captured_end_unix_ms: now_unix_ms(),
        child_status: status_record,
        samples_observed,
        retained_samples,
        settle_samples,
        minimum_mem_available_bytes,
        maximum_swap_used_bytes,
        maximum_target_tree_swap_bytes,
        maximum_clean_file_delta_bytes,
        maximum_dirty_writeback_delta_bytes,
        maximum_post_exit_current_drift_bytes,
        swap_free_byte_stable,
        swap_cached_byte_stable,
        violation,
        observation_error,
        output_validated,
        passed,
    })
}

fn validate_cell_output(
    spec: &CellSpec,
    bytes: &[u8],
    protected_nvme_kernel_name: &str,
) -> Result<()> {
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    if spec.load_class == "h7-repeat-two" {
        if value
            .pointer("/constructor")
            .and_then(serde_json::Value::as_str)
            != Some(&spec.constructor.replace('-', "_")[..])
            || value
                .pointer("/all_runs_passed")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
            || value
                .pointer("/repeat_requested")
                .and_then(serde_json::Value::as_u64)
                != Some(2)
        {
            bail!("H7 output identity or pass state is invalid");
        }
        let expected = value
            .pointer("/expected_token_ids")
            .and_then(serde_json::Value::as_array)
            .context("H7 output has no expected token IDs")?
            .iter()
            .map(|value| value.as_u64().context("invalid token ID"))
            .collect::<Result<Vec<_>>>()?;
        if expected != EXPECTED_TOKENS {
            bail!("H7 expected token identity changed");
        }
        let runs = value
            .pointer("/runs")
            .and_then(serde_json::Value::as_array)
            .context("H7 output has no runs")?;
        if runs.len() != 2 {
            bail!("H7 output does not contain exactly two runs");
        }
        for run in runs {
            let generated = run
                .get("generated_token_ids")
                .and_then(serde_json::Value::as_array)
                .context("H7 run has no generated tokens")?
                .iter()
                .map(|value| value.as_u64().context("invalid generated token"))
                .collect::<Result<Vec<_>>>()?;
            if generated != EXPECTED_TOKENS
                || run.get("passed").and_then(serde_json::Value::as_bool) != Some(true)
            {
                bail!("H7 run did not retain the exact continuation");
            }
        }
    } else {
        let expected_constructor = spec.constructor.replace('-', "_");
        let expected_schema = if spec.constructor == "capacity-one" {
            "gpt-oss-rs.heterogeneous-capacity-one-construction/v2"
        } else {
            "gpt-oss-rs.heterogeneous-construction/v6"
        };
        if value.pointer("/schema").and_then(serde_json::Value::as_str) != Some(expected_schema)
            || value
                .pointer("/passed")
                .and_then(serde_json::Value::as_bool)
                != Some(true)
            || value
                .pointer("/constructor")
                .and_then(serde_json::Value::as_str)
                != Some(expected_constructor.as_str())
            || value
                .pointer("/protected_nvme/kernel_name")
                .and_then(serde_json::Value::as_str)
                != Some(protected_nvme_kernel_name)
        {
            bail!("construction cell identity or pass state is invalid");
        }
        if spec.constructor == "capacity-one"
            && (value
                .pointer("/capacity_one/active_mapping_high_water")
                .and_then(serde_json::Value::as_u64)
                != Some(1)
                || value
                    .pointer("/capacity_one/publication_proof/active_source_mappings")
                    .and_then(serde_json::Value::as_u64)
                    != Some(0)
                || value
                    .pointer("/capacity_one/publication_proof/active_source_payload_fds")
                    .and_then(serde_json::Value::as_u64)
                    != Some(0)
                || value
                    .pointer("/capacity_one/publication_proof/source_inode_mappings")
                    .and_then(serde_json::Value::as_u64)
                    != Some(0)
                || value
                    .pointer("/capacity_one/publication_proof/source_inode_pss_bytes")
                    .and_then(serde_json::Value::as_u64)
                    != Some(0))
        {
            bail!("capacity-one construction proof is incomplete");
        }
    }
    Ok(())
}

fn validate_run_args(args: &RunArgs) -> Result<()> {
    validate_run_root(&args.run_root)?;
    if args.preflight.parent() != Some(args.run_root.as_path())
        || args.output.parent() != Some(args.run_root.as_path())
    {
        bail!("preflight and run evidence must be direct children of the R4 run root");
    }
    if !(100..=5_000).contains(&args.poll_interval_ms)
        || args.retain_interval_ms < args.poll_interval_ms
        || args.retain_interval_ms > 60_000
    {
        bail!("comparison observation cadence is outside the reviewed bounds");
    }
    for path in [
        &args.construct_executable,
        &args.control_executable,
        &args.model,
        &args.native_model,
        &args.mapping,
        &args.placement,
        &args.retained_trace,
    ] {
        if !path.exists() || path.is_symlink() {
            bail!(
                "comparison input is missing or a symlink: {}",
                path.display()
            );
        }
    }
    if args.model != Path::new("/data/models/openai/gpt-oss-20b")
        || args.native_model != Path::new("/data/models/openai/gpt-oss-20b/original")
    {
        bail!("R4 accepts only the retained local 20B model roots");
    }
    if sha256_file(&args.mapping)? != MAPPING_FILE_SHA256
        || sha256_file(&args.placement)? != PLACEMENT_SHA256
    {
        bail!("mapping or placement identity differs from the retained H7 authority");
    }
    validate_retained_trace(&args.retained_trace)?;
    Ok(())
}

fn validate_retained_trace(path: &Path) -> Result<()> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    if value
        .pointer("/source_cpu_trace_sha256")
        .and_then(serde_json::Value::as_str)
        != Some(ORIGINAL_CPU_TRACE_SHA256)
    {
        bail!("retained authority does not bind the original CPU trace");
    }
    let prompt = value
        .pointer("/prompt_token_ids")
        .and_then(serde_json::Value::as_array)
        .context("retained authority lacks prompt tokens")?;
    let generated = value
        .pointer("/generated_token_ids")
        .and_then(serde_json::Value::as_array)
        .context("retained authority lacks generated tokens")?;
    if prompt.len() != 63
        || generated
            .iter()
            .filter_map(serde_json::Value::as_u64)
            .collect::<Vec<_>>()
            != EXPECTED_TOKENS
    {
        bail!("retained authority token identity is invalid");
    }
    Ok(())
}

fn validate_preflight(evidence: &PreflightEvidence, max_age_seconds: u64) -> Result<()> {
    if evidence.schema != PREFLIGHT_SCHEMA
        || evidence.r2_policy_sha256 != R2_POLICY_SHA256
        || evidence.privileged_host_controls_changed
        || !evidence.observation_errors.is_empty()
    {
        bail!("R4 preflight identity is invalid");
    }
    if evidence.samples.len() < 5 {
        bail!("R4 preflight does not retain the five samples required by R2");
    }
    if !gpu_admission_failures(&evidence.gpus_before, "retained initial").is_empty()
        || !gpu_admission_failures(&evidence.gpus_after, "retained final").is_empty()
        || gpu_identities(&evidence.gpus_before) != gpu_identities(&evidence.gpus_after)
    {
        bail!("R4 preflight GPU admission is invalid");
    }
    if sample_chain_sha256(&evidence.samples)? != evidence.sample_chain_sha256 {
        bail!("R4 preflight sample chain is invalid");
    }
    let recomputed =
        analyze_r2_preflight(&evidence.samples, evidence.analysis.required_duration_ms);
    if !recomputed.passed || recomputed != evidence.analysis {
        bail!("R4 preflight analysis does not match its samples");
    }
    let max_age_ms = u128::from(max_age_seconds) * 1_000;
    if now_unix_ms().saturating_sub(evidence.captured_end_unix_ms) > max_age_ms {
        bail!("R4 preflight is stale");
    }
    Ok(())
}

fn validate_run_root(path: &Path) -> Result<()> {
    if !path.is_absolute() || path.is_symlink() || !path.is_dir() {
        bail!("R4 run root must be an existing absolute non-symlink directory");
    }
    let workspace = Path::new("/home/emmy/workspace");
    if fs::canonicalize(workspace)? != workspace {
        bail!("workspace root resolves unexpectedly");
    }
    let relative = path
        .strip_prefix(workspace)
        .context("R4 run root is outside workspace")?;
    let components = relative.components().collect::<Vec<_>>();
    if components.len() != 1 {
        bail!("R4 run root must be an immediate workspace child");
    }
    let name = match components[0] {
        Component::Normal(name) => name.to_string_lossy(),
        _ => bail!("R4 run root has a non-normal component"),
    };
    if !name.starts_with(RUN_PREFIX) {
        bail!("R4 run root has the wrong create-new namespace");
    }
    Ok(())
}

fn compare_caches(monolithic: &Path, capacity_one: &Path) -> Result<CacheComparison> {
    let monolithic_files = hash_cache_files(monolithic)?;
    let capacity_one_files = hash_cache_files(capacity_one)?;
    let exact_file_and_payload_identity = monolithic_files == capacity_one_files;
    Ok(CacheComparison {
        monolithic_files,
        capacity_one_files,
        exact_file_and_payload_identity,
    })
}

fn hash_cache_files(root: &Path) -> Result<BTreeMap<String, String>> {
    let mut pending = vec![root.to_owned()];
    let mut result = BTreeMap::new();
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                bail!("owner cache contains a symlink");
            }
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if file_type.is_file() {
                let relative = entry
                    .path()
                    .strip_prefix(root)?
                    .to_string_lossy()
                    .into_owned();
                result.insert(relative, sha256_file(&entry.path())?);
            } else {
                bail!("owner cache contains a non-regular object");
            }
        }
    }
    Ok(result)
}

fn update_sample_extrema(
    sample: &HostSnapshot,
    min_mem: &mut u64,
    max_swap: &mut u64,
    max_target_swap: &mut u64,
) {
    *min_mem = (*min_mem).min(sample.mem_available_bytes);
    *max_swap = (*max_swap).max(sample.swap_used_bytes);
    *max_target_swap = (*max_target_swap).max(sample.target_tree_vm_swap_bytes);
}

fn update_cgroup_extrema(
    current: &CgroupSnapshot,
    baseline: &CgroupSnapshot,
    max_clean_file: &mut u64,
    max_dirty_writeback: &mut u64,
) {
    *max_clean_file = (*max_clean_file).max(current.file_bytes.saturating_sub(baseline.file_bytes));
    *max_dirty_writeback = (*max_dirty_writeback).max(
        current
            .file_dirty_bytes
            .saturating_add(current.file_writeback_bytes)
            .saturating_sub(
                baseline
                    .file_dirty_bytes
                    .saturating_add(baseline.file_writeback_bytes),
            ),
    );
}

fn enforce_cgroup(
    current: &CgroupSnapshot,
    baseline: &CgroupSnapshot,
    post_exit: bool,
) -> Result<()> {
    if current.swap_current_bytes != 0 {
        bail!("comparison cgroup used swap");
    }
    if current.file_bytes.saturating_sub(baseline.file_bytes) > CLEAN_FILE_ALLOWANCE_BYTES {
        bail!("comparison cgroup clean-file allowance exceeded");
    }
    if current
        .file_dirty_bytes
        .saturating_add(current.file_writeback_bytes)
        .saturating_sub(
            baseline
                .file_dirty_bytes
                .saturating_add(baseline.file_writeback_bytes),
        )
        > DIRTY_WRITEBACK_ALLOWANCE_BYTES
    {
        bail!("comparison cgroup dirty/writeback allowance exceeded");
    }
    if post_exit
        && current
            .memory_current_bytes
            .saturating_sub(baseline.memory_current_bytes)
            > POST_EXIT_DRIFT_BYTES
    {
        bail!("comparison cgroup post-exit drift exceeded 64 MiB");
    }
    if post_exit && current.file_bytes.saturating_sub(baseline.file_bytes) > POST_EXIT_DRIFT_BYTES {
        bail!("comparison cgroup post-exit file drift exceeded 64 MiB");
    }
    Ok(())
}

fn cgroup_snapshot() -> Result<CgroupSnapshot> {
    let cgroup_text = fs::read_to_string("/proc/self/cgroup")?;
    let relative = cgroup_text
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .context("unified cgroup identity is missing")?;
    let root = Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/'));
    let stat = numeric_fields(&fs::read_to_string(root.join("memory.stat"))?)?;
    Ok(CgroupSnapshot {
        memory_current_bytes: read_u64(root.join("memory.current"))?,
        anon_bytes: stat
            .get("anon")
            .copied()
            .context("cgroup anon stat is missing")?,
        file_bytes: stat
            .get("file")
            .copied()
            .context("cgroup file stat is missing")?,
        file_mapped_bytes: stat
            .get("file_mapped")
            .copied()
            .context("cgroup file_mapped stat is missing")?,
        file_dirty_bytes: stat.get("file_dirty").copied().unwrap_or(0),
        file_writeback_bytes: stat.get("file_writeback").copied().unwrap_or(0),
        swap_current_bytes: read_u64(root.join("memory.swap.current"))?,
    })
}

fn numeric_fields(text: &str) -> Result<BTreeMap<String, u64>> {
    text.lines()
        .map(|line| {
            let (key, value) = line.split_once(' ').context("invalid numeric field")?;
            Ok((key.into(), value.parse()?))
        })
        .collect()
}

fn read_u64(path: PathBuf) -> Result<u64> {
    Ok(fs::read_to_string(path)?.trim().parse()?)
}

fn gpu_snapshot() -> Result<Vec<GpuSnapshot>> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=pci.bus_id,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !output.status.success() {
        bail!("nvidia-smi failed");
    }
    String::from_utf8(output.stdout)?
        .lines()
        .map(|line| {
            let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
            if fields.len() != 3 {
                bail!("unexpected GPU memory row");
            }
            Ok(GpuSnapshot {
                pci_bus_id: fields[0].to_ascii_lowercase(),
                memory_used_mib: fields[1].parse()?,
                memory_free_mib: fields[2].parse()?,
            })
        })
        .collect()
}

fn gpu_cleanup_within(before: &[GpuSnapshot], after: &[GpuSnapshot]) -> bool {
    before.len() == after.len()
        && before.iter().zip(after).all(|(before, after)| {
            before.pci_bus_id == after.pci_bus_id
                && after.memory_used_mib
                    <= before
                        .memory_used_mib
                        .saturating_add(GPU_CLEANUP_TOLERANCE_MIB)
        })
}

fn gpu_identities(gpus: &[GpuSnapshot]) -> Vec<&str> {
    gpus.iter().map(|gpu| gpu.pci_bus_id.as_str()).collect()
}

fn gpu_admission_failures(gpus: &[GpuSnapshot], phase: &str) -> Vec<String> {
    let mut failures = Vec::new();
    if gpus.len() != 2 {
        failures.push(format!("{phase} GPU count is not exactly two"));
    }
    let identities = gpu_identities(gpus);
    if identities.windows(2).any(|pair| pair[0] == pair[1]) {
        failures.push(format!("{phase} GPU PCI identities are not unique"));
    }
    for gpu in gpus {
        if gpu.memory_free_mib < GPU_ADMISSION_FREE_MIB {
            failures.push(format!(
                "{phase} GPU {} has {} MiB free, below the 4160-MiB admission reserve",
                gpu.pci_bus_id, gpu.memory_free_mib
            ));
        }
    }
    failures
}

fn install_signal_handlers() -> Result<()> {
    for requested in [SIGINT, SIGTERM] {
        // SAFETY: installs one static C-ABI handler that only performs
        // async-signal-safe operations.
        if unsafe { signal(requested, termination_handler) } == usize::MAX {
            bail!("failed to install comparison termination handler");
        }
    }
    Ok(())
}

fn configure_parent_death(command: &mut Command, expected_parent: c_int) {
    // SAFETY: this closure runs after fork and before exec, calling only libc
    // functions and returning an `io::Error` on failure.
    unsafe {
        command.pre_exec(move || {
            if prctl(PR_SET_PDEATHSIG, SIGKILL as c_ulong, 0, 0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            if getppid() != expected_parent {
                return Err(std::io::Error::other(
                    "comparison parent changed before exec",
                ));
            }
            Ok(())
        });
    }
}

fn terminate_child_group(child: &mut Child, pgid: c_int) -> Result<()> {
    // SAFETY: the negative PID targets only the child-created process group.
    unsafe { kill(-pgid, SIGINT) };
    let started = Instant::now();
    while started.elapsed() < Duration::from_secs(30) {
        if child.try_wait()?.is_some() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    // SAFETY: escalation remains scoped to the same process group.
    unsafe { kill(-pgid, SIGKILL) };
    child.wait()?;
    Ok(())
}

fn status_record(status: ExitStatus) -> ChildStatus {
    use std::os::unix::process::ExitStatusExt;
    ChildStatus {
        success: status.success(),
        code: status.code(),
        signal: status.signal(),
    }
}

fn available_bytes(path: &Path) -> Result<u64> {
    let output = Command::new("df")
        .args(["--output=avail", "-B1"])
        .arg(path)
        .output()?;
    if !output.status.success() {
        bail!("df failed for comparison run root");
    }
    String::from_utf8(output.stdout)?
        .lines()
        .nth(1)
        .context("df output has no available-byte row")?
        .trim()
        .parse()
        .context("invalid available-byte count")
}

fn ensure_new_output(path: &Path) -> Result<()> {
    if path.exists() || path.is_symlink() {
        bail!("output path already exists: {}", path.display());
    }
    if !path.parent().is_some_and(Path::is_dir) {
        bail!("output parent is not a directory");
    }
    Ok(())
}

fn write_json_new(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

fn git_text(arguments: &[&str]) -> Result<String> {
    let output = Command::new("git").args(arguments).output()?;
    if !output.status.success() {
        bail!("git command failed");
    }
    Ok(String::from_utf8(output.stdout)?.trim().into())
}

fn sample_chain_sha256(samples: &[HostSnapshot]) -> Result<String> {
    let mut hasher = Sha256::new();
    for sample in samples {
        let bytes = serde_json::to_vec(sample)?;
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn elapsed_ms(started: Instant) -> Result<u64> {
    u64::try_from(started.elapsed().as_millis()).context("elapsed duration overflows")
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpt_oss_bench::h8_watchdog::{CgroupMemory, CgroupScope, MemoryPressure};

    fn snapshot(elapsed_ms: u64) -> HostSnapshot {
        HostSnapshot {
            elapsed_ms,
            mem_available_bytes: MIN_MEM_AVAILABLE_BYTES + 1,
            swap_total_bytes: 1_000,
            swap_free_bytes: 900,
            swap_used_bytes: 100,
            swap_cached_bytes: 25,
            pressure: MemoryPressure::default(),
            target_tree_vm_swap_bytes: 0,
            attributed_process_vm_swap_bytes: 100,
            attribution: Vec::new(),
            proc_scan_complete: true,
            active_h8_process_found: false,
            cgroups: vec![CgroupMemory {
                scope: CgroupScope::GuardScope,
                memory_current_bytes: Some(1),
                swap_current_bytes: Some(0),
                swap_max: Some("max".into()),
                high_events: Some(0),
                max_events: Some(0),
                oom_events: Some(0),
                oom_kill_events: Some(0),
            }],
            swappiness: 60,
            protected_nvme_kernel_name: "nvme1n1".into(),
            protected_nvme_read_only: true,
            protected_nvme_mounted: false,
        }
    }

    #[test]
    fn r2_preflight_treats_swap_release_and_long_psi_as_diagnostic() {
        let mut samples = vec![
            snapshot(0),
            snapshot(40_000),
            snapshot(80_000),
            snapshot(120_000),
        ];
        samples[1].swap_free_bytes += 1;
        samples[1].swap_used_bytes -= 1;
        samples[2].swap_cached_bytes += 4_096;
        samples[2].pressure.some_avg60_millionths = 1;
        let analysis = analyze_r2_preflight(&samples, MIN_PREFLIGHT_DURATION_MS);
        assert!(analysis.passed, "{:?}", analysis.failures);
        assert!(!analysis.swap_free_byte_stable);
        assert!(!analysis.swap_cached_byte_stable);
    }

    #[test]
    fn r2_preflight_rejects_growth_avg10_and_cgroup_swap() {
        let base = vec![
            snapshot(0),
            snapshot(40_000),
            snapshot(80_000),
            snapshot(120_000),
        ];
        for mutate in [
            |sample: &mut HostSnapshot| sample.swap_used_bytes += 1,
            |sample: &mut HostSnapshot| sample.pressure.full_avg10_millionths = 1,
            |sample: &mut HostSnapshot| sample.cgroups[0].swap_current_bytes = Some(1),
        ] {
            let mut samples = base.clone();
            mutate(&mut samples[2]);
            assert!(!analyze_r2_preflight(&samples, MIN_PREFLIGHT_DURATION_MS).passed);
        }
    }

    #[test]
    fn retained_authority_is_exact() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/retained_20b_authority.json");
        validate_retained_trace(&path).unwrap();
    }

    #[test]
    fn construction_output_requires_current_schema_and_protected_identity() {
        let spec = CellSpec {
            name: "cold-monolithic-control".into(),
            constructor: "monolithic-control",
            load_class: "cold",
            executable: PathBuf::from("/bin/true"),
            arguments: Vec::new(),
            output: PathBuf::from("/tmp/unused.json"),
            memory_events: None,
        };
        let valid = serde_json::json!({
            "schema": "gpt-oss-rs.heterogeneous-construction/v6",
            "constructor": "monolithic_control",
            "protected_nvme": {
                "kernel_name": "nvme0n1",
                "read_only": true,
                "mounted": false
            },
            "passed": true
        });
        assert!(
            validate_cell_output(&spec, &serde_json::to_vec(&valid).unwrap(), "nvme0n1").is_ok()
        );

        let mut old_schema = valid.clone();
        old_schema["schema"] = "gpt-oss-rs.heterogeneous-construction/v5".into();
        assert!(
            validate_cell_output(&spec, &serde_json::to_vec(&old_schema).unwrap(), "nvme0n1")
                .is_err()
        );

        let mut changed_identity = valid;
        changed_identity["protected_nvme"]["kernel_name"] = "nvme1n1".into();
        assert!(validate_cell_output(
            &spec,
            &serde_json::to_vec(&changed_identity).unwrap(),
            "nvme0n1"
        )
        .is_err());
    }

    #[test]
    fn cgroup_allowances_and_post_exit_drift_are_fail_closed() {
        let baseline = CgroupSnapshot::default();
        assert!(enforce_cgroup(&baseline, &baseline, true).is_ok());
        let mut current = baseline.clone();
        current.file_bytes = CLEAN_FILE_ALLOWANCE_BYTES + 1;
        assert!(enforce_cgroup(&current, &baseline, false).is_err());
        current = baseline.clone();
        current.memory_current_bytes = POST_EXIT_DRIFT_BYTES + 1;
        assert!(enforce_cgroup(&current, &baseline, true).is_err());
        current = baseline.clone();
        current.file_bytes = POST_EXIT_DRIFT_BYTES + 1;
        assert!(enforce_cgroup(&current, &baseline, true).is_err());
    }

    #[test]
    fn gpu_admission_requires_two_unique_devices_and_the_frozen_reserve() {
        let valid = vec![
            GpuSnapshot {
                pci_bus_id: "0000:17:00.0".into(),
                memory_used_mib: 0,
                memory_free_mib: GPU_ADMISSION_FREE_MIB,
            },
            GpuSnapshot {
                pci_bus_id: "0000:65:00.0".into(),
                memory_used_mib: 0,
                memory_free_mib: GPU_ADMISSION_FREE_MIB,
            },
        ];
        assert!(gpu_admission_failures(&valid, "test").is_empty());
        let mut invalid = valid;
        invalid[1].pci_bus_id = invalid[0].pci_bus_id.clone();
        invalid[1].memory_free_mib -= 1;
        assert_eq!(gpu_admission_failures(&invalid, "test").len(), 2);
    }
}
