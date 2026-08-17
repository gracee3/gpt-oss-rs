//! Fail-closed host observation and child binding for an H8 construction run.
//!
//! This module deliberately does not load a checkpoint or initialize CUDA.  It
//! supplies the read-only host facts used by the standalone H8 watchdog and a
//! narrow environment binding consumed by `heterogeneous_construct --mode h8`.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const WATCHDOG_SCHEMA: &str = "gpt-oss-rs.heterogeneous-h8-watchdog/v1";
pub const PREFLIGHT_SCHEMA: &str = "gpt-oss-rs.heterogeneous-h8-preflight/v1";
pub const MIN_PREFLIGHT_DURATION_MS: u64 = 120_000;
pub const MIN_PREFLIGHT_SAMPLES: usize = 4;
pub const MIN_MEM_AVAILABLE_BYTES: u64 = 12 * 1024 * 1024 * 1024;
const EXPECTED_PROTECTED_NVME_MODEL: &str = "Samsung SSD 990 PRO 2TB";

pub const ENV_SCHEMA: &str = "GPT_OSS_H8_WATCHDOG_SCHEMA";
pub const ENV_PARENT_PID: &str = "GPT_OSS_H8_WATCHDOG_PARENT_PID";
pub const ENV_RUN_ID: &str = "GPT_OSS_H8_WATCHDOG_RUN_ID_SHA256";
pub const ENV_PREFLIGHT_SHA256: &str = "GPT_OSS_H8_WATCHDOG_PREFLIGHT_SHA256";
pub const ENV_EXECUTABLE_SHA256: &str = "GPT_OSS_H8_WATCHDOG_EXECUTABLE_SHA256";
pub const ENV_SWAP_BASELINE: &str = "GPT_OSS_H8_WATCHDOG_SWAP_BASELINE_BYTES";

#[derive(Debug, Clone)]
pub struct SystemPaths {
    pub proc_root: PathBuf,
    pub sys_root: PathBuf,
    pub cgroup_root: PathBuf,
}

impl Default for SystemPaths {
    fn default() -> Self {
        Self {
            proc_root: PathBuf::from("/proc"),
            sys_root: PathBuf::from("/sys"),
            cgroup_root: PathBuf::from("/sys/fs/cgroup"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HostSnapshot {
    pub elapsed_ms: u64,
    pub mem_available_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_free_bytes: u64,
    pub swap_used_bytes: u64,
    pub swap_cached_bytes: u64,
    pub pressure: MemoryPressure,
    pub target_tree_vm_swap_bytes: u64,
    pub attributed_process_vm_swap_bytes: u64,
    pub attribution: Vec<SwapAttribution>,
    pub proc_scan_complete: bool,
    pub active_h8_process_found: bool,
    pub cgroups: Vec<CgroupMemory>,
    pub swappiness: u64,
    #[serde(default)]
    pub protected_nvme_kernel_name: String,
    pub protected_nvme_read_only: bool,
    pub protected_nvme_mounted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProtectedNvmeState {
    pub kernel_name: String,
    pub read_only: bool,
    pub mounted: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryPressure {
    pub some_avg10_millionths: u64,
    pub some_avg60_millionths: u64,
    pub some_avg300_millionths: u64,
    pub full_avg10_millionths: u64,
    pub full_avg60_millionths: u64,
    pub full_avg300_millionths: u64,
}

impl MemoryPressure {
    pub fn is_zero(&self) -> bool {
        self.some_avg10_millionths == 0
            && self.some_avg60_millionths == 0
            && self.some_avg300_millionths == 0
            && self.full_avg10_millionths == 0
            && self.full_avg60_millionths == 0
            && self.full_avg300_millionths == 0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SwapCategory {
    TargetTree,
    Codex,
    ContainerRuntime,
    ShellOrSession,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SwapAttribution {
    pub category: SwapCategory,
    pub process_count: u64,
    pub vm_swap_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CgroupMemory {
    pub scope: CgroupScope,
    pub memory_current_bytes: Option<u64>,
    pub swap_current_bytes: Option<u64>,
    pub swap_max: Option<String>,
    pub high_events: Option<u64>,
    pub max_events: Option<u64>,
    pub oom_events: Option<u64>,
    pub oom_kill_events: Option<u64>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CgroupScope {
    GuardScope,
    UserSlice,
    SystemSlice,
    DockerService,
    ContainerdService,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreflightAnalysis {
    pub passed: bool,
    pub required_duration_ms: u64,
    pub observed_duration_ms: u64,
    pub sample_count: usize,
    pub baseline_swap_used_bytes: u64,
    pub baseline_swap_free_bytes: u64,
    pub baseline_swap_cached_bytes: u64,
    pub minimum_mem_available_bytes: u64,
    pub swap_free_byte_stable: bool,
    pub swap_cached_byte_stable: bool,
    pub global_swap_growth_zero: bool,
    pub target_tree_swap_zero: bool,
    pub pressure_zero: bool,
    pub proc_scan_complete: bool,
    pub no_active_h8_process: bool,
    pub protected_nvme_safe: bool,
    pub failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct H8WatchdogBinding {
    pub schema: String,
    pub run_id_sha256: String,
    pub preflight_sha256: String,
    pub watchdog_executable_sha256: String,
    pub swap_baseline_bytes: u64,
    pub direct_parent_validated: bool,
    pub parent_executable_validated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeGuardLimits {
    pub swap_baseline_bytes: u64,
    pub min_mem_available_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GuardViolation {
    pub reasons: Vec<String>,
}

/// The `/proc`-only portion of H8 process admission.
///
/// This is intentionally narrower than [`HostSnapshot`]: callers that only
/// need to detect an existing H8 process must not depend on this workstation's
/// protected-NVMe sysfs node. Production preflight and runtime admission still
/// use [`read_host_snapshot`] and therefore fail closed when any host safety
/// input, including the protected device, is unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H8ProcessScan {
    pub proc_scan_complete: bool,
    pub active_h8_process_found: bool,
}

pub fn read_h8_process_scan(child_root_pid: Option<u32>) -> Result<H8ProcessScan> {
    read_h8_process_scan_from(&SystemPaths::default(), child_root_pid)
}

pub fn read_h8_process_scan_from(
    paths: &SystemPaths,
    child_root_pid: Option<u32>,
) -> Result<H8ProcessScan> {
    let scan = scan_processes(&paths.proc_root, child_root_pid)?;
    Ok(H8ProcessScan {
        proc_scan_complete: scan.complete,
        active_h8_process_found: scan.active_h8_process_found,
    })
}

pub fn read_host_snapshot(child_root_pid: Option<u32>, elapsed_ms: u64) -> Result<HostSnapshot> {
    read_host_snapshot_from(&SystemPaths::default(), child_root_pid, elapsed_ms)
}

/// Resolve the protected Samsung namespace without opening a block device or
/// consulting serial/UUID data. The system namespace is discovered by tracing
/// the root mount's sysfs slave ancestry; the other expected namespace is the
/// protected device regardless of kernel enumeration order.
pub fn protected_nvme_state() -> Result<ProtectedNvmeState> {
    protected_nvme_state_from(&SystemPaths::default())
}

pub fn protected_nvme_state_from(paths: &SystemPaths) -> Result<ProtectedNvmeState> {
    let namespaces = expected_nvme_namespaces(paths)?;
    if namespaces.len() != 2 {
        bail!(
            "expected exactly two {EXPECTED_PROTECTED_NVME_MODEL} namespaces, found {}",
            namespaces.len()
        );
    }
    let expected = namespaces.iter().cloned().collect::<BTreeSet<_>>();
    let mountinfo = fs::read_to_string(paths.proc_root.join("self/mountinfo"))?;
    let (root_devices, mounted_devices) = mount_major_minors(&mountinfo)?;
    if root_devices.len() != 1 {
        bail!(
            "root mount resolves to {} distinct major:minor identities",
            root_devices.len()
        );
    }
    let root_major_minor = root_devices
        .iter()
        .next()
        .context("root mount is missing")?;
    let system_ancestors = block_namespace_ancestors(paths, root_major_minor, &expected)?;
    if system_ancestors.len() != 1 {
        bail!(
            "root block ancestry resolves to {} expected NVMe namespaces",
            system_ancestors.len()
        );
    }
    let system = system_ancestors
        .iter()
        .next()
        .context("system NVMe namespace is missing")?;
    let protected = namespaces
        .iter()
        .find(|name| *name != system)
        .context("protected NVMe namespace is ambiguous")?
        .clone();

    let mut mounted = false;
    for major_minor in mounted_devices {
        let ancestors = block_namespace_ancestors(paths, &major_minor, &expected)?;
        if ancestors.contains(&protected) {
            mounted = true;
            break;
        }
    }
    let ro = fs::read_to_string(paths.sys_root.join("block").join(&protected).join("ro"))?;
    let read_only = match ro.trim() {
        "0" => false,
        "1" => true,
        value => bail!("invalid read-only state for {protected}: {value}"),
    };
    if !read_only || mounted {
        bail!(
            "protected /dev/{protected} must be read-only and unmounted (read_only={read_only}, mounted={mounted})"
        );
    }
    Ok(ProtectedNvmeState {
        kernel_name: protected,
        read_only,
        mounted,
    })
}

pub fn read_host_snapshot_from(
    paths: &SystemPaths,
    child_root_pid: Option<u32>,
    elapsed_ms: u64,
) -> Result<HostSnapshot> {
    let meminfo = parse_kb_fields(&fs::read_to_string(paths.proc_root.join("meminfo"))?)?;
    let swap_total_bytes = required_value(&meminfo, "SwapTotal")?;
    let swap_free_bytes = required_value(&meminfo, "SwapFree")?;
    let scan = scan_processes(&paths.proc_root, child_root_pid)?;
    let pressure = parse_memory_pressure(&fs::read_to_string(
        paths.proc_root.join("pressure/memory"),
    )?)?;
    let cgroups = read_cgroups(paths)?;
    let swappiness = fs::read_to_string(paths.proc_root.join("sys/vm/swappiness"))?
        .trim()
        .parse()
        .context("invalid vm.swappiness")?;
    let protected_nvme = protected_nvme_state_from(paths)?;

    Ok(HostSnapshot {
        elapsed_ms,
        mem_available_bytes: required_value(&meminfo, "MemAvailable")?,
        swap_total_bytes,
        swap_free_bytes,
        swap_used_bytes: swap_total_bytes.saturating_sub(swap_free_bytes),
        swap_cached_bytes: required_value(&meminfo, "SwapCached")?,
        pressure,
        target_tree_vm_swap_bytes: scan.target_tree_vm_swap_bytes,
        attributed_process_vm_swap_bytes: scan.attributed_process_vm_swap_bytes,
        attribution: scan.attribution,
        proc_scan_complete: scan.complete,
        active_h8_process_found: scan.active_h8_process_found,
        cgroups,
        swappiness,
        protected_nvme_kernel_name: protected_nvme.kernel_name,
        protected_nvme_read_only: protected_nvme.read_only,
        protected_nvme_mounted: protected_nvme.mounted,
    })
}

fn expected_nvme_namespaces(paths: &SystemPaths) -> Result<Vec<String>> {
    let mut namespaces = Vec::new();
    for entry in fs::read_dir(paths.sys_root.join("block"))? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !is_nvme_namespace_name(&name) {
            continue;
        }
        let model_path = entry.path().join("device/model");
        let model = match fs::read_to_string(&model_path) {
            Ok(model) => model,
            Err(error) if error.kind() == ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        if model.trim() == EXPECTED_PROTECTED_NVME_MODEL {
            namespaces.push(name);
        }
    }
    namespaces.sort();
    Ok(namespaces)
}

fn is_nvme_namespace_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("nvme") else {
        return false;
    };
    let Some((controller, namespace)) = rest.split_once('n') else {
        return false;
    };
    !controller.is_empty()
        && controller.bytes().all(|byte| byte.is_ascii_digit())
        && !namespace.is_empty()
        && namespace.bytes().all(|byte| byte.is_ascii_digit())
}

fn mount_major_minors(mountinfo: &str) -> Result<(BTreeSet<String>, BTreeSet<String>)> {
    let mut roots = BTreeSet::new();
    let mut mounted = BTreeSet::new();
    for line in mountinfo.lines() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 6 {
            bail!("invalid mountinfo row");
        }
        let major_minor = fields[2];
        validate_major_minor(major_minor)?;
        mounted.insert(major_minor.to_owned());
        if fields[4] == "/" {
            roots.insert(major_minor.to_owned());
        }
    }
    if roots.is_empty() {
        bail!("root mount is absent from mountinfo");
    }
    Ok((roots, mounted))
}

fn validate_major_minor(value: &str) -> Result<()> {
    let (major, minor) = value.split_once(':').context("invalid mount major:minor")?;
    if major.is_empty()
        || minor.is_empty()
        || major.parse::<u32>().is_err()
        || minor.parse::<u32>().is_err()
    {
        bail!("invalid mount major:minor");
    }
    Ok(())
}

fn block_namespace_ancestors(
    paths: &SystemPaths,
    major_minor: &str,
    expected: &BTreeSet<String>,
) -> Result<BTreeSet<String>> {
    let path = paths.sys_root.join("dev/block").join(major_minor);
    let path = match fs::canonicalize(&path) {
        Ok(path) => path,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(BTreeSet::new()),
        Err(error) => return Err(error.into()),
    };
    let sys_root = fs::canonicalize(&paths.sys_root)?;
    let mut ancestors = BTreeSet::new();
    let mut visited = BTreeSet::new();
    collect_block_namespace_ancestors(&path, &sys_root, expected, &mut ancestors, &mut visited)?;
    Ok(ancestors)
}

fn collect_block_namespace_ancestors(
    path: &Path,
    sys_root: &Path,
    expected: &BTreeSet<String>,
    ancestors: &mut BTreeSet<String>,
    visited: &mut BTreeSet<PathBuf>,
) -> Result<()> {
    let canonical = fs::canonicalize(path)?;
    if !canonical.starts_with(sys_root) {
        bail!("block-device sysfs path escapes the sysfs root");
    }
    if !visited.insert(canonical.clone()) {
        return Ok(());
    }
    for component in canonical.components() {
        let name = component.as_os_str().to_string_lossy();
        if expected.contains(name.as_ref()) {
            ancestors.insert(name.into_owned());
        }
    }
    let slaves = canonical.join("slaves");
    let entries = match fs::read_dir(&slaves) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    for entry in entries {
        collect_block_namespace_ancestors(&entry?.path(), sys_root, expected, ancestors, visited)?;
    }
    Ok(())
}

pub fn analyze_preflight(samples: &[HostSnapshot], required_duration_ms: u64) -> PreflightAnalysis {
    let mut failures = Vec::new();
    let first = samples.first();
    let observed_duration_ms = match (first, samples.last()) {
        (Some(start), Some(end)) => end.elapsed_ms.saturating_sub(start.elapsed_ms),
        _ => 0,
    };
    if samples.len() < MIN_PREFLIGHT_SAMPLES {
        failures.push(format!(
            "preflight requires at least {MIN_PREFLIGHT_SAMPLES} samples"
        ));
    }
    if required_duration_ms < MIN_PREFLIGHT_DURATION_MS {
        failures.push(format!(
            "preflight configured duration must be at least {MIN_PREFLIGHT_DURATION_MS} ms"
        ));
    }
    if observed_duration_ms < required_duration_ms {
        failures.push("preflight did not span its configured duration".into());
    }

    let baseline_swap_used_bytes = first.map_or(0, |sample| sample.swap_used_bytes);
    let baseline_swap_free_bytes = first.map_or(0, |sample| sample.swap_free_bytes);
    let baseline_swap_cached_bytes = first.map_or(0, |sample| sample.swap_cached_bytes);
    let minimum_mem_available_bytes = samples
        .iter()
        .map(|sample| sample.mem_available_bytes)
        .min()
        .unwrap_or(0);
    let swap_free_byte_stable = samples
        .iter()
        .all(|sample| sample.swap_free_bytes == baseline_swap_free_bytes);
    let swap_cached_byte_stable = samples
        .iter()
        .all(|sample| sample.swap_cached_bytes == baseline_swap_cached_bytes);
    let global_swap_growth_zero = samples
        .iter()
        .all(|sample| sample.swap_used_bytes <= baseline_swap_used_bytes);
    let target_tree_swap_zero = samples
        .iter()
        .all(|sample| sample.target_tree_vm_swap_bytes == 0);
    let pressure_zero = samples.iter().all(|sample| sample.pressure.is_zero());
    let proc_scan_complete = samples.iter().all(|sample| sample.proc_scan_complete);
    let no_active_h8_process = samples.iter().all(|sample| !sample.active_h8_process_found);
    let protected_nvme_safe = samples
        .iter()
        .all(|sample| sample.protected_nvme_read_only && !sample.protected_nvme_mounted)
        && first.is_some_and(|sample| !sample.protected_nvme_kernel_name.is_empty())
        && samples.iter().all(|sample| {
            first.is_some_and(|first| {
                sample.protected_nvme_kernel_name == first.protected_nvme_kernel_name
            })
        });

    if !swap_free_byte_stable {
        failures.push("SwapFree changed during the stability window".into());
    }
    if !swap_cached_byte_stable {
        failures.push("SwapCached changed during the stability window".into());
    }
    if !global_swap_growth_zero {
        failures.push("global swap allocation grew during the stability window".into());
    }
    if !target_tree_swap_zero {
        failures.push("the watchdog process tree used swap".into());
    }
    if !pressure_zero {
        failures.push("memory PSI was nonzero during the stability window".into());
    }
    if !proc_scan_complete {
        failures.push("process swap attribution was incomplete".into());
    }
    if !no_active_h8_process {
        failures.push("an H8 construction process was already active".into());
    }
    if !protected_nvme_safe {
        failures.push(
            "the protected NVMe lacked one stable nonempty read-only unmounted identity".into(),
        );
    }
    if minimum_mem_available_bytes < MIN_MEM_AVAILABLE_BYTES {
        failures.push("MemAvailable fell below the H8 minimum".into());
    }

    PreflightAnalysis {
        passed: failures.is_empty(),
        required_duration_ms,
        observed_duration_ms,
        sample_count: samples.len(),
        baseline_swap_used_bytes,
        baseline_swap_free_bytes,
        baseline_swap_cached_bytes,
        minimum_mem_available_bytes,
        swap_free_byte_stable,
        swap_cached_byte_stable,
        global_swap_growth_zero,
        target_tree_swap_zero,
        pressure_zero,
        proc_scan_complete,
        no_active_h8_process,
        protected_nvme_safe,
        failures,
    }
}

/// Analyze the R2 retained-20B admission policy.
///
/// Unlike the older H8 policy, `SwapFree` and `SwapCached` movement is retained
/// as diagnostic evidence and is not itself a rejection.  The hard pressure
/// boundary is limited to the two `avg10` values frozen by R2; longer-window
/// values remain diagnostic.
pub fn analyze_r2_preflight(
    samples: &[HostSnapshot],
    required_duration_ms: u64,
) -> PreflightAnalysis {
    let mut analysis = analyze_preflight(samples, required_duration_ms);
    let pressure_avg10_zero = samples.iter().all(|sample| {
        sample.pressure.some_avg10_millionths == 0 && sample.pressure.full_avg10_millionths == 0
    });
    let guard_cgroup_swap_zero = samples.iter().all(guard_cgroup_swap_is_exact_zero);

    analysis.failures.retain(|failure| {
        failure != "SwapFree changed during the stability window"
            && failure != "SwapCached changed during the stability window"
            && (!pressure_avg10_zero
                || failure != "memory PSI was nonzero during the stability window")
    });
    analysis.pressure_zero = pressure_avg10_zero;
    if !guard_cgroup_swap_zero {
        analysis
            .failures
            .push("the comparison cgroup used swap".into());
    }
    analysis.passed = analysis.failures.is_empty();
    analysis
}

pub fn evaluate_runtime_guard(
    snapshot: &HostSnapshot,
    limits: &RuntimeGuardLimits,
) -> std::result::Result<(), GuardViolation> {
    let mut reasons = Vec::new();
    if snapshot.swap_used_bytes > limits.swap_baseline_bytes {
        reasons.push(format!(
            "global swap grew above baseline: baseline={} observed={}",
            limits.swap_baseline_bytes, snapshot.swap_used_bytes
        ));
    }
    if snapshot.target_tree_vm_swap_bytes != 0 {
        reasons.push(format!(
            "guarded child tree used {} swap bytes",
            snapshot.target_tree_vm_swap_bytes
        ));
    }
    if snapshot.mem_available_bytes < limits.min_mem_available_bytes {
        reasons.push(format!(
            "MemAvailable fell below minimum: minimum={} observed={}",
            limits.min_mem_available_bytes, snapshot.mem_available_bytes
        ));
    }
    if !snapshot.pressure.is_zero() {
        reasons.push("memory PSI became nonzero".into());
    }
    if !snapshot.proc_scan_complete {
        reasons.push("process swap attribution became incomplete".into());
    }
    if snapshot.protected_nvme_kernel_name.is_empty()
        || !snapshot.protected_nvme_read_only
        || snapshot.protected_nvme_mounted
    {
        reasons.push("protected NVMe state changed".into());
    }
    if reasons.is_empty() {
        Ok(())
    } else {
        Err(GuardViolation { reasons })
    }
}

/// Evaluate the live R2 retained-20B boundary without imposing the obsolete
/// exact `SwapFree`/`SwapCached` or long-window PSI requirements.
pub fn evaluate_r2_runtime_guard(
    snapshot: &HostSnapshot,
    limits: &RuntimeGuardLimits,
) -> std::result::Result<(), GuardViolation> {
    let mut reasons = Vec::new();
    if snapshot.swap_used_bytes > limits.swap_baseline_bytes {
        reasons.push(format!(
            "global swap grew above baseline: baseline={} observed={}",
            limits.swap_baseline_bytes, snapshot.swap_used_bytes
        ));
    }
    if snapshot.target_tree_vm_swap_bytes != 0 {
        reasons.push(format!(
            "guarded child tree used {} swap bytes",
            snapshot.target_tree_vm_swap_bytes
        ));
    }
    if !guard_cgroup_swap_is_exact_zero(snapshot) {
        reasons.push("comparison cgroup swap is nonzero or unreadable".into());
    }
    if snapshot.mem_available_bytes < limits.min_mem_available_bytes {
        reasons.push(format!(
            "MemAvailable fell below minimum: minimum={} observed={}",
            limits.min_mem_available_bytes, snapshot.mem_available_bytes
        ));
    }
    if snapshot.pressure.some_avg10_millionths != 0 || snapshot.pressure.full_avg10_millionths != 0
    {
        reasons.push("memory PSI avg10 became nonzero".into());
    }
    if !snapshot.proc_scan_complete {
        reasons.push("process swap attribution became incomplete".into());
    }
    if snapshot.protected_nvme_kernel_name.is_empty()
        || !snapshot.protected_nvme_read_only
        || snapshot.protected_nvme_mounted
    {
        reasons.push("protected NVMe state changed".into());
    }
    if reasons.is_empty() {
        Ok(())
    } else {
        Err(GuardViolation { reasons })
    }
}

fn guard_cgroup_swap_is_exact_zero(snapshot: &HostSnapshot) -> bool {
    let mut guard = snapshot
        .cgroups
        .iter()
        .filter(|cgroup| cgroup.scope == CgroupScope::GuardScope);
    guard
        .next()
        .is_some_and(|cgroup| cgroup.swap_current_bytes == Some(0))
        && guard.next().is_none()
}

pub fn require_h8_watchdog_binding(current_swap_used_bytes: u64) -> Result<H8WatchdogBinding> {
    let schema = required_env(ENV_SCHEMA)?;
    let expected_parent: u32 = required_env(ENV_PARENT_PID)?
        .parse()
        .context("invalid H8 watchdog parent PID")?;
    let actual_parent = current_parent_pid(Path::new("/proc"))?;
    let run_id_sha256 = required_hash_env(ENV_RUN_ID)?;
    let preflight_sha256 = required_hash_env(ENV_PREFLIGHT_SHA256)?;
    let watchdog_executable_sha256 = required_hash_env(ENV_EXECUTABLE_SHA256)?;
    let swap_baseline_bytes: u64 = required_env(ENV_SWAP_BASELINE)?
        .parse()
        .context("invalid H8 watchdog swap baseline")?;
    let actual_parent_hash = sha256_file(Path::new(&format!("/proc/{actual_parent}/exe")))?;
    validate_binding_values(BindingValues {
        schema,
        run_id_sha256,
        preflight_sha256,
        watchdog_executable_sha256,
        swap_baseline_bytes,
        expected_parent,
        actual_parent,
        actual_parent_hash,
        current_swap_used_bytes,
    })
}

struct BindingValues {
    schema: String,
    run_id_sha256: String,
    preflight_sha256: String,
    watchdog_executable_sha256: String,
    swap_baseline_bytes: u64,
    expected_parent: u32,
    actual_parent: u32,
    actual_parent_hash: String,
    current_swap_used_bytes: u64,
}

fn validate_binding_values(values: BindingValues) -> Result<H8WatchdogBinding> {
    if values.schema != WATCHDOG_SCHEMA {
        bail!("H8 watchdog schema mismatch");
    }
    for (label, value) in [
        ("run ID", values.run_id_sha256.as_str()),
        ("preflight", values.preflight_sha256.as_str()),
        (
            "watchdog executable",
            values.watchdog_executable_sha256.as_str(),
        ),
    ] {
        if !is_sha256(value) {
            bail!("invalid H8 watchdog {label} SHA-256");
        }
    }
    if values.expected_parent != values.actual_parent {
        bail!("H8 must be a direct child of its watchdog");
    }
    if values.current_swap_used_bytes != values.swap_baseline_bytes {
        bail!(
            "global swap changed between watchdog admission and H8 startup: baseline={} observed={}",
            values.swap_baseline_bytes,
            values.current_swap_used_bytes
        );
    }
    if values.actual_parent_hash != values.watchdog_executable_sha256 {
        bail!("H8 watchdog parent executable identity mismatch");
    }
    Ok(H8WatchdogBinding {
        schema: values.schema,
        run_id_sha256: values.run_id_sha256,
        preflight_sha256: values.preflight_sha256,
        watchdog_executable_sha256: values.watchdog_executable_sha256,
        swap_baseline_bytes: values.swap_baseline_bytes,
        direct_parent_validated: true,
        parent_executable_validated: true,
    })
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(sha256_bytes(&bytes))
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn required_env(name: &str) -> Result<String> {
    env::var(name).with_context(|| format!("H8 requires watchdog environment {name}"))
}

fn required_hash_env(name: &str) -> Result<String> {
    let value = required_env(name)?;
    if !is_sha256(&value) {
        bail!("invalid SHA-256 in watchdog environment {name}");
    }
    Ok(value)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
}

fn current_parent_pid(proc_root: &Path) -> Result<u32> {
    let fields = parse_status(&fs::read_to_string(proc_root.join("self/status"))?)?;
    fields
        .get("PPid")
        .context("/proc/self/status lacks PPid")?
        .parse()
        .context("invalid parent PID")
}

#[derive(Debug)]
struct ProcessScan {
    target_tree_vm_swap_bytes: u64,
    attributed_process_vm_swap_bytes: u64,
    attribution: Vec<SwapAttribution>,
    complete: bool,
    active_h8_process_found: bool,
}

#[derive(Debug)]
struct ProcessFact {
    pid: u32,
    parent: u32,
    name: String,
    vm_swap_bytes: u64,
    is_h8: bool,
}

fn scan_processes(proc_root: &Path, child_root_pid: Option<u32>) -> Result<ProcessScan> {
    let mut facts = Vec::new();
    let mut complete = true;
    for entry in fs::read_dir(proc_root)? {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                if error.kind() != ErrorKind::NotFound {
                    complete = false;
                }
                continue;
            }
        };
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        let status = match fs::read_to_string(entry.path().join("status")) {
            Ok(status) => status,
            Err(error) if error.kind() == ErrorKind::NotFound => continue,
            Err(_) => {
                complete = false;
                continue;
            }
        };
        let fields = match parse_status(&status) {
            Ok(fields) => fields,
            Err(_) => {
                complete = false;
                continue;
            }
        };
        let parent = fields
            .get("PPid")
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        let name = fields.get("Name").cloned().unwrap_or_default();
        let vm_swap_bytes = fields
            .get("VmSwap")
            .and_then(|value| parse_kib_value(value).ok())
            .unwrap_or(0);
        let is_h8 = match fs::read(entry.path().join("cmdline")) {
            Ok(command) if command_requests_exact_h8(&command) => {
                match fs::read_link(entry.path().join("exe")) {
                    Ok(executable) => executable_is_construct(&executable, &name),
                    Err(_) => {
                        // Only an exact H8-mode candidate needs executable
                        // identity. An unreadable candidate is ambiguous and
                        // therefore fails the scan closed; inaccessible exe
                        // links for unrelated root/kernel processes are never
                        // opened and do not poison attribution.
                        complete = false;
                        name.starts_with("heterogeneous_")
                    }
                }
            }
            Ok(_) => false,
            Err(error) if error.kind() == ErrorKind::NotFound => false,
            Err(_) => {
                complete = false;
                false
            }
        };
        facts.push(ProcessFact {
            pid,
            parent,
            name,
            vm_swap_bytes,
            is_h8,
        });
    }

    let mut target_tree = BTreeSet::new();
    if let Some(root) = child_root_pid {
        target_tree.insert(root);
        loop {
            let before = target_tree.len();
            for fact in &facts {
                if target_tree.contains(&fact.parent) {
                    target_tree.insert(fact.pid);
                }
            }
            if target_tree.len() == before {
                break;
            }
        }
    }

    let mut aggregated: BTreeMap<SwapCategory, (u64, u64)> = BTreeMap::new();
    let mut target_tree_vm_swap_bytes = 0_u64;
    let mut attributed_process_vm_swap_bytes = 0_u64;
    let mut active_h8_process_found = false;
    for fact in facts {
        let target = target_tree.contains(&fact.pid);
        if fact.is_h8 && !target {
            active_h8_process_found = true;
        }
        if target {
            target_tree_vm_swap_bytes =
                target_tree_vm_swap_bytes.saturating_add(fact.vm_swap_bytes);
        }
        if fact.vm_swap_bytes == 0 {
            continue;
        }
        attributed_process_vm_swap_bytes =
            attributed_process_vm_swap_bytes.saturating_add(fact.vm_swap_bytes);
        let category = if target {
            SwapCategory::TargetTree
        } else {
            categorize_process(&fact.name)
        };
        let entry = aggregated.entry(category).or_default();
        entry.0 = entry.0.saturating_add(1);
        entry.1 = entry.1.saturating_add(fact.vm_swap_bytes);
    }
    let attribution = aggregated
        .into_iter()
        .map(
            |(category, (process_count, vm_swap_bytes))| SwapAttribution {
                category,
                process_count,
                vm_swap_bytes,
            },
        )
        .collect();
    Ok(ProcessScan {
        target_tree_vm_swap_bytes,
        attributed_process_vm_swap_bytes,
        attribution,
        complete,
        active_h8_process_found,
    })
}

fn command_requests_exact_h8(command: &[u8]) -> bool {
    let arguments = command
        .split(|byte| *byte == 0)
        .filter(|argument| !argument.is_empty())
        .map(String::from_utf8_lossy)
        .collect::<Vec<_>>();
    let mut modes = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        let argument = arguments[index].as_ref();
        if argument == "--mode" {
            index = index.saturating_add(1);
            if let Some(value) = arguments.get(index) {
                modes.push(value.as_ref());
            } else {
                return false;
            }
        } else if let Some(value) = argument.strip_prefix("--mode=") {
            modes.push(value);
        }
        index = index.saturating_add(1);
    }
    modes == ["h8"]
}

fn executable_is_construct(executable: &Path, process_name: &str) -> bool {
    let exact_executable =
        executable.file_name().and_then(|name| name.to_str()) == Some("heterogeneous_construct");
    // Linux truncates `comm` to 15 bytes, so an fd-executed constructor is
    // reported as `heterogeneous_co`. Do not accept the broader
    // `heterogeneous_` prefix: the watchdog itself is `heterogeneous_h8` and
    // its argv contains the guarded child's trailing `--mode h8` arguments.
    let conservative_name_match = process_name == "heterogeneous_co";
    exact_executable || conservative_name_match
}

fn categorize_process(name: &str) -> SwapCategory {
    let name = name.to_ascii_lowercase();
    if name.contains("codex") {
        SwapCategory::Codex
    } else if name.contains("docker") || name.contains("containerd") {
        SwapCategory::ContainerRuntime
    } else if matches!(
        name.as_str(),
        "bash" | "sh" | "zsh" | "tmux" | "sshd" | "systemd" | "login"
    ) {
        SwapCategory::ShellOrSession
    } else {
        SwapCategory::Other
    }
}

fn parse_status(text: &str) -> Result<BTreeMap<String, String>> {
    let mut fields = BTreeMap::new();
    for line in text.lines() {
        if let Some((key, value)) = line.split_once(':') {
            fields.insert(key.to_owned(), value.trim().to_owned());
        }
    }
    if fields.is_empty() {
        bail!("empty process status")
    }
    Ok(fields)
}

fn parse_kb_fields(text: &str) -> Result<BTreeMap<String, u64>> {
    let mut fields = BTreeMap::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        fields.insert(key.to_owned(), parse_kib_value(value.trim())?);
    }
    Ok(fields)
}

fn parse_kib_value(value: &str) -> Result<u64> {
    let kib = value
        .split_whitespace()
        .next()
        .context("missing KiB value")?
        .parse::<u64>()?;
    kib.checked_mul(1024).context("KiB value overflows bytes")
}

fn required_value(values: &BTreeMap<String, u64>, key: &str) -> Result<u64> {
    values
        .get(key)
        .copied()
        .with_context(|| format!("missing {key}"))
}

fn parse_memory_pressure(text: &str) -> Result<MemoryPressure> {
    let mut rows = BTreeMap::new();
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let row = fields.next().context("pressure row has no name")?;
        let mut values = BTreeMap::new();
        for field in fields {
            if let Some((key, value)) = field.split_once('=') {
                values.insert(key, value);
            }
        }
        rows.insert(row, values);
    }
    let some = rows.get("some").context("pressure data lacks some row")?;
    let full = rows.get("full").context("pressure data lacks full row")?;
    Ok(MemoryPressure {
        some_avg10_millionths: parse_decimal_millionths(required_str(some, "avg10")?)?,
        some_avg60_millionths: parse_decimal_millionths(required_str(some, "avg60")?)?,
        some_avg300_millionths: parse_decimal_millionths(required_str(some, "avg300")?)?,
        full_avg10_millionths: parse_decimal_millionths(required_str(full, "avg10")?)?,
        full_avg60_millionths: parse_decimal_millionths(required_str(full, "avg60")?)?,
        full_avg300_millionths: parse_decimal_millionths(required_str(full, "avg300")?)?,
    })
}

fn required_str<'a>(values: &'a BTreeMap<&str, &str>, key: &str) -> Result<&'a str> {
    values
        .get(key)
        .copied()
        .with_context(|| format!("missing pressure {key}"))
}

fn parse_decimal_millionths(value: &str) -> Result<u64> {
    let (whole, fractional) = value.split_once('.').unwrap_or((value, ""));
    let whole: u64 = whole.parse()?;
    if fractional.len() > 6 || !fractional.bytes().all(|byte| byte.is_ascii_digit()) {
        bail!("invalid pressure decimal")
    }
    let mut padded = fractional.to_owned();
    while padded.len() < 6 {
        padded.push('0');
    }
    whole
        .checked_mul(1_000_000)
        .and_then(|scaled| {
            padded
                .parse::<u64>()
                .ok()
                .and_then(|part| scaled.checked_add(part))
        })
        .context("pressure decimal overflows")
}

fn read_cgroups(paths: &SystemPaths) -> Result<Vec<CgroupMemory>> {
    let current_relative = current_cgroup_relative(&paths.proc_root)?;
    let candidates = [
        (CgroupScope::GuardScope, current_relative),
        (CgroupScope::UserSlice, PathBuf::from("user.slice")),
        (CgroupScope::SystemSlice, PathBuf::from("system.slice")),
        (
            CgroupScope::DockerService,
            PathBuf::from("system.slice/docker.service"),
        ),
        (
            CgroupScope::ContainerdService,
            PathBuf::from("system.slice/containerd.service"),
        ),
    ];
    let mut result = Vec::with_capacity(candidates.len());
    for (scope, relative) in candidates {
        let path = paths.cgroup_root.join(relative);
        if !path.is_dir() {
            continue;
        }
        let events = read_numeric_fields_optional(&path.join("memory.events"))?;
        result.push(CgroupMemory {
            scope,
            memory_current_bytes: read_u64_optional(&path.join("memory.current"))?,
            swap_current_bytes: read_u64_optional(&path.join("memory.swap.current"))?,
            swap_max: read_string_optional(&path.join("memory.swap.max"))?,
            high_events: events
                .as_ref()
                .and_then(|values| values.get("high").copied()),
            max_events: events
                .as_ref()
                .and_then(|values| values.get("max").copied()),
            oom_events: events
                .as_ref()
                .and_then(|values| values.get("oom").copied()),
            oom_kill_events: events
                .as_ref()
                .and_then(|values| values.get("oom_kill").copied()),
        });
    }
    Ok(result)
}

fn current_cgroup_relative(proc_root: &Path) -> Result<PathBuf> {
    let text = fs::read_to_string(proc_root.join("self/cgroup"))?;
    let path = text
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .context("unified cgroup path is missing")?;
    let relative = Path::new(path.trim_start_matches('/'));
    if relative.components().any(|component| {
        matches!(
            component,
            std::path::Component::ParentDir | std::path::Component::RootDir
        )
    }) {
        bail!("invalid current cgroup path")
    }
    Ok(relative.to_owned())
}

fn read_u64_optional(path: &Path) -> Result<Option<u64>> {
    match fs::read_to_string(path) {
        Ok(value) => Ok(Some(value.trim().parse()?)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn read_string_optional(path: &Path) -> Result<Option<String>> {
    match fs::read_to_string(path) {
        Ok(value) => Ok(Some(value.trim().to_owned())),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn read_numeric_fields_optional(path: &Path) -> Result<Option<BTreeMap<String, u64>>> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mut values = BTreeMap::new();
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let key = fields.next().context("cgroup event lacks key")?;
        let value = fields.next().context("cgroup event lacks value")?.parse()?;
        values.insert(key.to_owned(), value);
    }
    Ok(Some(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    type SnapshotMutation = Box<dyn Fn(&mut HostSnapshot)>;

    fn process_status(name: &str, parent: u32) -> String {
        format!("Name:\t{name}\nPPid:\t{parent}\nVmSwap:\t0 kB\n")
    }

    fn snapshot(elapsed_ms: u64) -> HostSnapshot {
        HostSnapshot {
            elapsed_ms,
            mem_available_bytes: MIN_MEM_AVAILABLE_BYTES + 1,
            swap_total_bytes: 1000,
            swap_free_bytes: 900,
            swap_used_bytes: 100,
            swap_cached_bytes: 25,
            pressure: MemoryPressure::default(),
            target_tree_vm_swap_bytes: 0,
            attributed_process_vm_swap_bytes: 100,
            attribution: Vec::new(),
            proc_scan_complete: true,
            active_h8_process_found: false,
            cgroups: Vec::new(),
            swappiness: 60,
            protected_nvme_kernel_name: "nvme1n1".into(),
            protected_nvme_read_only: true,
            protected_nvme_mounted: false,
        }
    }

    struct NvmeFixture {
        root: PathBuf,
        paths: SystemPaths,
    }

    impl Drop for NvmeFixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).unwrap();
        }
    }

    fn nvme_fixture(
        system: &str,
        protected: &str,
        protected_ro: bool,
        protected_mounted: bool,
        ambiguous_root: bool,
    ) -> NvmeFixture {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "gpt-oss-protected-nvme-{}-{unique}",
            std::process::id()
        ));
        let paths = SystemPaths {
            proc_root: root.join("proc"),
            sys_root: root.join("sys"),
            cgroup_root: root.join("cgroup"),
        };
        fs::create_dir_all(paths.proc_root.join("self")).unwrap();
        fs::create_dir_all(paths.sys_root.join("block")).unwrap();
        fs::create_dir_all(paths.sys_root.join("dev/block")).unwrap();

        for (name, ro) in [(system, false), (protected, protected_ro)] {
            let disk = paths.sys_root.join("devices/block").join(name);
            fs::create_dir_all(disk.join("device")).unwrap();
            fs::write(
                disk.join("device/model"),
                format!("{EXPECTED_PROTECTED_NVME_MODEL}\n"),
            )
            .unwrap();
            fs::write(disk.join("ro"), if ro { "1\n" } else { "0\n" }).unwrap();
            symlink(
                Path::new("../devices/block").join(name),
                paths.sys_root.join("block").join(name),
            )
            .unwrap();
        }

        let system_partition_name = format!("{system}p3");
        let system_partition = paths
            .sys_root
            .join("devices/block")
            .join(system)
            .join(&system_partition_name);
        fs::create_dir_all(&system_partition).unwrap();
        let dm = paths.sys_root.join("devices/virtual/block/dm-0");
        fs::create_dir_all(dm.join("slaves")).unwrap();
        symlink(
            &system_partition,
            dm.join("slaves").join(&system_partition_name),
        )
        .unwrap();
        if ambiguous_root {
            symlink(
                paths.sys_root.join("devices/block").join(protected),
                dm.join("slaves").join(protected),
            )
            .unwrap();
        }
        symlink(
            Path::new("../../devices/virtual/block/dm-0"),
            paths.sys_root.join("dev/block/252:0"),
        )
        .unwrap();

        let mut mountinfo = "1 0 252:0 / / rw - ext4 /dev/dm-0 rw\n".to_owned();
        if protected_mounted {
            symlink(
                Path::new("../../devices/block").join(protected),
                paths.sys_root.join("dev/block/259:7"),
            )
            .unwrap();
            mountinfo.push_str("2 1 259:7 / /mnt/protected ro - ext4 /dev/nvme ro\n");
        }
        fs::write(paths.proc_root.join("self/mountinfo"), mountinfo).unwrap();
        NvmeFixture { root, paths }
    }

    #[test]
    fn protected_nvme_resolves_both_enumeration_orders_without_serial_data() {
        for (system, protected) in [("nvme0n1", "nvme1n1"), ("nvme1n1", "nvme0n1")] {
            let fixture = nvme_fixture(system, protected, true, false, false);
            let state = protected_nvme_state_from(&fixture.paths).unwrap();
            assert_eq!(state.kernel_name, protected);
            assert!(state.read_only);
            assert!(!state.mounted);
            assert!(!fixture
                .paths
                .sys_root
                .join("devices/block")
                .join(protected)
                .join("device/serial")
                .exists());
        }
    }

    #[test]
    fn protected_nvme_fails_closed_for_rw_mounted_and_ambiguous_topology() {
        let fixture = nvme_fixture("nvme0n1", "nvme1n1", false, false, false);
        assert!(protected_nvme_state_from(&fixture.paths).is_err());

        let fixture = nvme_fixture("nvme0n1", "nvme1n1", true, true, false);
        assert!(protected_nvme_state_from(&fixture.paths).is_err());

        let fixture = nvme_fixture("nvme0n1", "nvme1n1", true, false, true);
        assert!(protected_nvme_state_from(&fixture.paths).is_err());
    }

    #[test]
    fn preflight_requires_full_byte_stable_window() {
        let samples = [
            snapshot(0),
            snapshot(40_000),
            snapshot(80_000),
            snapshot(120_000),
        ];
        let result = analyze_preflight(&samples, MIN_PREFLIGHT_DURATION_MS);
        assert!(result.passed, "{:?}", result.failures);

        let mut changed = samples.clone();
        changed[2].swap_free_bytes -= 1;
        changed[2].swap_used_bytes += 1;
        let result = analyze_preflight(&changed, MIN_PREFLIGHT_DURATION_MS);
        assert!(!result.passed);
        assert!(!result.swap_free_byte_stable);
        assert!(!result.global_swap_growth_zero);
    }

    #[test]
    fn preflight_rejects_empty_or_changing_protected_nvme_identity() {
        let samples = [
            snapshot(0),
            snapshot(40_000),
            snapshot(80_000),
            snapshot(120_000),
        ];
        let mut empty = samples.clone();
        empty[0].protected_nvme_kernel_name.clear();
        assert!(!analyze_preflight(&empty, MIN_PREFLIGHT_DURATION_MS).passed);

        let mut changed = samples;
        changed[2].protected_nvme_kernel_name = "nvme0n1".into();
        assert!(!analyze_preflight(&changed, MIN_PREFLIGHT_DURATION_MS).passed);
    }

    #[test]
    fn preflight_duration_is_measured_from_first_retained_sample() {
        let mut samples = [
            snapshot(49),
            snapshot(30_060),
            snapshot(60_072),
            snapshot(90_085),
            snapshot(120_016),
        ];
        let result = analyze_preflight(&samples, MIN_PREFLIGHT_DURATION_MS);
        assert!(!result.passed);
        assert_eq!(result.observed_duration_ms, 119_967);

        samples[0].elapsed_ms = 0;
        samples[4].elapsed_ms = MIN_PREFLIGHT_DURATION_MS;
        let result = analyze_preflight(&samples, MIN_PREFLIGHT_DURATION_MS);
        assert!(result.passed, "{:?}", result.failures);
    }

    #[test]
    fn runtime_guard_is_fail_closed_for_each_resource_boundary() {
        let limits = RuntimeGuardLimits {
            swap_baseline_bytes: 100,
            min_mem_available_bytes: MIN_MEM_AVAILABLE_BYTES,
        };
        assert!(evaluate_runtime_guard(&snapshot(0), &limits).is_ok());

        let mutations: Vec<SnapshotMutation> = vec![
            Box::new(|sample| sample.swap_used_bytes = 101),
            Box::new(|sample| sample.target_tree_vm_swap_bytes = 1),
            Box::new(|sample| sample.mem_available_bytes = MIN_MEM_AVAILABLE_BYTES - 1),
            Box::new(|sample| sample.pressure.some_avg10_millionths = 1),
            Box::new(|sample| sample.proc_scan_complete = false),
            Box::new(|sample| sample.protected_nvme_read_only = false),
            Box::new(|sample| sample.protected_nvme_mounted = true),
        ];
        for mutate in mutations {
            let mut sample = snapshot(0);
            mutate(&mut sample);
            assert!(evaluate_runtime_guard(&sample, &limits).is_err());
        }
    }

    #[test]
    fn pressure_parser_is_exact_and_rejects_excess_precision() {
        let pressure = parse_memory_pressure(
            "some avg10=0.00 avg60=0.01 avg300=1.25 total=3\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=1\n",
        )
        .unwrap();
        assert_eq!(pressure.some_avg60_millionths, 10_000);
        assert_eq!(pressure.some_avg300_millionths, 1_250_000);
        assert!(!pressure.is_zero());
        assert!(parse_decimal_millionths("0.0000001").is_err());
    }

    #[test]
    fn command_detection_requires_construct_and_h8_mode() {
        let executable = Path::new("/tmp/heterogeneous_construct");
        assert!(command_requests_exact_h8(b"/proc/self/fd/9\0--mode\0h8\0"));
        assert!(command_requests_exact_h8(b"/proc/self/fd/9\0--mode=h8\0"));
        assert!(!command_requests_exact_h8(
            b"/tmp/heterogeneous_construct\0--mode\0warm\0"
        ));
        assert!(!command_requests_exact_h8(
            b"/tmp/heterogeneous_construct\0--mode\0h8\0--mode=warm\0"
        ));
        assert!(executable_is_construct(executable, "other"));
        assert!(executable_is_construct(
            Path::new("/proc/self/fd/9"),
            "heterogeneous_co"
        ));
        assert!(!executable_is_construct(Path::new("/tmp/other"), "other"));
    }

    #[test]
    fn system_paths_fixture_reads_exe_only_for_exact_h8_candidates() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "gpt-oss-h8-proc-scan-{}-{unique}",
            std::process::id()
        ));
        let paths = SystemPaths {
            proc_root: root.join("proc"),
            sys_root: root.join("sys"),
            cgroup_root: root.join("cgroup"),
        };
        let ordinary = paths.proc_root.join("101");
        fs::create_dir_all(&ordinary).unwrap();
        fs::write(ordinary.join("status"), process_status("root-service", 1)).unwrap();
        fs::write(
            ordinary.join("cmdline"),
            b"/usr/sbin/root-service\0--serve\0",
        )
        .unwrap();
        // A regular file makes read_link fail just as an inaccessible proc exe
        // would. The ordinary process must not make the scan incomplete because
        // its command line is not an exact H8 candidate.
        fs::write(ordinary.join("exe"), b"not a symlink").unwrap();

        let scan = read_h8_process_scan_from(&paths, None).unwrap();
        assert!(scan.proc_scan_complete);
        assert!(!scan.active_h8_process_found);

        let watchdog = paths.proc_root.join("103");
        fs::create_dir_all(&watchdog).unwrap();
        fs::write(
            watchdog.join("status"),
            process_status("heterogeneous_h8", 1),
        )
        .unwrap();
        fs::write(
            watchdog.join("cmdline"),
            b"/tmp/heterogeneous_h8_watchdog\0run\0--\0/tmp/heterogeneous_construct\0--mode\0h8\0",
        )
        .unwrap();
        std::os::unix::fs::symlink("/tmp/heterogeneous_h8_watchdog", watchdog.join("exe")).unwrap();

        let scan = read_h8_process_scan_from(&paths, None).unwrap();
        assert!(scan.proc_scan_complete);
        assert!(!scan.active_h8_process_found);

        let candidate = paths.proc_root.join("102");
        fs::create_dir_all(&candidate).unwrap();
        fs::write(
            candidate.join("status"),
            process_status("heterogeneous_co", 1),
        )
        .unwrap();
        fs::write(candidate.join("cmdline"), b"/proc/self/fd/9\0--mode\0h8\0").unwrap();
        fs::write(candidate.join("exe"), b"not a symlink").unwrap();

        let scan = read_h8_process_scan_from(&paths, None).unwrap();
        assert!(!scan.proc_scan_complete);
        assert!(scan.active_h8_process_found);

        fs::remove_file(candidate.join("exe")).unwrap();
        std::os::unix::fs::symlink(
            "/home/emmy/gpt-oss-rs/target/release/heterogeneous_construct",
            candidate.join("exe"),
        )
        .unwrap();
        let scan = read_h8_process_scan_from(&paths, None).unwrap();
        assert!(scan.proc_scan_complete);
        assert!(scan.active_h8_process_found);

        fs::write(
            paths.proc_root.join("meminfo"),
            "MemAvailable: 16777216 kB\nSwapTotal: 100 kB\nSwapFree: 90 kB\nSwapCached: 1 kB\n",
        )
        .unwrap();
        fs::create_dir_all(paths.proc_root.join("pressure")).unwrap();
        fs::write(
            paths.proc_root.join("pressure/memory"),
            "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
        )
        .unwrap();
        fs::create_dir_all(paths.proc_root.join("sys/vm")).unwrap();
        fs::write(paths.proc_root.join("sys/vm/swappiness"), "60\n").unwrap();
        fs::create_dir_all(paths.proc_root.join("self")).unwrap();
        fs::write(paths.proc_root.join("self/cgroup"), "0::/fixture\n").unwrap();
        fs::write(paths.proc_root.join("self/mountinfo"), "").unwrap();

        // The process-only API is portable, but production admission remains
        // fail-closed when the host-specific protected-device sysfs node is
        // absent (as it is on a generic GitHub-hosted runner).
        let error = read_host_snapshot_from(&paths, None, 0).unwrap_err();
        assert_eq!(
            error
                .downcast_ref::<std::io::Error>()
                .and_then(std::io::Error::raw_os_error),
            Some(2)
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn process_categories_do_not_retain_process_identity() {
        assert_eq!(categorize_process("codex"), SwapCategory::Codex);
        assert_eq!(
            categorize_process("dockerd"),
            SwapCategory::ContainerRuntime
        );
        assert_eq!(categorize_process("tmux"), SwapCategory::ShellOrSession);
        assert_eq!(categorize_process("private-app"), SwapCategory::Other);
    }

    #[test]
    fn child_binding_fails_closed_on_every_identity_boundary() {
        let hash = "a".repeat(64);
        let valid = || BindingValues {
            schema: WATCHDOG_SCHEMA.into(),
            run_id_sha256: hash.clone(),
            preflight_sha256: hash.clone(),
            watchdog_executable_sha256: hash.clone(),
            swap_baseline_bytes: 7,
            expected_parent: 11,
            actual_parent: 11,
            actual_parent_hash: hash.clone(),
            current_swap_used_bytes: 7,
        };
        assert!(validate_binding_values(valid()).is_ok());

        let mut wrong = valid();
        wrong.schema = "wrong".into();
        assert!(validate_binding_values(wrong).is_err());
        let mut wrong = valid();
        wrong.actual_parent = 12;
        assert!(validate_binding_values(wrong).is_err());
        let mut wrong = valid();
        wrong.actual_parent_hash = "b".repeat(64);
        assert!(validate_binding_values(wrong).is_err());
        let mut wrong = valid();
        wrong.current_swap_used_bytes = 8;
        assert!(validate_binding_values(wrong).is_err());
        let mut wrong = valid();
        wrong.preflight_sha256 = "not-a-hash".into();
        assert!(validate_binding_values(wrong).is_err());
    }
}
