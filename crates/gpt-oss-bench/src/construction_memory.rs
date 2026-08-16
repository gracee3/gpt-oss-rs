//! Bounded Linux memory evidence for owner-selective construction.
//!
//! Samples are taken synchronously at the constructor's existing stage
//! boundaries.  A create-new journal can persist each event before the next
//! allocation so a fail-closed abort still leaves the last complete sample.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use anyhow::{bail, Context, Result};
use gpt_oss_model_runner::model_loader::owner_selective::ConstructionLedger;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub use crate::construction_memory_policy::{
    CONSTRUCTION_MEMORY_EVENT_SCHEMA, MAX_CONSTRUCTION_MEMORY_EVENTS,
    MAX_CONSTRUCTION_MEMORY_EVENT_BYTES, MAX_CONSTRUCTION_MEMORY_JOURNAL_BYTES,
};

#[derive(Debug, Clone)]
pub struct ConstructionMemoryPaths {
    pub proc_root: PathBuf,
    pub cgroup_root: PathBuf,
}

impl Default for ConstructionMemoryPaths {
    fn default() -> Self {
        Self {
            proc_root: PathBuf::from("/proc"),
            cgroup_root: PathBuf::from("/sys/fs/cgroup"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConstructionMemoryIdentity {
    pub repository_head: String,
    pub executable_sha256: String,
    pub checkpoint_class: String,
    pub checkpoint_revision: String,
    pub checkpoint_metadata_sha256: String,
    pub checkpoint_mapping_sha256: String,
    pub placement_manifest_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProcessStatusMemory {
    pub vm_size_bytes: u64,
    pub vm_rss_bytes: u64,
    pub vm_hwm_bytes: u64,
    pub vm_swap_bytes: u64,
    pub rss_anon_bytes: u64,
    pub rss_file_bytes: u64,
    pub rss_shmem_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SmapsRollupMemory {
    pub rss_bytes: u64,
    pub pss_bytes: u64,
    pub pss_anon_bytes: u64,
    pub pss_file_bytes: u64,
    pub pss_shmem_bytes: u64,
    pub shared_clean_bytes: u64,
    pub shared_dirty_bytes: u64,
    pub private_clean_bytes: u64,
    pub private_dirty_bytes: u64,
    pub anonymous_bytes: u64,
    pub swap_bytes: u64,
    pub swap_pss_bytes: u64,
    pub locked_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GlobalMemoryInfo {
    pub mem_available_bytes: u64,
    pub cached_bytes: u64,
    pub swap_cached_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_free_bytes: u64,
    pub swap_used_bytes: u64,
    pub active_anon_bytes: u64,
    pub inactive_anon_bytes: u64,
    pub active_file_bytes: u64,
    pub inactive_file_bytes: u64,
    pub shmem_bytes: u64,
    pub sreclaimable_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VmstatSwapCounters {
    /// Cumulative pages read from swap since boot.
    pub pswpin_pages: u64,
    /// Cumulative pages written to swap since boot.
    pub pswpout_pages: u64,
    pub nr_swapcached_pages: u64,
    pub nr_anon_pages: u64,
    pub nr_file_pages: u64,
    pub nr_shmem_pages: u64,
    pub major_faults: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CgroupMemoryStat {
    pub anon_bytes: u64,
    pub file_bytes: u64,
    pub kernel_bytes: u64,
    pub pagetables_bytes: u64,
    pub shmem_bytes: u64,
    pub file_mapped_bytes: u64,
    pub file_dirty_bytes: u64,
    pub file_writeback_bytes: u64,
    pub swapcached_bytes: u64,
    pub inactive_anon_bytes: u64,
    pub active_anon_bytes: u64,
    pub inactive_file_bytes: u64,
    pub active_file_bytes: u64,
    pub unevictable_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CgroupMemoryEvents {
    pub high: u64,
    pub max: u64,
    pub oom: u64,
    pub oom_kill: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CurrentCgroupMemory {
    /// Hash of the unified cgroup path; the path itself may contain a session
    /// identifier and is intentionally not retained in publishable evidence.
    pub relative_path_sha256: String,
    pub memory_current_bytes: u64,
    pub memory_swap_current_bytes: u64,
    pub memory_peak_bytes: Option<u64>,
    pub stat: CgroupMemoryStat,
    pub events: CgroupMemoryEvents,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileAnonResidencyEvidence {
    pub self_status_rss_components_bytes: u64,
    pub self_status_rss_unclassified_bytes: u64,
    pub self_pss_components_bytes: u64,
    pub global_page_cache_estimate_bytes: u64,
    pub global_file_lru_bytes: u64,
    pub global_anon_lru_bytes: u64,
    pub cgroup_file_lru_bytes: u64,
    pub cgroup_anon_lru_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConstructionMemorySample {
    pub process_status: ProcessStatusMemory,
    pub smaps_rollup: SmapsRollupMemory,
    pub global_meminfo: GlobalMemoryInfo,
    pub vmstat: VmstatSwapCounters,
    pub current_cgroup: CurrentCgroupMemory,
    pub residency: FileAnonResidencyEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuResidencySample {
    pub pci_bus_id: String,
    pub used_mib: u64,
    pub free_mib: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConstructionMemoryEventPhase {
    BeforeCheckpointOpen,
    AfterCheckpointOpen,
    Stage,
    PostDrop,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConstructionMemoryEvent {
    pub schema: String,
    pub sequence: u32,
    pub captured_unix_ms: u128,
    pub run_label: String,
    pub phase: ConstructionMemoryEventPhase,
    pub elapsed_ms: u128,
    pub identity: ConstructionMemoryIdentity,
    pub checkpoint_mapped_address_bytes: Option<u64>,
    pub ledger: Option<ConstructionLedger>,
    pub memory: ConstructionMemorySample,
    pub gpus: Vec<GpuResidencySample>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConstructionMemoryJournalSummary {
    pub schema: String,
    pub persisted: bool,
    pub event_count: u32,
    pub encoded_event_bytes: u64,
    pub persisted_bytes: u64,
    pub max_events: u32,
    pub max_event_bytes: u64,
    pub max_total_bytes: u64,
    pub entries: Vec<ConstructionMemoryJournalEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConstructionMemoryJournalEntry {
    pub sequence: u32,
    pub filename: String,
    pub sha256: String,
    pub bytes: u64,
}

pub struct ConstructionMemoryRecorder {
    identity: ConstructionMemoryIdentity,
    paths: ConstructionMemoryPaths,
    journal_root: Option<PathBuf>,
    event_count: usize,
    encoded_event_bytes: usize,
    persisted_bytes: usize,
    entries: Vec<ConstructionMemoryJournalEntry>,
    max_events: usize,
    max_event_bytes: usize,
    max_total_bytes: usize,
}

impl ConstructionMemorySample {
    pub fn sample_self() -> Result<Self> {
        Self::sample_from(&ConstructionMemoryPaths::default())
    }

    pub fn sample_from(paths: &ConstructionMemoryPaths) -> Result<Self> {
        let status = parse_kib_fields(&fs::read_to_string(paths.proc_root.join("self/status"))?)?;
        let rollup = parse_kib_fields(&fs::read_to_string(
            paths.proc_root.join("self/smaps_rollup"),
        )?)?;
        let meminfo = parse_kib_fields(&fs::read_to_string(paths.proc_root.join("meminfo"))?)?;
        let vmstat = parse_numeric_fields(&fs::read_to_string(paths.proc_root.join("vmstat"))?)?;

        let process_status = ProcessStatusMemory {
            vm_size_bytes: required(&status, "VmSize")?,
            vm_rss_bytes: required(&status, "VmRSS")?,
            vm_hwm_bytes: required(&status, "VmHWM")?,
            vm_swap_bytes: required(&status, "VmSwap")?,
            rss_anon_bytes: required(&status, "RssAnon")?,
            rss_file_bytes: required(&status, "RssFile")?,
            rss_shmem_bytes: required(&status, "RssShmem")?,
        };
        let smaps_rollup = SmapsRollupMemory {
            rss_bytes: required(&rollup, "Rss")?,
            pss_bytes: required(&rollup, "Pss")?,
            pss_anon_bytes: required(&rollup, "Pss_Anon")?,
            pss_file_bytes: required(&rollup, "Pss_File")?,
            pss_shmem_bytes: required(&rollup, "Pss_Shmem")?,
            shared_clean_bytes: required(&rollup, "Shared_Clean")?,
            shared_dirty_bytes: required(&rollup, "Shared_Dirty")?,
            private_clean_bytes: required(&rollup, "Private_Clean")?,
            private_dirty_bytes: required(&rollup, "Private_Dirty")?,
            anonymous_bytes: required(&rollup, "Anonymous")?,
            swap_bytes: required(&rollup, "Swap")?,
            swap_pss_bytes: required(&rollup, "SwapPss")?,
            locked_bytes: required(&rollup, "Locked")?,
        };
        let swap_total_bytes = required(&meminfo, "SwapTotal")?;
        let swap_free_bytes = required(&meminfo, "SwapFree")?;
        let global_meminfo = GlobalMemoryInfo {
            mem_available_bytes: required(&meminfo, "MemAvailable")?,
            cached_bytes: required(&meminfo, "Cached")?,
            swap_cached_bytes: required(&meminfo, "SwapCached")?,
            swap_total_bytes,
            swap_free_bytes,
            swap_used_bytes: swap_total_bytes
                .checked_sub(swap_free_bytes)
                .context("global SwapFree exceeds SwapTotal while sampling construction memory")?,
            active_anon_bytes: required(&meminfo, "Active(anon)")?,
            inactive_anon_bytes: required(&meminfo, "Inactive(anon)")?,
            active_file_bytes: required(&meminfo, "Active(file)")?,
            inactive_file_bytes: required(&meminfo, "Inactive(file)")?,
            shmem_bytes: required(&meminfo, "Shmem")?,
            sreclaimable_bytes: required(&meminfo, "SReclaimable")?,
        };
        let vmstat = VmstatSwapCounters {
            pswpin_pages: required(&vmstat, "pswpin")?,
            pswpout_pages: required(&vmstat, "pswpout")?,
            nr_swapcached_pages: required(&vmstat, "nr_swapcached")?,
            nr_anon_pages: required(&vmstat, "nr_anon_pages")?,
            nr_file_pages: required(&vmstat, "nr_file_pages")?,
            nr_shmem_pages: required(&vmstat, "nr_shmem")?,
            major_faults: required(&vmstat, "pgmajfault")?,
        };
        let current_cgroup = read_current_cgroup(paths)?;
        let residency = derive_residency(
            &process_status,
            &smaps_rollup,
            &global_meminfo,
            &current_cgroup,
        )?;
        Ok(Self {
            process_status,
            smaps_rollup,
            global_meminfo,
            vmstat,
            current_cgroup,
            residency,
        })
    }
}

impl ConstructionMemoryRecorder {
    pub fn new(identity: ConstructionMemoryIdentity, journal_root: Option<&Path>) -> Result<Self> {
        Self::new_with_limits(
            identity,
            ConstructionMemoryPaths::default(),
            journal_root,
            MAX_CONSTRUCTION_MEMORY_EVENTS,
            MAX_CONSTRUCTION_MEMORY_EVENT_BYTES,
            MAX_CONSTRUCTION_MEMORY_JOURNAL_BYTES,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_limits(
        identity: ConstructionMemoryIdentity,
        paths: ConstructionMemoryPaths,
        journal_root: Option<&Path>,
        max_events: usize,
        max_event_bytes: usize,
        max_total_bytes: usize,
    ) -> Result<Self> {
        if max_events == 0 || max_event_bytes == 0 || max_total_bytes < max_event_bytes {
            bail!("construction memory journal limits are invalid");
        }
        let journal_root = match journal_root {
            Some(root) => {
                let parent = root
                    .parent()
                    .context("construction memory journal root has no parent")?;
                if !parent.is_dir() {
                    bail!("construction memory journal parent is not a directory");
                }
                fs::create_dir(root).with_context(|| {
                    format!(
                        "construction memory journal must be a new directory: {}",
                        root.display()
                    )
                })?;
                Some(root.to_owned())
            }
            None => None,
        };
        Ok(Self {
            identity,
            paths,
            journal_root,
            event_count: 0,
            encoded_event_bytes: 0,
            persisted_bytes: 0,
            entries: Vec::with_capacity(max_events),
            max_events,
            max_event_bytes,
            max_total_bytes,
        })
    }

    pub fn capture(
        &mut self,
        run_label: &str,
        phase: ConstructionMemoryEventPhase,
        elapsed_ms: u128,
        ledger: ConstructionLedger,
        gpus: Vec<GpuResidencySample>,
    ) -> Result<ConstructionMemoryEvent> {
        self.capture_inner(run_label, phase, elapsed_ms, None, Some(ledger), gpus)
    }

    pub fn capture_checkpoint_boundary(
        &mut self,
        run_label: &str,
        phase: ConstructionMemoryEventPhase,
        elapsed_ms: u128,
        checkpoint_mapped_address_bytes: Option<u64>,
        gpus: Vec<GpuResidencySample>,
    ) -> Result<ConstructionMemoryEvent> {
        if !matches!(
            phase,
            ConstructionMemoryEventPhase::BeforeCheckpointOpen
                | ConstructionMemoryEventPhase::AfterCheckpointOpen
        ) {
            bail!("checkpoint boundary capture requires a checkpoint-open phase");
        }
        if matches!(phase, ConstructionMemoryEventPhase::BeforeCheckpointOpen)
            && checkpoint_mapped_address_bytes.is_some()
        {
            bail!("before-open checkpoint event cannot claim mapped bytes");
        }
        if matches!(phase, ConstructionMemoryEventPhase::AfterCheckpointOpen)
            && checkpoint_mapped_address_bytes.is_none()
        {
            bail!("after-open checkpoint event requires mapped bytes");
        }
        self.capture_inner(
            run_label,
            phase,
            elapsed_ms,
            checkpoint_mapped_address_bytes,
            None,
            gpus,
        )
    }

    fn capture_inner(
        &mut self,
        run_label: &str,
        phase: ConstructionMemoryEventPhase,
        elapsed_ms: u128,
        checkpoint_mapped_address_bytes: Option<u64>,
        ledger: Option<ConstructionLedger>,
        gpus: Vec<GpuResidencySample>,
    ) -> Result<ConstructionMemoryEvent> {
        validate_run_label(run_label)?;
        if matches!(
            phase,
            ConstructionMemoryEventPhase::Stage | ConstructionMemoryEventPhase::PostDrop
        ) != ledger.is_some()
        {
            bail!("construction stage event and ledger presence differ");
        }
        if self.event_count >= self.max_events {
            bail!("construction memory event count exceeds the hard bound");
        }
        let sequence = u32::try_from(self.event_count).context("event sequence exceeds u32")?;
        let event = ConstructionMemoryEvent {
            schema: CONSTRUCTION_MEMORY_EVENT_SCHEMA.into(),
            sequence,
            captured_unix_ms: now_unix_ms()?,
            run_label: run_label.into(),
            phase,
            elapsed_ms,
            identity: self.identity.clone(),
            checkpoint_mapped_address_bytes,
            ledger,
            memory: ConstructionMemorySample::sample_from(&self.paths)?,
            gpus,
        };
        let bytes = event_bytes(&event)?;
        if bytes.len() > self.max_event_bytes {
            bail!("construction memory event exceeds the per-event byte bound");
        }
        let next_total = self
            .encoded_event_bytes
            .checked_add(bytes.len())
            .context("construction memory journal byte count overflows")?;
        if next_total > self.max_total_bytes {
            bail!("construction memory journal exceeds the total byte bound");
        }
        if let Some(root) = &self.journal_root {
            let boundary = event_boundary_name(&event);
            let entry = write_event_new(root, sequence, run_label, &boundary, &bytes)?;
            self.persisted_bytes = self
                .persisted_bytes
                .checked_add(bytes.len())
                .context("persisted construction memory byte count overflows")?;
            self.entries.push(entry);
        }
        self.encoded_event_bytes = next_total;
        self.event_count += 1;
        Ok(event)
    }

    pub fn summary(&self) -> Result<ConstructionMemoryJournalSummary> {
        Ok(ConstructionMemoryJournalSummary {
            schema: CONSTRUCTION_MEMORY_EVENT_SCHEMA.into(),
            persisted: self.journal_root.is_some(),
            event_count: u32::try_from(self.event_count).context("event count exceeds u32")?,
            encoded_event_bytes: u64::try_from(self.encoded_event_bytes)
                .context("encoded event bytes exceed u64")?,
            persisted_bytes: u64::try_from(self.persisted_bytes)
                .context("persisted event bytes exceed u64")?,
            max_events: u32::try_from(self.max_events).context("event bound exceeds u32")?,
            max_event_bytes: u64::try_from(self.max_event_bytes)
                .context("event byte bound exceeds u64")?,
            max_total_bytes: u64::try_from(self.max_total_bytes)
                .context("journal byte bound exceeds u64")?,
            entries: self.entries.clone(),
        })
    }
}

fn read_current_cgroup(paths: &ConstructionMemoryPaths) -> Result<CurrentCgroupMemory> {
    let cgroup_text = fs::read_to_string(paths.proc_root.join("self/cgroup"))?;
    let relative = cgroup_text
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .context("unified cgroup path is missing")?
        .trim_start_matches('/');
    let relative_path = Path::new(relative);
    if relative_path
        .components()
        .any(|component| matches!(component, Component::ParentDir | Component::RootDir))
    {
        bail!("unified cgroup path escapes the cgroup root");
    }
    let root = paths.cgroup_root.join(relative_path);
    let memory_current_bytes = read_u64(&root.join("memory.current"))?;
    let memory_swap_current_bytes = read_u64(&root.join("memory.swap.current"))?;
    let memory_peak_bytes = read_optional_u64(&root.join("memory.peak"))?;
    let stat = parse_numeric_fields(&fs::read_to_string(root.join("memory.stat"))?)?;
    let events = parse_numeric_fields(&fs::read_to_string(root.join("memory.events"))?)?;
    Ok(CurrentCgroupMemory {
        relative_path_sha256: format!("{:x}", Sha256::digest(relative.as_bytes())),
        memory_current_bytes,
        memory_swap_current_bytes,
        memory_peak_bytes,
        stat: CgroupMemoryStat {
            anon_bytes: required(&stat, "anon")?,
            file_bytes: required(&stat, "file")?,
            kernel_bytes: required(&stat, "kernel")?,
            pagetables_bytes: required(&stat, "pagetables")?,
            shmem_bytes: required(&stat, "shmem")?,
            file_mapped_bytes: required(&stat, "file_mapped")?,
            file_dirty_bytes: required(&stat, "file_dirty")?,
            file_writeback_bytes: required(&stat, "file_writeback")?,
            swapcached_bytes: required(&stat, "swapcached")?,
            inactive_anon_bytes: required(&stat, "inactive_anon")?,
            active_anon_bytes: required(&stat, "active_anon")?,
            inactive_file_bytes: required(&stat, "inactive_file")?,
            active_file_bytes: required(&stat, "active_file")?,
            unevictable_bytes: required(&stat, "unevictable")?,
        },
        events: CgroupMemoryEvents {
            high: required(&events, "high")?,
            max: required(&events, "max")?,
            oom: required(&events, "oom")?,
            oom_kill: required(&events, "oom_kill")?,
        },
    })
}

fn derive_residency(
    status: &ProcessStatusMemory,
    rollup: &SmapsRollupMemory,
    global: &GlobalMemoryInfo,
    cgroup: &CurrentCgroupMemory,
) -> Result<FileAnonResidencyEvidence> {
    let self_status_rss_components_bytes = checked_sum(&[
        status.rss_anon_bytes,
        status.rss_file_bytes,
        status.rss_shmem_bytes,
    ])?;
    let self_pss_components_bytes = checked_sum(&[
        rollup.pss_anon_bytes,
        rollup.pss_file_bytes,
        rollup.pss_shmem_bytes,
    ])?;
    let global_page_cache_estimate_bytes = global
        .cached_bytes
        .checked_add(global.sreclaimable_bytes)
        .and_then(|bytes| bytes.checked_sub(global.shmem_bytes))
        .context("global page-cache estimate overflows or underflows")?;
    Ok(FileAnonResidencyEvidence {
        self_status_rss_components_bytes,
        self_status_rss_unclassified_bytes: status
            .vm_rss_bytes
            .saturating_sub(self_status_rss_components_bytes),
        self_pss_components_bytes,
        global_page_cache_estimate_bytes,
        global_file_lru_bytes: checked_sum(&[
            global.active_file_bytes,
            global.inactive_file_bytes,
        ])?,
        global_anon_lru_bytes: checked_sum(&[
            global.active_anon_bytes,
            global.inactive_anon_bytes,
        ])?,
        cgroup_file_lru_bytes: checked_sum(&[
            cgroup.stat.active_file_bytes,
            cgroup.stat.inactive_file_bytes,
        ])?,
        cgroup_anon_lru_bytes: checked_sum(&[
            cgroup.stat.active_anon_bytes,
            cgroup.stat.inactive_anon_bytes,
        ])?,
    })
}

fn parse_kib_fields(text: &str) -> Result<BTreeMap<String, u64>> {
    let mut result = BTreeMap::new();
    for line in text.lines() {
        let Some((name, values)) = line.split_once(':') else {
            continue;
        };
        let mut values = values.split_whitespace();
        let Some(number) = values.next() else {
            continue;
        };
        let Ok(number) = number.parse::<u64>() else {
            continue;
        };
        if values.next() != Some("kB") {
            continue;
        }
        result.insert(
            name.into(),
            number
                .checked_mul(1024)
                .context("kB field overflows bytes")?,
        );
    }
    Ok(result)
}

fn parse_numeric_fields(text: &str) -> Result<BTreeMap<String, u64>> {
    let mut result = BTreeMap::new();
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let Some(name) = fields.next() else {
            continue;
        };
        let Some(value) = fields.next() else {
            continue;
        };
        let value = value
            .parse::<u64>()
            .with_context(|| format!("invalid numeric field {name}"))?;
        result.insert(name.into(), value);
    }
    Ok(result)
}

fn required(values: &BTreeMap<String, u64>, name: &str) -> Result<u64> {
    values
        .get(name)
        .copied()
        .with_context(|| format!("required memory field {name} is missing"))
}

fn checked_sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(*value)
            .context("memory residency sum overflows")
    })
}

fn read_u64(path: &Path) -> Result<u64> {
    fs::read_to_string(path)?
        .trim()
        .parse()
        .with_context(|| format!("invalid integer in {}", path.display()))
}

fn read_optional_u64(path: &Path) -> Result<Option<u64>> {
    match fs::read_to_string(path) {
        Ok(value) => {
            Ok(Some(value.trim().parse().with_context(|| {
                format!("invalid integer in {}", path.display())
            })?))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn now_unix_ms() -> Result<u128> {
    Ok(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system clock precedes the Unix epoch")?
        .as_millis())
}

fn validate_run_label(label: &str) -> Result<()> {
    if label.is_empty()
        || label.len() > 32
        || !label
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    {
        bail!("construction memory run label is not bounded lowercase ASCII");
    }
    Ok(())
}

fn event_bytes(event: &ConstructionMemoryEvent) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(event)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn event_boundary_name(event: &ConstructionMemoryEvent) -> String {
    event.ledger.as_ref().map_or_else(
        || {
            match event.phase {
                ConstructionMemoryEventPhase::BeforeCheckpointOpen => "before_checkpoint_open",
                ConstructionMemoryEventPhase::AfterCheckpointOpen => "after_checkpoint_open",
                ConstructionMemoryEventPhase::Stage => "stage_without_ledger",
                ConstructionMemoryEventPhase::PostDrop => "post_drop_without_ledger",
            }
            .into()
        },
        |ledger| format!("{:?}", ledger.stage).to_ascii_lowercase(),
    )
}

fn write_event_new(
    root: &Path,
    sequence: u32,
    run_label: &str,
    boundary: &str,
    bytes: &[u8],
) -> Result<ConstructionMemoryJournalEntry> {
    let filename = format!("{sequence:03}-{run_label}-{boundary}.json");
    let final_path = root.join(&filename);
    let temporary_path = root.join(format!(".{sequence:03}-{run_label}-{boundary}.tmp"));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        // A same-filesystem hard link gives create-new publication semantics:
        // unlike rename, it cannot replace a concurrently-created final name.
        fs::hard_link(&temporary_path, &final_path)?;
        fs::remove_file(&temporary_path)?;
        File::open(root)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }
    result?;
    Ok(ConstructionMemoryJournalEntry {
        sequence,
        filename,
        sha256: format!("{:x}", Sha256::digest(bytes)),
        bytes: u64::try_from(bytes.len()).context("event length exceeds u64")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpt_oss_model_runner::model_loader::owner_selective::ConstructionStage;

    fn unique_root(label: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "gpt-oss-construction-memory-{label}-{}-{unique}",
            std::process::id()
        ))
    }

    fn identity() -> ConstructionMemoryIdentity {
        ConstructionMemoryIdentity {
            repository_head: "a".repeat(40),
            executable_sha256: "b".repeat(64),
            checkpoint_class: "fixture".into(),
            checkpoint_revision: "revision".into(),
            checkpoint_metadata_sha256: "c".repeat(64),
            checkpoint_mapping_sha256: "d".repeat(64),
            placement_manifest_sha256: "e".repeat(64),
        }
    }

    fn fixture() -> (PathBuf, ConstructionMemoryPaths) {
        let root = unique_root("fixture");
        let proc_root = root.join("proc");
        let cgroup_root = root.join("cgroup");
        fs::create_dir_all(proc_root.join("self")).unwrap();
        fs::create_dir_all(cgroup_root.join("scope")).unwrap();
        fs::write(
            proc_root.join("self/status"),
            "VmSize: 100 kB\nVmRSS: 80 kB\nVmHWM: 90 kB\nVmSwap: 0 kB\nRssAnon: 50 kB\nRssFile: 25 kB\nRssShmem: 5 kB\n",
        )
        .unwrap();
        fs::write(
            proc_root.join("self/smaps_rollup"),
            "rollup\nRss: 80 kB\nPss: 70 kB\nPss_Anon: 45 kB\nPss_File: 20 kB\nPss_Shmem: 5 kB\nShared_Clean: 10 kB\nShared_Dirty: 1 kB\nPrivate_Clean: 20 kB\nPrivate_Dirty: 49 kB\nAnonymous: 50 kB\nSwap: 0 kB\nSwapPss: 0 kB\nLocked: 0 kB\n",
        )
        .unwrap();
        fs::write(
            proc_root.join("meminfo"),
            "MemAvailable: 1000 kB\nCached: 500 kB\nSwapCached: 2 kB\nSwapTotal: 100 kB\nSwapFree: 90 kB\nActive(anon): 40 kB\nInactive(anon): 30 kB\nActive(file): 200 kB\nInactive(file): 250 kB\nShmem: 10 kB\nSReclaimable: 20 kB\n",
        )
        .unwrap();
        fs::write(
            proc_root.join("vmstat"),
            "pswpin 1\npswpout 2\nnr_swapcached 3\nnr_anon_pages 4\nnr_file_pages 5\nnr_shmem 6\npgmajfault 7\n",
        )
        .unwrap();
        fs::write(proc_root.join("self/cgroup"), "0::/scope\n").unwrap();
        fs::write(cgroup_root.join("scope/memory.current"), "10000\n").unwrap();
        fs::write(cgroup_root.join("scope/memory.swap.current"), "0\n").unwrap();
        fs::write(cgroup_root.join("scope/memory.peak"), "12000\n").unwrap();
        fs::write(
            cgroup_root.join("scope/memory.stat"),
            "anon 100\nfile 200\nkernel 30\npagetables 4\nshmem 5\nfile_mapped 6\nfile_dirty 7\nfile_writeback 8\nswapcached 9\ninactive_anon 10\nactive_anon 11\ninactive_file 12\nactive_file 13\nunevictable 14\n",
        )
        .unwrap();
        fs::write(
            cgroup_root.join("scope/memory.events"),
            "low 0\nhigh 1\nmax 2\noom 3\noom_kill 4\n",
        )
        .unwrap();
        (
            root,
            ConstructionMemoryPaths {
                proc_root,
                cgroup_root,
            },
        )
    }

    #[test]
    fn fixture_samples_all_required_residency_domains() {
        let (root, paths) = fixture();
        let sample = ConstructionMemorySample::sample_from(&paths).unwrap();
        assert_eq!(sample.process_status.vm_rss_bytes, 80 * 1024);
        assert_eq!(sample.smaps_rollup.pss_file_bytes, 20 * 1024);
        assert_eq!(sample.global_meminfo.swap_used_bytes, 10 * 1024);
        assert_eq!(sample.vmstat.pswpout_pages, 2);
        assert_eq!(sample.current_cgroup.memory_current_bytes, 10_000);
        assert_eq!(sample.current_cgroup.stat.file_bytes, 200);
        assert_eq!(
            sample.residency.global_page_cache_estimate_bytes,
            510 * 1024
        );
        assert_eq!(sample.residency.cgroup_file_lru_bytes, 25);
        assert!(!sample.current_cgroup.relative_path_sha256.contains("scope"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_required_field_and_cgroup_escape_fail_closed() {
        let (root, paths) = fixture();
        fs::write(paths.proc_root.join("vmstat"), "pswpin 1\n").unwrap();
        assert!(ConstructionMemorySample::sample_from(&paths).is_err());
        let (_, complete_paths) = fixture();
        fs::write(
            complete_paths.proc_root.join("self/cgroup"),
            "0::/../escape\n",
        )
        .unwrap();
        assert!(ConstructionMemorySample::sample_from(&complete_paths).is_err());
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(complete_paths.proc_root.parent().unwrap()).unwrap();
    }

    #[test]
    fn journal_is_create_new_atomic_and_hard_bounded() {
        let (root, paths) = fixture();
        let journal = root.join("events");
        let mut recorder = ConstructionMemoryRecorder::new_with_limits(
            identity(),
            paths.clone(),
            Some(&journal),
            1,
            MAX_CONSTRUCTION_MEMORY_EVENT_BYTES,
            MAX_CONSTRUCTION_MEMORY_EVENT_BYTES,
        )
        .unwrap();
        let event = recorder
            .capture(
                "cold",
                ConstructionMemoryEventPhase::Stage,
                1,
                ConstructionLedger {
                    stage: ConstructionStage::Identity,
                    mapped_address_bytes: 0,
                    layer_owner_dense_bytes: 0,
                    layer_owner_expert_bytes: 0,
                    remote_gpu_expert_bytes: 0,
                    cpu_x8_bytes: 0,
                    pinned_bytes: 0,
                    construction_temporary_high_water_bytes: 0,
                    layer_owner_experts: 0,
                    remote_gpu_experts: 0,
                    cpu_experts: 0,
                    execution_reserve_reviewed: false,
                    execution_runtime_resources_materialized_at_construction: false,
                    layer_owner_execution_materialized_before_admission_bytes: 0,
                    remote_gpu_execution_materialized_before_admission_bytes: 0,
                    layer_owner_execution_planned_bytes: 0,
                    remote_gpu_execution_planned_bytes: 0,
                },
                Vec::new(),
            )
            .unwrap();
        assert_eq!(event.sequence, 0);
        assert_eq!(fs::read_dir(&journal).unwrap().count(), 1);
        assert!(recorder
            .capture(
                "cold",
                ConstructionMemoryEventPhase::Stage,
                2,
                event.ledger.clone().unwrap(),
                Vec::new(),
            )
            .is_err());
        assert!(ConstructionMemoryRecorder::new(identity(), Some(&journal)).is_err());
        let summary = recorder.summary().unwrap();
        assert_eq!(summary.event_count, 1);
        assert!(summary.persisted);
        assert!(summary.persisted_bytes > 0);
        assert_eq!(summary.persisted_bytes, summary.encoded_event_bytes);
        assert_eq!(summary.entries.len(), 1);
        assert_eq!(summary.entries[0].sha256.len(), 64);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn disabled_journal_still_enforces_encoded_byte_bound() {
        let (root, paths) = fixture();
        let mut recorder =
            ConstructionMemoryRecorder::new_with_limits(identity(), paths, None, 2, 1, 2).unwrap();
        let result = recorder.capture(
            "cold",
            ConstructionMemoryEventPhase::Stage,
            1,
            ConstructionLedger {
                stage: ConstructionStage::Identity,
                mapped_address_bytes: 0,
                layer_owner_dense_bytes: 0,
                layer_owner_expert_bytes: 0,
                remote_gpu_expert_bytes: 0,
                cpu_x8_bytes: 0,
                pinned_bytes: 0,
                construction_temporary_high_water_bytes: 0,
                layer_owner_experts: 0,
                remote_gpu_experts: 0,
                cpu_experts: 0,
                execution_reserve_reviewed: false,
                execution_runtime_resources_materialized_at_construction: false,
                layer_owner_execution_materialized_before_admission_bytes: 0,
                remote_gpu_execution_materialized_before_admission_bytes: 0,
                layer_owner_execution_planned_bytes: 0,
                remote_gpu_execution_planned_bytes: 0,
            },
            Vec::new(),
        );
        assert!(result.is_err());
        assert_eq!(recorder.summary().unwrap().event_count, 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn checkpoint_boundaries_require_truthful_mapping_state() {
        let (root, paths) = fixture();
        let mut recorder = ConstructionMemoryRecorder::new_with_limits(
            identity(),
            paths,
            None,
            2,
            MAX_CONSTRUCTION_MEMORY_EVENT_BYTES,
            MAX_CONSTRUCTION_MEMORY_JOURNAL_BYTES,
        )
        .unwrap();
        assert!(recorder
            .capture_checkpoint_boundary(
                "cold",
                ConstructionMemoryEventPhase::BeforeCheckpointOpen,
                0,
                Some(1),
                Vec::new(),
            )
            .is_err());
        let event = recorder
            .capture_checkpoint_boundary(
                "cold",
                ConstructionMemoryEventPhase::AfterCheckpointOpen,
                1,
                Some(123),
                Vec::new(),
            )
            .unwrap();
        assert_eq!(event.checkpoint_mapped_address_bytes, Some(123));
        assert!(event.ledger.is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn live_self_sample_has_consistent_required_fields() {
        let sample = ConstructionMemorySample::sample_self().unwrap();
        assert!(sample.process_status.vm_size_bytes >= sample.process_status.vm_rss_bytes);
        assert!(sample.global_meminfo.swap_total_bytes >= sample.global_meminfo.swap_free_bytes);
        assert_eq!(sample.current_cgroup.relative_path_sha256.len(), 64);
    }
}
