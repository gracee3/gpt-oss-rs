//! Bounded, metadata-only SafeTensors shard catalog.
//!
//! Catalog construction reads the optional index and the eight-byte length plus
//! JSON header of each shard. It never maps or reads tensor payload bytes. A
//! separate callback-scoped API maps exactly one validated shard at a time for
//! the explicit capacity-one construction mode.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use memmap2::Mmap;
#[cfg(feature = "cuda")]
use memmap2::UncheckedAdvice as MmapUncheckedAdvice;
use serde::de::{Error as DeError, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

use gpt_oss_core::error::{LLMError, Result};

use crate::cpu_tensor_store::map_cataloged_immutable_shard;

use super::dtype::DType;

pub const SAFETENSORS_CATALOG_SCHEMA_V1: &str = "gpt-oss-rs.safetensors-shard-catalog/v1";
pub const MAX_SAFETENSORS_INDEX_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_SAFETENSORS_HEADER_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_SAFETENSORS_TOTAL_HEADER_BYTES: u64 = 128 * 1024 * 1024;
pub const MAX_SAFETENSORS_SHARDS: usize = 4_096;
pub const MAX_SAFETENSORS_TENSORS: usize = 1_000_000;
pub const MAX_SAFETENSORS_TENSOR_NAME_BYTES: usize = 4_096;
pub const MAX_SAFETENSORS_TENSOR_RANK: usize = 64;
pub const MAX_SHARD_RELEASE_REPORTS: usize = MAX_SAFETENSORS_SHARDS;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ProcessSourceMemorySample {
    pub rss_bytes: u64,
    pub pss_bytes: u64,
    pub pss_file_bytes: u64,
    pub vm_hwm_bytes: u64,
    pub vm_swap_bytes: u64,
    pub source_inode_mapping_count: usize,
    pub source_inode_pss_bytes: u64,
    pub mem_available_bytes: u64,
    pub global_swap_used_bytes: u64,
    pub global_swap_cached_bytes: u64,
    pub cgroup_memory_current_bytes: u64,
    pub cgroup_memory_file_bytes: u64,
    pub cgroup_memory_anon_bytes: u64,
    pub cgroup_swap_current_bytes: u64,
    /// `memory.pressure` `some avg10`, scaled by one million.
    pub memory_psi_some_avg10_micros: u64,
    /// `memory.pressure` `full avg10`, scaled by one million.
    pub memory_psi_full_avg10_micros: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdviceTelemetry {
    pub kind: String,
    pub offset: u64,
    pub length: u64,
    pub succeeded: bool,
    pub errno: Option<i32>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ShardReleaseLogicalLedger {
    pub partial_store_current_count: usize,
    pub partial_store_current_bytes: u64,
    pub partial_store_high_water_count: usize,
    pub partial_store_high_water_bytes: u64,
    pub pinned_construction_bytes: u64,
    pub anonymous_temporary_high_water_bytes: u64,
    pub output_logical_bytes: u64,
    pub device_destination_logical_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ShardReleaseTelemetry {
    pub shard_index: usize,
    pub source_file_name: String,
    pub source_device: u64,
    pub source_inode: u64,
    pub source_file_bytes: u64,
    pub mapping_address: usize,
    pub mapping_bytes: u64,
    pub covered_action_count: usize,
    pub covered_native_bytes: u64,
    pub terminal_receipts: usize,
    pub terminal_action_ids_sha256: Vec<String>,
    pub logical_ledger: ShardReleaseLogicalLedger,
    pub terminal_audit_complete: bool,
    pub pre_release: ProcessSourceMemorySample,
    pub post_release: ProcessSourceMemorySample,
    pub mmap_advice: AdviceTelemetry,
    pub file_advice: AdviceTelemetry,
    pub unmap_close_duration_micros: u128,
    pub mapping_removed: bool,
    pub fd_closed: bool,
    pub cleanup_outcome: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SafeTensorFileIdentity {
    pub file_name: String,
    pub file_length: u64,
    pub header_sha256: String,
    pub data_start: u64,
    pub payload_length: u64,
    /// Process-local device guard. Deliberately excluded from catalog identity.
    #[serde(skip)]
    pub device: u64,
    /// Process-local inode guard. Deliberately excluded from catalog identity.
    #[serde(skip)]
    pub inode: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SafeTensorDescriptor {
    pub name: String,
    pub shard_index: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Byte range relative to the SafeTensors data section.
    pub data_offsets: [u64; 2],
    /// Checked byte range relative to the beginning of the shard file.
    pub absolute_range: [u64; 2],
}

impl SafeTensorDescriptor {
    pub fn byte_len(&self) -> u64 {
        self.absolute_range[1] - self.absolute_range[0]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SafeTensorShardDescriptor {
    pub identity: SafeTensorFileIdentity,
    pub tensor_count: usize,
    #[serde(skip)]
    path: PathBuf,
}

impl SafeTensorShardDescriptor {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ShardMappingActivity {
    pub current: usize,
    pub high_water: usize,
    pub current_mapped_bytes: u64,
    pub mapped_byte_high_water: u64,
    pub quarantined: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MappingLifecycle {
    Idle,
    Active,
    Quarantined,
}

struct MappingActivityState {
    lifecycle: MappingLifecycle,
    high_water: usize,
    current_mapped_bytes: u64,
    mapped_byte_high_water: u64,
}

/// Immutable metadata catalog. Payload bytes are not part of its identity.
pub struct SafeTensorShardCatalog {
    root: PathBuf,
    shards: Vec<SafeTensorShardDescriptor>,
    tensors: BTreeMap<String, SafeTensorDescriptor>,
    metadata_sha256: String,
    index_sha256: Option<String>,
    index_declared_payload_bytes: Option<u64>,
    total_file_bytes: u64,
    total_payload_bytes: u64,
    total_header_bytes_read: u64,
    mapping_activity: Mutex<MappingActivityState>,
    release_reports: Mutex<Vec<ShardReleaseTelemetry>>,
    release_report_overflow: Mutex<bool>,
}

impl SafeTensorShardCatalog {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        validate_real_directory(root)?;
        let discovered = discover_shards(root)?;
        let index = read_optional_index(root)?;
        validate_indexed_shard_set(&discovered, index.shards.as_ref())?;

        let mut shards = Vec::with_capacity(discovered.len());
        let mut tensors = BTreeMap::new();
        let mut total_file_bytes = 0_u64;
        let mut total_payload_bytes = 0_u64;
        let mut total_header_bytes_read = 0_u64;
        for (shard_index, (file_name, path)) in discovered.into_iter().enumerate() {
            let (shard, parsed) = parse_shard(path, file_name, shard_index)?;
            total_file_bytes = total_file_bytes
                .checked_add(shard.identity.file_length)
                .ok_or_else(|| model_error("catalog file-byte total overflows"))?;
            total_payload_bytes = total_payload_bytes
                .checked_add(shard.identity.payload_length)
                .ok_or_else(|| model_error("catalog payload-byte total overflows"))?;
            total_header_bytes_read = total_header_bytes_read
                .checked_add(shard.identity.data_start)
                .ok_or_else(|| model_error("catalog header-byte total overflows"))?;
            if total_header_bytes_read > MAX_SAFETENSORS_TOTAL_HEADER_BYTES {
                return Err(model_error(format!(
                    "SafeTensors headers exceed the {}-byte catalog bound",
                    MAX_SAFETENSORS_TOTAL_HEADER_BYTES
                )));
            }
            for descriptor in parsed {
                if tensors
                    .insert(descriptor.name.clone(), descriptor)
                    .is_some()
                {
                    return Err(model_error("tensor occurs in more than one shard"));
                }
                if tensors.len() > MAX_SAFETENSORS_TENSORS {
                    return Err(model_error(
                        "SafeTensors catalog tensor count exceeds bound",
                    ));
                }
            }
            shards.push(shard);
        }
        if index
            .declared_payload_bytes
            .is_some_and(|declared| declared != total_payload_bytes)
        {
            return Err(model_error(
                "SafeTensors index total_size differs from validated shard payload bytes",
            ));
        }
        validate_index_mapping(&tensors, &shards, index.weights.as_ref())?;
        let metadata_sha256 = catalog_identity(&shards, &tensors, index.sha256.as_deref())?;
        let release_report_capacity = shards.len();
        Ok(Self {
            root: root.to_path_buf(),
            shards,
            tensors,
            metadata_sha256,
            index_sha256: index.sha256,
            index_declared_payload_bytes: index.declared_payload_bytes,
            total_file_bytes,
            total_payload_bytes,
            total_header_bytes_read,
            mapping_activity: Mutex::new(MappingActivityState {
                lifecycle: MappingLifecycle::Idle,
                high_water: 0,
                current_mapped_bytes: 0,
                mapped_byte_high_water: 0,
            }),
            release_reports: Mutex::new(Vec::with_capacity(release_report_capacity)),
            release_report_overflow: Mutex::new(false),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn shards(&self) -> &[SafeTensorShardDescriptor] {
        &self.shards
    }

    pub fn tensors(&self) -> impl Iterator<Item = &SafeTensorDescriptor> {
        self.tensors.values()
    }

    pub fn tensor(&self, name: &str) -> Result<&SafeTensorDescriptor> {
        self.tensors
            .get(name)
            .ok_or_else(|| model_error(format!("missing SafeTensors catalog tensor {name}")))
    }

    pub fn metadata_sha256(&self) -> &str {
        &self.metadata_sha256
    }

    pub fn index_sha256(&self) -> Option<&str> {
        self.index_sha256.as_deref()
    }

    pub const fn index_declared_payload_bytes(&self) -> Option<u64> {
        self.index_declared_payload_bytes
    }

    pub const fn total_file_bytes(&self) -> u64 {
        self.total_file_bytes
    }

    pub const fn total_payload_bytes(&self) -> u64 {
        self.total_payload_bytes
    }

    pub const fn total_header_bytes_read(&self) -> u64 {
        self.total_header_bytes_read
    }

    pub fn mapping_activity(&self) -> ShardMappingActivity {
        let state = self
            .mapping_activity
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ShardMappingActivity {
            current: usize::from(state.lifecycle != MappingLifecycle::Idle),
            high_water: state.high_water,
            current_mapped_bytes: state.current_mapped_bytes,
            mapped_byte_high_water: state.mapped_byte_high_water,
            quarantined: state.lifecycle == MappingLifecycle::Quarantined,
        }
    }

    pub fn active_source_payload_fds(&self) -> usize {
        self.mapping_activity().current
    }

    pub fn release_reports(&self) -> Vec<ShardReleaseTelemetry> {
        self.release_reports
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn release_report_overflowed(&self) -> bool {
        *self
            .release_report_overflow
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn record_release_report(&self, report: ShardReleaseTelemetry) {
        let mut reports = self
            .release_reports
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if reports.len() < MAX_SHARD_RELEASE_REPORTS {
            reports.push(report);
        } else {
            *self
                .release_report_overflow
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = true;
        }
    }

    /// Map one shard for the duration of `use_mapping` only.
    ///
    /// The return type cannot borrow from `ScopedShardMapping`, so a tensor
    /// slice cannot escape the callback. A nested or concurrent call fails
    /// before opening another file. Cleanup is RAII-based on success, error,
    /// and unwind. The checkpoint root has the same operational immutability
    /// requirement as `CpuTensorStore`: no external actor may mutate or replace
    /// a shard while this callback holds its mapping.
    pub fn with_mapped_shard<R>(
        &self,
        shard_index: usize,
        use_mapping: impl FnOnce(&ScopedShardMapping<'_>) -> Result<R>,
    ) -> Result<R> {
        let mapped = self.map_scoped_shard(shard_index)?;
        let result = {
            let scope = ScopedShardMapping {
                shard_index,
                descriptor: mapped.descriptor(),
                tensors: &self.tensors,
                mapping: mapped.mapping(),
            };
            use_mapping(&scope)
        };
        drop(mapped);
        result
    }

    pub(crate) fn map_scoped_shard(&self, shard_index: usize) -> Result<CatalogMappedShard<'_>> {
        let descriptor = self
            .shards
            .get(shard_index)
            .ok_or_else(|| model_error("shard index is outside the catalog"))?;
        let mut activity = ActiveMappingGuard::acquire(self)?;
        let mut file = open_validated_shard(descriptor)?;
        let observed_header = read_shard_header(&mut file, descriptor.identity.file_length)?;
        if observed_header.header_sha256 != descriptor.identity.header_sha256
            || observed_header.data_start != descriptor.identity.data_start
        {
            return Err(model_error(format!(
                "shard {} header identity changed after catalog construction",
                descriptor.identity.file_name
            )));
        }
        let mapping = map_cataloged_immutable_shard(
            &file,
            descriptor.path(),
            descriptor.identity.file_length,
            descriptor.identity.device,
            descriptor.identity.inode,
        )?;
        activity.mark_mapped(descriptor.identity.file_length);
        Ok(CatalogMappedShard {
            shard_index,
            descriptor,
            mapping: Some(mapping),
            file: Some(file),
            activity: Some(activity),
            terminal_audit: None,
        })
    }
}

/// One catalog-admitted mapping. Normal drop unmaps before releasing capacity;
/// quarantine intentionally retains the mapping for process life and poisons
/// this catalog instance permanently. The shard's external immutability
/// obligation therefore also lasts until process exit: holding an open `File`
/// would not prevent another actor from mutating the mapped inode.
pub(crate) struct CatalogMappedShard<'a> {
    shard_index: usize,
    descriptor: &'a SafeTensorShardDescriptor,
    mapping: Option<Mmap>,
    file: Option<File>,
    activity: Option<ActiveMappingGuard<'a>>,
    terminal_audit: Option<ShardTerminalAudit>,
}

#[derive(Debug, Clone)]
struct ShardTerminalAudit {
    covered_action_count: usize,
    covered_native_bytes: u64,
    terminal_receipts: usize,
    terminal_action_ids_sha256: Vec<String>,
    logical_ledger: ShardReleaseLogicalLedger,
}

impl CatalogMappedShard<'_> {
    pub(crate) const fn descriptor(&self) -> &SafeTensorShardDescriptor {
        self.descriptor
    }

    pub(crate) fn mapping(&self) -> &Mmap {
        self.mapping
            .as_ref()
            .expect("catalog mapping is present until release or quarantine")
    }

    pub(crate) fn checked_bytes(&self, absolute_range: [u64; 2]) -> Result<&[u8]> {
        let start = usize::try_from(absolute_range[0])
            .map_err(|_| model_error("mapped action start exceeds usize"))?;
        let end = usize::try_from(absolute_range[1])
            .map_err(|_| model_error("mapped action end exceeds usize"))?;
        self.mapping()
            .get(start..end)
            .ok_or_else(|| model_error("mapped action range is unavailable"))
    }

    pub(crate) fn record_terminal_audit(
        &mut self,
        covered_action_count: usize,
        covered_native_bytes: u64,
        terminal_receipts: usize,
        terminal_action_ids_sha256: Vec<String>,
        logical_ledger: ShardReleaseLogicalLedger,
    ) -> Result<()> {
        if self.terminal_audit.is_some()
            || covered_action_count == 0
            || covered_native_bytes == 0
            || covered_native_bytes != self.descriptor.identity.payload_length
            || terminal_receipts > covered_action_count
            || terminal_action_ids_sha256.len() != terminal_receipts
            || terminal_action_ids_sha256.iter().any(|identity| {
                identity.len() != 64 || !identity.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        {
            return Err(model_error("invalid or duplicate shard terminal audit"));
        }
        self.terminal_audit = Some(ShardTerminalAudit {
            covered_action_count,
            covered_native_bytes,
            terminal_receipts,
            terminal_action_ids_sha256,
            logical_ledger,
        });
        Ok(())
    }

    pub(crate) fn quarantine_for_process_lifetime(&mut self) {
        if let Some(activity) = self.activity.as_mut() {
            activity.quarantine();
        }
        if let Some(mapping) = self.mapping.as_ref() {
            let pre_release = process_source_memory_sample(self.descriptor.identity.inode);
            let terminal = self.terminal_audit.take().unwrap_or(ShardTerminalAudit {
                covered_action_count: 0,
                covered_native_bytes: 0,
                terminal_receipts: 0,
                terminal_action_ids_sha256: Vec::new(),
                logical_ledger: ShardReleaseLogicalLedger::default(),
            });
            let not_attempted = AdviceTelemetry {
                kind: "not_attempted_unproven_terminal".into(),
                offset: 0,
                length: mapping.len() as u64,
                succeeded: false,
                errno: None,
                error: Some("source ownership terminality is unproven".into()),
            };
            let report = ShardReleaseTelemetry {
                shard_index: self.shard_index,
                source_file_name: self.descriptor.identity.file_name.clone(),
                source_device: self.descriptor.identity.device,
                source_inode: self.descriptor.identity.inode,
                source_file_bytes: self.descriptor.identity.file_length,
                mapping_address: mapping.as_ptr() as usize,
                mapping_bytes: mapping.len() as u64,
                covered_action_count: terminal.covered_action_count,
                covered_native_bytes: terminal.covered_native_bytes,
                terminal_receipts: terminal.terminal_receipts,
                terminal_action_ids_sha256: terminal.terminal_action_ids_sha256,
                logical_ledger: terminal.logical_ledger,
                terminal_audit_complete: false,
                pre_release: pre_release.clone(),
                post_release: pre_release,
                mmap_advice: not_attempted.clone(),
                file_advice: not_attempted,
                unmap_close_duration_micros: 0,
                mapping_removed: false,
                fd_closed: false,
                cleanup_outcome: "quarantined_for_process_lifetime".into(),
            };
            if let Some(activity) = self.activity.as_ref() {
                activity.catalog.record_release_report(report);
            }
        }
        if let Some(mapping) = self.mapping.take() {
            std::mem::forget(mapping);
        }
        if let Some(file) = self.file.take() {
            std::mem::forget(file);
        }
        drop(self.activity.take());
    }
}

impl Drop for CatalogMappedShard<'_> {
    fn drop(&mut self) {
        let Some(mapping) = self.mapping.take() else {
            drop(self.file.take());
            drop(self.activity.take());
            return;
        };
        let mapping_address = mapping.as_ptr() as usize;
        let mapping_bytes = mapping.len() as u64;
        let pre_release = process_source_memory_sample(self.descriptor.identity.inode);
        let started = Instant::now();
        let mmap_advice_result = madvise_dontneed(&mapping);
        let mmap_advice = advice_telemetry(
            "madv_dontneed",
            0,
            mapping_bytes,
            mmap_advice_result.as_ref().err(),
        );
        drop(mapping);
        let mapping_removed = true;
        let file_advice_result = self.file.as_ref().map_or(Ok(()), |file| {
            gpt_oss_cpu_kernels::posix_fadvise_dontneed(file, 0, 0)
        });
        let file_advice = advice_telemetry(
            "posix_fadv_dontneed",
            0,
            mapping_bytes,
            file_advice_result.as_ref().err(),
        );
        drop(self.file.take());
        let fd_closed = true;
        let post_release = process_source_memory_sample(self.descriptor.identity.inode);
        let terminal_audit_complete = self.terminal_audit.is_some();
        let terminal = self.terminal_audit.take().unwrap_or(ShardTerminalAudit {
            covered_action_count: 0,
            covered_native_bytes: 0,
            terminal_receipts: 0,
            terminal_action_ids_sha256: Vec::new(),
            logical_ledger: ShardReleaseLogicalLedger::default(),
        });
        let report = ShardReleaseTelemetry {
            shard_index: self.shard_index,
            source_file_name: self.descriptor.identity.file_name.clone(),
            source_device: self.descriptor.identity.device,
            source_inode: self.descriptor.identity.inode,
            source_file_bytes: self.descriptor.identity.file_length,
            mapping_address,
            mapping_bytes,
            covered_action_count: terminal.covered_action_count,
            covered_native_bytes: terminal.covered_native_bytes,
            terminal_receipts: terminal.terminal_receipts,
            terminal_action_ids_sha256: terminal.terminal_action_ids_sha256,
            logical_ledger: terminal.logical_ledger,
            terminal_audit_complete,
            pre_release,
            post_release,
            mmap_advice,
            file_advice,
            unmap_close_duration_micros: started.elapsed().as_micros(),
            mapping_removed,
            fd_closed,
            cleanup_outcome: if terminal_audit_complete {
                "released_after_terminal_audit"
            } else {
                "released_during_cancellation"
            }
            .into(),
        };
        if let Some(activity) = self.activity.as_ref() {
            activity.catalog.record_release_report(report);
        }
        drop(self.activity.take());
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn madvise_dontneed(mapping: &Mmap) -> std::io::Result<()> {
    // SAFETY: this is called only from `CatalogMappedShard::drop` after the
    // scoped consumer callback and every action borrow have ended. This drop
    // owns the mapping exclusively and does not access it after advice.
    unsafe { mapping.unchecked_advise(MmapUncheckedAdvice::DontNeed) }
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn madvise_dontneed(_mapping: &Mmap) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "MADV_DONTNEED release advice is enabled on the CUDA construction build",
    ))
}

pub(crate) fn advice_telemetry(
    kind: &str,
    offset: u64,
    length: u64,
    error: Option<&std::io::Error>,
) -> AdviceTelemetry {
    AdviceTelemetry {
        kind: kind.into(),
        offset,
        length,
        succeeded: error.is_none(),
        errno: error.and_then(std::io::Error::raw_os_error),
        error: error.map(ToString::to_string),
    }
}

pub(crate) fn process_source_memory_sample(source_inode: u64) -> ProcessSourceMemorySample {
    let mut sample = ProcessSourceMemorySample::default();
    if let Ok(rollup) = std::fs::read_to_string("/proc/self/smaps_rollup") {
        sample.rss_bytes = proc_kib_field(&rollup, "Rss:");
        sample.pss_bytes = proc_kib_field(&rollup, "Pss:");
        sample.pss_file_bytes = proc_kib_field(&rollup, "Pss_File:");
    }
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        sample.vm_hwm_bytes = proc_kib_field(&status, "VmHWM:");
        sample.vm_swap_bytes = proc_kib_field(&status, "VmSwap:");
    }
    if let Ok(smaps) = std::fs::read_to_string("/proc/self/smaps") {
        let mut source_section = false;
        for line in smaps.lines() {
            if is_smaps_header(line) {
                source_section = line
                    .split_whitespace()
                    .nth(4)
                    .and_then(|inode| inode.parse::<u64>().ok())
                    == Some(source_inode);
                if source_section {
                    sample.source_inode_mapping_count += 1;
                }
            } else if source_section && line.starts_with("Pss:") {
                sample.source_inode_pss_bytes = sample
                    .source_inode_pss_bytes
                    .saturating_add(proc_kib_line(line));
            }
        }
    }
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        sample.mem_available_bytes = proc_kib_field(&meminfo, "MemAvailable:");
        let swap_total = proc_kib_field(&meminfo, "SwapTotal:");
        let swap_free = proc_kib_field(&meminfo, "SwapFree:");
        sample.global_swap_used_bytes = swap_total.saturating_sub(swap_free);
        sample.global_swap_cached_bytes = proc_kib_field(&meminfo, "SwapCached:");
    }
    if let Some(cgroup) = current_cgroup_v2_path() {
        sample.cgroup_memory_current_bytes = read_u64_file(&cgroup.join("memory.current"));
        sample.cgroup_swap_current_bytes = read_u64_file(&cgroup.join("memory.swap.current"));
        if let Ok(stat) = std::fs::read_to_string(cgroup.join("memory.stat")) {
            sample.cgroup_memory_file_bytes = space_value(&stat, "file");
            sample.cgroup_memory_anon_bytes = space_value(&stat, "anon");
        }
    }
    if let Ok(pressure) = std::fs::read_to_string("/proc/pressure/memory") {
        sample.memory_psi_some_avg10_micros = psi_avg10_micros(&pressure, "some");
        sample.memory_psi_full_avg10_micros = psi_avg10_micros(&pressure, "full");
    }
    sample
}

fn current_cgroup_v2_path() -> Option<PathBuf> {
    let cgroup = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let relative = cgroup.lines().find_map(|line| line.strip_prefix("0::"))?;
    Some(Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/')))
}

fn read_u64_file(path: &Path) -> u64 {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

fn space_value(contents: &str, field: &str) -> u64 {
    contents
        .lines()
        .find_map(|line| {
            let mut fields = line.split_ascii_whitespace();
            (fields.next() == Some(field))
                .then(|| fields.next()?.parse::<u64>().ok())
                .flatten()
        })
        .unwrap_or(0)
}

fn psi_avg10_micros(contents: &str, class: &str) -> u64 {
    contents
        .lines()
        .find(|line| line.starts_with(class))
        .and_then(|line| {
            line.split_ascii_whitespace()
                .find_map(|field| field.strip_prefix("avg10="))
        })
        .and_then(|value| value.parse::<f64>().ok())
        .map(|value| (value * 1_000_000.0).round() as u64)
        .unwrap_or(0)
}

fn is_smaps_header(line: &str) -> bool {
    line.split_whitespace().next().is_some_and(|range| {
        range.contains('-')
            && range
                .bytes()
                .all(|byte| byte == b'-' || byte.is_ascii_hexdigit())
    })
}

fn proc_kib_field(contents: &str, field: &str) -> u64 {
    contents
        .lines()
        .find(|line| line.starts_with(field))
        .map_or(0, proc_kib_line)
}

fn proc_kib_line(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|kib| kib.checked_mul(1024))
        .unwrap_or(0)
}

/// Borrowed view valid only inside [`SafeTensorShardCatalog::with_mapped_shard`].
pub struct ScopedShardMapping<'a> {
    shard_index: usize,
    descriptor: &'a SafeTensorShardDescriptor,
    tensors: &'a BTreeMap<String, SafeTensorDescriptor>,
    mapping: &'a Mmap,
}

impl<'a> ScopedShardMapping<'a> {
    pub const fn descriptor(&self) -> &'a SafeTensorShardDescriptor {
        self.descriptor
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| model_error(format!("missing mapped tensor {name}")))?;
        if tensor.shard_index != self.shard_index {
            return Err(model_error(format!(
                "tensor {name} belongs to shard {}, not {}",
                tensor.shard_index, self.shard_index
            )));
        }
        let start = usize::try_from(tensor.absolute_range[0])
            .map_err(|_| model_error("mapped tensor start exceeds usize"))?;
        let end = usize::try_from(tensor.absolute_range[1])
            .map_err(|_| model_error("mapped tensor end exceeds usize"))?;
        self.mapping
            .get(start..end)
            .ok_or_else(|| model_error(format!("mapped tensor {name} range is unavailable")))
    }
}

struct ActiveMappingGuard<'a> {
    catalog: &'a SafeTensorShardCatalog,
    mapped_bytes: u64,
    release_on_drop: bool,
}

impl<'a> ActiveMappingGuard<'a> {
    fn acquire(catalog: &'a SafeTensorShardCatalog) -> Result<Self> {
        let mut state = catalog
            .mapping_activity
            .lock()
            .map_err(|_| model_error("shard mapping activity lock is poisoned"))?;
        match state.lifecycle {
            MappingLifecycle::Idle => state.lifecycle = MappingLifecycle::Active,
            MappingLifecycle::Active => {
                return Err(model_error("a shard mapping is already active"))
            }
            MappingLifecycle::Quarantined => {
                return Err(model_error("shard catalog is permanently quarantined"))
            }
        }
        state.high_water = state.high_water.max(1);
        drop(state);
        Ok(Self {
            catalog,
            mapped_bytes: 0,
            release_on_drop: true,
        })
    }

    fn mark_mapped(&mut self, mapped_bytes: u64) {
        let mut state = self
            .catalog
            .mapping_activity
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(state.lifecycle, MappingLifecycle::Active);
        debug_assert_eq!(state.current_mapped_bytes, 0);
        state.current_mapped_bytes = mapped_bytes;
        state.mapped_byte_high_water = state.mapped_byte_high_water.max(mapped_bytes);
        self.mapped_bytes = mapped_bytes;
    }

    fn quarantine(&mut self) {
        let mut state = self
            .catalog
            .mapping_activity
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert!(matches!(
            state.lifecycle,
            MappingLifecycle::Active | MappingLifecycle::Quarantined
        ));
        debug_assert_eq!(state.current_mapped_bytes, self.mapped_bytes);
        state.lifecycle = MappingLifecycle::Quarantined;
        self.release_on_drop = false;
    }
}

impl Drop for ActiveMappingGuard<'_> {
    fn drop(&mut self) {
        if !self.release_on_drop {
            return;
        }
        let mut state = self
            .catalog
            .mapping_activity
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        debug_assert_eq!(state.lifecycle, MappingLifecycle::Active);
        state.current_mapped_bytes = 0;
        state.lifecycle = MappingLifecycle::Idle;
    }
}

#[derive(Debug, Deserialize)]
struct SafeTensorsIndex {
    #[serde(default)]
    metadata: SafeTensorsIndexMetadata,
    #[serde(deserialize_with = "deserialize_unique_string_map")]
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Default, Deserialize)]
struct SafeTensorsIndexMetadata {
    total_size: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

struct ReadHeader {
    bytes: Vec<u8>,
    data_start: u64,
    header_sha256: String,
}

fn validate_real_directory(root: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(root)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(model_error(
            "SafeTensors catalog root must be a real directory",
        ));
    }
    Ok(())
}

fn discover_shards(root: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut shards = BTreeMap::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path
            .extension()
            .is_none_or(|extension| extension != "safetensors")
        {
            continue;
        }
        let os_name = entry.file_name();
        let name = os_name
            .to_str()
            .ok_or_else(|| model_error("SafeTensors shard name is not UTF-8"))?;
        let name = validated_leaf_name(name)?;
        let metadata = std::fs::symlink_metadata(&path)?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(model_error(format!(
                "SafeTensors shard {name} must be a regular non-symlink file"
            )));
        }
        if shards.insert(name, path).is_some() {
            return Err(model_error("duplicate SafeTensors shard name"));
        }
        if shards.len() > MAX_SAFETENSORS_SHARDS {
            return Err(model_error("SafeTensors shard count exceeds bound"));
        }
    }
    if shards.is_empty() {
        return Err(model_error("SafeTensors catalog contains no shards"));
    }
    Ok(shards.into_iter().collect())
}

type IndexedWeights = BTreeMap<String, String>;

struct OptionalIndex {
    weights: Option<IndexedWeights>,
    shards: Option<BTreeSet<String>>,
    sha256: Option<String>,
    declared_payload_bytes: Option<u64>,
}

fn read_optional_index(root: &Path) -> Result<OptionalIndex> {
    let path = root.join("model.safetensors.index.json");
    match std::fs::symlink_metadata(&path) {
        Ok(metadata) => {
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                return Err(model_error(
                    "SafeTensors index must be a regular non-symlink file",
                ));
            }
            let bytes = read_bounded_file(&path, MAX_SAFETENSORS_INDEX_BYTES)?;
            let index: SafeTensorsIndex = serde_json::from_slice(&bytes)
                .map_err(|error| model_error(format!("invalid SafeTensors index: {error}")))?;
            if index.weight_map.is_empty() {
                return Err(model_error("SafeTensors index weight_map is empty"));
            }
            if index.weight_map.len() > MAX_SAFETENSORS_TENSORS {
                return Err(model_error("SafeTensors index tensor count exceeds bound"));
            }
            let mut shard_names = BTreeSet::new();
            for (tensor, shard) in &index.weight_map {
                validate_tensor_name(tensor)?;
                shard_names.insert(validated_leaf_name(shard)?);
            }
            Ok(OptionalIndex {
                weights: Some(index.weight_map),
                shards: Some(shard_names),
                sha256: Some(hash_bytes(&bytes)),
                declared_payload_bytes: index.metadata.total_size,
            })
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(OptionalIndex {
            weights: None,
            shards: None,
            sha256: None,
            declared_payload_bytes: None,
        }),
        Err(error) => Err(error.into()),
    }
}

fn validate_indexed_shard_set(
    discovered: &[(String, PathBuf)],
    indexed: Option<&BTreeSet<String>>,
) -> Result<()> {
    let observed = discovered
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<BTreeSet<_>>();
    match indexed {
        Some(expected) if expected != &observed => Err(model_error(
            "SafeTensors index shard set has missing or extra files",
        )),
        None if observed.len() != 1 => Err(model_error(
            "multiple SafeTensors shards require model.safetensors.index.json",
        )),
        _ => Ok(()),
    }
}

fn validate_index_mapping(
    tensors: &BTreeMap<String, SafeTensorDescriptor>,
    shards: &[SafeTensorShardDescriptor],
    indexed: Option<&IndexedWeights>,
) -> Result<()> {
    let Some(indexed) = indexed else {
        return Ok(());
    };
    if indexed.len() != tensors.len() || indexed.keys().ne(tensors.keys()) {
        return Err(model_error(
            "SafeTensors index and header tensor sets have missing or extra mappings",
        ));
    }
    for (name, expected_shard) in indexed {
        let tensor = tensors
            .get(name)
            .ok_or_else(|| model_error("indexed tensor is absent from headers"))?;
        if shards[tensor.shard_index].identity.file_name != *expected_shard {
            return Err(model_error(format!(
                "SafeTensors index maps tensor {name} to the wrong shard"
            )));
        }
    }
    Ok(())
}

fn parse_shard(
    path: PathBuf,
    file_name: String,
    shard_index: usize,
) -> Result<(SafeTensorShardDescriptor, Vec<SafeTensorDescriptor>)> {
    let before = std::fs::symlink_metadata(&path)?;
    if !before.is_file() || before.file_type().is_symlink() {
        return Err(model_error(format!(
            "shard {file_name} is not an unambiguous file"
        )));
    }
    let mut file = File::open(&path)?;
    let opened = file.metadata()?;
    validate_same_file(&before, &opened, &file_name)?;
    let file_length = opened.len();
    let header = read_shard_header(&mut file, file_length)?;
    let payload_length = file_length
        .checked_sub(header.data_start)
        .ok_or_else(|| model_error("SafeTensors data start exceeds file length"))?;
    let parsed = serde_json::from_slice::<UniqueValueMap>(&header.bytes)
        .map_err(|error| model_error(format!("invalid SafeTensors header: {error}")))?
        .0;
    let mut tensors = Vec::with_capacity(parsed.len());
    for (name, value) in parsed {
        if name == "__metadata__" {
            continue;
        }
        validate_tensor_name(&name)?;
        let tensor: HeaderTensor = serde_json::from_value(value).map_err(|error| {
            model_error(format!("invalid SafeTensors metadata for {name}: {error}"))
        })?;
        if tensor.shape.len() > MAX_SAFETENSORS_TENSOR_RANK {
            return Err(model_error(format!("tensor {name} rank exceeds bound")));
        }
        let dtype = DType::from_safetensors_str(&tensor.dtype).ok_or_else(|| {
            model_error(format!("unsupported SafeTensors dtype {}", tensor.dtype))
        })?;
        let elements = tensor.shape.iter().try_fold(1_u64, |product, dimension| {
            u64::try_from(*dimension)
                .ok()
                .and_then(|dimension| product.checked_mul(dimension))
        });
        let expected_bytes = elements
            .and_then(|count| count.checked_mul(dtype.size_of() as u64))
            .ok_or_else(|| model_error(format!("tensor {name} byte size overflows")))?;
        let [relative_start, relative_end] = tensor.data_offsets;
        if expected_bytes == 0 || relative_start > relative_end {
            return Err(model_error(format!(
                "tensor {name} has an invalid empty/range shape"
            )));
        }
        if relative_end
            .checked_sub(relative_start)
            .is_none_or(|length| length != expected_bytes)
        {
            return Err(model_error(format!(
                "tensor {name} byte range differs from shape"
            )));
        }
        let absolute_start = header
            .data_start
            .checked_add(relative_start)
            .ok_or_else(|| model_error(format!("tensor {name} absolute start overflows")))?;
        let absolute_end = header
            .data_start
            .checked_add(relative_end)
            .ok_or_else(|| model_error(format!("tensor {name} absolute end overflows")))?;
        if relative_end > payload_length || absolute_end > file_length {
            return Err(model_error(format!(
                "tensor {name} range exceeds shard file"
            )));
        }
        tensors.push(SafeTensorDescriptor {
            name,
            shard_index,
            dtype: dtype.to_string(),
            shape: tensor.shape,
            data_offsets: [relative_start, relative_end],
            absolute_range: [absolute_start, absolute_end],
        });
    }
    validate_range_partition(&file_name, payload_length, &tensors)?;
    let tensor_count = tensors.len();
    Ok((
        SafeTensorShardDescriptor {
            identity: SafeTensorFileIdentity {
                file_name,
                file_length,
                header_sha256: header.header_sha256,
                data_start: header.data_start,
                payload_length,
                device: metadata_device(&opened),
                inode: metadata_inode(&opened),
            },
            tensor_count,
            path,
        },
        tensors,
    ))
}

fn read_shard_header(reader: &mut (impl Read + Seek), file_length: u64) -> Result<ReadHeader> {
    reader.seek(SeekFrom::Start(0))?;
    let mut length_bytes = [0_u8; 8];
    reader.read_exact(&mut length_bytes)?;
    let header_length = u64::from_le_bytes(length_bytes);
    if header_length == 0 || header_length > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(model_error(format!(
            "SafeTensors header length {header_length} is outside the reviewed bound"
        )));
    }
    let data_start = 8_u64
        .checked_add(header_length)
        .ok_or_else(|| model_error("SafeTensors header length overflows"))?;
    if data_start > file_length {
        return Err(model_error("SafeTensors header exceeds file length"));
    }
    let length = usize::try_from(header_length)
        .map_err(|_| model_error("SafeTensors header exceeds addressable memory"))?;
    let mut bytes = vec![0_u8; length];
    reader.read_exact(&mut bytes)?;
    Ok(ReadHeader {
        header_sha256: hash_bytes(&bytes),
        bytes,
        data_start,
    })
}

fn validate_range_partition(
    file_name: &str,
    payload_length: u64,
    tensors: &[SafeTensorDescriptor],
) -> Result<()> {
    let mut ordered = tensors.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|tensor| (tensor.data_offsets[0], tensor.data_offsets[1], &tensor.name));
    let mut next = 0_u64;
    for tensor in ordered {
        if tensor.data_offsets[0] != next {
            return Err(model_error(format!(
                "shard {file_name} tensor ranges overlap or leave a forbidden gap"
            )));
        }
        next = tensor.data_offsets[1];
    }
    if next != payload_length {
        return Err(model_error(format!(
            "shard {file_name} tensor ranges do not cover the payload exactly"
        )));
    }
    Ok(())
}

fn open_validated_shard(descriptor: &SafeTensorShardDescriptor) -> Result<File> {
    let before = std::fs::symlink_metadata(&descriptor.path)?;
    if !before.is_file() || before.file_type().is_symlink() {
        return Err(model_error("catalog shard path became ambiguous"));
    }
    let file = OpenOptions::new().read(true).open(&descriptor.path)?;
    let opened = file.metadata()?;
    validate_same_file(&before, &opened, &descriptor.identity.file_name)?;
    if opened.len() != descriptor.identity.file_length
        || metadata_device(&opened) != descriptor.identity.device
        || metadata_inode(&opened) != descriptor.identity.inode
    {
        return Err(model_error(format!(
            "shard {} file identity changed after catalog construction",
            descriptor.identity.file_name
        )));
    }
    Ok(file)
}

#[cfg(unix)]
fn validate_same_file(
    before: &std::fs::Metadata,
    opened: &std::fs::Metadata,
    label: &str,
) -> Result<()> {
    use std::os::unix::fs::MetadataExt;
    if before.dev() != opened.dev() || before.ino() != opened.ino() {
        return Err(model_error(format!(
            "file identity changed while opening {label}"
        )));
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_same_file(
    before: &std::fs::Metadata,
    opened: &std::fs::Metadata,
    label: &str,
) -> Result<()> {
    if before.len() != opened.len() {
        return Err(model_error(format!(
            "file length changed while opening {label}"
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn metadata_device(metadata: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    metadata.dev()
}

#[cfg(not(unix))]
fn metadata_device(_metadata: &std::fs::Metadata) -> u64 {
    0
}

#[cfg(unix)]
fn metadata_inode(metadata: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    metadata.ino()
}

#[cfg(not(unix))]
fn metadata_inode(_metadata: &std::fs::Metadata) -> u64 {
    0
}

fn read_bounded_file(path: &Path, maximum: u64) -> Result<Vec<u8>> {
    let before = std::fs::symlink_metadata(path)?;
    if !before.is_file() || before.file_type().is_symlink() {
        return Err(model_error(
            "bounded metadata path is not an unambiguous file",
        ));
    }
    let mut file = File::open(path)?;
    let opened = file.metadata()?;
    validate_same_file(&before, &opened, &path.display().to_string())?;
    read_bounded_opened_file(path, &mut file, &opened, maximum)
}

fn read_bounded_opened_file(
    path: &Path,
    file: &mut File,
    opened: &std::fs::Metadata,
    maximum: u64,
) -> Result<Vec<u8>> {
    let length = opened.len();
    if length == 0 || length > maximum {
        return Err(model_error(format!(
            "metadata file {} length is outside the reviewed bound",
            path.display()
        )));
    }
    let bytes = read_declared_length_plus_one(file, length)?;
    let after = file.metadata()?;
    validate_same_file(opened, &after, &path.display().to_string())?;
    let after_path = std::fs::symlink_metadata(path)?;
    if !after_path.is_file() || after_path.file_type().is_symlink() {
        return Err(model_error(
            "bounded metadata path became ambiguous while reading",
        ));
    }
    validate_same_file(&after_path, &after, &path.display().to_string())?;
    let bytes_read =
        u64::try_from(bytes.len()).map_err(|_| model_error("metadata read length exceeds u64"))?;
    if after.len() != length || after_path.len() != length || bytes_read != length {
        return Err(model_error("metadata file length changed while reading"));
    }
    Ok(bytes)
}

fn read_declared_length_plus_one(reader: &mut impl Read, declared_length: u64) -> Result<Vec<u8>> {
    let read_limit = declared_length
        .checked_add(1)
        .ok_or_else(|| model_error("metadata read bound overflows"))?;
    let mut bytes = Vec::with_capacity(
        usize::try_from(read_limit).map_err(|_| model_error("metadata length exceeds usize"))?,
    );
    reader.take(read_limit).read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn validated_leaf_name(name: &str) -> Result<String> {
    let path = Path::new(name);
    let mut components = path.components();
    if name.is_empty()
        || name.contains(['/', '\\'])
        || !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
        || path
            .extension()
            .is_none_or(|extension| extension != "safetensors")
    {
        return Err(model_error(format!(
            "invalid SafeTensors shard name {name:?}"
        )));
    }
    Ok(name.to_owned())
}

struct UniqueValueMap(BTreeMap<String, serde_json::Value>);

impl<'de> Deserialize<'de> for UniqueValueMap {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueValueMapVisitor;

        impl<'de> Visitor<'de> for UniqueValueMapVisitor {
            type Value = UniqueValueMap;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object with unique keys")
            }

            fn visit_map<M>(self, mut access: M) -> std::result::Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut values = BTreeMap::new();
                while let Some((key, value)) = access.next_entry::<String, serde_json::Value>()? {
                    if values.insert(key.clone(), value).is_some() {
                        return Err(M::Error::custom(format!("duplicate JSON key {key}")));
                    }
                }
                Ok(UniqueValueMap(values))
            }
        }

        deserializer.deserialize_map(UniqueValueMapVisitor)
    }
}

fn deserialize_unique_string_map<'de, D>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct UniqueStringMapVisitor;

    impl<'de> Visitor<'de> for UniqueStringMapVisitor {
        type Value = BTreeMap<String, String>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a string map with unique keys")
        }

        fn visit_map<M>(self, mut access: M) -> std::result::Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut values = BTreeMap::new();
            while let Some((key, value)) = access.next_entry::<String, String>()? {
                if values.insert(key.clone(), value).is_some() {
                    return Err(M::Error::custom(format!("duplicate JSON key {key}")));
                }
            }
            Ok(values)
        }
    }

    deserializer.deserialize_map(UniqueStringMapVisitor)
}

fn validate_tensor_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > MAX_SAFETENSORS_TENSOR_NAME_BYTES
        || name
            .bytes()
            .any(|byte| byte == 0 || byte.is_ascii_control())
    {
        return Err(model_error(
            "SafeTensors tensor name is invalid or exceeds bound",
        ));
    }
    Ok(())
}

fn catalog_identity(
    shards: &[SafeTensorShardDescriptor],
    tensors: &BTreeMap<String, SafeTensorDescriptor>,
    index_sha256: Option<&str>,
) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(SAFETENSORS_CATALOG_SCHEMA_V1.as_bytes());
    match index_sha256 {
        Some(hash) => {
            digest.update([1]);
            digest.update(hash.as_bytes());
        }
        None => digest.update([0]),
    }
    for shard in shards {
        update_len_prefixed(&mut digest, shard.identity.file_name.as_bytes())?;
        digest.update(shard.identity.file_length.to_le_bytes());
        digest.update(shard.identity.data_start.to_le_bytes());
        digest.update(shard.identity.payload_length.to_le_bytes());
        digest.update(shard.identity.header_sha256.as_bytes());
    }
    for tensor in tensors.values() {
        update_len_prefixed(&mut digest, tensor.name.as_bytes())?;
        digest.update(
            u64::try_from(tensor.shard_index)
                .map_err(|_| model_error("shard index exceeds u64"))?
                .to_le_bytes(),
        );
        update_len_prefixed(&mut digest, tensor.dtype.as_bytes())?;
        digest.update(
            u64::try_from(tensor.shape.len())
                .map_err(|_| model_error("tensor rank exceeds u64"))?
                .to_le_bytes(),
        );
        for dimension in &tensor.shape {
            digest.update(
                u64::try_from(*dimension)
                    .map_err(|_| model_error("tensor dimension exceeds u64"))?
                    .to_le_bytes(),
            );
        }
        digest.update(tensor.absolute_range[0].to_le_bytes());
        digest.update(tensor.absolute_range[1].to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn update_len_prefixed(digest: &mut Sha256, bytes: &[u8]) -> Result<()> {
    digest.update(
        u64::try_from(bytes.len())
            .map_err(|_| model_error("catalog identity field exceeds u64"))?
            .to_le_bytes(),
    );
    digest.update(bytes);
    Ok(())
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn model_error(message: impl Into<String>) -> LLMError {
    LLMError::ModelError(message.into())
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[cfg(unix)]
    use std::ffi::OsString;
    #[cfg(unix)]
    use std::os::unix::ffi::OsStringExt;

    use tempfile::tempdir;

    use super::*;

    #[derive(Clone)]
    struct TensorFixture<'a> {
        name: &'a str,
        dtype: &'a str,
        shape: &'a [usize],
        payload: &'a [u8],
    }

    #[derive(Serialize)]
    struct HeaderTensorFixture<'a> {
        dtype: &'a str,
        shape: &'a [usize],
        data_offsets: [u64; 2],
    }

    fn write_shard(path: &Path, tensors: &[TensorFixture<'_>]) {
        let mut offset = 0_u64;
        let mut header: BTreeMap<String, HeaderTensorFixture<'_>> = BTreeMap::new();
        for tensor in tensors {
            let start = offset;
            offset += tensor.payload.len() as u64;
            header.insert(
                tensor.name.into(),
                HeaderTensorFixture {
                    dtype: tensor.dtype,
                    shape: tensor.shape,
                    data_offsets: [start, offset],
                },
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        for tensor in tensors {
            file.write_all(tensor.payload).unwrap();
        }
        file.sync_all().unwrap();
    }

    fn write_index(root: &Path, mappings: &[(&str, &str)]) {
        let weight_map = mappings
            .iter()
            .map(|(tensor, shard)| ((*tensor).to_owned(), (*shard).to_owned()))
            .collect::<BTreeMap<_, _>>();
        std::fs::write(
            root.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
        )
        .unwrap();
    }

    fn write_raw_shard(path: &Path, header: &[u8], payload: &[u8]) {
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(payload).unwrap();
        file.sync_all().unwrap();
    }

    fn two_shard_fixture(root: &Path, reverse_creation: bool) {
        let first = || {
            write_shard(
                &root.join("model-00001-of-00002.safetensors"),
                &[TensorFixture {
                    name: "a",
                    dtype: "U8",
                    shape: &[3],
                    payload: &[1, 2, 3],
                }],
            )
        };
        let second = || {
            write_shard(
                &root.join("model-00002-of-00002.safetensors"),
                &[TensorFixture {
                    name: "b",
                    dtype: "BF16",
                    shape: &[2],
                    payload: &[4, 5, 6, 7],
                }],
            )
        };
        if reverse_creation {
            second();
            first();
        } else {
            first();
            second();
        }
        write_index(
            root,
            &[
                ("a", "model-00001-of-00002.safetensors"),
                ("b", "model-00002-of-00002.safetensors"),
            ],
        );
    }

    #[test]
    fn catalog_is_bounded_deterministic_and_payload_independent() {
        let first = tempdir().unwrap();
        let second = tempdir().unwrap();
        two_shard_fixture(first.path(), false);
        two_shard_fixture(second.path(), true);
        let first_catalog = SafeTensorShardCatalog::open(first.path()).unwrap();
        let second_catalog = SafeTensorShardCatalog::open(second.path()).unwrap();
        assert_eq!(
            first_catalog.metadata_sha256(),
            second_catalog.metadata_sha256()
        );
        assert_eq!(
            first_catalog.metadata_sha256(),
            "08650b3d5ccf811e80d040b84b75c03a3c09ee364ce033a3054a95e6b5375e08"
        );
        assert_eq!(
            first_catalog.index_sha256(),
            Some("fcd62bc8c750aa36660232e02d85d166f5fa56080d2c6b0161421c7984d1e89f")
        );
        assert_eq!(first_catalog.shards().len(), 2);
        assert_eq!(first_catalog.tensors().count(), 2);
        assert_eq!(first_catalog.total_payload_bytes(), 7);
        assert!(first_catalog.total_header_bytes_read() < 4_096);
        assert_eq!(
            first_catalog.tensor("a").unwrap().absolute_range[1]
                - first_catalog.tensor("a").unwrap().absolute_range[0],
            3
        );

        let shard = second.path().join("model-00001-of-00002.safetensors");
        let data_start = second_catalog.shards()[0].identity.data_start;
        let mut file = OpenOptions::new().write(true).open(shard).unwrap();
        file.seek(SeekFrom::Start(data_start)).unwrap();
        file.write_all(&[9, 9, 9]).unwrap();
        file.sync_all().unwrap();
        let payload_changed = SafeTensorShardCatalog::open(second.path()).unwrap();
        assert_eq!(
            second_catalog.metadata_sha256(),
            payload_changed.metadata_sha256(),
            "catalog identity intentionally excludes payload contents/residency"
        );
    }

    #[test]
    fn serialized_catalog_descriptors_exclude_process_local_identity() {
        let root = tempdir().unwrap();
        write_shard(
            &root.path().join("model.safetensors"),
            &[TensorFixture {
                name: "a",
                dtype: "U8",
                shape: &[1],
                payload: &[1],
            }],
        );
        let catalog = SafeTensorShardCatalog::open(root.path()).unwrap();
        let serialized = serde_json::to_string(catalog.shards()).unwrap();
        assert!(serialized.contains("model.safetensors"));
        assert!(!serialized.contains("device"));
        assert!(!serialized.contains("inode"));
        assert!(!serialized.contains("path"));
        assert!(!serialized.contains(&root.path().display().to_string()));
    }

    #[test]
    fn header_parser_never_reads_claimed_payload() {
        struct HeaderOnlyReader {
            inner: Cursor<Vec<u8>>,
            maximum_read_position: u64,
        }
        impl Read for HeaderOnlyReader {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                let read = self.inner.read(buffer)?;
                self.maximum_read_position = self.maximum_read_position.max(self.inner.position());
                Ok(read)
            }
        }
        impl Seek for HeaderOnlyReader {
            fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
                self.inner.seek(position)
            }
        }
        let header = br#"{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        let mut reader = HeaderOnlyReader {
            inner: Cursor::new(bytes),
            maximum_read_position: 0,
        };
        let parsed = read_shard_header(&mut reader, 8 + header.len() as u64 + 4).unwrap();
        assert_eq!(parsed.data_start, 8 + header.len() as u64);
        assert_eq!(reader.maximum_read_position, parsed.data_start);
    }

    #[test]
    fn bounded_metadata_reader_reads_declared_length_plus_at_most_one() {
        struct CountingReader {
            inner: Cursor<Vec<u8>>,
            maximum_read_position: u64,
        }
        impl Read for CountingReader {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                let read = self.inner.read(buffer)?;
                self.maximum_read_position = self.maximum_read_position.max(self.inner.position());
                Ok(read)
            }
        }

        let mut reader = CountingReader {
            inner: Cursor::new(vec![7; 64]),
            maximum_read_position: 0,
        };
        let bytes = read_declared_length_plus_one(&mut reader, 5).unwrap();
        assert_eq!(bytes.len(), 6);
        assert_eq!(reader.maximum_read_position, 6);

        let root = tempdir().unwrap();
        let path = root.path().join("metadata.json");
        std::fs::write(&path, b"{}").unwrap();
        let mut file = File::open(&path).unwrap();
        let opened = file.metadata().unwrap();
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"x")
            .unwrap();
        assert!(read_bounded_opened_file(&path, &mut file, &opened, 32).is_err());
    }

    #[test]
    fn catalog_rejects_bad_index_paths_sets_and_symlinks() {
        for bad_name in [
            "../escape.safetensors",
            "/absolute.safetensors",
            "a/b.safetensors",
        ] {
            let root = tempdir().unwrap();
            write_shard(
                &root.path().join("model.safetensors"),
                &[TensorFixture {
                    name: "a",
                    dtype: "U8",
                    shape: &[1],
                    payload: &[1],
                }],
            );
            write_index(root.path(), &[("a", bad_name)]);
            assert!(SafeTensorShardCatalog::open(root.path()).is_err());
        }

        let root = tempdir().unwrap();
        two_shard_fixture(root.path(), false);
        write_index(root.path(), &[("a", "model-00001-of-00002.safetensors")]);
        assert!(SafeTensorShardCatalog::open(root.path()).is_err());

        let root = tempdir().unwrap();
        two_shard_fixture(root.path(), false);
        write_index(
            root.path(),
            &[
                ("a", "model-00002-of-00002.safetensors"),
                ("b", "model-00001-of-00002.safetensors"),
            ],
        );
        assert!(SafeTensorShardCatalog::open(root.path()).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let root = tempdir().unwrap();
            let outside = root.path().join("outside");
            write_shard(
                &outside,
                &[TensorFixture {
                    name: "a",
                    dtype: "U8",
                    shape: &[1],
                    payload: &[1],
                }],
            );
            symlink(&outside, root.path().join("model.safetensors")).unwrap();
            assert!(SafeTensorShardCatalog::open(root.path()).is_err());

            let root = tempdir().unwrap();
            let non_utf8 = OsString::from_vec(b"bad-\xff.safetensors".to_vec());
            write_shard(
                &root.path().join(non_utf8),
                &[TensorFixture {
                    name: "a",
                    dtype: "U8",
                    shape: &[1],
                    payload: &[1],
                }],
            );
            assert!(SafeTensorShardCatalog::open(root.path()).is_err());
        }
    }

    #[test]
    fn catalog_rejects_malformed_oversized_overflowing_and_overlapping_headers() {
        let root = tempdir().unwrap();
        let oversized = root.path().join("model.safetensors");
        std::fs::write(&oversized, (MAX_SAFETENSORS_HEADER_BYTES + 1).to_le_bytes()).unwrap();
        assert!(SafeTensorShardCatalog::open(root.path()).is_err());

        std::fs::write(&oversized, 4_u64.to_le_bytes()).unwrap();
        let mut file = OpenOptions::new().append(true).open(&oversized).unwrap();
        file.write_all(b"nope").unwrap();
        drop(file);
        assert!(SafeTensorShardCatalog::open(root.path()).is_err());

        let malformed = serde_json::json!({
            "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
            "b": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]}
        });
        let header = serde_json::to_vec(&malformed).unwrap();
        let mut file = File::create(&oversized).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&[0; 3]).unwrap();
        drop(file);
        assert!(SafeTensorShardCatalog::open(root.path()).is_err());

        let out_of_file = serde_json::to_vec(&serde_json::json!({
            "a": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]}
        }))
        .unwrap();
        write_raw_shard(&oversized, &out_of_file, &[0; 4]);
        assert!(SafeTensorShardCatalog::open(root.path()).is_err());

        let overflow = serde_json::json!({
            "a": {"dtype": "F32", "shape": [usize::MAX, 2], "data_offsets": [0, 4]}
        });
        let header = serde_json::to_vec(&overflow).unwrap();
        let mut file = File::create(&oversized).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&[0; 4]).unwrap();
        drop(file);
        assert!(SafeTensorShardCatalog::open(root.path()).is_err());
    }

    #[test]
    fn catalog_rejects_duplicate_header_index_and_cross_shard_tensor_names() {
        let root = tempdir().unwrap();
        let duplicate_header = br#"{
            "a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},
            "a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}
        }"#;
        write_raw_shard(
            &root.path().join("model.safetensors"),
            duplicate_header,
            &[1],
        );
        assert!(SafeTensorShardCatalog::open(root.path()).is_err());

        let root = tempdir().unwrap();
        write_shard(
            &root.path().join("model.safetensors"),
            &[TensorFixture {
                name: "a",
                dtype: "U8",
                shape: &[1],
                payload: &[1],
            }],
        );
        std::fs::write(
            root.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"a":"model.safetensors","a":"model.safetensors"}}"#,
        )
        .unwrap();
        assert!(SafeTensorShardCatalog::open(root.path()).is_err());

        let root = tempdir().unwrap();
        for shard in [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ] {
            write_shard(
                &root.path().join(shard),
                &[TensorFixture {
                    name: "a",
                    dtype: "U8",
                    shape: &[1],
                    payload: &[1],
                }],
            );
        }
        write_index(
            root.path(),
            &[
                ("a", "model-00001-of-00002.safetensors"),
                ("b", "model-00002-of-00002.safetensors"),
            ],
        );
        assert!(SafeTensorShardCatalog::open(root.path()).is_err());
    }

    #[test]
    fn catalog_rejects_oversized_index_and_wrong_declared_payload_size() {
        let root = tempdir().unwrap();
        write_shard(
            &root.path().join("model.safetensors"),
            &[TensorFixture {
                name: "a",
                dtype: "U8",
                shape: &[1],
                payload: &[1],
            }],
        );
        File::create(root.path().join("model.safetensors.index.json"))
            .unwrap()
            .set_len(MAX_SAFETENSORS_INDEX_BYTES + 1)
            .unwrap();
        assert!(SafeTensorShardCatalog::open(root.path()).is_err());

        std::fs::write(
            root.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "metadata": {"total_size": 2},
                "weight_map": {"a": "model.safetensors"}
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(SafeTensorShardCatalog::open(root.path()).is_err());
    }

    #[test]
    fn scoped_mapping_is_single_active_and_cleans_success_error_and_panic() {
        let root = tempdir().unwrap();
        write_shard(
            &root.path().join("model.safetensors"),
            &[TensorFixture {
                name: "a",
                dtype: "U8",
                shape: &[3],
                payload: &[1, 2, 3],
            }],
        );
        let catalog = SafeTensorShardCatalog::open(root.path()).unwrap();
        let file_length = catalog.shards()[0].identity.file_length;
        let copied = catalog
            .with_mapped_shard(0, |mapping| {
                assert_eq!(catalog.mapping_activity().current, 1);
                assert!(catalog.with_mapped_shard(0, |_| Ok(())).is_err());
                Ok(mapping.tensor_bytes("a")?.to_vec())
            })
            .unwrap();
        assert_eq!(copied, [1, 2, 3]);
        assert_eq!(
            catalog.mapping_activity(),
            ShardMappingActivity {
                current: 0,
                high_water: 1,
                current_mapped_bytes: 0,
                mapped_byte_high_water: file_length,
                quarantined: false,
            }
        );

        let result: Result<()> =
            catalog.with_mapped_shard(0, |_| Err(model_error("synthetic mapped callback failure")));
        assert!(result.is_err());
        assert_eq!(catalog.mapping_activity().current, 0);

        let panic_result = catch_unwind(AssertUnwindSafe(|| {
            let _ = catalog.with_mapped_shard::<()>(0, |_| panic!("synthetic panic"));
        }));
        assert!(panic_result.is_err());
        assert_eq!(catalog.mapping_activity().current, 0);
        assert_eq!(
            catalog
                .with_mapped_shard(0, |mapping| Ok(mapping.tensor_bytes("a")?[0]))
                .unwrap(),
            1
        );
    }

    #[test]
    fn scoped_mapping_rejects_replaced_path_and_changed_header() {
        let root = tempdir().unwrap();
        let shard = root.path().join("model.safetensors");
        let tensor = TensorFixture {
            name: "a",
            dtype: "U8",
            shape: &[1],
            payload: &[1],
        };
        write_shard(&shard, std::slice::from_ref(&tensor));
        let catalog = SafeTensorShardCatalog::open(root.path()).unwrap();
        std::fs::rename(&shard, root.path().join("original.safetensors")).unwrap();
        write_shard(&shard, std::slice::from_ref(&tensor));
        assert!(catalog.with_mapped_shard(0, |_| Ok(())).is_err());
        assert_eq!(catalog.mapping_activity().current, 0);

        std::fs::remove_file(root.path().join("original.safetensors")).unwrap();
        let catalog = SafeTensorShardCatalog::open(root.path()).unwrap();
        let changed = TensorFixture {
            name: "b",
            dtype: "U8",
            shape: &[1],
            payload: &[1],
        };
        write_shard(&shard, &[changed]);
        assert!(catalog.with_mapped_shard(0, |_| Ok(())).is_err());
        assert_eq!(catalog.mapping_activity().current, 0);
    }
}
