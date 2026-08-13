//! Opt-in bounded CPU execution profiling.
//!
//! The disabled path owns no storage and never reads a clock. Enabled paths
//! append fixed-width records to a preallocated slab; formatting and I/O are
//! explicit cold-path operations.

use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use sha2::{Digest, Sha256};

use gpt_oss_core::prelude::{LLMError, Result};
use gpt_oss_cpu_kernels::{KernelPath, Mxfp4MatmulBackend};

pub const EXECUTION_PROFILE_SCHEMA: &str = "gpt-oss-rs.execution-profile/v1";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuProfilePhase {
    Prefill,
    Decode,
    Mixed,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuProfileOperation {
    InputNormalization,
    QueryProjection,
    KeyProjection,
    ValueProjection,
    Attention,
    AttentionOutputProjection,
    PostAttentionNormalization,
    RouterProjection,
    Routing,
    Q8Preparation,
    ResidualQ8Preparation,
    GateUpProjection,
    SwiGlu,
    DownProjection,
    ExpertWeightingAccumulation,
    FinalNormalization,
    LmHeadProjection,
    Fallback,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuProfileProjectionRole {
    Query,
    Key,
    Value,
    AttentionOutput,
    Router,
    GateUp,
    Down,
    LmHead,
    #[default]
    None,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuProfileAttentionClass {
    Full,
    Sliding,
    #[default]
    None,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuProfilePreparationState {
    Cold,
    Warm,
    Prepared,
    #[default]
    NotApplicable,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuProfileResidencyState {
    Hit,
    Miss,
    Bypass,
    Fault,
    #[default]
    Disabled,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuProfileFallbackReason {
    UnsupportedIsa,
    ProfileMismatch,
    UnobservedShape,
    XeFault,
    Capacity,
    Validation,
    ExecutionFailure,
    #[default]
    None,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuProfileTransactionState {
    Prepared,
    Failed,
    #[default]
    Pending,
}

/// Fixed-layout hot-path record. No field owns heap storage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct CpuExecutionProfileRecord {
    pub sequence: u64,
    pub phase: CpuProfilePhase,
    pub prefill_rows: u32,
    pub decode_rows: u32,
    pub operation: CpuProfileOperation,
    pub layer: u16,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub expert_bucket_m: u32,
    pub context_length: u32,
    pub requested_cpu_kernel: u8,
    pub effective_cpu_kernel: u8,
    pub requested_matrix_backend: u8,
    pub effective_matrix_backend: u8,
    pub thread_count: u16,
    pub projection_role: CpuProfileProjectionRole,
    pub attention_class: CpuProfileAttentionClass,
    pub preparation_state: CpuProfilePreparationState,
    pub residency_state: CpuProfileResidencyState,
    pub fallback_reason: CpuProfileFallbackReason,
    pub transaction_state: CpuProfileTransactionState,
    pub start_ns: u64,
    pub duration_ns: u64,
    pub scratch_bytes: u64,
    pub scratch_high_water_bytes: u64,
    pub resident_bytes: u64,
    pub resident_high_water_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CpuExecutionProfileDocument {
    pub schema: &'static str,
    pub repository_commit: String,
    pub repository_dirty: bool,
    pub model_revision: String,
    pub model_source_hashes: Vec<String>,
    pub workload_id: Option<String>,
    pub hardware_profile_key: String,
    pub cpu_identity: serde_json::Value,
    pub dispatch: serde_json::Value,
    pub xe_runtime: Option<serde_json::Value>,
    pub command: Vec<String>,
    pub timer: &'static str,
    pub record_bytes: usize,
    pub record_capacity: usize,
    pub records_written: usize,
    pub records_dropped: u64,
    pub truncated: bool,
    pub start_unix_ns: u128,
    pub end_unix_ns: u128,
    pub records_sha256: String,
    pub records: Vec<CpuExecutionProfileRecord>,
}

#[derive(Debug, Clone)]
pub struct CpuExecutionProfileMetadata {
    pub model_revision: String,
    pub model_source_hashes: Vec<String>,
    pub workload_id: Option<String>,
    pub hardware_profile_key: String,
    pub cpu_identity: serde_json::Value,
    pub dispatch: serde_json::Value,
    pub xe_runtime: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CpuProfileRecordSpec {
    pub operation: CpuProfileOperation,
    pub layer: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub expert_bucket_m: usize,
    pub projection_role: CpuProfileProjectionRole,
    pub attention_class: CpuProfileAttentionClass,
    pub preparation_state: CpuProfilePreparationState,
    pub residency_state: CpuProfileResidencyState,
    pub fallback_reason: CpuProfileFallbackReason,
    pub scratch_bytes: usize,
    pub resident_bytes: usize,
    pub effective_matrix_backend: Option<Mxfp4MatmulBackend>,
}

#[derive(Debug, Clone, Copy, Default)]
struct BatchContext {
    phase: CpuProfilePhase,
    prefill_rows: u32,
    decode_rows: u32,
    context_length: u32,
    first_record: usize,
}

pub(crate) struct CpuProfileClock(Instant);

#[derive(Debug)]
pub(crate) struct CpuExecutionProfiler {
    origin: Instant,
    start_unix_ns: u128,
    records: Box<[CpuExecutionProfileRecord]>,
    len: usize,
    dropped: u64,
    sequence: u64,
    batch: BatchContext,
    requested_kernel: KernelPath,
    effective_kernel: KernelPath,
    requested_matrix: Mxfp4MatmulBackend,
    effective_matrix: Mxfp4MatmulBackend,
    thread_count: usize,
    scratch_high_water: usize,
    resident_high_water: usize,
    command: Vec<String>,
}

impl CpuExecutionProfiler {
    pub(crate) fn new(
        capacity_bytes: usize,
        requested_kernel: KernelPath,
        effective_kernel: KernelPath,
        requested_matrix: Mxfp4MatmulBackend,
        thread_count: usize,
    ) -> Result<Self> {
        let record_bytes = std::mem::size_of::<CpuExecutionProfileRecord>();
        let capacity = capacity_bytes / record_bytes;
        if capacity == 0 {
            return Err(LLMError::ConfigError(format!(
                "CPU profile capacity must hold at least one {record_bytes}-byte record"
            )));
        }
        Ok(Self {
            origin: Instant::now(),
            start_unix_ns: unix_ns(),
            records: vec![CpuExecutionProfileRecord::default(); capacity].into_boxed_slice(),
            len: 0,
            dropped: 0,
            sequence: 0,
            batch: BatchContext::default(),
            requested_kernel,
            effective_kernel,
            requested_matrix,
            effective_matrix: requested_matrix,
            thread_count,
            scratch_high_water: 0,
            resident_high_water: 0,
            command: std::env::args().collect(),
        })
    }

    pub(crate) fn begin_batch(
        &mut self,
        phase: CpuProfilePhase,
        prefill_rows: usize,
        decode_rows: usize,
        context_length: usize,
    ) {
        self.batch = BatchContext {
            phase,
            prefill_rows: saturating_u32(prefill_rows),
            decode_rows: saturating_u32(decode_rows),
            context_length: saturating_u32(context_length),
            first_record: self.len,
        };
    }

    pub(crate) fn start(&self) -> CpuProfileClock {
        CpuProfileClock(Instant::now())
    }

    pub(crate) fn append(&mut self, start: CpuProfileClock, spec: CpuProfileRecordSpec) {
        self.scratch_high_water = self.scratch_high_water.max(spec.scratch_bytes);
        self.resident_high_water = self.resident_high_water.max(spec.resident_bytes);
        let now = Instant::now();
        let record = CpuExecutionProfileRecord {
            sequence: self.sequence,
            phase: self.batch.phase,
            prefill_rows: self.batch.prefill_rows,
            decode_rows: self.batch.decode_rows,
            operation: spec.operation,
            layer: spec.layer.try_into().unwrap_or(u16::MAX),
            m: saturating_u32(spec.m),
            n: saturating_u32(spec.n),
            k: saturating_u32(spec.k),
            expert_bucket_m: saturating_u32(spec.expert_bucket_m),
            context_length: self.batch.context_length,
            requested_cpu_kernel: kernel_code(self.requested_kernel),
            effective_cpu_kernel: kernel_code(self.effective_kernel),
            requested_matrix_backend: matrix_code(self.requested_matrix),
            effective_matrix_backend: matrix_code(
                spec.effective_matrix_backend
                    .unwrap_or(self.effective_matrix),
            ),
            thread_count: self.thread_count.try_into().unwrap_or(u16::MAX),
            projection_role: spec.projection_role,
            attention_class: spec.attention_class,
            preparation_state: spec.preparation_state,
            residency_state: spec.residency_state,
            fallback_reason: spec.fallback_reason,
            transaction_state: CpuProfileTransactionState::Pending,
            start_ns: saturating_u64(start.0.duration_since(self.origin).as_nanos()),
            duration_ns: saturating_u64(now.duration_since(start.0).as_nanos()),
            scratch_bytes: saturating_u64(spec.scratch_bytes as u128),
            scratch_high_water_bytes: saturating_u64(self.scratch_high_water as u128),
            resident_bytes: saturating_u64(spec.resident_bytes as u128),
            resident_high_water_bytes: saturating_u64(self.resident_high_water as u128),
        };
        self.sequence = self.sequence.saturating_add(1);
        if self.len == self.records.len() {
            self.dropped = self.dropped.saturating_add(1);
        } else {
            self.records[self.len] = record;
            self.len += 1;
        }
    }

    pub(crate) fn finish_batch(&mut self, success: bool) {
        let state = if success {
            CpuProfileTransactionState::Prepared
        } else {
            CpuProfileTransactionState::Failed
        };
        for record in &mut self.records[self.batch.first_record.min(self.len)..self.len] {
            record.transaction_state = state;
        }
    }

    pub(crate) fn document(
        &self,
        metadata: CpuExecutionProfileMetadata,
    ) -> Result<CpuExecutionProfileDocument> {
        let records = self.records[..self.len].to_vec();
        let encoded = serde_json::to_vec(&records).map_err(|error| {
            LLMError::ModelError(format!(
                "CPU execution profile serialization failed: {error}"
            ))
        })?;
        let (repository_commit, repository_dirty) = repository_identity();
        Ok(CpuExecutionProfileDocument {
            schema: EXECUTION_PROFILE_SCHEMA,
            repository_commit,
            repository_dirty,
            model_revision: metadata.model_revision,
            model_source_hashes: metadata.model_source_hashes,
            workload_id: metadata.workload_id,
            hardware_profile_key: metadata.hardware_profile_key,
            cpu_identity: metadata.cpu_identity,
            dispatch: metadata.dispatch,
            xe_runtime: metadata.xe_runtime,
            command: self.command.clone(),
            timer: "std::time::Instant monotonic; exclusive coarse operation boundaries",
            record_bytes: std::mem::size_of::<CpuExecutionProfileRecord>(),
            record_capacity: self.records.len(),
            records_written: self.len,
            records_dropped: self.dropped,
            truncated: self.dropped != 0,
            start_unix_ns: self.start_unix_ns,
            end_unix_ns: unix_ns(),
            records_sha256: format!("{:x}", Sha256::digest(encoded)),
            records,
        })
    }

    pub(crate) fn write(&self, path: &Path, metadata: CpuExecutionProfileMetadata) -> Result<()> {
        let document = self.document(metadata)?;
        let bytes = serde_json::to_vec_pretty(&document).map_err(|error| {
            LLMError::ModelError(format!(
                "CPU execution profile serialization failed: {error}"
            ))
        })?;
        atomic_write(path, &bytes)
    }

    #[cfg(test)]
    pub(crate) fn counts(&self) -> (usize, usize, u64) {
        (self.records.len(), self.len, self.dropped)
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            LLMError::ConfigError("CPU profile output requires a UTF-8 file name".into())
        })?;
    let temporary = parent.join(format!(".{name}.{}.tmp", std::process::id()));
    let result = (|| -> std::io::Result<()> {
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        output.write_all(bytes)?;
        output.write_all(b"\n")?;
        output.sync_all()?;
        std::fs::hard_link(&temporary, path)?;
        std::fs::remove_file(&temporary)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result.map_err(Into::into)
}

fn repository_identity() -> (String, bool) {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|value| value.status.success())
        .and_then(|value| String::from_utf8(value.stdout).ok())
        .map(|value| value.trim().to_owned())
        .unwrap_or_else(|| "unknown".into());
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .is_some_and(|value| value.status.success() && !value.stdout.is_empty());
    (commit, dirty)
}

fn unix_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |value| value.as_nanos())
}

const fn kernel_code(value: KernelPath) -> u8 {
    match value {
        KernelPath::Auto => 0,
        KernelPath::Scalar => 1,
        KernelPath::Avx2 => 2,
        KernelPath::Avx512Vnni => 3,
    }
}

const fn matrix_code(value: Mxfp4MatmulBackend) -> u8 {
    match value {
        Mxfp4MatmulBackend::Auto => 0,
        Mxfp4MatmulBackend::Scalar => 1,
        Mxfp4MatmulBackend::Avx2 => 2,
        Mxfp4MatmulBackend::Avx512Vnni => 3,
        Mxfp4MatmulBackend::AmxInt8 => 4,
    }
}

const fn saturating_u32(value: usize) -> u32 {
    if value > u32::MAX as usize {
        u32::MAX
    } else {
        value as u32
    }
}
const fn saturating_u64(value: u128) -> u64 {
    if value > u64::MAX as u128 {
        u64::MAX
    } else {
        value as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> CpuExecutionProfileMetadata {
        CpuExecutionProfileMetadata {
            model_revision: "test".into(),
            model_source_hashes: vec!["a".repeat(64)],
            workload_id: Some("fixture".into()),
            hardware_profile_key: "synthetic".into(),
            cpu_identity: serde_json::json!({"family": 6}),
            dispatch: serde_json::json!({"matrix": "scalar"}),
            xe_runtime: None,
        }
    }

    #[test]
    fn fixed_slab_drops_without_growing_and_labels_failed_batches() {
        let bytes = std::mem::size_of::<CpuExecutionProfileRecord>() * 2;
        let mut profiler = CpuExecutionProfiler::new(
            bytes,
            KernelPath::Scalar,
            KernelPath::Scalar,
            Mxfp4MatmulBackend::Scalar,
            4,
        )
        .unwrap();
        profiler.begin_batch(CpuProfilePhase::Mixed, 2, 1, 9);
        for _ in 0..3 {
            let start = profiler.start();
            profiler.append(
                start,
                CpuProfileRecordSpec {
                    operation: CpuProfileOperation::Routing,
                    m: 3,
                    ..CpuProfileRecordSpec::default()
                },
            );
        }
        profiler.finish_batch(false);
        assert_eq!(profiler.counts(), (2, 2, 1));
        let document = profiler.document(metadata()).unwrap();
        assert!(document.truncated);
        assert_eq!(document.records[0].phase, CpuProfilePhase::Mixed);
        assert_eq!(document.records[0].prefill_rows, 2);
        assert_eq!(document.records[0].decode_rows, 1);
        assert!(document
            .records
            .iter()
            .all(|record| record.transaction_state == CpuProfileTransactionState::Failed));
    }

    #[test]
    fn atomic_output_refuses_to_replace_an_existing_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("profile.json");
        std::fs::write(&path, b"existing").unwrap();
        let profiler = CpuExecutionProfiler::new(
            4096,
            KernelPath::Scalar,
            KernelPath::Scalar,
            Mxfp4MatmulBackend::Scalar,
            1,
        )
        .unwrap();
        assert!(profiler.write(&path, metadata()).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"existing");
    }
}
