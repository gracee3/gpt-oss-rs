#![deny(unsafe_op_in_unsafe_fn)]
//! Internal Intel Iris Xe acceleration for validated GPT-OSS MXFP4 expert
//! projections.
//!
//! The crate is deliberately not published. OpenCL is resolved at runtime, so
//! binaries built with this crate retain no OpenCL link dependency. All unsafe
//! code is confined to the platform loader module.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[cfg(target_os = "linux")]
mod opencl;

/// Exact X8 source bytes. Do not copy or normalize this file: its hash is part
/// of the runtime and native-cache identity.
pub const KERNEL_SOURCE: &[u8] = include_bytes!("../../../tools/xe-research/kernels/mxfp4.cl");
/// Immutable X8 ABI v2 bytes.
pub const KERNEL_ABI_V2: &[u8] =
    include_bytes!("../../../tools/xe-research/fixtures/kernel-abi-v2.json");
pub const PROMOTION_RECORD_BYTES: &[u8] = include_bytes!("../promotion-record.json");

pub const KERNEL_SOURCE_SHA256: &str =
    "dd467aa3fed7a4a4f5ef5c811bf478bf392eb943c633c2480fce0ac101dfedf6";
pub const KERNEL_ABI_SHA256: &str =
    "62f13d73158f6b136993f5031b75f14cce2484cddb8482d2f0972119949bfcbe";
pub const BUILD_OPTIONS: &str = "-cl-std=CL3.0 -DXE_ENABLE_DP4A=1";
pub const EXPECTED_VENDOR_ID: u32 = 0x8086;
pub const EXPECTED_DEVICE_ID: u32 = 0x9a49;
pub const XE_TILE: usize = 32;
pub const XE_WEIGHT_PLANES: usize = 17;
pub const XE_ACTIVATION_RECORD_BYTES: usize = 72;
pub const DEFAULT_MAX_RESIDENT_MIB: usize = 128;
pub const AUTO_MIN_ROWS: usize = 4;
pub const WORKGROUP_SIZE: usize = 32;

pub const PROJECTION_TOTAL: &str = "gpt_oss_xe_projection_total";
pub const PHASE_DURATION_SECONDS: &str = "gpt_oss_xe_phase_duration_seconds";
pub const TRANSFER_BYTES_TOTAL: &str = "gpt_oss_xe_transfer_bytes_total";
pub const CIRCUIT_BREAKER: &str = "gpt_oss_xe_circuit_breaker";
pub const EXPERT_CACHE_TOTAL: &str = "gpt_oss_xe_expert_cache_total";
pub const EXPERT_CACHE_RESIDENT_BYTES: &str = "gpt_oss_xe_expert_cache_resident_bytes";

// A runtime OpenCL fault disables every current and future attachment in this
// process. Restart is the only production reset boundary.
static PROCESS_XE_CIRCUIT_OPEN: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum XeError {
    #[error("Xe is unsupported on this platform: {0}")]
    Unsupported(String),
    #[error("Xe capability validation failed: {0}")]
    Capability(String),
    #[error("Xe artifact validation failed: {0}")]
    Artifact(String),
    #[error("Xe projection dimensions are invalid: {0}")]
    Dimensions(String),
    #[error("Xe resident memory limit is insufficient: {0}")]
    ResidentLimit(String),
    #[error("Xe OpenCL operation failed: {0}")]
    Runtime(String),
    #[error("Xe circuit breaker is open")]
    CircuitOpen,
    #[error("Xe shutdown failed: {0}")]
    Shutdown(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentMode {
    Automatic,
    Explicit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationClass {
    ValidatedAutomatic,
    ValidatedExplicit,
    UnvalidatedExplicit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionRole {
    GateUp,
    Down,
}

impl ProjectionRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GateUp => "gate_up",
            Self::Down => "down",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelVariant {
    Tile32M1,
    Tile32M2,
    Tile32M4,
}

impl KernelVariant {
    pub const fn entry_point(self) -> &'static str {
        match self {
            Self::Tile32M1 => "mxfp4_tile32_m1_v2",
            Self::Tile32M2 => "mxfp4_tile32_m2_v2",
            Self::Tile32M4 => "mxfp4_tile32_m4_v2",
        }
    }

    pub const fn rows_per_dispatch(self) -> usize {
        match self {
            Self::Tile32M1 => 1,
            Self::Tile32M2 => 2,
            Self::Tile32M4 => 4,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromotionRecord {
    pub schema: String,
    pub automatic_enabled: bool,
    pub reason: String,
    pub pci_vendor_id: String,
    pub pci_device_id: String,
    pub driver_version: String,
    pub opencl_loader_sha256: String,
    pub opencl_driver_sha256: String,
    pub igc_sha256: String,
    pub kernel_source_sha256: String,
    pub kernel_abi_sha256: String,
    pub build_options: String,
    pub gate_up_min_rows: usize,
    pub down_min_rows: usize,
    pub workgroup_size: usize,
    pub evidence: serde_json::Value,
}

pub fn promotion_record() -> Result<PromotionRecord, XeError> {
    let record: PromotionRecord = serde_json::from_slice(PROMOTION_RECORD_BYTES)
        .map_err(|error| XeError::Artifact(format!("invalid promotion record: {error}")))?;
    validate_embedded_artifacts(&record)?;
    Ok(record)
}

pub fn automatic_promotion_enabled() -> bool {
    promotion_record().is_ok_and(|record| record.automatic_enabled)
}

fn validate_embedded_artifacts(record: &PromotionRecord) -> Result<(), XeError> {
    let source_hash = sha256_bytes(KERNEL_SOURCE);
    let abi_hash = sha256_bytes(KERNEL_ABI_V2);
    if source_hash != KERNEL_SOURCE_SHA256 || source_hash != record.kernel_source_sha256 {
        return Err(XeError::Artifact(format!(
            "kernel source hash {source_hash} does not match the immutable record"
        )));
    }
    if abi_hash != KERNEL_ABI_SHA256 || abi_hash != record.kernel_abi_sha256 {
        return Err(XeError::Artifact(format!(
            "kernel ABI hash {abi_hash} does not match the immutable record"
        )));
    }
    let abi: serde_json::Value = serde_json::from_slice(KERNEL_ABI_V2)
        .map_err(|error| XeError::Artifact(format!("invalid ABI v2 JSON: {error}")))?;
    if record.schema != "gpt-oss-rs.xe-auto-promotion/v1"
        || record.pci_vendor_id != format!("{EXPECTED_VENDOR_ID:04x}")
        || record.pci_device_id != format!("{EXPECTED_DEVICE_ID:04x}")
        || abi.get("schema").and_then(serde_json::Value::as_str)
            != Some("gpt-oss-rs.xe-kernel-abi/v2")
        || record.build_options != BUILD_OPTIONS
        || record.gate_up_min_rows != AUTO_MIN_ROWS
        || record.down_min_rows != AUTO_MIN_ROWS
        || record.workgroup_size != WORKGROUP_SIZE
    {
        return Err(XeError::Artifact(
            "promotion record and immutable device/ABI/dispatch policy disagree".into(),
        ));
    }
    if record.automatic_enabled
        && record
            .evidence
            .get("production_gate")
            .and_then(serde_json::Value::as_str)
            != Some("pass")
    {
        return Err(XeError::Artifact(
            "automatic Xe selection requires a passing production gate".into(),
        ));
    }
    for entry in [
        "mxfp4_tile32_m1_v2",
        "mxfp4_tile32_m2_v2",
        "mxfp4_tile32_m4_v2",
    ] {
        if abi
            .pointer("/entry_points")
            .and_then(|entries| entries.get(entry))
            .is_none()
        {
            return Err(XeError::Artifact(format!(
                "immutable ABI v2 is missing {entry}"
            )));
        }
    }
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ActivationRecordV2 {
    pub primary: [i8; 32],
    pub residual: [i8; 32],
    pub primary_scale: f32,
    pub residual_scale: f32,
}

const _: () = assert!(std::mem::size_of::<ActivationRecordV2>() == XE_ACTIVATION_RECORD_BYTES);

#[derive(Debug, Clone)]
pub struct AttachConfig {
    pub mode: AttachmentMode,
    pub cache_root: PathBuf,
    pub max_resident_bytes: usize,
    pub expert_cache_bytes: usize,
    pub max_columns: usize,
    pub max_blocks: usize,
}

impl AttachConfig {
    pub fn new(
        mode: AttachmentMode,
        cache_root: impl Into<PathBuf>,
        max_resident_bytes: usize,
        max_columns: usize,
        max_blocks: usize,
    ) -> Self {
        Self {
            mode,
            cache_root: cache_root.into(),
            max_resident_bytes,
            expert_cache_bytes: 0,
            max_columns,
            max_blocks,
        }
    }

    pub fn with_expert_cache_bytes(mut self, bytes: usize) -> Self {
        self.expert_cache_bytes = bytes;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XeIdentity {
    pub pci_vendor_id: String,
    pub pci_device_id: String,
    pub driver_version: String,
    pub device_version: String,
    pub opencl_loader_sha256: String,
    pub opencl_driver_sha256: String,
    pub igc_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XeMemoryDescriptor {
    pub max_resident_bytes: usize,
    pub device_resident_bytes: usize,
    pub host_staging_bound_bytes: usize,
    pub weight_capacity_bytes: usize,
    pub bias_capacity_bytes: usize,
    pub activation_capacity_bytes: usize,
    pub output_capacity_bytes: usize,
    pub max_rows_per_chunk: usize,
    pub expert_cache_capacity_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XeDescriptor {
    pub effective_backend: String,
    pub validation_class: ValidationClass,
    pub identity: XeIdentity,
    pub source_sha256: String,
    pub abi_sha256: String,
    pub build_options: String,
    pub native_cache_key: String,
    pub native_cache_hit: bool,
    pub gate_up_min_rows: usize,
    pub down_min_rows: usize,
    pub workgroup_size: usize,
    pub memory: XeMemoryDescriptor,
    pub runtime_fault_policy: String,
}

#[derive(Debug)]
pub struct ProjectionRequest<'a> {
    pub role: ProjectionRole,
    pub rows: usize,
    pub columns: usize,
    pub blocks: usize,
    pub weights_v2: &'a [u8],
    pub activations_v2: &'a [ActivationRecordV2],
    pub bias: &'a [f32],
}

/// Stable model/tensor coordinates for one immutable expert projection.
/// Runtime, kernel, ABI, build, PCI, and driver identity are added internally.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExpertCacheIdentity {
    pub model_source_key: String,
    pub tensor_source_key: String,
    pub layer: u16,
    pub expert: u16,
    pub role: ProjectionRole,
    pub columns: usize,
    pub blocks: usize,
    pub weight_layout_version: u32,
}

#[derive(Debug)]
pub struct ResidentProjectionRequest<'a> {
    pub identity: ExpertCacheIdentity,
    pub role: ProjectionRole,
    pub rows: usize,
    pub columns: usize,
    pub blocks: usize,
    pub activations_v2: &'a [ActivationRecordV2],
    pub bias: &'a [f32],
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XeResidencyState {
    Hit,
    Miss,
    Bypass,
    Fault,
    #[default]
    Disabled,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct XeResidencyStats {
    pub capacity_bytes: usize,
    pub resident_bytes: usize,
    pub resident_high_water_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub bypasses: u64,
    pub evictions: u64,
    pub repacks_avoided: u64,
    pub upload_bytes_avoided: u64,
    pub uploaded_bytes: u64,
    pub faults: u64,
}

#[derive(Debug)]
pub struct ProjectionResult {
    pub output: Vec<f32>,
    pub residency: XeResidencyState,
    pub residency_stats: XeResidencyStats,
}

#[derive(Debug)]
struct LruEntry<K, V> {
    key: K,
    value: V,
    bytes: usize,
    last_used: u64,
}

/// Small deterministic LRU ledger. Device objects stay runtime-owned and are
/// returned to the caller exactly once on eviction or shutdown.
#[derive(Debug)]
pub(crate) struct BoundedLru<K, V> {
    entries: Vec<LruEntry<K, V>>,
    tick: u64,
    stats: XeResidencyStats,
}

impl<K: Eq, V: Copy> BoundedLru<K, V> {
    pub(crate) fn new(capacity_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            tick: 0,
            stats: XeResidencyStats {
                capacity_bytes,
                ..XeResidencyStats::default()
            },
        }
    }

    pub(crate) fn lookup(&mut self, key: &K, expected_bytes: usize) -> Option<V> {
        self.tick = self.tick.saturating_add(1);
        if let Some(entry) = self.entries.iter_mut().find(|entry| &entry.key == key) {
            entry.last_used = self.tick;
            self.stats.hits = self.stats.hits.saturating_add(1);
            self.stats.repacks_avoided = self.stats.repacks_avoided.saturating_add(1);
            self.stats.upload_bytes_avoided = self
                .stats
                .upload_bytes_avoided
                .saturating_add(expected_bytes as u64);
            Some(entry.value)
        } else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            None
        }
    }

    pub(crate) fn can_reside(&self, bytes: usize) -> bool {
        bytes <= self.stats.capacity_bytes && bytes != 0
    }

    pub(crate) fn record_bypass(&mut self) {
        self.stats.bypasses = self.stats.bypasses.saturating_add(1);
    }

    pub(crate) fn record_fault(&mut self) {
        self.stats.faults = self.stats.faults.saturating_add(1);
    }

    pub(crate) fn record_upload(&mut self, bytes: usize) {
        self.stats.uploaded_bytes = self.stats.uploaded_bytes.saturating_add(bytes as u64);
    }

    /// Reserve logical capacity before a device allocation is attempted. This
    /// prevents a miss from transiently exceeding the configured bound.
    pub(crate) fn evict_for(&mut self, bytes: usize) -> Vec<V> {
        debug_assert!(self.can_reside(bytes));
        let mut evicted = Vec::new();
        while self.stats.resident_bytes.saturating_add(bytes) > self.stats.capacity_bytes {
            let index = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(index, entry)| (entry.last_used, *index))
                .map(|(index, _)| index)
                .expect("an entry must exist when resident bytes exceed capacity");
            let entry = self.entries.remove(index);
            self.stats.resident_bytes -= entry.bytes;
            self.stats.evictions = self.stats.evictions.saturating_add(1);
            evicted.push(entry.value);
        }
        evicted
    }

    pub(crate) fn insert_reserved(&mut self, key: K, value: V, bytes: usize) {
        debug_assert!(self.can_reside(bytes));
        debug_assert!(self.stats.resident_bytes.saturating_add(bytes) <= self.stats.capacity_bytes);
        self.tick = self.tick.saturating_add(1);
        self.stats.resident_bytes += bytes;
        self.stats.resident_high_water_bytes = self
            .stats
            .resident_high_water_bytes
            .max(self.stats.resident_bytes);
        self.record_upload(bytes);
        self.entries.push(LruEntry {
            key,
            value,
            bytes,
            last_used: self.tick,
        });
    }

    pub(crate) const fn stats(&self) -> XeResidencyStats {
        self.stats
    }

    pub(crate) fn reset_measurements(&mut self) {
        self.stats = XeResidencyStats {
            capacity_bytes: self.stats.capacity_bytes,
            resident_bytes: self.stats.resident_bytes,
            resident_high_water_bytes: self.stats.resident_bytes,
            ..XeResidencyStats::default()
        };
    }

    pub(crate) fn clear(&mut self) -> Vec<V> {
        self.stats.resident_bytes = 0;
        self.entries.drain(..).map(|entry| entry.value).collect()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct PhaseTiming {
    weight: Duration,
    repack: Duration,
    upload: Duration,
    activation: Duration,
    submit_wait: Duration,
    readback: Duration,
    residency: XeResidencyState,
    uploaded_weight_bias_bytes: usize,
}

trait ProjectionRuntime: Send {
    fn descriptor(&self) -> &XeDescriptor;
    fn project(
        &mut self,
        request: &ProjectionRequest<'_>,
        variant: KernelVariant,
        output: &mut [f32],
    ) -> Result<PhaseTiming, XeError>;
    fn project_resident(
        &mut self,
        request: &ResidentProjectionRequest<'_>,
        variant: KernelVariant,
        output: &mut [f32],
        repack: &mut dyn FnMut() -> Result<Vec<u8>, XeError>,
    ) -> Result<PhaseTiming, XeError> {
        let weights = repack()?;
        self.project(
            &ProjectionRequest {
                role: request.role,
                rows: request.rows,
                columns: request.columns,
                blocks: request.blocks,
                weights_v2: &weights,
                activations_v2: request.activations_v2,
                bias: request.bias,
            },
            variant,
            output,
        )
    }
    fn residency_stats(&self) -> XeResidencyStats {
        XeResidencyStats::default()
    }
    fn record_residency_fault(&mut self) {}
    fn drain(&mut self) -> Result<(), XeError>;
    fn shutdown(&mut self) -> Result<(), XeError>;
}

/// A non-cloneable, process-attached Xe engine. Its in-order queue is guarded
/// by a mutex and every operation is synchronous through a terminal event.
pub struct XeProjectionEngine {
    runtime: Mutex<Box<dyn ProjectionRuntime>>,
    gate_up_min_rows: usize,
    down_min_rows: usize,
}

impl std::fmt::Debug for XeProjectionEngine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("XeProjectionEngine")
            .field(
                "circuit_open",
                &PROCESS_XE_CIRCUIT_OPEN.load(Ordering::Acquire),
            )
            .finish_non_exhaustive()
    }
}

impl XeProjectionEngine {
    pub fn attach(config: AttachConfig) -> Result<Self, XeError> {
        if PROCESS_XE_CIRCUIT_OPEN.load(Ordering::Acquire) {
            return Err(XeError::CircuitOpen);
        }
        let record = promotion_record()?;
        if config.mode == AttachmentMode::Automatic && !record.automatic_enabled {
            return Err(XeError::Unsupported(format!(
                "automatic Xe dispatch is not promoted: {}",
                record.reason
            )));
        }
        validate_shape_capacity(&config)?;
        #[cfg(target_os = "linux")]
        let runtime = opencl::OpenClRuntime::attach(&config, &record)?;
        #[cfg(not(target_os = "linux"))]
        let runtime = return Err(XeError::Unsupported(format!(
            "runtime OpenCL loading is not implemented for {}",
            std::env::consts::OS
        )));
        let engine = Self {
            runtime: Mutex::new(Box::new(runtime)),
            gate_up_min_rows: record.gate_up_min_rows,
            down_min_rows: record.down_min_rows,
        };
        metrics::gauge!(CIRCUIT_BREAKER).set(0.0);
        Ok(engine)
    }

    pub fn descriptor(&self) -> Result<XeDescriptor, XeError> {
        self.runtime
            .lock()
            .map_err(|_| XeError::Runtime("Xe queue mutex is poisoned".into()))
            .map(|runtime| runtime.descriptor().clone())
    }

    pub fn should_accelerate(&self, role: ProjectionRole, rows: usize) -> bool {
        !PROCESS_XE_CIRCUIT_OPEN.load(Ordering::Acquire)
            && rows
                >= match role {
                    ProjectionRole::GateUp => self.gate_up_min_rows,
                    ProjectionRole::Down => self.down_min_rows,
                }
    }

    pub fn project(&self, request: ProjectionRequest<'_>) -> Result<Vec<f32>, XeError> {
        validate_request(&request)?;
        if request.rows < AUTO_MIN_ROWS {
            return Err(XeError::Dimensions(format!(
                "automatic hybrid policy keeps M={} on CPU",
                request.rows
            )));
        }
        let variant = KernelVariant::Tile32M4;
        self.execute_projection(request, variant)
            .map(|result| result.output)
    }

    /// Forced-only resident projection. The repack closure is invoked exactly
    /// once on a miss or bypass and is never invoked on a cache hit.
    pub fn project_resident(
        &self,
        request: ResidentProjectionRequest<'_>,
        mut repack: impl FnMut() -> Result<Vec<u8>, XeError>,
    ) -> Result<ProjectionResult, XeError> {
        validate_resident_request(&request)?;
        if request.rows < AUTO_MIN_ROWS {
            return Err(XeError::Dimensions(format!(
                "explicit hybrid policy keeps M={} on CPU",
                request.rows
            )));
        }
        self.execute_resident_projection(request, KernelVariant::Tile32M4, &mut repack)
    }

    /// Explicit benchmark/test control for all three validated tile kernels.
    pub fn project_with_variant(
        &self,
        request: ProjectionRequest<'_>,
        variant: KernelVariant,
    ) -> Result<Vec<f32>, XeError> {
        validate_request(&request)?;
        let divisor = variant.rows_per_dispatch();
        if !request.rows.is_multiple_of(divisor) {
            return Err(XeError::Dimensions(format!(
                "{} requires rows divisible by {divisor}",
                variant.entry_point()
            )));
        }
        self.execute_projection(request, variant)
            .map(|result| result.output)
    }

    pub fn circuit_is_open(&self) -> bool {
        PROCESS_XE_CIRCUIT_OPEN.load(Ordering::Acquire)
    }

    pub fn shutdown(&self) -> Result<(), XeError> {
        self.runtime
            .lock()
            .map_err(|_| XeError::Shutdown("Xe queue mutex is poisoned".into()))?
            .shutdown()
    }

    pub fn residency_stats(&self) -> Result<XeResidencyStats, XeError> {
        self.runtime
            .lock()
            .map_err(|_| XeError::Runtime("Xe queue mutex is poisoned".into()))
            .map(|runtime| runtime.residency_stats())
    }

    fn execute_projection(
        &self,
        request: ProjectionRequest<'_>,
        variant: KernelVariant,
    ) -> Result<ProjectionResult, XeError> {
        if PROCESS_XE_CIRCUIT_OPEN.load(Ordering::Acquire) {
            return Err(XeError::CircuitOpen);
        }
        let output_len = request
            .rows
            .checked_mul(request.columns)
            .ok_or_else(|| XeError::Dimensions("output extent overflows".into()))?;
        let mut output = vec![0.0; output_len];
        let result = self
            .runtime
            .lock()
            .map_err(|_| XeError::Runtime("Xe queue mutex is poisoned".into()))
            .and_then(|mut runtime| runtime.project(&request, variant, &mut output));
        match result {
            Ok(timing) => {
                record_projection_metrics(&request, "xe", "ok", timing);
                let stats = self.residency_stats()?;
                Ok(ProjectionResult {
                    output,
                    residency: timing.residency,
                    residency_stats: stats,
                })
            }
            Err(error) => {
                if let Ok(mut runtime) = self.runtime.lock() {
                    let _ = runtime.drain();
                }
                PROCESS_XE_CIRCUIT_OPEN.store(true, Ordering::Release);
                metrics::gauge!(CIRCUIT_BREAKER).set(1.0);
                metrics::counter!(
                    PROJECTION_TOTAL,
                    "role" => request.role.as_str(),
                    "backend" => "xe",
                    "result" => "fault"
                )
                .increment(1);
                Err(error)
            }
        }
    }

    fn execute_resident_projection(
        &self,
        request: ResidentProjectionRequest<'_>,
        variant: KernelVariant,
        repack: &mut dyn FnMut() -> Result<Vec<u8>, XeError>,
    ) -> Result<ProjectionResult, XeError> {
        if PROCESS_XE_CIRCUIT_OPEN.load(Ordering::Acquire) {
            return Err(XeError::CircuitOpen);
        }
        let output_len = request
            .rows
            .checked_mul(request.columns)
            .ok_or_else(|| XeError::Dimensions("output extent overflows".into()))?;
        let mut output = vec![0.0; output_len];
        let result = self
            .runtime
            .lock()
            .map_err(|_| XeError::Runtime("Xe queue mutex is poisoned".into()))
            .and_then(|mut runtime| {
                runtime.project_resident(&request, variant, &mut output, repack)
            });
        match result {
            Ok(timing) => {
                record_resident_projection_metrics(&request, "xe", "ok", timing);
                let stats = self.residency_stats()?;
                Ok(ProjectionResult {
                    output,
                    residency: timing.residency,
                    residency_stats: stats,
                })
            }
            Err(error) => {
                if let Ok(mut runtime) = self.runtime.lock() {
                    runtime.record_residency_fault();
                    let _ = runtime.drain();
                }
                PROCESS_XE_CIRCUIT_OPEN.store(true, Ordering::Release);
                metrics::gauge!(CIRCUIT_BREAKER).set(1.0);
                metrics::counter!(
                    PROJECTION_TOTAL,
                    "role" => request.role.as_str(),
                    "backend" => "xe",
                    "result" => "fault"
                )
                .increment(1);
                Err(error)
            }
        }
    }
}

impl Drop for XeProjectionEngine {
    fn drop(&mut self) {
        if let Ok(runtime) = self.runtime.get_mut() {
            let _ = runtime.shutdown();
        }
    }
}

fn record_projection_metrics(
    request: &ProjectionRequest<'_>,
    backend: &'static str,
    result: &'static str,
    timing: PhaseTiming,
) {
    metrics::counter!(
        PROJECTION_TOTAL,
        "role" => request.role.as_str(),
        "backend" => backend,
        "result" => result
    )
    .increment(1);
    for (phase, duration) in [
        ("weight", timing.weight),
        ("repack", timing.repack),
        ("upload", timing.upload),
        ("activation", timing.activation),
        ("submit_wait", timing.submit_wait),
        ("readback", timing.readback),
    ] {
        metrics::histogram!(
            PHASE_DURATION_SECONDS,
            "role" => request.role.as_str(),
            "phase" => phase
        )
        .record(duration.as_secs_f64());
    }
    for (direction, bytes) in [
        ("weights_bias_uploaded", timing.uploaded_weight_bias_bytes),
        ("activations", std::mem::size_of_val(request.activations_v2)),
        ("output", request.rows * request.columns * 4),
    ] {
        metrics::counter!(
            TRANSFER_BYTES_TOTAL,
            "role" => request.role.as_str(),
            "direction" => direction
        )
        .increment(bytes as u64);
    }
    record_residency_metrics(request.role, timing.residency);
}

fn record_resident_projection_metrics(
    request: &ResidentProjectionRequest<'_>,
    backend: &'static str,
    result: &'static str,
    timing: PhaseTiming,
) {
    metrics::counter!(
        PROJECTION_TOTAL,
        "role" => request.role.as_str(),
        "backend" => backend,
        "result" => result
    )
    .increment(1);
    for (phase, duration) in [
        ("weight", timing.weight),
        ("repack", timing.repack),
        ("upload", timing.upload),
        ("activation", timing.activation),
        ("submit_wait", timing.submit_wait),
        ("readback", timing.readback),
    ] {
        metrics::histogram!(
            PHASE_DURATION_SECONDS,
            "role" => request.role.as_str(),
            "phase" => phase
        )
        .record(duration.as_secs_f64());
    }
    for (direction, bytes) in [
        ("weights_bias_uploaded", timing.uploaded_weight_bias_bytes),
        ("activations", std::mem::size_of_val(request.activations_v2)),
        ("output", request.rows * request.columns * 4),
    ] {
        metrics::counter!(
            TRANSFER_BYTES_TOTAL,
            "role" => request.role.as_str(),
            "direction" => direction
        )
        .increment(bytes as u64);
    }
    record_residency_metrics(request.role, timing.residency);
}

fn record_residency_metrics(role: ProjectionRole, state: XeResidencyState) {
    let state = match state {
        XeResidencyState::Hit => "hit",
        XeResidencyState::Miss => "miss",
        XeResidencyState::Bypass => "bypass",
        XeResidencyState::Fault => "fault",
        XeResidencyState::Disabled => "disabled",
    };
    metrics::counter!(
        EXPERT_CACHE_TOTAL,
        "role" => role.as_str(),
        "result" => state
    )
    .increment(1);
}

pub fn record_cpu_fallback(role: ProjectionRole) {
    metrics::counter!(
        PROJECTION_TOTAL,
        "role" => role.as_str(),
        "backend" => "cpu",
        "result" => "fallback"
    )
    .increment(1);
}

fn validate_shape_capacity(config: &AttachConfig) -> Result<(), XeError> {
    if config.expert_cache_bytes != 0 && config.mode != AttachmentMode::Explicit {
        return Err(XeError::Dimensions(
            "Xe expert residency is legal only for explicit attachment".into(),
        ));
    }
    if config.max_columns == 0
        || config.max_blocks == 0
        || !config.max_columns.is_multiple_of(XE_TILE)
    {
        return Err(XeError::Dimensions(
            "maximum columns/blocks must be non-zero and columns divisible by 32".into(),
        ));
    }
    let weight = config
        .max_columns
        .checked_mul(config.max_blocks)
        .and_then(|value| value.checked_mul(XE_WEIGHT_PLANES))
        .ok_or_else(|| XeError::Dimensions("weight capacity overflows".into()))?;
    let bias = config
        .max_columns
        .checked_mul(4)
        .ok_or_else(|| XeError::Dimensions("bias capacity overflows".into()))?;
    let one_row = config
        .max_blocks
        .checked_mul(XE_ACTIVATION_RECORD_BYTES)
        .and_then(|value| value.checked_add(config.max_columns * 4))
        .ok_or_else(|| XeError::Dimensions("streaming row capacity overflows".into()))?;
    let fixed = weight
        .checked_add(bias)
        .ok_or_else(|| XeError::Dimensions("fixed resident capacity overflows".into()))?;
    let minimum =
        fixed
            .checked_add(one_row.checked_mul(4).ok_or_else(|| {
                XeError::Dimensions("minimum streaming capacity overflows".into())
            })?)
            .ok_or_else(|| XeError::Dimensions("minimum resident capacity overflows".into()))?;
    if config.max_resident_bytes < minimum {
        return Err(XeError::ResidentLimit(format!(
            "{} bytes cannot hold the largest expert and four streaming rows; need at least {minimum}",
            config.max_resident_bytes
        )));
    }
    config
        .max_resident_bytes
        .checked_add(config.expert_cache_bytes)
        .ok_or_else(|| XeError::Dimensions("combined Xe resident capacity overflows".into()))?;
    Ok(())
}

fn validate_request(request: &ProjectionRequest<'_>) -> Result<(), XeError> {
    if request.rows == 0
        || request.columns == 0
        || request.blocks == 0
        || !request.columns.is_multiple_of(XE_TILE)
    {
        return Err(XeError::Dimensions(
            "rows/columns/blocks must be non-zero and columns divisible by 32".into(),
        ));
    }
    let expected_weights = request
        .columns
        .checked_mul(request.blocks)
        .and_then(|value| value.checked_mul(XE_WEIGHT_PLANES))
        .ok_or_else(|| XeError::Dimensions("weight extent overflows".into()))?;
    let expected_activations = request
        .rows
        .checked_mul(request.blocks)
        .ok_or_else(|| XeError::Dimensions("activation extent overflows".into()))?;
    if request.weights_v2.len() != expected_weights
        || request.activations_v2.len() != expected_activations
        || request.bias.len() != request.columns
    {
        return Err(XeError::Dimensions(format!(
            "extent mismatch: weights {}/{expected_weights}, activations {}/{expected_activations}, bias {}/{}",
            request.weights_v2.len(),
            request.activations_v2.len(),
            request.bias.len(),
            request.columns
        )));
    }
    Ok(())
}

fn validate_resident_request(request: &ResidentProjectionRequest<'_>) -> Result<(), XeError> {
    if request.identity.role != request.role
        || request.identity.columns != request.columns
        || request.identity.blocks != request.blocks
        || request.rows == 0
        || request.columns == 0
        || request.blocks == 0
        || !request.columns.is_multiple_of(XE_TILE)
    {
        return Err(XeError::Dimensions(
            "resident request identity and dimensions disagree".into(),
        ));
    }
    let expected_activations = request
        .rows
        .checked_mul(request.blocks)
        .ok_or_else(|| XeError::Dimensions("activation extent overflows".into()))?;
    if request.activations_v2.len() != expected_activations || request.bias.len() != request.columns
    {
        return Err(XeError::Dimensions(
            "resident activation or bias extent mismatch".into(),
        ));
    }
    Ok(())
}

pub fn repack_v2<F>(columns: usize, blocks: usize, mut block: F) -> Result<Vec<u8>, XeError>
where
    F: FnMut(usize, usize) -> Result<(u8, [u8; 16]), XeError>,
{
    if columns == 0 || blocks == 0 || !columns.is_multiple_of(XE_TILE) {
        return Err(XeError::Dimensions(
            "Xe v2 repack requires non-zero dimensions and columns divisible by 32".into(),
        ));
    }
    let len = columns
        .checked_mul(blocks)
        .and_then(|value| value.checked_mul(XE_WEIGHT_PLANES))
        .ok_or_else(|| XeError::Dimensions("Xe v2 repack extent overflows".into()))?;
    let mut output = vec![0_u8; len];
    for tile in 0..columns / XE_TILE {
        for k_block in 0..blocks {
            let destination = (tile * blocks + k_block) * XE_WEIGHT_PLANES * XE_TILE;
            for lane in 0..XE_TILE {
                let (scale, packed) = block(tile * XE_TILE + lane, k_block)?;
                output[destination + lane] = scale;
                for (byte, value) in packed.into_iter().enumerate() {
                    output[destination + (byte + 1) * XE_TILE + lane] = value;
                }
            }
        }
    }
    Ok(output)
}

pub fn register_metric_descriptions() {
    metrics::describe_counter!(PROJECTION_TOTAL, "Bounded CPU+Xe projection outcomes");
    metrics::describe_histogram!(
        PHASE_DURATION_SECONDS,
        metrics::Unit::Seconds,
        "Xe projection phase durations"
    );
    metrics::describe_counter!(
        TRANSFER_BYTES_TOTAL,
        metrics::Unit::Bytes,
        "Xe projection transfer bytes"
    );
    metrics::describe_gauge!(CIRCUIT_BREAKER, "Process-wide Xe circuit breaker state");
    metrics::describe_counter!(EXPERT_CACHE_TOTAL, "Bounded Xe expert residency outcomes");
    metrics::describe_gauge!(
        EXPERT_CACHE_RESIDENT_BYTES,
        metrics::Unit::Bytes,
        "Current Xe expert-cache resident bytes"
    );
}

/// Lightweight exact-stack probe used only if the checked-in promotion record
/// enables automatic dispatch.
pub fn probe_automatic(cache_root: &Path) -> Result<XeDescriptor, XeError> {
    if !automatic_promotion_enabled() {
        return Err(XeError::Unsupported(
            "checked-in Xe automatic promotion record is disabled".into(),
        ));
    }
    let engine = XeProjectionEngine::attach(AttachConfig::new(
        AttachmentMode::Automatic,
        cache_root,
        1024 * 1024,
        32,
        1,
    ))?;
    engine.descriptor()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::Arc;

    static TEST_PROCESS_CIRCUIT_LOCK: Mutex<()> = Mutex::new(());

    #[derive(Default)]
    struct FakeCounts {
        projects: AtomicUsize,
        drains: AtomicUsize,
        shutdowns: AtomicUsize,
        residency_faults: AtomicUsize,
    }

    struct FailingRuntime {
        descriptor: XeDescriptor,
        counts: Arc<FakeCounts>,
        failure: &'static str,
        shutdown_error: bool,
        shutdown: bool,
    }

    impl ProjectionRuntime for FailingRuntime {
        fn descriptor(&self) -> &XeDescriptor {
            &self.descriptor
        }

        fn project(
            &mut self,
            _request: &ProjectionRequest<'_>,
            _variant: KernelVariant,
            _output: &mut [f32],
        ) -> Result<PhaseTiming, XeError> {
            self.counts.projects.fetch_add(1, AtomicOrdering::SeqCst);
            Err(XeError::Runtime(format!(
                "injected {} failure",
                self.failure
            )))
        }

        fn drain(&mut self) -> Result<(), XeError> {
            self.counts.drains.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(())
        }

        fn record_residency_fault(&mut self) {
            self.counts
                .residency_faults
                .fetch_add(1, AtomicOrdering::SeqCst);
        }

        fn shutdown(&mut self) -> Result<(), XeError> {
            if !self.shutdown {
                self.shutdown = true;
                self.counts.shutdowns.fetch_add(1, AtomicOrdering::SeqCst);
                if self.shutdown_error {
                    return Err(XeError::Shutdown("injected teardown failure".into()));
                }
            }
            Ok(())
        }
    }

    fn fake_descriptor() -> XeDescriptor {
        XeDescriptor {
            effective_backend: "cpu_xe".into(),
            validation_class: ValidationClass::UnvalidatedExplicit,
            identity: XeIdentity {
                pci_vendor_id: "8086".into(),
                pci_device_id: "9a49".into(),
                driver_version: "test".into(),
                device_version: "test".into(),
                opencl_loader_sha256: "0".repeat(64),
                opencl_driver_sha256: "0".repeat(64),
                igc_sha256: "0".repeat(64),
            },
            source_sha256: KERNEL_SOURCE_SHA256.into(),
            abi_sha256: KERNEL_ABI_SHA256.into(),
            build_options: BUILD_OPTIONS.into(),
            native_cache_key: "test".into(),
            native_cache_hit: false,
            gate_up_min_rows: 4,
            down_min_rows: 4,
            workgroup_size: WORKGROUP_SIZE,
            memory: XeMemoryDescriptor {
                max_resident_bytes: 1 << 20,
                device_resident_bytes: 1 << 20,
                host_staging_bound_bytes: 1 << 20,
                weight_capacity_bytes: 544,
                bias_capacity_bytes: 128,
                activation_capacity_bytes: 288,
                output_capacity_bytes: 512,
                max_rows_per_chunk: 4,
                expert_cache_capacity_bytes: 0,
            },
            runtime_fault_policy: "test".into(),
        }
    }

    fn test_weights(columns: usize, blocks: usize) -> (Vec<u8>, Vec<[u8; 16]>) {
        let mut canonical = vec![[0_u8; 16]; columns * blocks];
        let packed = repack_v2(columns, blocks, |column, block| {
            let mut bytes = [0_u8; 16];
            for (index, value) in bytes.iter_mut().enumerate() {
                let low = ((column + block + index) % 15) as u8;
                let high = ((column * 3 + block + index + 1) % 15) as u8;
                *value = low | (high << 4);
            }
            canonical[column * blocks + block] = bytes;
            Ok((127, bytes))
        })
        .unwrap();
        (packed, canonical)
    }

    fn test_activations(rows: usize, blocks: usize) -> Vec<ActivationRecordV2> {
        let mut activations = vec![ActivationRecordV2::zeroed(); rows * blocks];
        for row in 0..rows {
            for block in 0..blocks {
                let record = &mut activations[row * blocks + block];
                for lane in 0..32 {
                    record.primary[lane] = ((row + block + lane) % 29) as i8 - 14;
                    record.residual[lane] = ((row * 3 + block + lane) % 11) as i8 - 5;
                }
                record.primary_scale = 0.03125 * ((block % 3) + 1) as f32;
                record.residual_scale = 0.0078125 * ((row % 3) + 1) as f32;
            }
        }
        activations
    }

    fn e2m1_x2(nibble: u8) -> i8 {
        const VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
        VALUES[(nibble & 0x0f) as usize]
    }

    fn scalar_projection(
        rows: usize,
        columns: usize,
        blocks: usize,
        canonical: &[[u8; 16]],
        activations: &[ActivationRecordV2],
        bias: &[f32],
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                let mut value = bias[column];
                for block in 0..blocks {
                    let record = &activations[row * blocks + block];
                    let bytes = canonical[column * blocks + block];
                    let mut primary = 0_i32;
                    let mut residual = 0_i32;
                    for (index, packed) in bytes.into_iter().enumerate() {
                        let low = e2m1_x2(packed) as i32;
                        let high = e2m1_x2(packed >> 4) as i32;
                        primary += low * record.primary[index * 2] as i32
                            + high * record.primary[index * 2 + 1] as i32;
                        residual += low * record.residual[index * 2] as i32
                            + high * record.residual[index * 2 + 1] as i32;
                    }
                    value += primary as f32 * 0.5 * record.primary_scale
                        + residual as f32 * 0.5 * record.residual_scale;
                }
                output[row * columns + column] = value;
            }
        }
        output
    }

    fn assert_projection_matches(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(actual.is_finite(), "non-finite output at {index}");
            let actual_ordered = actual.to_bits() as i32;
            let expected_ordered = expected.to_bits() as i32;
            let ulp = actual_ordered.abs_diff(expected_ordered);
            assert!(
                (actual - expected).abs() <= 1e-6 || ulp <= 4,
                "projection mismatch at {index}: {actual} != {expected} ({ulp} ULP)"
            );
            assert_eq!(
                half::bf16::from_f32(actual).to_bits(),
                half::bf16::from_f32(expected).to_bits(),
                "BF16 boundary mismatch at {index}"
            );
        }
    }

    #[test]
    fn embedded_hashes_and_abi_are_immutable() {
        let record = promotion_record().unwrap();
        assert_eq!(record.schema, "gpt-oss-rs.xe-auto-promotion/v1");
        assert!(!record.automatic_enabled);
    }

    #[test]
    fn automatic_record_requires_a_passing_production_gate() {
        let mut record: PromotionRecord = serde_json::from_slice(PROMOTION_RECORD_BYTES).unwrap();
        record.automatic_enabled = true;
        assert!(matches!(
            validate_embedded_artifacts(&record),
            Err(XeError::Artifact(message))
                if message.contains("passing production gate")
        ));
    }

    #[test]
    fn v2_layout_round_trips_every_plane_and_lane() {
        let records = repack_v2(64, 3, |row, block| {
            Ok(((row * 7 + block) as u8, [row as u8 + block as u8; 16]))
        })
        .unwrap();
        for row in 0..64 {
            for block in 0..3 {
                let tile = row / 32;
                let lane = row % 32;
                let base = (tile * 3 + block) * 17 * 32;
                assert_eq!(records[base + lane], (row * 7 + block) as u8);
                for byte in 0..16 {
                    assert_eq!(
                        records[base + (byte + 1) * 32 + lane],
                        row as u8 + block as u8
                    );
                }
            }
        }
    }

    #[test]
    fn activation_abi_is_exactly_72_bytes_and_eight_aligned() {
        assert_eq!(std::mem::size_of::<ActivationRecordV2>(), 72);
        assert_eq!(std::mem::align_of::<ActivationRecordV2>(), 8);
    }

    #[test]
    fn rejects_overflow_and_small_resident_caps() {
        let overflow = AttachConfig::new(
            AttachmentMode::Explicit,
            ".",
            usize::MAX,
            usize::MAX - 31,
            usize::MAX,
        );
        assert!(matches!(
            validate_shape_capacity(&overflow),
            Err(XeError::Dimensions(_))
        ));
        let small = AttachConfig::new(AttachmentMode::Explicit, ".", 1024, 5760, 90);
        assert!(matches!(
            validate_shape_capacity(&small),
            Err(XeError::ResidentLimit(_))
        ));
        let automatic_cache = AttachConfig::new(AttachmentMode::Automatic, ".", 1 << 20, 32, 1)
            .with_expert_cache_bytes(1024);
        assert!(matches!(
            validate_shape_capacity(&automatic_cache),
            Err(XeError::Dimensions(message)) if message.contains("explicit")
        ));
    }

    #[test]
    fn bounded_lru_is_exact_deterministic_and_identity_scoped() {
        let mut cache = BoundedLru::new(100);
        assert!(cache.lookup(&"a", 40).is_none());
        assert!(cache.evict_for(40).is_empty());
        cache.insert_reserved("a", 1_u8, 40);
        assert!(cache.evict_for(40).is_empty());
        cache.insert_reserved("b", 2_u8, 40);
        assert_eq!(cache.lookup(&"a", 40), Some(1));

        // a was just touched, so b is the deterministic eviction victim.
        assert_eq!(cache.evict_for(40), vec![2]);
        cache.insert_reserved("c", 3_u8, 40);
        assert_eq!(cache.lookup(&"b", 40), None);
        assert_eq!(cache.lookup(&"a", 40), Some(1));
        assert_eq!(cache.lookup(&"c", 40), Some(3));
        assert_eq!(cache.lookup(&"identity-drift", 40), None);
        let stats = cache.stats();
        assert_eq!(stats.capacity_bytes, 100);
        assert_eq!(stats.resident_bytes, 80);
        assert_eq!(stats.resident_high_water_bytes, 80);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.hits, 3);
        assert_eq!(stats.misses, 3);
        assert_eq!(stats.repacks_avoided, 3);
        assert_eq!(stats.upload_bytes_avoided, 120);
        assert_eq!(stats.uploaded_bytes, 120);
        assert_eq!(cache.clear().len(), 2);
        assert_eq!(cache.stats().resident_bytes, 0);
    }

    #[test]
    fn zero_or_too_small_cache_bypasses_without_residency() {
        let mut disabled = BoundedLru::<u8, u8>::new(0);
        assert!(!disabled.can_reside(1));
        disabled.record_bypass();
        assert_eq!(disabled.stats().bypasses, 1);
        assert_eq!(disabled.stats().resident_bytes, 0);

        let bounded = BoundedLru::<u8, u8>::new(16);
        assert!(!bounded.can_reside(17));
        assert!(bounded.can_reside(16));
    }

    #[test]
    fn cache_identity_separates_projection_roles_and_tensor_sources() {
        let identity = ExpertCacheIdentity {
            model_source_key: "model".into(),
            tensor_source_key: "gate".into(),
            layer: 1,
            expert: 2,
            role: ProjectionRole::GateUp,
            columns: 32,
            blocks: 1,
            weight_layout_version: 2,
        };
        let mut cache = BoundedLru::new(2048);
        cache.insert_reserved(identity.clone(), 7_u8, 672);
        let mut down = identity.clone();
        down.role = ProjectionRole::Down;
        down.tensor_source_key = "down".into();
        assert_eq!(cache.lookup(&down, 672), None);
        assert_eq!(cache.lookup(&identity, 672), Some(7));
    }

    #[test]
    fn malformed_requests_are_rejected_before_runtime() {
        let request = ProjectionRequest {
            role: ProjectionRole::GateUp,
            rows: 4,
            columns: 32,
            blocks: 1,
            weights_v2: &[],
            activations_v2: &[],
            bias: &[],
        };
        assert!(matches!(
            validate_request(&request),
            Err(XeError::Dimensions(_))
        ));
    }

    #[test]
    fn injected_runtime_stages_drain_trip_once_and_never_release_early() {
        let _guard = TEST_PROCESS_CIRCUIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let weights = vec![0_u8; 32 * 17];
        let activations = vec![ActivationRecordV2::zeroed(); 4];
        let bias = vec![0.0_f32; 32];
        let request = || ProjectionRequest {
            role: ProjectionRole::GateUp,
            rows: 4,
            columns: 32,
            blocks: 1,
            weights_v2: &weights,
            activations_v2: &activations,
            bias: &bias,
        };
        for failure in [
            "build",
            "allocation",
            "upload",
            "argument setup",
            "submit",
            "wait",
            "readback",
        ] {
            PROCESS_XE_CIRCUIT_OPEN.store(false, Ordering::Release);
            let counts = Arc::new(FakeCounts::default());
            let engine = XeProjectionEngine {
                runtime: Mutex::new(Box::new(FailingRuntime {
                    descriptor: fake_descriptor(),
                    counts: counts.clone(),
                    failure,
                    shutdown_error: false,
                    shutdown: false,
                })),
                gate_up_min_rows: AUTO_MIN_ROWS,
                down_min_rows: AUTO_MIN_ROWS,
            };
            assert!(matches!(
                engine.project(request()),
                Err(XeError::Runtime(message)) if message.contains(failure)
            ));
            assert!(engine.circuit_is_open());
            assert_eq!(counts.projects.load(AtomicOrdering::SeqCst), 1);
            assert_eq!(counts.drains.load(AtomicOrdering::SeqCst), 1);
            assert_eq!(counts.shutdowns.load(AtomicOrdering::SeqCst), 0);
            assert_eq!(engine.project(request()), Err(XeError::CircuitOpen));
            assert_eq!(counts.projects.load(AtomicOrdering::SeqCst), 1);
            engine.shutdown().unwrap();
            engine.shutdown().unwrap();
            assert_eq!(counts.shutdowns.load(AtomicOrdering::SeqCst), 1);
        }
        assert!(matches!(
            XeProjectionEngine::attach(AttachConfig::new(
                AttachmentMode::Explicit,
                ".",
                1 << 20,
                32,
                1,
            )),
            Err(XeError::CircuitOpen)
        ));
        PROCESS_XE_CIRCUIT_OPEN.store(false, Ordering::Release);
        metrics::gauge!(CIRCUIT_BREAKER).set(0.0);
    }

    #[test]
    fn resident_fault_drains_opens_breaker_and_never_retries_repack() {
        let _guard = TEST_PROCESS_CIRCUIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        PROCESS_XE_CIRCUIT_OPEN.store(false, Ordering::Release);
        let counts = Arc::new(FakeCounts::default());
        let engine = XeProjectionEngine {
            runtime: Mutex::new(Box::new(FailingRuntime {
                descriptor: fake_descriptor(),
                counts: counts.clone(),
                failure: "resident submit",
                shutdown_error: false,
                shutdown: false,
            })),
            gate_up_min_rows: AUTO_MIN_ROWS,
            down_min_rows: AUTO_MIN_ROWS,
        };
        let activations = vec![ActivationRecordV2::zeroed(); 4];
        let bias = vec![0.0_f32; 32];
        let request = ResidentProjectionRequest {
            identity: ExpertCacheIdentity {
                model_source_key: "model".into(),
                tensor_source_key: "tensor".into(),
                layer: 0,
                expert: 0,
                role: ProjectionRole::GateUp,
                columns: 32,
                blocks: 1,
                weight_layout_version: 2,
            },
            role: ProjectionRole::GateUp,
            rows: 4,
            columns: 32,
            blocks: 1,
            activations_v2: &activations,
            bias: &bias,
        };
        let repacks = AtomicUsize::new(0);
        assert!(matches!(
            engine.project_resident(request, || {
                repacks.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(vec![0_u8; 32 * XE_WEIGHT_PLANES])
            }),
            Err(XeError::Runtime(message)) if message.contains("resident submit")
        ));
        assert!(engine.circuit_is_open());
        assert_eq!(repacks.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(counts.projects.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(counts.drains.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(counts.residency_faults.load(AtomicOrdering::SeqCst), 1);
        engine.shutdown().unwrap();
        PROCESS_XE_CIRCUIT_OPEN.store(false, Ordering::Release);
        metrics::gauge!(CIRCUIT_BREAKER).set(0.0);
    }

    #[test]
    fn injected_teardown_failure_is_reported_once_and_shutdown_remains_idempotent() {
        let counts = Arc::new(FakeCounts::default());
        let engine = XeProjectionEngine {
            runtime: Mutex::new(Box::new(FailingRuntime {
                descriptor: fake_descriptor(),
                counts: counts.clone(),
                failure: "unused",
                shutdown_error: true,
                shutdown: false,
            })),
            gate_up_min_rows: AUTO_MIN_ROWS,
            down_min_rows: AUTO_MIN_ROWS,
        };
        assert!(matches!(engine.shutdown(), Err(XeError::Shutdown(_))));
        engine.shutdown().unwrap();
        assert_eq!(counts.shutdowns.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn opt_in_live_attachment_runs_startup_numerical_test_and_shutdown() {
        if std::env::var_os("GPT_OSS_XE_LIVE_TEST").is_none() {
            return;
        }
        let _guard = TEST_PROCESS_CIRCUIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        PROCESS_XE_CIRCUIT_OPEN.store(false, Ordering::Release);
        let cache = tempfile::tempdir().unwrap();
        let config =
            || AttachConfig::new(AttachmentMode::Explicit, cache.path(), 1024 * 1024, 32, 1);
        let first = XeProjectionEngine::attach(config()).unwrap();
        let descriptor = first.descriptor().unwrap();
        assert_eq!(descriptor.identity.pci_vendor_id, "8086");
        assert_eq!(descriptor.identity.pci_device_id, "9a49");
        assert!(!descriptor.native_cache_hit);
        first.shutdown().unwrap();
        first.shutdown().unwrap();
        drop(first);

        let second = XeProjectionEngine::attach(config()).unwrap();
        let descriptor = second.descriptor().unwrap();
        assert!(descriptor.native_cache_hit);
        second.shutdown().unwrap();
        drop(second);

        let program = cache
            .path()
            .join("xe/native")
            .join(&descriptor.native_cache_key)
            .join("program.bin");
        std::fs::write(program, b"corrupt native cache").unwrap();
        let recovered = XeProjectionEngine::attach(config()).unwrap();
        assert!(!recovered.descriptor().unwrap().native_cache_hit);
        recovered.shutdown().unwrap();
    }

    #[test]
    fn opt_in_live_projection_matrix_covers_policy_padding_chunks_and_real_shapes() {
        if std::env::var_os("GPT_OSS_XE_LIVE_TEST").is_none() {
            return;
        }
        let _guard = TEST_PROCESS_CIRCUIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        PROCESS_XE_CIRCUIT_OPEN.store(false, Ordering::Release);
        let cache = tempfile::tempdir().unwrap();
        let engine = XeProjectionEngine::attach(AttachConfig::new(
            AttachmentMode::Explicit,
            cache.path(),
            4096,
            32,
            1,
        ))
        .unwrap();
        for rows in 1..AUTO_MIN_ROWS {
            assert!(!engine.should_accelerate(ProjectionRole::GateUp, rows));
            assert!(!engine.should_accelerate(ProjectionRole::Down, rows));
        }
        assert!(engine.should_accelerate(ProjectionRole::GateUp, AUTO_MIN_ROWS));
        assert!(engine.should_accelerate(ProjectionRole::Down, AUTO_MIN_ROWS));
        let (weights, canonical) = test_weights(32, 1);
        let bias = (0..32)
            .map(|column| column as f32 / 1024.0)
            .collect::<Vec<_>>();
        for rows in [4, 5, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
            let activations = test_activations(rows, 1);
            let expected = scalar_projection(rows, 32, 1, &canonical, &activations, &bias);
            let actual = engine
                .project(ProjectionRequest {
                    role: ProjectionRole::GateUp,
                    rows,
                    columns: 32,
                    blocks: 1,
                    weights_v2: &weights,
                    activations_v2: &activations,
                    bias: &bias,
                })
                .unwrap();
            assert_projection_matches(&actual, &expected);
        }
        for (variant, rows) in [(KernelVariant::Tile32M1, 1), (KernelVariant::Tile32M2, 2)] {
            let activations = test_activations(rows, 1);
            let expected = scalar_projection(rows, 32, 1, &canonical, &activations, &bias);
            let actual = engine
                .project_with_variant(
                    ProjectionRequest {
                        role: ProjectionRole::Down,
                        rows,
                        columns: 32,
                        blocks: 1,
                        weights_v2: &weights,
                        activations_v2: &activations,
                        bias: &bias,
                    },
                    variant,
                )
                .unwrap();
            assert_projection_matches(&actual, &expected);
        }
        assert_eq!(engine.residency_stats().unwrap().uploaded_bytes, 14 * 672);
        engine.shutdown().unwrap();

        let real_engine = XeProjectionEngine::attach(AttachConfig::new(
            AttachmentMode::Explicit,
            cache.path(),
            128 * 1024 * 1024,
            5760,
            90,
        ))
        .unwrap();
        for (role, columns, blocks) in [
            (ProjectionRole::GateUp, 5760, 90),
            (ProjectionRole::Down, 2880, 90),
        ] {
            let rows = 4;
            let (weights, canonical) = test_weights(columns, blocks);
            let activations = test_activations(rows, blocks);
            let bias = (0..columns)
                .map(|column| column as f32 / 65536.0)
                .collect::<Vec<_>>();
            let expected =
                scalar_projection(rows, columns, blocks, &canonical, &activations, &bias);
            let actual = real_engine
                .project(ProjectionRequest {
                    role,
                    rows,
                    columns,
                    blocks,
                    weights_v2: &weights,
                    activations_v2: &activations,
                    bias: &bias,
                })
                .unwrap();
            assert_projection_matches(&actual, &expected);
        }
        real_engine.shutdown().unwrap();
    }

    #[test]
    fn opt_in_live_residency_covers_hits_identity_eviction_and_shutdown() {
        if std::env::var_os("GPT_OSS_XE_LIVE_TEST").is_none() {
            return;
        }
        let _guard = TEST_PROCESS_CIRCUIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        PROCESS_XE_CIRCUIT_OPEN.store(false, Ordering::Release);
        let cache_root = tempfile::tempdir().unwrap();
        let engine = XeProjectionEngine::attach(
            AttachConfig::new(
                AttachmentMode::Explicit,
                cache_root.path(),
                1024 * 1024,
                32,
                1,
            )
            .with_expert_cache_bytes(1344),
        )
        .unwrap();
        let (weights, canonical) = test_weights(32, 1);
        let activations = test_activations(4, 1);
        let bias = (0..32)
            .map(|column| column as f32 / 1024.0)
            .collect::<Vec<_>>();
        let expected = scalar_projection(4, 32, 1, &canonical, &activations, &bias);
        let identity = |expert| ExpertCacheIdentity {
            model_source_key: "model-a".into(),
            tensor_source_key: format!("tensor-{expert}"),
            layer: 0,
            expert,
            role: ProjectionRole::GateUp,
            columns: 32,
            blocks: 1,
            weight_layout_version: 2,
        };
        let request = |identity| ResidentProjectionRequest {
            identity,
            role: ProjectionRole::GateUp,
            rows: 4,
            columns: 32,
            blocks: 1,
            activations_v2: &activations,
            bias: &bias,
        };
        let repacks = AtomicUsize::new(0);
        let mut repack = || {
            repacks.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(weights.clone())
        };

        let cold = engine
            .project_resident(request(identity(0)), &mut repack)
            .unwrap();
        assert_eq!(cold.residency, XeResidencyState::Miss);
        assert_projection_matches(&cold.output, &expected);
        let warm = engine
            .project_resident(request(identity(0)), &mut repack)
            .unwrap();
        assert_eq!(warm.residency, XeResidencyState::Hit);
        assert_projection_matches(&warm.output, &expected);
        assert_eq!(repacks.load(AtomicOrdering::SeqCst), 1);

        // Two exact 672-byte entries fit; the third deterministically evicts.
        assert_eq!(
            engine
                .project_resident(request(identity(1)), &mut repack)
                .unwrap()
                .residency,
            XeResidencyState::Miss
        );
        assert_eq!(
            engine
                .project_resident(request(identity(2)), &mut repack)
                .unwrap()
                .residency,
            XeResidencyState::Miss
        );
        let stats = engine.residency_stats().unwrap();
        assert_eq!(stats.resident_bytes, 1344);
        assert_eq!(stats.resident_high_water_bytes, 1344);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 3);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.repacks_avoided, 1);
        assert_eq!(stats.upload_bytes_avoided, 672);
        assert_eq!(stats.uploaded_bytes, 2016);
        engine.shutdown().unwrap();
        engine.shutdown().unwrap();
        assert_eq!(engine.residency_stats().unwrap().resident_bytes, 0);
    }
}
