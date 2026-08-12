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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentMode {
    Automatic,
    Explicit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationClass {
    ValidatedAutomatic,
    ValidatedExplicit,
    UnvalidatedExplicit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    if abi.get("schema").and_then(serde_json::Value::as_str) != Some("gpt-oss-rs.xe-kernel-abi/v2")
        || record.build_options != BUILD_OPTIONS
        || record.workgroup_size != WORKGROUP_SIZE
    {
        return Err(XeError::Artifact(
            "promotion record and immutable ABI/build policy disagree".into(),
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
            max_columns,
            max_blocks,
        }
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

#[derive(Debug, Clone, Copy, Default)]
struct PhaseTiming {
    weight: Duration,
    activation: Duration,
    submit_wait: Duration,
    readback: Duration,
}

trait ProjectionRuntime: Send {
    fn descriptor(&self) -> &XeDescriptor;
    fn project(
        &mut self,
        request: &ProjectionRequest<'_>,
        variant: KernelVariant,
        output: &mut [f32],
    ) -> Result<PhaseTiming, XeError>;
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

    fn execute_projection(
        &self,
        request: ProjectionRequest<'_>,
        variant: KernelVariant,
    ) -> Result<Vec<f32>, XeError> {
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
                Ok(output)
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
        ("weights", request.weights_v2.len()),
        ("activations", std::mem::size_of_val(request.activations_v2)),
        ("bias", std::mem::size_of_val(request.bias)),
        ("output", request.rows * request.columns * 4),
    ] {
        metrics::counter!(
            TRANSFER_BYTES_TOTAL,
            "role" => request.role.as_str(),
            "direction" => direction
        )
        .increment(bytes as u64);
    }
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
    }

    struct FailingRuntime {
        descriptor: XeDescriptor,
        counts: Arc<FakeCounts>,
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
            Err(XeError::Runtime("injected readback failure".into()))
        }

        fn drain(&mut self) -> Result<(), XeError> {
            self.counts.drains.fetch_add(1, AtomicOrdering::SeqCst);
            Ok(())
        }

        fn shutdown(&mut self) -> Result<(), XeError> {
            if !self.shutdown {
                self.shutdown = true;
                self.counts.shutdowns.fetch_add(1, AtomicOrdering::SeqCst);
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
    fn runtime_fault_drains_once_trips_process_circuit_and_never_releases_early() {
        let _guard = TEST_PROCESS_CIRCUIT_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        PROCESS_XE_CIRCUIT_OPEN.store(false, Ordering::Release);
        let counts = Arc::new(FakeCounts::default());
        let engine = XeProjectionEngine {
            runtime: Mutex::new(Box::new(FailingRuntime {
                descriptor: fake_descriptor(),
                counts: counts.clone(),
                shutdown: false,
            })),
            gate_up_min_rows: AUTO_MIN_ROWS,
            down_min_rows: AUTO_MIN_ROWS,
        };
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
        assert!(matches!(
            engine.project(request()),
            Err(XeError::Runtime(_))
        ));
        assert!(engine.circuit_is_open());
        assert_eq!(counts.projects.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(counts.drains.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(counts.shutdowns.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(engine.project(request()), Err(XeError::CircuitOpen));
        assert_eq!(counts.projects.load(AtomicOrdering::SeqCst), 1);
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
        engine.shutdown().unwrap();
        engine.shutdown().unwrap();
        assert_eq!(counts.shutdowns.load(AtomicOrdering::SeqCst), 1);
        PROCESS_XE_CIRCUIT_OPEN.store(false, Ordering::Release);
        metrics::gauge!(CIRCUIT_BREAKER).set(0.0);
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
        engine.shutdown().unwrap();

        let real_engine = XeProjectionEngine::attach(AttachConfig::new(
            AttachmentMode::Explicit,
            cache.path(),
            128 * 1024 * 1024,
            5760,
            90,
        ))
        .unwrap();
        for (role, columns) in [(ProjectionRole::GateUp, 5760), (ProjectionRole::Down, 2880)] {
            let blocks = 90;
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
}
