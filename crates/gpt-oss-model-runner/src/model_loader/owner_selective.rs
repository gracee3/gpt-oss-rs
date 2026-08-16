#![allow(unsafe_code)]
//! Owner-selective construction for the exact heterogeneous GPT-OSS path.
//!
//! Construction is explicit and detached from the current model forward path.
//! It validates the complete manifest before allocation, retains native MXFP4
//! only on the assigned GPU, and creates x8 records only for CPU owners.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::{CudaSlice, CudaStream, PinnedHostSlice};
use gpt_oss_core::error::{LLMError, Result};
use gpt_oss_gpu::device::{list_devices, GpuDevice};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cpu_repack::{
    CpuOwnerLayerRecord, CpuOwnerRepackCache, OWNER_EXPERT_BYTES, OWNER_REPACK_TEMP_BYTES_MAX,
};
use crate::heterogeneous::cuda_expert::{
    CudaSelectedExpertExecutor, CudaSelectedExpertWeights, NativeMxfp4ExpertView,
    GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES,
};
use crate::heterogeneous::placement::{
    ExpertOwner, GptOssExpertKey, GptOssExpertPlacementManifestV1, ResolvedExpertPlacement,
    CONSERVATIVE_OWNER_EXPERT_BYTES,
};

use super::gpt_oss_native::GptOssCheckpointView;

pub const OWNER_SELECTIVE_PINNED_UPLOAD_BYTES: usize = 16 * 1024 * 1024;
pub const OWNER_SELECTIVE_TEMPORARY_CAP_BYTES: usize = 256 * 1024 * 1024;
pub const OWNER_SELECTIVE_GPU_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;

static PINNED_CURRENT_BYTES: AtomicU64 = AtomicU64::new(0);
static PINNED_HIGH_WATER_BYTES: AtomicU64 = AtomicU64::new(0);

pub fn owner_selective_pinned_current_bytes() -> u64 {
    PINNED_CURRENT_BYTES.load(Ordering::Acquire)
}

pub fn owner_selective_pinned_high_water_bytes() -> u64 {
    PINNED_HIGH_WATER_BYTES.load(Ordering::Acquire)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstructionStage {
    Identity,
    RuntimeBaseline,
    Mappings,
    LayerOwnerDense,
    GpuExperts,
    CpuExperts,
    ExecutionReserve,
    Publish,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnerSelectiveEnvelope {
    pub native_mapped_payload_bytes: u64,
    pub non_expert_payload_bytes: u64,
    pub checkpoint_expert_payload_bytes: u64,
    pub layer_owner_experts: u32,
    pub remote_gpu_experts: u32,
    pub cpu_experts: u32,
    pub layer_owner_native_expert_bytes: u64,
    pub remote_gpu_native_expert_bytes: u64,
    pub cpu_x8_record_bytes: u64,
    pub layer_owner_conservative_admission_bytes: u64,
    pub remote_gpu_conservative_admission_bytes: u64,
    pub host_conservative_owner_bytes: u64,
    pub layer_owner_execution_reserve_bytes: u64,
    pub remote_gpu_execution_reserve_bytes: u64,
    pub pinned_upload_bytes: u64,
    pub construction_temporary_cap_bytes: u64,
}

impl OwnerSelectiveEnvelope {
    pub fn from_checkpoint_and_placement(
        checkpoint: &GptOssCheckpointView,
        placement: &ResolvedExpertPlacement,
    ) -> Result<Self> {
        let counts = placement.counts();
        let checked = |count: u32, bytes: u64, label: &str| {
            u64::from(count)
                .checked_mul(bytes)
                .ok_or_else(|| LLMError::ModelError(format!("{label} owner byte count overflows")))
        };
        Ok(Self {
            native_mapped_payload_bytes: checkpoint.mapped_payload_bytes(),
            non_expert_payload_bytes: checkpoint.non_expert_payload_bytes(),
            checkpoint_expert_payload_bytes: checkpoint.expert_payload_bytes(),
            layer_owner_experts: counts.layer_owner_gpu,
            remote_gpu_experts: counts.remote_gpu,
            cpu_experts: counts.cpu,
            layer_owner_native_expert_bytes: checked(
                counts.layer_owner_gpu,
                GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES as u64,
                "layer-owner GPU",
            )?,
            remote_gpu_native_expert_bytes: checked(
                counts.remote_gpu,
                GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES as u64,
                "remote GPU",
            )?,
            cpu_x8_record_bytes: checked(counts.cpu, OWNER_EXPERT_BYTES as u64, "CPU x8")?,
            layer_owner_conservative_admission_bytes: checked(
                counts.layer_owner_gpu,
                CONSERVATIVE_OWNER_EXPERT_BYTES,
                "layer-owner GPU conservative",
            )?,
            remote_gpu_conservative_admission_bytes: checked(
                counts.remote_gpu,
                CONSERVATIVE_OWNER_EXPERT_BYTES,
                "remote GPU conservative",
            )?,
            host_conservative_owner_bytes: checked(
                counts.cpu,
                CONSERVATIVE_OWNER_EXPERT_BYTES,
                "CPU conservative",
            )?,
            layer_owner_execution_reserve_bytes: OWNER_SELECTIVE_GPU_RESERVE_BYTES,
            remote_gpu_execution_reserve_bytes: OWNER_SELECTIVE_GPU_RESERVE_BYTES,
            pinned_upload_bytes: OWNER_SELECTIVE_PINNED_UPLOAD_BYTES as u64,
            construction_temporary_cap_bytes: OWNER_SELECTIVE_TEMPORARY_CAP_BYTES as u64,
        })
    }

    pub fn layer_owner_required_free_bytes(&self) -> Result<u64> {
        self.non_expert_payload_bytes
            .checked_add(self.layer_owner_conservative_admission_bytes)
            .and_then(|bytes| bytes.checked_add(self.layer_owner_execution_reserve_bytes))
            .ok_or_else(|| LLMError::ModelError("layer-owner admission bytes overflow".into()))
    }

    pub fn remote_gpu_required_free_bytes(&self) -> Result<u64> {
        self.remote_gpu_conservative_admission_bytes
            .checked_add(self.remote_gpu_execution_reserve_bytes)
            .ok_or_else(|| LLMError::ModelError("remote-GPU admission bytes overflow".into()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstructionLedger {
    pub stage: ConstructionStage,
    pub mapped_address_bytes: u64,
    pub layer_owner_dense_bytes: u64,
    pub layer_owner_expert_bytes: u64,
    pub remote_gpu_expert_bytes: u64,
    pub cpu_x8_bytes: u64,
    pub pinned_bytes: u64,
    pub construction_temporary_high_water_bytes: u64,
    pub layer_owner_experts: u32,
    pub remote_gpu_experts: u32,
    pub cpu_experts: u32,
}

impl ConstructionLedger {
    fn new() -> Self {
        Self {
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
        }
    }
}

/// Immutable GPU tensor retained by the layer owner.
pub struct LayerOwnerDenseTensor {
    pub name: String,
    pub logical_bytes: u64,
    allocation: CudaSlice<u8>,
}

impl LayerOwnerDenseTensor {
    pub fn device_bytes(&self) -> usize {
        self.allocation.len()
    }
}

/// A fully materialized but execution-detached owner topology.
pub struct OwnerSelectiveModel {
    // Fields are declared in construction rollback order because Rust drops
    // struct fields in declaration order after `Drop::drop`: CPU records,
    // remote/local experts, dense weights, contexts, then source mappings.
    cpu_layers: BTreeMap<u16, CpuOwnerLayerRecord>,
    remote_gpu_experts: BTreeMap<GptOssExpertKey, CudaSelectedExpertWeights>,
    layer_owner_experts: BTreeMap<GptOssExpertKey, CudaSelectedExpertWeights>,
    layer_owner_dense: Vec<LayerOwnerDenseTensor>,
    remote_executor: CudaSelectedExpertExecutor,
    layer_owner_executor: CudaSelectedExpertExecutor,
    checkpoint: GptOssCheckpointView,
    placement: ResolvedExpertPlacement,
    envelope: OwnerSelectiveEnvelope,
    ledger: ConstructionLedger,
}

impl OwnerSelectiveModel {
    pub fn checkpoint(&self) -> &GptOssCheckpointView {
        &self.checkpoint
    }

    pub fn placement(&self) -> &ResolvedExpertPlacement {
        &self.placement
    }

    pub const fn envelope(&self) -> &OwnerSelectiveEnvelope {
        &self.envelope
    }

    pub const fn ledger(&self) -> &ConstructionLedger {
        &self.ledger
    }

    pub fn layer_owner_dense(&self) -> &[LayerOwnerDenseTensor] {
        &self.layer_owner_dense
    }

    pub fn layer_owner_expert_count(&self) -> usize {
        self.layer_owner_experts.len()
    }

    pub fn remote_gpu_expert_count(&self) -> usize {
        self.remote_gpu_experts.len()
    }

    pub fn cpu_layer_records(&self) -> impl Iterator<Item = (&u16, &CpuOwnerLayerRecord)> {
        self.cpu_layers.iter()
    }

    pub fn device_memory_info(&self) -> Result<[(usize, usize); 2]> {
        Ok([
            self.layer_owner_executor.memory_info()?,
            self.remote_executor.memory_info()?,
        ])
    }

    /// Drain both construction streams before reverse-order field teardown.
    pub fn drain(&self) -> Result<()> {
        self.layer_owner_executor
            .stream()
            .synchronize()
            .map_err(cuda_error("layer-owner construction drain"))?;
        self.remote_executor
            .stream()
            .synchronize()
            .map_err(cuda_error("remote-GPU construction drain"))?;
        Ok(())
    }
}

impl Drop for OwnerSelectiveModel {
    fn drop(&mut self) {
        // CudaSlice and pinned allocations also synchronize on drop. An
        // explicit best-effort drain keeps reverse-order teardown deterministic
        // even when a caller does not invoke `drain` itself.
        let _ = self.layer_owner_executor.stream().synchronize();
        let _ = self.remote_executor.stream().synchronize();
    }
}

pub struct OwnerSelectiveConstructor {
    cache_root: PathBuf,
}

impl OwnerSelectiveConstructor {
    pub fn new(cache_root: impl Into<PathBuf>) -> Self {
        Self {
            cache_root: cache_root.into(),
        }
    }

    pub fn cache_root(&self) -> &Path {
        &self.cache_root
    }

    pub fn validate(
        &self,
        checkpoint: &GptOssCheckpointView,
        manifest: &GptOssExpertPlacementManifestV1,
        devices: &[GpuDevice],
    ) -> Result<(ResolvedExpertPlacement, OwnerSelectiveEnvelope)> {
        validate_manifest_identity(checkpoint, manifest)?;
        let placement = manifest
            .validate(devices)
            .map_err(|error| LLMError::ModelError(format!("placement manifest: {error}")))?;
        let envelope =
            OwnerSelectiveEnvelope::from_checkpoint_and_placement(checkpoint, &placement)?;
        if OWNER_REPACK_TEMP_BYTES_MAX > OWNER_SELECTIVE_TEMPORARY_CAP_BYTES {
            return Err(LLMError::ModelError(format!(
                "owner x8 temporary {} exceeds construction cap {}",
                OWNER_REPACK_TEMP_BYTES_MAX, OWNER_SELECTIVE_TEMPORARY_CAP_BYTES
            )));
        }
        Ok((placement, envelope))
    }

    pub fn construct<F>(
        &self,
        checkpoint: GptOssCheckpointView,
        manifest: &GptOssExpertPlacementManifestV1,
        mut observe: F,
    ) -> Result<OwnerSelectiveModel>
    where
        F: FnMut(&ConstructionLedger) -> Result<()>,
    {
        self.construct_inner(checkpoint, manifest, &mut observe, None)
    }

    /// Run the real constructor with one deterministic post-resource fault.
    /// This API is unavailable unless the explicit integration-test feature is
    /// enabled and is never selected by a production runtime feature.
    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn construct_with_fault<F>(
        &self,
        checkpoint: GptOssCheckpointView,
        manifest: &GptOssExpertPlacementManifestV1,
        fault: ConstructionStage,
        mut observe: F,
    ) -> Result<OwnerSelectiveModel>
    where
        F: FnMut(&ConstructionLedger) -> Result<()>,
    {
        self.construct_inner(checkpoint, manifest, &mut observe, Some(fault))
    }

    fn construct_inner<F>(
        &self,
        checkpoint: GptOssCheckpointView,
        manifest: &GptOssExpertPlacementManifestV1,
        observe: &mut F,
        injected_fault: Option<ConstructionStage>,
    ) -> Result<OwnerSelectiveModel>
    where
        F: FnMut(&ConstructionLedger) -> Result<()>,
    {
        let devices = list_devices();
        let (placement, envelope) = self.validate(&checkpoint, manifest, &devices)?;
        let mut ledger = ConstructionLedger::new();
        observe(&ledger)?;
        inject_construction_fault(injected_fault, ConstructionStage::Identity)?;

        let layer_owner_executor =
            CudaSelectedExpertExecutor::new(placement.layer_owner().stable_id.clone())?;
        let remote_executor =
            CudaSelectedExpertExecutor::new(placement.remote_worker().stable_id.clone())?;
        ledger.stage = ConstructionStage::RuntimeBaseline;
        observe(&ledger)?;
        inject_construction_fault(injected_fault, ConstructionStage::RuntimeBaseline)?;

        let (layer_free, layer_total) = layer_owner_executor.memory_info()?;
        let (remote_free, remote_total) = remote_executor.memory_info()?;
        require_device_admission(
            "layer-owner GPU",
            layer_free as u64,
            layer_total as u64,
            envelope.layer_owner_required_free_bytes()?,
        )?;
        require_device_admission(
            "remote GPU",
            remote_free as u64,
            remote_total as u64,
            envelope.remote_gpu_required_free_bytes()?,
        )?;

        ledger.stage = ConstructionStage::Mappings;
        ledger.mapped_address_bytes = checkpoint.mapped_payload_bytes();
        observe(&ledger)?;
        inject_construction_fault(injected_fault, ConstructionStage::Mappings)?;

        let mut pinned = allocate_pinned(layer_owner_executor.stream())?;
        ledger.pinned_bytes = OWNER_SELECTIVE_PINNED_UPLOAD_BYTES as u64;
        let mut layer_owner_dense = Vec::new();
        for mapping in checkpoint
            .mappings()
            .filter(|mapping| !mapping.runtime.contains(".mlp.experts."))
        {
            let source = checkpoint.tensor(&mapping.runtime)?;
            let allocation = upload_pinned_chunks(
                layer_owner_executor.stream(),
                &mut pinned,
                source.bytes(),
                &mapping.runtime,
            )?;
            ledger.layer_owner_dense_bytes = ledger
                .layer_owner_dense_bytes
                .checked_add(mapping.bytes as u64)
                .ok_or_else(|| LLMError::ModelError("dense ledger overflows".into()))?;
            layer_owner_dense.push(LayerOwnerDenseTensor {
                name: mapping.runtime.clone(),
                logical_bytes: mapping.bytes as u64,
                allocation,
            });
            if injected_fault == Some(ConstructionStage::LayerOwnerDense) {
                ledger.stage = ConstructionStage::LayerOwnerDense;
                observe(&ledger)?;
                inject_construction_fault(injected_fault, ConstructionStage::LayerOwnerDense)?;
            }
        }
        ledger.stage = ConstructionStage::LayerOwnerDense;
        observe(&ledger)?;

        let mut layer_owner_experts = BTreeMap::new();
        let mut remote_gpu_experts = BTreeMap::new();
        let mut cpu_by_layer = BTreeMap::<u16, Vec<u16>>::new();
        let mut layer_owner_work = Vec::new();
        let mut remote_gpu_work = Vec::new();
        for (key, owner) in placement.assignments() {
            match owner {
                ExpertOwner::Cpu { .. } => {
                    cpu_by_layer.entry(key.layer).or_default().push(key.expert);
                }
                ExpertOwner::LayerOwnerGpu { .. } => {
                    layer_owner_work.push((*key, owner.clone()));
                }
                ExpertOwner::RemoteGpu { .. } => {
                    remote_gpu_work.push((*key, owner.clone()));
                }
            }
        }

        // Reuse the layer-owner lease for all six native surfaces of every
        // local expert. No mmap slice is passed directly to CUDA on this path.
        for (key, owner) in layer_owner_work {
            let identity = expert_identity(&checkpoint, key)?;
            let source = native_expert_view(&checkpoint, key, &identity)?;
            let weights =
                layer_owner_executor.upload_expert_staged(owner, source, &mut pinned.allocation)?;
            if layer_owner_experts.insert(key, weights).is_some() {
                return Err(LLMError::ModelError("duplicate layer-owner expert".into()));
            }
            ledger.layer_owner_experts += 1;
            ledger.layer_owner_expert_bytes += GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES as u64;
            if injected_fault == Some(ConstructionStage::GpuExperts) {
                ledger.stage = ConstructionStage::GpuExperts;
                observe(&ledger)?;
                inject_construction_fault(injected_fault, ConstructionStage::GpuExperts)?;
            }
        }
        drop(pinned);

        // The GPU0 lease is gone before the equal-size GPU1 lease is created,
        // keeping the construction pinned high-water at exactly 16 MiB and
        // binding the staging allocation to the destination CUDA context.
        let mut remote_pinned = allocate_pinned(remote_executor.stream())?;
        for (key, owner) in remote_gpu_work {
            let identity = expert_identity(&checkpoint, key)?;
            let source = native_expert_view(&checkpoint, key, &identity)?;
            let weights = remote_executor.upload_expert_staged(
                owner,
                source,
                &mut remote_pinned.allocation,
            )?;
            if remote_gpu_experts.insert(key, weights).is_some() {
                return Err(LLMError::ModelError("duplicate remote-GPU expert".into()));
            }
            ledger.remote_gpu_experts += 1;
            ledger.remote_gpu_expert_bytes += GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES as u64;
        }
        drop(remote_pinned);
        ledger.stage = ConstructionStage::GpuExperts;
        observe(&ledger)?;

        ledger.pinned_bytes = 0;
        let cache = CpuOwnerRepackCache::new(
            &self.cache_root,
            checkpoint.revision(),
            checkpoint.mapping_sha256(),
            placement.manifest_hash(),
            envelope
                .cpu_x8_record_bytes
                .checked_add(checkpoint.config().num_hidden_layers as u64 * 1024 * 1024)
                .ok_or_else(|| LLMError::ModelError("owner x8 cache cap overflows".into()))?,
        )?;
        let mut cpu_layers = BTreeMap::new();
        for (layer, experts) in cpu_by_layer {
            let record = cache.open_or_create_layer(&checkpoint, layer, &experts)?;
            for expert in &experts {
                let view = record.expert_view(*expert)?;
                if view.gate_up.rows() != 5_760
                    || view.gate_up.blocks() != 90
                    || view.down.rows() != 2_880
                    || view.down.blocks() != 90
                    || view.gate_up_bias.len() != 5_760
                    || view.down_bias.len() != 2_880
                {
                    return Err(LLMError::ModelError(format!(
                        "owner x8 expert ({layer},{expert}) has invalid execution dimensions"
                    )));
                }
            }
            ledger.cpu_experts += experts.len() as u32;
            ledger.cpu_x8_bytes = ledger
                .cpu_x8_bytes
                .checked_add(record.payload_bytes())
                .ok_or_else(|| LLMError::ModelError("CPU x8 ledger overflows".into()))?;
            ledger.construction_temporary_high_water_bytes = ledger
                .construction_temporary_high_water_bytes
                .max(OWNER_REPACK_TEMP_BYTES_MAX as u64);
            cpu_layers.insert(layer, record);
            if injected_fault == Some(ConstructionStage::CpuExperts) {
                ledger.stage = ConstructionStage::CpuExperts;
                observe(&ledger)?;
                inject_construction_fault(injected_fault, ConstructionStage::CpuExperts)?;
            }
        }
        ledger.stage = ConstructionStage::CpuExperts;
        observe(&ledger)?;

        ledger.stage = ConstructionStage::ExecutionReserve;
        verify_materialized(
            &placement,
            &layer_owner_experts,
            &remote_gpu_experts,
            &cpu_layers,
        )?;
        observe(&ledger)?;
        inject_construction_fault(injected_fault, ConstructionStage::ExecutionReserve)?;

        if ledger.layer_owner_dense_bytes != envelope.non_expert_payload_bytes
            || ledger.layer_owner_expert_bytes != envelope.layer_owner_native_expert_bytes
            || ledger.remote_gpu_expert_bytes != envelope.remote_gpu_native_expert_bytes
            || ledger.cpu_x8_bytes != envelope.cpu_x8_record_bytes
        {
            return Err(LLMError::ModelError(format!(
                "owner-selective byte ledger mismatch: ledger={ledger:?} envelope={envelope:?}"
            )));
        }
        ledger.stage = ConstructionStage::Publish;
        observe(&ledger)?;
        inject_construction_fault(injected_fault, ConstructionStage::Publish)?;
        Ok(OwnerSelectiveModel {
            cpu_layers,
            remote_gpu_experts,
            layer_owner_experts,
            layer_owner_dense,
            remote_executor,
            layer_owner_executor,
            checkpoint,
            placement,
            envelope,
            ledger,
        })
    }
}

fn inject_construction_fault(
    injected: Option<ConstructionStage>,
    reached: ConstructionStage,
) -> Result<()> {
    if injected == Some(reached) {
        return Err(LLMError::ModelError(format!(
            "injected owner-selective construction failure at {reached:?}"
        )));
    }
    Ok(())
}

fn validate_manifest_identity(
    checkpoint: &GptOssCheckpointView,
    manifest: &GptOssExpertPlacementManifestV1,
) -> Result<()> {
    let config = checkpoint.config();
    if manifest.model.revision != checkpoint.revision()
        || manifest.model.config_sha256 != checkpoint.config_sha256()
        || manifest.model.index_sha256 != checkpoint.metadata_sha256()
        || manifest.model.mapping_sha256 != checkpoint.mapping_sha256()
        || usize::from(manifest.model.num_layers) != config.num_hidden_layers
        || usize::from(manifest.model.experts_per_layer) != config.num_experts
        || usize::from(manifest.model.hidden_size) != config.hidden_size
        || usize::from(manifest.model.intermediate_size) != config.intermediate_size
        || usize::from(manifest.model.top_k) != config.experts_per_token
    {
        return Err(LLMError::ModelError(
            "placement manifest does not identify the opened native checkpoint exactly".into(),
        ));
    }
    Ok(())
}

fn require_device_admission(label: &str, free: u64, total: u64, required: u64) -> Result<()> {
    if required > total || required > free {
        return Err(LLMError::GpuError(format!(
            "{label} admission requires {required} bytes, driver reports free={free} total={total}"
        )));
    }
    Ok(())
}

struct TrackedPinnedHostSlice {
    allocation: PinnedHostSlice<u8>,
    bytes: u64,
}

impl Drop for TrackedPinnedHostSlice {
    fn drop(&mut self) {
        PINNED_CURRENT_BYTES.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

fn allocate_pinned(stream: &std::sync::Arc<CudaStream>) -> Result<TrackedPinnedHostSlice> {
    // SAFETY: every byte is initialized before it is supplied to CUDA and the
    // pinned allocation remains live until all copies are synchronized.
    let allocation = unsafe {
        stream
            .context()
            .alloc_pinned::<u8>(OWNER_SELECTIVE_PINNED_UPLOAD_BYTES)
    }
    .map_err(cuda_error("owner-selective pinned upload allocation"))?;
    let bytes = OWNER_SELECTIVE_PINNED_UPLOAD_BYTES as u64;
    let current = PINNED_CURRENT_BYTES.fetch_add(bytes, Ordering::AcqRel) + bytes;
    PINNED_HIGH_WATER_BYTES.fetch_max(current, Ordering::AcqRel);
    Ok(TrackedPinnedHostSlice { allocation, bytes })
}

fn upload_pinned_chunks(
    stream: &std::sync::Arc<CudaStream>,
    pinned: &mut TrackedPinnedHostSlice,
    source: &[u8],
    label: &str,
) -> Result<CudaSlice<u8>> {
    // SAFETY: the uninitialized device allocation is written completely by
    // the chunk loop before it becomes part of a published model.
    let mut destination = unsafe { stream.alloc::<u8>(source.len()) }
        .map_err(cuda_error("owner-selective dense allocation"))?;
    for (chunk_index, source_chunk) in source.chunks(pinned.allocation.len()).enumerate() {
        let start = chunk_index * pinned.allocation.len();
        let end = start + source_chunk.len();
        pinned
            .allocation
            .as_mut_slice()
            .map_err(cuda_error("owner-selective pinned write access"))?[..source_chunk.len()]
            .copy_from_slice(source_chunk);
        let mut target = destination.slice_mut(start..end);
        if source_chunk.len() == pinned.allocation.len() {
            stream
                .memcpy_htod(&pinned.allocation, &mut target)
                .map_err(cuda_error("owner-selective pinned H2D"))?;
        } else {
            // The backing address remains page-locked. cudarc cannot express a
            // subview of PinnedHostSlice, so the bounded tail uses a borrowed
            // slice and is synchronized before reuse/drop.
            let tail = &pinned
                .allocation
                .as_slice()
                .map_err(cuda_error("owner-selective pinned tail access"))?[..source_chunk.len()];
            stream
                .memcpy_htod(tail, &mut target)
                .map_err(cuda_error("owner-selective pinned tail H2D"))?;
        }
        stream
            .synchronize()
            .map_err(cuda_error("owner-selective dense upload drain"))?;
    }
    if destination.len() != source.len() {
        return Err(LLMError::GpuError(format!(
            "dense tensor {label} allocation length mismatch"
        )));
    }
    Ok(destination)
}

fn expert_component<'a>(
    checkpoint: &'a GptOssCheckpointView,
    key: GptOssExpertKey,
    suffix: &str,
    stride: usize,
) -> Result<&'a [u8]> {
    let name = format!("model.layers.{}.mlp.experts.{suffix}", key.layer);
    let bytes = checkpoint.tensor(&name)?.bytes();
    let start = usize::from(key.expert)
        .checked_mul(stride)
        .ok_or_else(|| LLMError::ModelError("expert component range overflows".into()))?;
    let end = start
        .checked_add(stride)
        .ok_or_else(|| LLMError::ModelError("expert component range overflows".into()))?;
    bytes.get(start..end).ok_or_else(|| {
        LLMError::ModelError(format!(
            "expert component {name}[{start}..{end}] is out of range"
        ))
    })
}

fn expert_identity(checkpoint: &GptOssCheckpointView, key: GptOssExpertKey) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(b"gpt-oss-rs-native-expert-v1");
    digest.update(key.layer.to_le_bytes());
    digest.update(key.expert.to_le_bytes());
    for (suffix, stride) in expert_components() {
        digest.update(expert_component(checkpoint, key, suffix, stride)?);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn native_expert_view<'a>(
    checkpoint: &'a GptOssCheckpointView,
    key: GptOssExpertKey,
    identity: &'a str,
) -> Result<NativeMxfp4ExpertView<'a>> {
    let gate_up_bias = expert_component(checkpoint, key, "gate_up_proj_bias", 11_520)?;
    let down_bias = expert_component(checkpoint, key, "down_proj_bias", 5_760)?;
    let gate_up_bias_bf16_bits: &[u16] = bytemuck::try_cast_slice(gate_up_bias)
        .map_err(|error| LLMError::ModelError(format!("gate/up BF16 bias: {error}")))?;
    let down_bias_bf16_bits: &[u16] = bytemuck::try_cast_slice(down_bias)
        .map_err(|error| LLMError::ModelError(format!("down BF16 bias: {error}")))?;
    Ok(NativeMxfp4ExpertView {
        key,
        gate_up_blocks: expert_component(checkpoint, key, "gate_up_proj_blocks", 8_294_400)?,
        gate_up_scales: expert_component(checkpoint, key, "gate_up_proj_scales", 518_400)?,
        gate_up_bias_bf16_bits,
        down_blocks: expert_component(checkpoint, key, "down_proj_blocks", 4_147_200)?,
        down_scales: expert_component(checkpoint, key, "down_proj_scales", 259_200)?,
        down_bias_bf16_bits,
        identity_sha256: identity,
    })
}

fn expert_components() -> [(&'static str, usize); 6] {
    [
        ("gate_up_proj_blocks", 8_294_400),
        ("gate_up_proj_scales", 518_400),
        ("gate_up_proj_bias", 11_520),
        ("down_proj_blocks", 4_147_200),
        ("down_proj_scales", 259_200),
        ("down_proj_bias", 5_760),
    ]
}

fn verify_materialized(
    placement: &ResolvedExpertPlacement,
    layer_owner: &BTreeMap<GptOssExpertKey, CudaSelectedExpertWeights>,
    remote: &BTreeMap<GptOssExpertKey, CudaSelectedExpertWeights>,
    cpu_layers: &BTreeMap<u16, CpuOwnerLayerRecord>,
) -> Result<()> {
    let cpu = cpu_layers
        .iter()
        .flat_map(|(layer, record)| {
            record
                .expert_ids()
                .iter()
                .map(move |expert| GptOssExpertKey {
                    layer: *layer,
                    expert: *expert,
                })
        })
        .collect::<BTreeSet<_>>();
    for (key, owner) in placement.assignments() {
        let representations = usize::from(cpu.contains(key))
            + usize::from(layer_owner.contains_key(key))
            + usize::from(remote.contains_key(key));
        if representations != 1 {
            return Err(LLMError::ModelError(format!(
                "expert ({},{}) materialized {representations} times",
                key.layer, key.expert
            )));
        }
        let owner_matches = match owner {
            ExpertOwner::Cpu { .. } => cpu.contains(key),
            ExpertOwner::LayerOwnerGpu { .. } => layer_owner.contains_key(key),
            ExpertOwner::RemoteGpu { .. } => remote.contains_key(key),
        };
        if !owner_matches {
            return Err(LLMError::ModelError(format!(
                "expert ({},{}) materialized on the wrong owner",
                key.layer, key.expert
            )));
        }
    }
    Ok(())
}

fn cuda_error(context: &'static str) -> impl Fn(cudarc::driver::DriverError) -> LLMError {
    move |error| LLMError::GpuError(format!("{context}: {error}"))
}
