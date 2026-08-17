#![allow(unsafe_code)]
//! Owner-selective construction for the exact heterogeneous GPT-OSS path.
//!
//! Construction is explicit and detached from the current model forward path.
//! It validates the complete manifest before allocation, retains native MXFP4
//! only on the assigned GPU, and creates x8 records only for CPU owners.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, PinnedHostSlice};
use gpt_oss_core::error::{LLMError, Result};
use gpt_oss_gpu::device::{list_devices, GpuDevice};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cpu_repack::{
    CpuOwnerExpertActionExpectation, CpuOwnerExpertSource, CpuOwnerLayerRecord,
    CpuOwnerLayerRecordValidation, CpuOwnerRecordReleaseTelemetry, CpuOwnerRepackCache,
    OWNER_EXPERT_BYTES, OWNER_REPACK_TEMP_BYTES_MAX,
};
use crate::heterogeneous::contract::{GptOssPhase, GPT_OSS_TOP_K};
use crate::heterogeneous::control::heterogeneous_control_shell_device_bytes;
use crate::heterogeneous::cuda_expert::{
    CudaSelectedExpertExecutor, CudaSelectedExpertWeights, NativeMxfp4ExpertView,
    GPT_OSS_SELECTED_EXPERT_EXECUTOR_BYTES, GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES,
    GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES,
};
use crate::heterogeneous::packing::{relay_pinned_capacity_bytes, H4_PREFILL_PINNED_CAP_BYTES};
use crate::heterogeneous::placement::{
    ExpertOwner, GptOssExpertKey, GptOssExpertPlacementManifestV1, ResolvedExpertPlacement,
    CONSERVATIVE_OWNER_EXPERT_BYTES,
};
use crate::heterogeneous::reduction::GPT_OSS_REDUCER_OWNED_DEVICE_BYTES;
use crate::heterogeneous::relay::result_relay_owned_device_bytes;
use crate::heterogeneous::router::{
    exact_router_owned_device_bytes, exact_router_weight_surface_bytes, ResidentExactRouterWeights,
};

use super::capacity_one::{
    all_surfaces, ExactActionCoverage, ExpertPartialKey, ExpertPartialPlan, ExpertPartialStore,
    OwnerPartialHighWater, OwnerSelectivePublicationProof, WarmRecordElisionProof,
    CAPACITY_ONE_POLICY_SHA256, R2_DISK_RESERVE_BYTES, RETAINED_120B_SPLIT_BOUND_BYTES,
    RETAINED_MAX_DIRTY_CPU_OUTPUT_BYTES, RETAINED_MAX_PINNED_CONSTRUCTION_BYTES,
    RETAINED_MAX_SOURCE_MAPPING_BYTES,
};
use super::gpt_oss_native::{
    GptOssCheckpointReleaseEvidence, GptOssCheckpointView, GptOssNativeCatalogMap,
    GptOssNativeConfig,
};
use super::shard_catalog::SafeTensorShardCatalog;
use super::shard_catalog::{ShardReleaseLogicalLedger, ShardReleaseTelemetry};
use super::shard_consumer_plan::{
    GptOssExpertSurface, GptOssShardConsumer, GptOssShardConsumerPlan,
};
use super::shard_transaction::ScopedShardAction;

pub const OWNER_SELECTIVE_PINNED_UPLOAD_BYTES: usize = 16 * 1024 * 1024;
pub const OWNER_SELECTIVE_TEMPORARY_CAP_BYTES: usize = 256 * 1024 * 1024;
pub const OWNER_SELECTIVE_GPU_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub const OWNER_SELECTIVE_PROOF_CONTEXT_CAP: usize = 4_096;
pub const OWNER_SELECTIVE_DECODE_MAX_ROWS: usize = 1;

static PINNED_CURRENT_BYTES: AtomicU64 = AtomicU64::new(0);
static PINNED_HIGH_WATER_BYTES: AtomicU64 = AtomicU64::new(0);
static CAPACITY_ONE_FATAL_QUARANTINE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionReserveDisposition {
    /// H8 samples free VRAM after each selected-expert executor is created.
    /// K/V, router, relay, reducer, result-slot, and pinned-relay resources
    /// remain a reviewed deferred plan at that admission boundary.
    PostExecutorAdmissionRuntimePlanReviewed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceExecutionReserveLedger {
    /// Deferred-allocation admission cap. H8 samples free memory after the
    /// selected-expert executor and its CUDA context/modules already exist.
    pub reserve_cap_bytes: u64,
    pub kv_cache_bytes: u64,
    pub layer_owner_shell_fixed_bytes: u64,
    pub router_bytes: u64,
    pub relay_result_arena_bytes: u64,
    pub reduction_bytes: u64,
    pub selected_expert_executor_bytes: u64,
    pub result_slot_bytes: u64,
    /// Exact selected-expert executor buffers required to exist before the H8
    /// free-memory admission sample. The plan itself does not allocate them.
    pub materialized_before_admission_bytes: u64,
    /// Exact runtime buffers reviewed here and deferred until the
    /// heterogeneous execution runtime is constructed after admission.
    pub reviewed_deferred_after_admission_bytes: u64,
    pub planned_owned_bytes: u64,
    /// Allocator retention/fragmentation, later modules/cuBLAS state and the
    /// hard OOM safety remainder remain unmaterialized inside this term.
    pub runtime_and_safety_remainder_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionReservePlan {
    pub disposition: ExecutionReserveDisposition,
    pub context_cap: u32,
    pub max_dispatch_rows: u32,
    pub num_layers: u32,
    pub remote_result_layers: u32,
    pub layer_owner: DeviceExecutionReserveLedger,
    pub remote_gpu: DeviceExecutionReserveLedger,
    pub decode_pinned_relay_raw_capacity_bytes: u64,
    pub decode_pinned_relay_cap_bytes: u64,
    pub decode_pinned_relay_materialized_at_construction: bool,
    /// The first proof does not materialize prefill relay pools. This is the
    /// reviewed hard policy cap, not an allocation or a raw-byte claim.
    pub prefill_pinned_relay_cap_bytes: u64,
    pub prefill_pinned_relay_materialized_at_construction: bool,
}

impl ExecutionReservePlan {
    pub fn from_config(
        config: &super::gpt_oss_native::GptOssNativeConfig,
        remote_layers: usize,
    ) -> Result<Self> {
        Self::from_config_with_policy(
            config,
            remote_layers,
            OWNER_SELECTIVE_PROOF_CONTEXT_CAP,
            OWNER_SELECTIVE_DECODE_MAX_ROWS,
            OWNER_SELECTIVE_GPU_RESERVE_BYTES,
        )
    }

    fn from_config_with_policy(
        config: &super::gpt_oss_native::GptOssNativeConfig,
        remote_layers: usize,
        context_cap: usize,
        max_dispatch_rows: usize,
        gpu_reserve_bytes: u64,
    ) -> Result<Self> {
        config.validate()?;
        if remote_layers > config.num_hidden_layers {
            return Err(LLMError::ModelError(format!(
                "remote execution-result layer count {remote_layers} exceeds model layers {}",
                config.num_hidden_layers
            )));
        }
        if config.num_hidden_layers == 0
            || config.vocab_size == 0
            || !matches!(config.num_experts, 32 | 128)
            || config.experts_per_token != GPT_OSS_TOP_K
            || config.hidden_size != crate::heterogeneous::contract::GPT_OSS_HIDDEN_SIZE
            || config.intermediate_size != crate::heterogeneous::cuda_expert::INTERMEDIATE_SIZE
            || config.head_dim != 64
            || config.num_attention_heads != 64
            || config.num_key_value_heads != 8
        {
            return Err(LLMError::ModelError(
                "execution reserve dimensions do not match the fixed GPT-OSS runtime".into(),
            ));
        }
        let u64_value = |value: usize, label: &str| {
            u64::try_from(value)
                .map_err(|_| LLMError::ModelError(format!("{label} does not fit u64")))
        };
        let product = |label: &str, factors: &[u64]| {
            factors.iter().try_fold(1_u64, |value, factor| {
                value.checked_mul(*factor).ok_or_else(|| {
                    LLMError::ModelError(format!("{label} byte arithmetic overflows"))
                })
            })
        };
        let sum = |label: &str, values: &[u64]| {
            values.iter().try_fold(0_u64, |value, term| {
                value.checked_add(*term).ok_or_else(|| {
                    LLMError::ModelError(format!("{label} byte arithmetic overflows"))
                })
            })
        };

        let layers = u64_value(config.num_hidden_layers, "layer count")?;
        let remote_layers_u64 = u64_value(remote_layers, "remote layer count")?;
        let head_dim = u64_value(config.head_dim, "head dimension")?;
        let kv_width = product(
            "K/V width",
            &[
                u64_value(config.num_key_value_heads, "K/V heads")?,
                head_dim,
            ],
        )?;
        let top_k = u64_value(GPT_OSS_TOP_K, "top-k")?;
        let bf16_bytes = u64::try_from(size_of::<u16>()).expect("u16 size fits u64");

        let kv_cache_bytes = product(
            "K/V cache",
            &[
                layers,
                u64_value(context_cap, "context cap")?,
                kv_width,
                2,
                bf16_bytes,
            ],
        )?;
        let shell_total = u64_value(
            heterogeneous_control_shell_device_bytes(
                config.num_hidden_layers,
                config.vocab_size,
                context_cap,
            )?,
            "control shell bytes",
        )?;
        let layer_owner_shell_fixed_bytes = shell_total
            .checked_sub(kv_cache_bytes)
            .ok_or_else(|| LLMError::ModelError("control shell K/V split underflows".into()))?;

        let router_per_layer = u64_value(
            exact_router_owned_device_bytes(config.num_experts, max_dispatch_rows)?,
            "router bytes",
        )?;
        let router_bytes = product("all-layer router", &[layers, router_per_layer])?;
        let relay_per_layer = u64_value(
            result_relay_owned_device_bytes(max_dispatch_rows)?,
            "relay arena bytes",
        )?;
        let relay_result_arena_bytes =
            product("all-layer relay arena", &[layers, relay_per_layer])?;
        let reduction_bytes = product(
            "all-layer reducer",
            &[
                layers,
                u64_value(GPT_OSS_REDUCER_OWNED_DEVICE_BYTES, "reducer bytes")?,
            ],
        )?;
        let selected_expert_executor_bytes = u64_value(
            GPT_OSS_SELECTED_EXPERT_EXECUTOR_BYTES,
            "selected-expert executor bytes",
        )?;
        let one_layer_result_slots = product(
            "result slots",
            &[
                top_k,
                u64_value(GPT_OSS_SELECTED_EXPERT_OUTPUT_BYTES, "expert output bytes")?,
            ],
        )?;
        let layer_owner_result_slot_bytes = product(
            "layer-owner result slots",
            &[layers, one_layer_result_slots],
        )?;
        let remote_result_slot_bytes = product(
            "remote result slots",
            &[remote_layers_u64, one_layer_result_slots],
        )?;

        let device = |label: &str,
                      kv_cache_bytes: u64,
                      layer_owner_shell_fixed_bytes: u64,
                      router_bytes: u64,
                      relay_result_arena_bytes: u64,
                      reduction_bytes: u64,
                      result_slot_bytes: u64|
         -> Result<DeviceExecutionReserveLedger> {
            let planned_owned_bytes = sum(
                label,
                &[
                    kv_cache_bytes,
                    layer_owner_shell_fixed_bytes,
                    router_bytes,
                    relay_result_arena_bytes,
                    reduction_bytes,
                    selected_expert_executor_bytes,
                    result_slot_bytes,
                ],
            )?;
            let materialized_before_admission_bytes = selected_expert_executor_bytes;
            let reviewed_deferred_after_admission_bytes = planned_owned_bytes
                .checked_sub(materialized_before_admission_bytes)
                .ok_or_else(|| {
                    LLMError::ModelError(format!(
                        "{label} materialized execution bytes exceed planned bytes"
                    ))
                })?;
            let runtime_and_safety_remainder_bytes = gpu_reserve_bytes
                .checked_sub(reviewed_deferred_after_admission_bytes)
                .ok_or_else(|| {
                    LLMError::MemoryError(format!(
                        "{label} deferred execution bytes {reviewed_deferred_after_admission_bytes} exceed reserve {}",
                        gpu_reserve_bytes
                    ))
                })?;
            Ok(DeviceExecutionReserveLedger {
                reserve_cap_bytes: gpu_reserve_bytes,
                kv_cache_bytes,
                layer_owner_shell_fixed_bytes,
                router_bytes,
                relay_result_arena_bytes,
                reduction_bytes,
                selected_expert_executor_bytes,
                result_slot_bytes,
                materialized_before_admission_bytes,
                reviewed_deferred_after_admission_bytes,
                planned_owned_bytes,
                runtime_and_safety_remainder_bytes,
            })
        };

        let (decode_pinned_raw, decode_pinned_cap) =
            relay_pinned_capacity_bytes(GptOssPhase::Decode, max_dispatch_rows)?;
        if decode_pinned_raw > decode_pinned_cap {
            return Err(LLMError::MemoryError(
                "decode pinned relay raw capacity exceeds cap".into(),
            ));
        }

        let plan = Self {
            disposition: ExecutionReserveDisposition::PostExecutorAdmissionRuntimePlanReviewed,
            context_cap: u32::try_from(context_cap)
                .map_err(|_| LLMError::ModelError("proof context cap does not fit u32".into()))?,
            max_dispatch_rows: u32::try_from(max_dispatch_rows)
                .map_err(|_| LLMError::ModelError("decode row cap does not fit u32".into()))?,
            num_layers: u32::try_from(config.num_hidden_layers)
                .map_err(|_| LLMError::ModelError("layer count does not fit u32".into()))?,
            remote_result_layers: u32::try_from(remote_layers)
                .map_err(|_| LLMError::ModelError("remote layer count does not fit u32".into()))?,
            layer_owner: device(
                "layer-owner execution plan",
                kv_cache_bytes,
                layer_owner_shell_fixed_bytes,
                router_bytes,
                relay_result_arena_bytes,
                reduction_bytes,
                layer_owner_result_slot_bytes,
            )?,
            remote_gpu: device(
                "remote-GPU execution plan",
                0,
                0,
                0,
                0,
                0,
                remote_result_slot_bytes,
            )?,
            decode_pinned_relay_raw_capacity_bytes: u64_value(
                decode_pinned_raw,
                "decode pinned relay raw capacity",
            )?,
            decode_pinned_relay_cap_bytes: u64_value(decode_pinned_cap, "decode pinned relay cap")?,
            decode_pinned_relay_materialized_at_construction: false,
            prefill_pinned_relay_cap_bytes: u64_value(
                H4_PREFILL_PINNED_CAP_BYTES,
                "prefill pinned relay bytes",
            )?,
            prefill_pinned_relay_materialized_at_construction: false,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<()> {
        if self.disposition != ExecutionReserveDisposition::PostExecutorAdmissionRuntimePlanReviewed
            || self.context_cap == 0
            || self.max_dispatch_rows == 0
            || self.num_layers == 0
            || self.remote_result_layers > self.num_layers
            || self.decode_pinned_relay_materialized_at_construction
            || self.prefill_pinned_relay_materialized_at_construction
            || self.decode_pinned_relay_raw_capacity_bytes > self.decode_pinned_relay_cap_bytes
        {
            return Err(LLMError::ModelError(
                "execution reserve plan has invalid policy or materialization state".into(),
            ));
        }
        for (label, device) in [
            ("layer owner", &self.layer_owner),
            ("remote GPU", &self.remote_gpu),
        ] {
            let exact_categories = device
                .kv_cache_bytes
                .checked_add(device.layer_owner_shell_fixed_bytes)
                .and_then(|bytes| bytes.checked_add(device.router_bytes))
                .and_then(|bytes| bytes.checked_add(device.relay_result_arena_bytes))
                .and_then(|bytes| bytes.checked_add(device.reduction_bytes))
                .and_then(|bytes| bytes.checked_add(device.selected_expert_executor_bytes))
                .and_then(|bytes| bytes.checked_add(device.result_slot_bytes))
                .ok_or_else(|| {
                    LLMError::ModelError(format!("{label} execution reserve validation overflows"))
                })?;
            let materialization_split = device
                .materialized_before_admission_bytes
                .checked_add(device.reviewed_deferred_after_admission_bytes)
                .ok_or_else(|| {
                    LLMError::ModelError(format!(
                        "{label} execution materialization split overflows"
                    ))
                })?;
            let inclusive_reserve = device
                .reviewed_deferred_after_admission_bytes
                .checked_add(device.runtime_and_safety_remainder_bytes)
                .ok_or_else(|| {
                    LLMError::ModelError(format!("{label} inclusive execution reserve overflows"))
                })?;
            if exact_categories != device.planned_owned_bytes
                || materialization_split != device.planned_owned_bytes
                || device.materialized_before_admission_bytes
                    != device.selected_expert_executor_bytes
                || inclusive_reserve != device.reserve_cap_bytes
            {
                return Err(LLMError::ModelError(format!(
                    "{label} execution reserve ledger is inconsistent"
                )));
            }
        }
        Ok(())
    }
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
    pub execution_reserve_plan: ExecutionReservePlan,
}

impl OwnerSelectiveEnvelope {
    pub fn from_checkpoint_and_placement(
        checkpoint: &GptOssCheckpointView,
        placement: &ResolvedExpertPlacement,
    ) -> Result<Self> {
        Self::from_native_parts_and_placement(
            checkpoint.config(),
            checkpoint.mapped_payload_bytes(),
            checkpoint.expert_payload_bytes(),
            checkpoint.non_expert_payload_bytes(),
            placement,
        )
    }

    pub fn from_catalog_map_and_placement(
        native: &GptOssNativeCatalogMap,
        placement: &ResolvedExpertPlacement,
    ) -> Result<Self> {
        Self::from_native_parts_and_placement(
            native.config(),
            native.mapped_payload_bytes(),
            native.expert_payload_bytes(),
            native.non_expert_payload_bytes(),
            placement,
        )
    }

    fn from_native_parts_and_placement(
        config: &GptOssNativeConfig,
        mapped_payload_bytes: u64,
        expert_payload_bytes: u64,
        non_expert_payload_bytes: u64,
        placement: &ResolvedExpertPlacement,
    ) -> Result<Self> {
        let counts = placement.counts();
        let checked = |count: u32, bytes: u64, label: &str| {
            u64::from(count)
                .checked_mul(bytes)
                .ok_or_else(|| LLMError::ModelError(format!("{label} owner byte count overflows")))
        };
        let remote_layers = placement
            .assignments()
            .filter_map(|(key, owner)| {
                matches!(owner, ExpertOwner::RemoteGpu { .. }).then_some(key.layer)
            })
            .collect::<BTreeSet<_>>()
            .len();
        let execution_reserve_plan = ExecutionReservePlan::from_config(config, remote_layers)?;
        Ok(Self {
            native_mapped_payload_bytes: mapped_payload_bytes,
            non_expert_payload_bytes,
            checkpoint_expert_payload_bytes: expert_payload_bytes,
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
            execution_reserve_plan,
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
    pub execution_reserve_reviewed: bool,
    pub execution_runtime_resources_materialized_at_construction: bool,
    pub layer_owner_execution_materialized_before_admission_bytes: u64,
    pub remote_gpu_execution_materialized_before_admission_bytes: u64,
    pub layer_owner_execution_planned_bytes: u64,
    pub remote_gpu_execution_planned_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CapacityOneConstructionEvidence {
    pub policy_sha256: String,
    pub catalog_sha256: String,
    pub plan_sha256: String,
    pub active_mapping_high_water: usize,
    pub mapped_byte_high_water: u64,
    pub plan_partial_high_water_count: usize,
    pub plan_partial_high_water_bytes: u64,
    pub plan_owner_partial_high_waters: Vec<OwnerPartialHighWater>,
    pub partial_high_water_count: usize,
    pub partial_high_water_bytes: u64,
    pub warm_elision_proofs: Vec<WarmRecordElisionProof>,
    pub shard_releases: Vec<ShardReleaseTelemetry>,
    pub publication_proof: OwnerSelectivePublicationProof,
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
            execution_reserve_reviewed: false,
            execution_runtime_resources_materialized_at_construction: false,
            layer_owner_execution_materialized_before_admission_bytes: 0,
            remote_gpu_execution_materialized_before_admission_bytes: 0,
            layer_owner_execution_planned_bytes: 0,
            remote_gpu_execution_planned_bytes: 0,
        }
    }
}

/// Payload-free native identity retained after construction drops the
/// checkpoint mappings.
#[derive(Debug, Clone)]
pub struct OwnerSelectiveNativeMetadata {
    config: GptOssNativeConfig,
    revision: String,
    config_sha256: String,
    metadata_sha256: String,
    mapping_sha256: String,
}

impl OwnerSelectiveNativeMetadata {
    fn from_checkpoint(checkpoint: &GptOssCheckpointView) -> Result<Self> {
        let metadata = Self {
            config: checkpoint.config().clone(),
            revision: checkpoint.revision().to_owned(),
            config_sha256: checkpoint.config_sha256().to_owned(),
            metadata_sha256: checkpoint.metadata_sha256().to_owned(),
            mapping_sha256: checkpoint.mapping_sha256().to_owned(),
        };
        metadata.validate()?;
        Ok(metadata)
    }

    fn from_catalog_map(native: &GptOssNativeCatalogMap) -> Result<Self> {
        let metadata = Self {
            config: native.config().clone(),
            revision: native.revision().to_owned(),
            config_sha256: native.config_sha256().to_owned(),
            metadata_sha256: native.metadata_sha256().to_owned(),
            mapping_sha256: native.mapping_sha256().to_owned(),
        };
        metadata.validate()?;
        Ok(metadata)
    }

    fn validate(&self) -> Result<()> {
        self.config.validate()?;
        if self.revision.trim().is_empty()
            || !is_sha256(&self.config_sha256)
            || !is_sha256(&self.metadata_sha256)
            || !is_sha256(&self.mapping_sha256)
        {
            return Err(LLMError::ModelError(
                "owner-selective native identity metadata is invalid".into(),
            ));
        }
        Ok(())
    }

    pub const fn config(&self) -> &GptOssNativeConfig {
        &self.config
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    pub fn metadata_sha256(&self) -> &str {
        &self.metadata_sha256
    }

    pub fn mapping_sha256(&self) -> &str {
        &self.mapping_sha256
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RouterDenseComponent {
    Weight,
    Bias,
}

fn classify_router_dense_tensor(
    name: &str,
    num_layers: usize,
) -> Result<Option<(usize, RouterDenseComponent)>> {
    if !name.contains(".mlp.router.") {
        return Ok(None);
    }
    let rest = name
        .strip_prefix("model.layers.")
        .ok_or_else(|| LLMError::ModelError(format!("malformed router tensor name {name}")))?;
    let (layer_text, suffix) = rest
        .split_once('.')
        .ok_or_else(|| LLMError::ModelError(format!("malformed router tensor name {name}")))?;
    let layer = layer_text
        .parse::<usize>()
        .map_err(|_| LLMError::ModelError(format!("invalid router layer in {name}")))?;
    if layer.to_string() != layer_text || layer >= num_layers {
        return Err(LLMError::ModelError(format!(
            "router tensor {name} identifies an invalid layer"
        )));
    }
    let component = match suffix {
        "mlp.router.weight" => RouterDenseComponent::Weight,
        "mlp.router.bias" => RouterDenseComponent::Bias,
        _ => {
            return Err(LLMError::ModelError(format!(
                "unsupported router tensor component {name}"
            )))
        }
    };
    Ok(Some((layer, component)))
}

struct RouterPairSlot<T> {
    weight: Option<T>,
    bias: Option<T>,
}

struct RouterPairAccumulator<T> {
    slots: Vec<RouterPairSlot<T>>,
}

impl<T> RouterPairAccumulator<T> {
    fn new(num_layers: usize) -> Result<Self> {
        if num_layers == 0 || num_layers > usize::from(u16::MAX) {
            return Err(LLMError::ModelError(
                "router publication layer count is invalid".into(),
            ));
        }
        Ok(Self {
            slots: (0..num_layers)
                .map(|_| RouterPairSlot {
                    weight: None,
                    bias: None,
                })
                .collect(),
        })
    }

    fn insert(&mut self, layer: usize, component: RouterDenseComponent, value: T) -> Result<()> {
        let slot = self.slots.get_mut(layer).ok_or_else(|| {
            LLMError::ModelError(format!("router publication layer {layer} is out of range"))
        })?;
        let target = match component {
            RouterDenseComponent::Weight => &mut slot.weight,
            RouterDenseComponent::Bias => &mut slot.bias,
        };
        if target.replace(value).is_some() {
            return Err(LLMError::ModelError(format!(
                "duplicate router {component:?} for layer {layer}"
            )));
        }
        Ok(())
    }

    fn finish(self) -> Result<Vec<(usize, T, T)>> {
        self.slots
            .into_iter()
            .enumerate()
            .map(|(layer, slot)| {
                let weight = slot.weight.ok_or_else(|| {
                    LLMError::ModelError(format!("missing router weight for layer {layer}"))
                })?;
                let bias = slot.bias.ok_or_else(|| {
                    LLMError::ModelError(format!("missing router bias for layer {layer}"))
                })?;
                Ok((layer, weight, bias))
            })
            .collect()
    }
}

/// Deterministically ordered resident router sources. The owned allocations
/// may be consumed exactly once by the heterogeneous runtime.
pub struct ResidentExactRouterSources {
    expected_layers: usize,
    expected_experts: usize,
    stable_device: gpt_oss_gpu::device::StableCudaDeviceId,
    layers: Option<Vec<(usize, ResidentExactRouterWeights)>>,
}

impl ResidentExactRouterSources {
    pub fn new(
        expected_layers: usize,
        expected_experts: usize,
        stable_device: gpt_oss_gpu::device::StableCudaDeviceId,
        layers: Vec<(usize, ResidentExactRouterWeights)>,
    ) -> Result<Self> {
        if expected_layers == 0 || expected_layers > usize::from(u16::MAX) {
            return Err(LLMError::ModelError(
                "resident router publication layer count is invalid".into(),
            ));
        }
        validate_router_publication_order(expected_layers, layers.iter().map(|(layer, _)| *layer))?;
        if !matches!(expected_experts, 32 | 128)
            || layers.iter().any(|(_, source)| {
                source.experts() != expected_experts || source.stable_device() != &stable_device
            })
        {
            return Err(LLMError::ModelError(
                "resident router publication identity or shape mismatch".into(),
            ));
        }
        Ok(Self {
            expected_layers,
            expected_experts,
            stable_device,
            layers: Some(layers),
        })
    }

    pub fn available_layers(&self) -> usize {
        self.layers.as_ref().map_or(0, Vec::len)
    }

    pub fn source_tensor_count(&self) -> Result<usize> {
        self.available_layers().checked_mul(2).ok_or_else(|| {
            LLMError::ModelError("resident router source tensor count overflows".into())
        })
    }

    pub fn device_bytes(&self) -> Result<usize> {
        self.layers
            .as_ref()
            .into_iter()
            .flatten()
            .try_fold(0_usize, |total, (_, source)| {
                total.checked_add(source.device_bytes()?).ok_or_else(|| {
                    LLMError::ModelError("resident router byte total overflows".into())
                })
            })
    }

    pub fn take_ordered(&mut self) -> Result<Vec<ResidentExactRouterWeights>> {
        let layers = self.layers.take().ok_or_else(|| {
            LLMError::ModelError("resident router sources were already consumed".into())
        })?;
        validate_router_publication_order(
            self.expected_layers,
            layers.iter().map(|(layer, _)| *layer),
        )?;
        if !matches!(self.expected_experts, 32 | 128)
            || layers.iter().any(|(_, source)| {
                source.experts() != self.expected_experts
                    || source.stable_device() != &self.stable_device
            })
        {
            return Err(LLMError::ModelError(
                "resident router sources changed before consumption".into(),
            ));
        }
        Ok(layers.into_iter().map(|(_, source)| source).collect())
    }

    fn quarantine_for_process_lifetime(&mut self) {
        if let Some(layers) = self.layers.take() {
            std::mem::forget(layers);
        }
    }
}

fn validate_router_publication_order(
    expected_layers: usize,
    observed_layers: impl IntoIterator<Item = usize>,
) -> Result<()> {
    let observed = observed_layers.into_iter().collect::<Vec<_>>();
    if observed.len() != expected_layers
        || observed
            .iter()
            .enumerate()
            .any(|(expected, observed)| expected != *observed)
    {
        return Err(LLMError::ModelError(format!(
            "resident router publication order mismatch: observed={observed:?} expected=0..{expected_layers}"
        )));
    }
    Ok(())
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

    pub(crate) fn allocation(&self) -> &CudaSlice<u8> {
        &self.allocation
    }
}

/// A fully materialized owner-weight topology. Its selected-expert executor
/// buffers also exist; the remaining execution resources are an explicitly
/// reviewed, not-yet-materialized reserve plan.
pub struct OwnerSelectiveModel {
    // Fields are declared in construction rollback order because Rust drops
    // struct fields in declaration order after `Drop::drop`: CPU records,
    // remote/local experts, dense weights, contexts, then source mappings.
    cpu_layers: BTreeMap<u16, CpuOwnerLayerRecord>,
    remote_gpu_experts: BTreeMap<GptOssExpertKey, Arc<CudaSelectedExpertWeights>>,
    layer_owner_experts: BTreeMap<GptOssExpertKey, Arc<CudaSelectedExpertWeights>>,
    resident_router_sources: ResidentExactRouterSources,
    layer_owner_dense: Vec<LayerOwnerDenseTensor>,
    remote_executor: Option<CudaSelectedExpertExecutor>,
    layer_owner_executor: Option<CudaSelectedExpertExecutor>,
    execution_quarantined: bool,
    native_metadata: OwnerSelectiveNativeMetadata,
    placement: ResolvedExpertPlacement,
    envelope: OwnerSelectiveEnvelope,
    ledger: ConstructionLedger,
    capacity_one_evidence: Option<CapacityOneConstructionEvidence>,
    checkpoint_release_evidence: Option<GptOssCheckpointReleaseEvidence>,
}

pub(crate) struct OwnerSelectiveExecutionParts<'a> {
    pub cpu_layers: &'a BTreeMap<u16, CpuOwnerLayerRecord>,
    pub remote_gpu_experts: &'a BTreeMap<GptOssExpertKey, Arc<CudaSelectedExpertWeights>>,
    pub layer_owner_experts: &'a BTreeMap<GptOssExpertKey, Arc<CudaSelectedExpertWeights>>,
    pub remote_executor: &'a mut CudaSelectedExpertExecutor,
    pub layer_owner_executor: &'a mut CudaSelectedExpertExecutor,
}

impl OwnerSelectiveModel {
    pub const fn native_metadata(&self) -> &OwnerSelectiveNativeMetadata {
        &self.native_metadata
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

    pub fn capacity_one_evidence(&self) -> Option<&CapacityOneConstructionEvidence> {
        self.capacity_one_evidence.as_ref()
    }

    pub fn checkpoint_release_evidence(&self) -> Option<&GptOssCheckpointReleaseEvidence> {
        self.checkpoint_release_evidence.as_ref()
    }

    pub fn layer_owner_dense(&self) -> &[LayerOwnerDenseTensor] {
        &self.layer_owner_dense
    }

    pub fn resident_router_source_layers(&self) -> usize {
        self.resident_router_sources.available_layers()
    }

    pub fn resident_router_source_tensor_count(&self) -> Result<usize> {
        self.resident_router_sources.source_tensor_count()
    }

    pub fn resident_router_source_device_bytes(&self) -> Result<usize> {
        self.resident_router_sources.device_bytes()
    }

    /// Transfer every layer's router pair exactly once to the control runtime.
    pub fn take_resident_exact_router_weights(
        &mut self,
    ) -> Result<Vec<ResidentExactRouterWeights>> {
        self.resident_router_sources.take_ordered()
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

    /// Release every runtime CPU-record mapping after the caller has drained
    /// and stopped all execution consumers.
    pub fn release_cpu_record_mappings_with_advice(
        &mut self,
    ) -> Result<Vec<CpuOwnerRecordReleaseTelemetry>> {
        self.drain()?;
        let releases = std::mem::take(&mut self.cpu_layers)
            .into_values()
            .map(CpuOwnerLayerRecord::release_with_advice)
            .collect::<Vec<_>>();
        if releases.iter().any(|release| {
            !release.mapping_removed
                || !release.fd_closed
                || release.post_release.source_inode_mapping_count != 0
                || release.post_release.source_inode_pss_bytes != 0
        }) {
            return Err(LLMError::ModelError(
                "CPU owner-record release proof is incomplete".into(),
            ));
        }
        Ok(releases)
    }

    pub fn device_memory_info(&self) -> Result<[(usize, usize); 2]> {
        Ok([
            self.layer_owner_executor
                .as_ref()
                .ok_or_else(execution_quarantined)?
                .memory_info()?,
            self.remote_executor
                .as_ref()
                .ok_or_else(execution_quarantined)?
                .memory_info()?,
        ])
    }

    /// Narrow execution-only split used by the H6 routed-expert coordinator.
    /// It exposes no construction, checkpoint, or placement mutation and keeps
    /// every resident expert handle owned by this model.
    pub(crate) fn execution_parts(&mut self) -> OwnerSelectiveExecutionParts<'_> {
        OwnerSelectiveExecutionParts {
            cpu_layers: &self.cpu_layers,
            remote_gpu_experts: &self.remote_gpu_experts,
            layer_owner_experts: &self.layer_owner_experts,
            remote_executor: self
                .remote_executor
                .as_mut()
                .expect("owner-selective execution is not quarantined"),
            layer_owner_executor: self
                .layer_owner_executor
                .as_mut()
                .expect("owner-selective execution is not quarantined"),
        }
    }

    /// Fail closed after an unproven CUDA drain. Resident weights, streams,
    /// modules, scratch and result targets may still be referenced, so retain
    /// every GPU execution object for process lifetime and make the model
    /// permanently unusable.
    pub(crate) fn quarantine_execution(&mut self) {
        if self.execution_quarantined {
            return;
        }
        self.execution_quarantined = true;
        if let Some(executor) = self.layer_owner_executor.take() {
            std::mem::forget(executor);
        }
        if let Some(executor) = self.remote_executor.take() {
            std::mem::forget(executor);
        }
        std::mem::forget(std::mem::take(&mut self.layer_owner_experts));
        std::mem::forget(std::mem::take(&mut self.remote_gpu_experts));
        self.resident_router_sources
            .quarantine_for_process_lifetime();
        std::mem::forget(std::mem::take(&mut self.layer_owner_dense));
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub const fn execution_quarantined_for_test(&self) -> bool {
        self.execution_quarantined
    }

    /// Drain both construction streams before reverse-order field teardown.
    pub fn drain(&mut self) -> Result<()> {
        if self
            .layer_owner_executor
            .as_ref()
            .is_some_and(CudaSelectedExpertExecutor::owned_drain_unproven)
            || self
                .remote_executor
                .as_ref()
                .is_some_and(CudaSelectedExpertExecutor::owned_drain_unproven)
        {
            self.quarantine_execution();
            return Err(LLMError::GpuError(
                "owner-selective execution has an unproven CUDA drain and requires quarantine"
                    .into(),
            ));
        }
        let layer_owner_drain = self
            .layer_owner_executor
            .as_ref()
            .ok_or_else(execution_quarantined)?
            .stream()
            .synchronize();
        if let Err(error) = layer_owner_drain {
            self.quarantine_execution();
            return Err(cuda_error("layer-owner construction drain")(error));
        }
        let remote_drain = self
            .remote_executor
            .as_ref()
            .ok_or_else(execution_quarantined)?
            .stream()
            .synchronize();
        match remote_drain {
            Ok(()) => Ok(()),
            Err(error) => {
                self.quarantine_execution();
                Err(cuda_error("remote-GPU construction drain")(error))
            }
        }
    }
}

impl Drop for OwnerSelectiveModel {
    fn drop(&mut self) {
        if self
            .layer_owner_executor
            .as_ref()
            .is_some_and(CudaSelectedExpertExecutor::owned_drain_unproven)
            || self
                .remote_executor
                .as_ref()
                .is_some_and(CudaSelectedExpertExecutor::owned_drain_unproven)
        {
            self.quarantine_execution();
            return;
        }
        // CudaSlice and pinned allocations also synchronize on drop. An
        // explicit best-effort drain keeps reverse-order teardown deterministic
        // even when a caller does not invoke `drain` itself.
        let layer_owner_drained = self
            .layer_owner_executor
            .as_ref()
            .is_none_or(|executor| executor.stream().synchronize().is_ok());
        let remote_drained = self
            .remote_executor
            .as_ref()
            .is_none_or(|executor| executor.stream().synchronize().is_ok());
        if !layer_owner_drained || !remote_drained {
            self.quarantine_execution();
        }
    }
}

fn execution_quarantined() -> LLMError {
    LLMError::GpuError("owner-selective execution topology is quarantined".into())
}

pub struct OwnerSelectiveConstructor {
    cache_root: PathBuf,
}

#[derive(Debug, Clone)]
struct PlanActionLocator {
    shard_index: usize,
    action_index: usize,
    action_id_sha256: String,
}

struct StreamingPlanIndex {
    dense_by_shard: Vec<Vec<usize>>,
    expert_actions: BTreeMap<ExpertPartialKey, BTreeMap<GptOssExpertSurface, PlanActionLocator>>,
    cpu_by_layer: BTreeMap<u16, Vec<ExpertPartialKey>>,
    gpu_completion_by_shard: Vec<Vec<ExpertPartialKey>>,
    cpu_layer_completion_shard: BTreeMap<u16, usize>,
}

impl StreamingPlanIndex {
    fn build(plan: &GptOssShardConsumerPlan) -> Result<Self> {
        plan.validate_identity()?;
        let mut dense_by_shard = vec![Vec::new(); plan.shards().len()];
        let mut expert_actions =
            BTreeMap::<ExpertPartialKey, BTreeMap<GptOssExpertSurface, PlanActionLocator>>::new();
        for (shard_index, shard) in plan.shards().iter().enumerate() {
            for (action_index, action) in shard.actions.iter().enumerate() {
                match &action.consumer {
                    GptOssShardConsumer::LayerOwnerDense { .. } => {
                        dense_by_shard[shard_index].push(action_index);
                    }
                    GptOssShardConsumer::OwnedExpert {
                        key,
                        owner,
                        surface,
                    } => {
                        let identity = ExpertPartialKey {
                            key: *key,
                            owner: owner.clone(),
                        };
                        if expert_actions
                            .entry(identity)
                            .or_default()
                            .insert(
                                *surface,
                                PlanActionLocator {
                                    shard_index,
                                    action_index,
                                    action_id_sha256: action.action_id_sha256.clone(),
                                },
                            )
                            .is_some()
                        {
                            return Err(LLMError::ModelError(
                                "streaming plan contains a duplicate expert surface".into(),
                            ));
                        }
                    }
                }
            }
        }
        let expected = all_surfaces().into_iter().collect::<BTreeSet<_>>();
        let mut cpu_by_layer = BTreeMap::<u16, Vec<ExpertPartialKey>>::new();
        let mut gpu_completion_by_shard = vec![Vec::new(); plan.shards().len()];
        for (identity, surfaces) in &expert_actions {
            if surfaces.keys().copied().collect::<BTreeSet<_>>() != expected {
                return Err(LLMError::ModelError(format!(
                    "streaming expert ({},{}) lacks exact six-surface coverage",
                    identity.key.layer, identity.key.expert
                )));
            }
            let completion_shard = surfaces
                .values()
                .map(|locator| locator.shard_index)
                .max()
                .expect("six surfaces have a completion shard");
            match identity.owner {
                ExpertOwner::Cpu { .. } => {
                    cpu_by_layer
                        .entry(identity.key.layer)
                        .or_default()
                        .push(identity.clone());
                }
                ExpertOwner::LayerOwnerGpu { .. } | ExpertOwner::RemoteGpu { .. } => {
                    gpu_completion_by_shard[completion_shard].push(identity.clone());
                }
            }
        }
        for experts in cpu_by_layer.values_mut() {
            experts.sort_by_key(|identity| identity.key.expert);
        }
        for experts in &mut gpu_completion_by_shard {
            experts.sort();
        }
        let mut cpu_layer_completion_shard = BTreeMap::new();
        for (layer, identities) in &cpu_by_layer {
            let completion = identities
                .iter()
                .flat_map(|identity| expert_actions[identity].values())
                .map(|locator| locator.shard_index)
                .max()
                .ok_or_else(|| LLMError::ModelError("CPU layer has no actions".into()))?;
            for identity in identities {
                let surfaces = &expert_actions[identity];
                let expert_completion = surfaces
                    .values()
                    .map(|locator| locator.shard_index)
                    .max()
                    .expect("six surfaces have a completion shard");
                if expert_completion != completion {
                    return Err(LLMError::ModelError(format!(
                        "CPU layer {layer} cannot be completed in one mapped shard"
                    )));
                }
                for (surface, locator) in surfaces {
                    if locator.shard_index != completion
                        && !(*surface == GptOssExpertSurface::GateUpBias
                            && locator.shard_index.checked_add(1) == Some(completion))
                    {
                        return Err(LLMError::ModelError(format!(
                            "CPU layer {layer} would retain a non-bias native surface"
                        )));
                    }
                }
            }
            cpu_layer_completion_shard.insert(*layer, completion);
        }
        Ok(Self {
            dense_by_shard,
            expert_actions,
            cpu_by_layer,
            gpu_completion_by_shard,
            cpu_layer_completion_shard,
        })
    }

    fn action_ids(&self, identity: &ExpertPartialKey) -> [String; 6] {
        let surfaces = &self.expert_actions[identity];
        all_surfaces().map(|surface| surfaces[&surface].action_id_sha256.clone())
    }

    fn current_action_indices(
        &self,
        identity: &ExpertPartialKey,
        shard_index: usize,
    ) -> Vec<usize> {
        let mut indices = self.expert_actions[identity]
            .values()
            .filter(|locator| locator.shard_index == shard_index)
            .map(|locator| locator.action_index)
            .collect::<Vec<_>>();
        indices.sort_unstable();
        indices
    }
}

struct ExpertSurfaceBytes<'a> {
    gate_up_bias: &'a [u8],
    gate_up_blocks: &'a [u8],
    gate_up_scales: &'a [u8],
    down_bias: &'a [u8],
    down_blocks: &'a [u8],
    down_scales: &'a [u8],
}

fn with_expert_surface_bytes<R>(
    identity: &ExpertPartialKey,
    actions: &[ScopedShardAction<'_>],
    owned_split_bias: Option<&[u8]>,
    use_surfaces: impl FnOnce(ExpertSurfaceBytes<'_>) -> Result<R>,
) -> Result<R> {
    let mut gate_up_bias = owned_split_bias;
    let mut gate_up_blocks = None;
    let mut gate_up_scales = None;
    let mut down_bias = None;
    let mut down_blocks = None;
    let mut down_scales = None;
    for scoped in actions {
        let GptOssShardConsumer::OwnedExpert {
            key,
            owner,
            surface,
        } = &scoped.action().consumer
        else {
            return Err(LLMError::ModelError(
                "expert assembly received a dense action".into(),
            ));
        };
        if *key != identity.key || *owner != identity.owner {
            return Err(LLMError::ModelError(
                "expert assembly action key or owner mismatch".into(),
            ));
        }
        let target = match surface {
            GptOssExpertSurface::GateUpBias => &mut gate_up_bias,
            GptOssExpertSurface::GateUpBlocks => &mut gate_up_blocks,
            GptOssExpertSurface::GateUpScales => &mut gate_up_scales,
            GptOssExpertSurface::DownBias => &mut down_bias,
            GptOssExpertSurface::DownBlocks => &mut down_blocks,
            GptOssExpertSurface::DownScales => &mut down_scales,
        };
        if target.replace(scoped.bytes()).is_some() {
            return Err(LLMError::ModelError(
                "expert assembly received a duplicate surface".into(),
            ));
        }
    }
    use_surfaces(ExpertSurfaceBytes {
        gate_up_bias: gate_up_bias
            .ok_or_else(|| LLMError::ModelError("expert gate/up bias is missing".into()))?,
        gate_up_blocks: gate_up_blocks
            .ok_or_else(|| LLMError::ModelError("expert gate/up blocks are missing".into()))?,
        gate_up_scales: gate_up_scales
            .ok_or_else(|| LLMError::ModelError("expert gate/up scales are missing".into()))?,
        down_bias: down_bias
            .ok_or_else(|| LLMError::ModelError("expert down bias is missing".into()))?,
        down_blocks: down_blocks
            .ok_or_else(|| LLMError::ModelError("expert down blocks are missing".into()))?,
        down_scales: down_scales
            .ok_or_else(|| LLMError::ModelError("expert down scales are missing".into()))?,
    })
}

fn expert_identity_from_surfaces(
    key: GptOssExpertKey,
    surfaces: &ExpertSurfaceBytes<'_>,
) -> String {
    let mut digest = Sha256::new();
    digest.update(b"gpt-oss-rs-native-expert-v1");
    digest.update(key.layer.to_le_bytes());
    digest.update(key.expert.to_le_bytes());
    for bytes in [
        surfaces.gate_up_blocks,
        surfaces.gate_up_scales,
        surfaces.gate_up_bias,
        surfaces.down_blocks,
        surfaces.down_scales,
        surfaces.down_bias,
    ] {
        digest.update(bytes);
    }
    format!("{:x}", digest.finalize())
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

    /// Production capacity-one constructor. The source root is cataloged from
    /// bounded headers and never opened through `GptOssCheckpointView`.
    pub fn construct_capacity_one<F>(
        &self,
        source_root: &Path,
        manifest: &GptOssExpertPlacementManifestV1,
        policy_sha256: &str,
        mut observe: F,
    ) -> Result<OwnerSelectiveModel>
    where
        F: FnMut(&ConstructionLedger) -> Result<()>,
    {
        if policy_sha256 != CAPACITY_ONE_POLICY_SHA256 {
            return Err(LLMError::ModelError(
                "capacity-one construction policy identity is unsupported".into(),
            ));
        }
        if CAPACITY_ONE_FATAL_QUARANTINE.load(Ordering::Acquire) {
            return Err(LLMError::GpuError(
                "this process is quarantined after an unproven capacity-one CUDA terminal state"
                    .into(),
            ));
        }
        let catalog = SafeTensorShardCatalog::open(source_root)?;
        if catalog
            .shards()
            .iter()
            .any(|shard| shard.identity.file_length > RETAINED_MAX_SOURCE_MAPPING_BYTES)
        {
            return Err(LLMError::MemoryError(
                "capacity-one source shard exceeds the frozen mapping window".into(),
            ));
        }
        let native = GptOssNativeCatalogMap::from_source_root(source_root, &catalog)?;
        validate_manifest_identity_catalog(&native, manifest)?;
        let devices = list_devices();
        let placement = manifest
            .validate(&devices)
            .map_err(|error| LLMError::ModelError(format!("placement manifest: {error}")))?;
        let envelope = OwnerSelectiveEnvelope::from_catalog_map_and_placement(&native, &placement)?;
        if OWNER_REPACK_TEMP_BYTES_MAX > OWNER_SELECTIVE_TEMPORARY_CAP_BYTES {
            return Err(LLMError::ModelError(
                "owner x8 conversion scratch exceeds construction cap".into(),
            ));
        }
        let plan = GptOssShardConsumerPlan::build(&catalog, &native, manifest)?;
        let partial_plan = ExpertPartialPlan::derive(&plan, RETAINED_120B_SPLIT_BOUND_BYTES)?;
        let plan_partial_high_water_count = partial_plan.derived_high_water_count();
        let plan_partial_high_water_bytes = partial_plan.derived_high_water_bytes();
        let plan_owner_partial_high_waters = partial_plan.derived_owner_high_waters().to_vec();
        let split_entries = partial_plan
            .entries()
            .map(|entry| (entry.identity.clone(), entry.clone()))
            .collect::<BTreeMap<_, _>>();
        let plan_index = StreamingPlanIndex::build(&plan)?;
        let native_metadata = OwnerSelectiveNativeMetadata::from_catalog_map(&native)?;
        let mut coverage = ExactActionCoverage::new(&plan)?;
        let cache = CpuOwnerRepackCache::new(
            &self.cache_root,
            native.revision(),
            native.mapping_sha256(),
            placement.manifest_hash(),
            envelope
                .cpu_x8_record_bytes
                .checked_add(native.config().num_hidden_layers as u64 * 1024 * 1024)
                .ok_or_else(|| LLMError::ModelError("owner x8 cache cap overflows".into()))?,
        )?;

        let mut warm_validations = BTreeMap::<u16, CpuOwnerLayerRecordValidation>::new();
        let mut cold_layers = BTreeSet::new();
        let mut warm_elision_proofs = Vec::new();
        let mut warm_cpu_layers = BTreeSet::new();
        for (layer, identities) in &plan_index.cpu_by_layer {
            let expert_ids = identities
                .iter()
                .map(|identity| identity.key.expert)
                .collect::<Vec<_>>();
            let layer_payload_bytes = u64::try_from(expert_ids.len())
                .ok()
                .and_then(|count| count.checked_mul(OWNER_EXPERT_BYTES as u64))
                .ok_or_else(|| LLMError::ModelError("CPU layer output bytes overflow".into()))?;
            if layer_payload_bytes > RETAINED_MAX_DIRTY_CPU_OUTPUT_BYTES {
                return Err(LLMError::MemoryError(
                    "capacity-one CPU layer output exceeds the frozen dirty bound".into(),
                ));
            }
            match cache.validate_layer_without_mapping(
                *layer,
                &expert_ids,
                native.config().num_hidden_layers,
                native.config().num_experts,
            )? {
                Some(validation) => {
                    let mut action_ids_sha256 = Vec::with_capacity(identities.len() * 6);
                    let mut native_bytes = 0_u64;
                    for identity in identities {
                        for surface in all_surfaces() {
                            let locator = &plan_index.expert_actions[identity][&surface];
                            let action =
                                &plan.shards()[locator.shard_index].actions[locator.action_index];
                            coverage.elide(&locator.action_id_sha256, action.byte_len()?)?;
                            native_bytes = native_bytes
                                .checked_add(action.byte_len()?)
                                .ok_or_else(|| {
                                    LLMError::ModelError(
                                        "warm elision native bytes overflow".into(),
                                    )
                                })?;
                            action_ids_sha256.push(locator.action_id_sha256.clone());
                        }
                    }
                    warm_elision_proofs.push(WarmRecordElisionProof {
                        catalog_sha256: catalog.metadata_sha256().into(),
                        source_revision: native.revision().into(),
                        mapping_sha256: native.mapping_sha256().into(),
                        placement_sha256: placement.manifest_hash().into(),
                        placement_epoch: placement.placement_epoch(),
                        format_version: crate::cpu_repack::OWNER_REPACK_FORMAT_VERSION,
                        layer: *layer,
                        ordered_expert_ids: expert_ids,
                        record_identity_sha256: validation.record_identity_sha256.clone(),
                        action_ids_sha256,
                        native_bytes,
                    });
                    warm_cpu_layers.insert(*layer);
                    warm_validations.insert(*layer, validation);
                }
                None => {
                    cold_layers.insert(*layer);
                }
            }
        }

        let mut ledger = ConstructionLedger::new();
        observe(&ledger)?;
        let layer_owner_executor =
            CudaSelectedExpertExecutor::new(placement.layer_owner().stable_id.clone())?;
        let remote_executor =
            CudaSelectedExpertExecutor::new(placement.remote_worker().stable_id.clone())?;
        ledger.stage = ConstructionStage::RuntimeBaseline;
        observe(&ledger)?;
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

        let mut layer_pinned = allocate_pinned(layer_owner_executor.stream())?;
        let mut remote_pinned = allocate_pinned(remote_executor.stream())?;
        ledger.pinned_bytes = (2 * OWNER_SELECTIVE_PINNED_UPLOAD_BYTES) as u64;
        if ledger.pinned_bytes > RETAINED_MAX_PINNED_CONSTRUCTION_BYTES {
            return Err(LLMError::MemoryError(
                "capacity-one pinned construction bound exceeded".into(),
            ));
        }
        let mut layer_owner_dense = Vec::new();
        let mut router_pairs =
            RouterPairAccumulator::<CudaSlice<u8>>::new(native.config().num_hidden_layers)?;
        let (router_weight_bytes, router_bias_bytes) =
            exact_router_weight_surface_bytes(native.config().num_experts)?;
        let mut layer_owner_experts = BTreeMap::new();
        let mut remote_gpu_experts = BTreeMap::new();
        let mut cold_validations = BTreeMap::<u16, CpuOwnerLayerRecordValidation>::new();
        let mut partial_store = ExpertPartialStore::new(partial_plan);
        let mut remaining_cold_record_bytes =
            cold_layers.iter().try_fold(0_u64, |total, layer| {
                let expert_ids = plan_index.cpu_by_layer[layer]
                    .iter()
                    .map(|identity| identity.key.expert)
                    .collect::<Vec<_>>();
                total
                    .checked_add(cache.expected_layer_file_bytes(
                        *layer,
                        &expert_ids,
                        native.config().num_hidden_layers,
                        native.config().num_experts,
                    )?)
                    .ok_or_else(|| LLMError::ModelError("cold record bytes overflow".into()))
            })?;
        let mut fatal_cuda = false;

        for shard_index in 0..plan.shards().len() {
            ledger.stage = ConstructionStage::Mappings;
            let shard_result =
                catalog.with_scoped_shard_transaction(&plan, shard_index, |transaction| {
                    let warm_elided_indices = plan.shards()[shard_index]
                        .actions
                        .iter()
                        .enumerate()
                        .filter_map(|(index, action)| match &action.consumer {
                            GptOssShardConsumer::OwnedExpert {
                                key,
                                owner: ExpertOwner::Cpu { .. },
                                ..
                            } if warm_cpu_layers.contains(&key.layer) => Some(index),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    if !warm_elided_indices.is_empty() {
                        transaction.record_warm_elisions(&warm_elided_indices)?;
                    }
                    for split in split_entries.values().filter(|entry| {
                        entry.earlier_shard_index == shard_index
                            && !(matches!(entry.identity.owner, ExpertOwner::Cpu { .. })
                                && warm_cpu_layers.contains(&entry.identity.key.layer))
                    }) {
                        let locator = &plan_index.expert_actions[&split.identity]
                            [&GptOssExpertSurface::GateUpBias];
                        let action = &plan.shards()[shard_index].actions[locator.action_index];
                        transaction.with_synchronous_action(locator.action_index, |scoped| {
                            partial_store.insert_bias(shard_index, scoped.action(), scoped.bytes())
                        })?;
                        coverage.consume(action)?;
                    }

                    for action_index in &plan_index.dense_by_shard[shard_index] {
                        let action = &plan.shards()[shard_index].actions[*action_index];
                        let GptOssShardConsumer::LayerOwnerDense { runtime_tensor } =
                            &action.consumer
                        else {
                            return Err(LLMError::ModelError(
                                "dense action index identifies an expert".into(),
                            ));
                        };
                        let mut upload_terminal_unproven = false;
                        let uploaded =
                            transaction.with_synchronous_action(*action_index, |scoped| {
                                upload_pinned_chunks_classified(
                                    layer_owner_executor.stream(),
                                    &mut layer_pinned,
                                    scoped.bytes(),
                                    runtime_tensor,
                                )
                                .map_err(|failure| {
                                    upload_terminal_unproven = failure.terminal_unproven;
                                    failure.error
                                })
                            });
                        let allocation = match uploaded {
                            Ok(allocation) => allocation,
                            Err(error) => {
                                if upload_terminal_unproven {
                                    fatal_cuda = true;
                                    transaction.quarantine_unproven_terminal();
                                }
                                return Err(error);
                            }
                        };
                        ledger.layer_owner_dense_bytes = ledger
                            .layer_owner_dense_bytes
                            .checked_add(action.byte_len()?)
                            .ok_or_else(|| LLMError::ModelError("dense ledger overflows".into()))?;
                        if let Some((layer, component)) = classify_router_dense_tensor(
                            runtime_tensor,
                            native.config().num_hidden_layers,
                        )? {
                            let expected = match component {
                                RouterDenseComponent::Weight => router_weight_bytes,
                                RouterDenseComponent::Bias => router_bias_bytes,
                            };
                            if action.byte_len()? != expected as u64 {
                                return Err(LLMError::ModelError(format!(
                                    "router tensor {runtime_tensor} byte length mismatch"
                                )));
                            }
                            router_pairs.insert(layer, component, allocation)?;
                        } else {
                            layer_owner_dense.push(LayerOwnerDenseTensor {
                                name: runtime_tensor.clone(),
                                logical_bytes: action.byte_len()?,
                                allocation,
                            });
                        }
                        coverage.consume(action)?;
                    }

                    for identity in &plan_index.gpu_completion_by_shard[shard_index] {
                        let split = split_entries.get(identity);
                        let current_indices =
                            plan_index.current_action_indices(identity, shard_index);
                        let current_action_ids = current_indices
                            .iter()
                            .map(|index| {
                                plan.shards()[shard_index].actions[*index]
                                    .action_id_sha256
                                    .clone()
                            })
                            .collect::<Vec<_>>();
                        let completed_bias = split
                            .map(|entry| {
                                partial_store.take_for_completion(
                                    identity,
                                    shard_index,
                                    &entry.later_action_ids_sha256,
                                )
                            })
                            .transpose()?;
                        let mut upload_terminal_unproven = false;
                        let uploaded =
                            transaction.with_synchronous_actions(&current_indices, |actions| {
                                with_expert_surface_bytes(
                                    identity,
                                    actions,
                                    completed_bias.as_ref().map(|bias| bias.bytes.as_slice()),
                                    |surfaces| {
                                        let expert_identity =
                                            expert_identity_from_surfaces(identity.key, &surfaces);
                                        let gate_up_bias_bf16_bits =
                                            bytemuck::try_cast_slice(surfaces.gate_up_bias)
                                                .map_err(|error| {
                                                    LLMError::ModelError(format!(
                                                        "gate/up BF16 bias: {error}"
                                                    ))
                                                })?;
                                        let down_bias_bf16_bits = bytemuck::try_cast_slice(
                                            surfaces.down_bias,
                                        )
                                        .map_err(|error| {
                                            LLMError::ModelError(format!("down BF16 bias: {error}"))
                                        })?;
                                        let source = NativeMxfp4ExpertView {
                                            key: identity.key,
                                            gate_up_blocks: surfaces.gate_up_blocks,
                                            gate_up_scales: surfaces.gate_up_scales,
                                            gate_up_bias_bf16_bits,
                                            down_blocks: surfaces.down_blocks,
                                            down_scales: surfaces.down_scales,
                                            down_bias_bf16_bits,
                                            identity_sha256: &expert_identity,
                                        };
                                        match &identity.owner {
                                            ExpertOwner::LayerOwnerGpu { .. } => {
                                                layer_owner_executor
                                                    .upload_expert_staged_classified(
                                                        identity.owner.clone(),
                                                        source,
                                                        &mut layer_pinned.allocation,
                                                    )
                                                    .map_err(|failure| {
                                                        upload_terminal_unproven =
                                                            failure.terminal_unproven;
                                                        failure.error
                                                    })
                                            }
                                            ExpertOwner::RemoteGpu { .. } => remote_executor
                                                .upload_expert_staged_classified(
                                                    identity.owner.clone(),
                                                    source,
                                                    &mut remote_pinned.allocation,
                                                )
                                                .map_err(|failure| {
                                                    upload_terminal_unproven =
                                                        failure.terminal_unproven;
                                                    failure.error
                                                }),
                                            ExpertOwner::Cpu { .. } => Err(LLMError::ModelError(
                                                "CPU expert entered GPU completion".into(),
                                            )),
                                        }
                                    },
                                )
                            });
                        let weights = match uploaded {
                            Ok(weights) => Arc::new(weights),
                            Err(error) => {
                                if upload_terminal_unproven {
                                    fatal_cuda = true;
                                    transaction.quarantine_unproven_terminal();
                                }
                                return Err(error);
                            }
                        };
                        match &identity.owner {
                            ExpertOwner::LayerOwnerGpu { .. } => {
                                if layer_owner_experts.insert(identity.key, weights).is_some() {
                                    return Err(LLMError::ModelError(
                                        "duplicate layer-owner expert".into(),
                                    ));
                                }
                                ledger.layer_owner_experts += 1;
                                ledger.layer_owner_expert_bytes +=
                                    GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES as u64;
                            }
                            ExpertOwner::RemoteGpu { .. } => {
                                if remote_gpu_experts.insert(identity.key, weights).is_some() {
                                    return Err(LLMError::ModelError(
                                        "duplicate remote-GPU expert".into(),
                                    ));
                                }
                                ledger.remote_gpu_experts += 1;
                                ledger.remote_gpu_expert_bytes +=
                                    GPT_OSS_SELECTED_EXPERT_PAYLOAD_BYTES as u64;
                            }
                            ExpertOwner::Cpu { .. } => unreachable!(),
                        }
                        for action_id in current_action_ids {
                            let action = plan.shards()[shard_index]
                                .actions
                                .iter()
                                .find(|action| action.action_id_sha256 == action_id)
                                .expect("current action ID belongs to shard");
                            coverage.consume(action)?;
                        }
                    }

                    for (layer, completion_shard) in &plan_index.cpu_layer_completion_shard {
                        if *completion_shard != shard_index || warm_cpu_layers.contains(layer) {
                            continue;
                        }
                        let identities = &plan_index.cpu_by_layer[layer];
                        let expectations = identities
                            .iter()
                            .map(|identity| {
                                let ids = plan_index.action_ids(identity);
                                CpuOwnerExpertActionExpectation {
                                    expert_id: identity.key.expert,
                                    gate_up_bias_action: ids[0].clone(),
                                    gate_up_blocks_action: ids[1].clone(),
                                    gate_up_scales_action: ids[2].clone(),
                                    down_bias_action: ids[3].clone(),
                                    down_blocks_action: ids[4].clone(),
                                    down_scales_action: ids[5].clone(),
                                }
                            })
                            .collect::<Vec<_>>();
                        let layer_payload_bytes = u64::try_from(identities.len())
                            .ok()
                            .and_then(|count| count.checked_mul(OWNER_EXPERT_BYTES as u64))
                            .ok_or_else(|| {
                                LLMError::ModelError("CPU layer output bytes overflow".into())
                            })?;
                        if layer_payload_bytes > RETAINED_MAX_DIRTY_CPU_OUTPUT_BYTES {
                            return Err(LLMError::MemoryError(
                                "capacity-one CPU layer output exceeds the frozen dirty bound"
                                    .into(),
                            ));
                        }
                        let mut record_transaction = cache.begin_layer_transaction(
                            *layer,
                            expectations,
                            native.config().num_hidden_layers,
                            native.config().num_experts,
                            remaining_cold_record_bytes,
                            R2_DISK_RESERVE_BYTES,
                        )?;
                        for identity in identities {
                            let ids = plan_index.action_ids(identity);
                            let id_refs = ids.each_ref().map(String::as_str);
                            let current_indices =
                                plan_index.current_action_indices(identity, shard_index);
                            let split = split_entries.get(identity);
                            let completed_bias = split
                                .map(|entry| {
                                    partial_store.take_for_completion(
                                        identity,
                                        shard_index,
                                        &entry.later_action_ids_sha256,
                                    )
                                })
                                .transpose()?;
                            transaction.with_synchronous_actions(&current_indices, |actions| {
                                with_expert_surface_bytes(
                                    identity,
                                    actions,
                                    completed_bias.as_ref().map(|bias| bias.bytes.as_slice()),
                                    |surfaces| {
                                        record_transaction.accept_complete_expert(
                                            identity.key.expert,
                                            id_refs,
                                            CpuOwnerExpertSource {
                                                gate_up_bias: surfaces.gate_up_bias,
                                                gate_up_blocks: surfaces.gate_up_blocks,
                                                gate_up_scales: surfaces.gate_up_scales,
                                                down_bias: surfaces.down_bias,
                                                down_blocks: surfaces.down_blocks,
                                                down_scales: surfaces.down_scales,
                                            },
                                        )
                                    },
                                )
                            })?;
                            for action_index in current_indices {
                                coverage
                                    .consume(&plan.shards()[shard_index].actions[action_index])?;
                            }
                        }
                        let validation = record_transaction.finish(true)?;
                        if validation.payload_bytes > RETAINED_MAX_DIRTY_CPU_OUTPUT_BYTES {
                            return Err(LLMError::MemoryError(
                                "capacity-one CPU layer output exceeds the frozen dirty bound"
                                    .into(),
                            ));
                        }
                        remaining_cold_record_bytes = remaining_cold_record_bytes
                            .checked_sub(validation.file_bytes)
                            .ok_or_else(|| {
                                LLMError::ModelError("cold record bytes underflow".into())
                            })?;
                        ledger.cpu_experts += identities.len() as u32;
                        ledger.cpu_x8_bytes = ledger
                            .cpu_x8_bytes
                            .checked_add(validation.payload_bytes)
                            .ok_or_else(|| {
                                LLMError::ModelError("CPU x8 ledger overflows".into())
                            })?;
                        ledger.construction_temporary_high_water_bytes = ledger
                            .construction_temporary_high_water_bytes
                            .max(OWNER_REPACK_TEMP_BYTES_MAX as u64);
                        cold_validations.insert(*layer, validation);
                    }
                    let partial = partial_store.stats();
                    transaction.record_terminal_audit(ShardReleaseLogicalLedger {
                        partial_store_current_count: partial.current_count,
                        partial_store_current_bytes: partial.current_bytes,
                        partial_store_high_water_count: partial.high_water_count,
                        partial_store_high_water_bytes: partial.high_water_bytes,
                        pinned_construction_bytes: ledger.pinned_bytes,
                        anonymous_temporary_high_water_bytes: ledger
                            .construction_temporary_high_water_bytes,
                        output_logical_bytes: ledger.cpu_x8_bytes,
                        device_destination_logical_bytes: ledger
                            .layer_owner_dense_bytes
                            .checked_add(ledger.layer_owner_expert_bytes)
                            .and_then(|bytes| bytes.checked_add(ledger.remote_gpu_expert_bytes))
                            .ok_or_else(|| {
                                LLMError::ModelError(
                                    "device destination logical ledger overflows".into(),
                                )
                            })?,
                    })?;
                    Ok(())
                });
            if let Err(error) = shard_result {
                if fatal_cuda {
                    CAPACITY_ONE_FATAL_QUARANTINE.store(true, Ordering::Release);
                    std::mem::forget(layer_pinned);
                    std::mem::forget(remote_pinned);
                    std::mem::forget(layer_owner_experts);
                    std::mem::forget(remote_gpu_experts);
                    std::mem::forget(layer_owner_dense);
                    std::mem::forget(router_pairs);
                    std::mem::forget(layer_owner_executor);
                    std::mem::forget(remote_executor);
                    return Err(LLMError::GpuError(format!(
                        "capacity-one CUDA terminal ownership is unproven; process quarantined ({error})"
                    )));
                }
                return Err(error);
            }
            ledger.mapped_address_bytes = catalog.mapping_activity().current_mapped_bytes;
            observe(&ledger)?;
        }

        coverage.validate_complete()?;
        partial_store.require_empty()?;
        if remaining_cold_record_bytes != 0 {
            return Err(LLMError::ModelError(
                "cold CPU record byte ledger did not reach zero".into(),
            ));
        }
        drop(layer_pinned);
        drop(remote_pinned);
        ledger.pinned_bytes = 0;

        let resident_router_layers = router_pairs
            .finish()?
            .into_iter()
            .map(|(layer, weight, bias)| {
                Ok((
                    layer,
                    ResidentExactRouterWeights::new(
                        placement.layer_owner().stable_id.clone(),
                        native.config().num_experts,
                        weight,
                        bias,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let resident_router_sources = ResidentExactRouterSources::new(
            native.config().num_hidden_layers,
            native.config().num_experts,
            placement.layer_owner().stable_id.clone(),
            resident_router_layers,
        )?;

        if catalog.mapping_activity().current != 0 || catalog.active_source_payload_fds() != 0 {
            return Err(LLMError::ModelError(
                "source mappings or payload fds remain before CPU runtime mapping".into(),
            ));
        }
        let mut all_validations = BTreeMap::new();
        all_validations.extend(warm_validations);
        all_validations.extend(cold_validations);
        let mut cpu_layers = BTreeMap::new();
        for (layer, identities) in &plan_index.cpu_by_layer {
            let expert_ids = identities
                .iter()
                .map(|identity| identity.key.expert)
                .collect::<Vec<_>>();
            let fresh = cache
                .validate_layer_without_mapping(
                    *layer,
                    &expert_ids,
                    native.config().num_hidden_layers,
                    native.config().num_experts,
                )?
                .ok_or_else(|| {
                    LLMError::ModelError(format!(
                        "owner x8 layer {layer} disappeared before runtime mapping"
                    ))
                })?;
            if all_validations.get(layer) != Some(&fresh) {
                return Err(LLMError::ModelError(format!(
                    "owner x8 layer {layer} identity changed before runtime mapping"
                )));
            }
            let record = cache.map_validated_layer(&fresh, catalog.mapping_activity().current)?;
            for expert in &expert_ids {
                let view = record.expert_view(*expert)?;
                if view.gate_up.rows() != 5_760
                    || view.gate_up.blocks() != 90
                    || view.down.rows() != 2_880
                    || view.down.blocks() != 90
                {
                    return Err(LLMError::ModelError(
                        "owner x8 runtime record dimensions are invalid".into(),
                    ));
                }
            }
            if warm_cpu_layers.contains(layer) {
                ledger.cpu_experts += identities.len() as u32;
                ledger.cpu_x8_bytes = ledger
                    .cpu_x8_bytes
                    .checked_add(record.payload_bytes())
                    .ok_or_else(|| LLMError::ModelError("CPU x8 ledger overflows".into()))?;
            }
            cpu_layers.insert(*layer, record);
        }

        ledger.stage = ConstructionStage::ExecutionReserve;
        verify_materialized_expert_weights(
            &placement,
            &layer_owner_experts,
            &remote_gpu_experts,
            &cpu_layers,
        )?;
        envelope.execution_reserve_plan.validate()?;
        ledger.execution_reserve_reviewed = true;
        ledger.execution_runtime_resources_materialized_at_construction = false;
        ledger.layer_owner_execution_materialized_before_admission_bytes = envelope
            .execution_reserve_plan
            .layer_owner
            .materialized_before_admission_bytes;
        ledger.remote_gpu_execution_materialized_before_admission_bytes = envelope
            .execution_reserve_plan
            .remote_gpu
            .materialized_before_admission_bytes;
        ledger.layer_owner_execution_planned_bytes = envelope
            .execution_reserve_plan
            .layer_owner
            .planned_owned_bytes;
        ledger.remote_gpu_execution_planned_bytes = envelope
            .execution_reserve_plan
            .remote_gpu
            .planned_owned_bytes;
        observe(&ledger)?;

        if ledger.layer_owner_dense_bytes != envelope.non_expert_payload_bytes
            || ledger.layer_owner_expert_bytes != envelope.layer_owner_native_expert_bytes
            || ledger.remote_gpu_expert_bytes != envelope.remote_gpu_native_expert_bytes
            || ledger.cpu_x8_bytes != envelope.cpu_x8_record_bytes
        {
            return Err(LLMError::ModelError(
                "capacity-one owner-selective byte ledger mismatch".into(),
            ));
        }
        let release_reports = catalog.release_reports();
        let mapping = catalog.mapping_activity();
        let partial_stats = partial_store.stats();
        let expected_cpu_experts = plan_index
            .cpu_by_layer
            .values()
            .map(Vec::len)
            .sum::<usize>();
        let publication_proof = OwnerSelectivePublicationProof {
            catalog_identity_exact: plan.catalog_sha256() == catalog.metadata_sha256(),
            action_coverage_complete: coverage.validate_complete().is_ok(),
            warm_elision_complete: coverage.elided_count()
                == warm_elision_proofs
                    .iter()
                    .map(|proof| proof.action_ids_sha256.len())
                    .sum::<usize>(),
            active_source_mappings: mapping.current,
            active_source_payload_fds: catalog.active_source_payload_fds(),
            source_payload_views: mapping.current,
            borrowed_source_slices: 0,
            source_inode_mappings: release_reports
                .iter()
                .map(|report| report.post_release.source_inode_mapping_count)
                .sum(),
            source_inode_pss_bytes: release_reports
                .iter()
                .map(|report| report.post_release.source_inode_pss_bytes)
                .sum(),
            partial_store_entries: partial_stats.current_count,
            partial_store_bytes: partial_stats.current_bytes,
            incomplete_cpu_experts: expected_cpu_experts
                .saturating_sub(ledger.cpu_experts as usize),
            task_temporaries: cache.capacity_one_temporary_count()?,
            pending_cuda_receipts: 0,
            quarantined_cuda_receipts: 0,
            cold_records_directory_synced: all_validations.len() == plan_index.cpu_by_layer.len(),
            records_freshly_validated: cpu_layers.len() == plan_index.cpu_by_layer.len(),
            runtime_maps_after_source_release: mapping.current == 0,
            stable_device_ownership_complete: true,
            bounded_journal_complete: release_reports.len() == plan.shards().len()
                && !catalog.release_report_overflowed()
                && release_reports
                    .iter()
                    .all(|report| report.terminal_audit_complete),
            visibility_contract_unchanged: true,
        };
        publication_proof.validate()?;
        let capacity_one_evidence = CapacityOneConstructionEvidence {
            policy_sha256: policy_sha256.into(),
            catalog_sha256: catalog.metadata_sha256().into(),
            plan_sha256: plan.plan_sha256().into(),
            active_mapping_high_water: mapping.high_water,
            mapped_byte_high_water: mapping.mapped_byte_high_water,
            plan_partial_high_water_count,
            plan_partial_high_water_bytes,
            plan_owner_partial_high_waters,
            partial_high_water_count: partial_stats.high_water_count,
            partial_high_water_bytes: partial_stats.high_water_bytes,
            warm_elision_proofs,
            shard_releases: release_reports,
            publication_proof,
        };
        if capacity_one_evidence.active_mapping_high_water != 1 {
            return Err(LLMError::ModelError(
                "capacity-one mapping high-water is not exactly one".into(),
            ));
        }
        ledger.stage = ConstructionStage::Publish;
        observe(&ledger)?;
        Ok(OwnerSelectiveModel {
            cpu_layers,
            remote_gpu_experts,
            layer_owner_experts,
            resident_router_sources,
            layer_owner_dense,
            remote_executor: Some(remote_executor),
            layer_owner_executor: Some(layer_owner_executor),
            execution_quarantined: false,
            native_metadata,
            placement,
            envelope,
            ledger,
            capacity_one_evidence: Some(capacity_one_evidence),
            checkpoint_release_evidence: None,
        })
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
        let native_metadata = OwnerSelectiveNativeMetadata::from_checkpoint(&checkpoint)?;
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
        let mut router_pairs =
            RouterPairAccumulator::<CudaSlice<u8>>::new(checkpoint.config().num_hidden_layers)?;
        let (router_weight_bytes, router_bias_bytes) =
            exact_router_weight_surface_bytes(checkpoint.config().num_experts)?;
        for mapping in checkpoint
            .mappings()
            .filter(|mapping| !mapping.runtime.contains(".mlp.experts."))
        {
            let router_component = classify_router_dense_tensor(
                &mapping.runtime,
                checkpoint.config().num_hidden_layers,
            )?;
            if let Some((_, component)) = router_component {
                let expected = match component {
                    RouterDenseComponent::Weight => router_weight_bytes,
                    RouterDenseComponent::Bias => router_bias_bytes,
                };
                if mapping.bytes != expected {
                    return Err(LLMError::ModelError(format!(
                        "router tensor {} has {} bytes, expected {expected}",
                        mapping.runtime, mapping.bytes
                    )));
                }
            }
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
            if let Some((layer, component)) = router_component {
                router_pairs.insert(layer, component, allocation)?;
            } else {
                layer_owner_dense.push(LayerOwnerDenseTensor {
                    name: mapping.runtime.clone(),
                    logical_bytes: mapping.bytes as u64,
                    allocation,
                });
            }
            if injected_fault == Some(ConstructionStage::LayerOwnerDense) {
                ledger.stage = ConstructionStage::LayerOwnerDense;
                observe(&ledger)?;
                inject_construction_fault(injected_fault, ConstructionStage::LayerOwnerDense)?;
            }
        }
        let resident_router_layers = router_pairs
            .finish()?
            .into_iter()
            .map(|(layer, weight, bias)| {
                Ok((
                    layer,
                    ResidentExactRouterWeights::new(
                        placement.layer_owner().stable_id.clone(),
                        checkpoint.config().num_experts,
                        weight,
                        bias,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let resident_router_sources = ResidentExactRouterSources::new(
            checkpoint.config().num_hidden_layers,
            checkpoint.config().num_experts,
            placement.layer_owner().stable_id.clone(),
            resident_router_layers,
        )?;
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
            if layer_owner_experts.insert(key, Arc::new(weights)).is_some() {
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
            if remote_gpu_experts.insert(key, Arc::new(weights)).is_some() {
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
        verify_materialized_expert_weights(
            &placement,
            &layer_owner_experts,
            &remote_gpu_experts,
            &cpu_layers,
        )?;
        envelope.execution_reserve_plan.validate()?;
        ledger.execution_reserve_reviewed = true;
        ledger.execution_runtime_resources_materialized_at_construction = false;
        ledger.layer_owner_execution_materialized_before_admission_bytes = envelope
            .execution_reserve_plan
            .layer_owner
            .materialized_before_admission_bytes;
        ledger.remote_gpu_execution_materialized_before_admission_bytes = envelope
            .execution_reserve_plan
            .remote_gpu
            .materialized_before_admission_bytes;
        ledger.layer_owner_execution_planned_bytes = envelope
            .execution_reserve_plan
            .layer_owner
            .planned_owned_bytes;
        ledger.remote_gpu_execution_planned_bytes = envelope
            .execution_reserve_plan
            .remote_gpu
            .planned_owned_bytes;
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
        let checkpoint_release_evidence = checkpoint.release_with_advice()?;
        let model = OwnerSelectiveModel {
            cpu_layers,
            remote_gpu_experts,
            layer_owner_experts,
            resident_router_sources,
            layer_owner_dense,
            remote_executor: Some(remote_executor),
            layer_owner_executor: Some(layer_owner_executor),
            execution_quarantined: false,
            native_metadata,
            placement,
            envelope,
            ledger,
            capacity_one_evidence: None,
            checkpoint_release_evidence: Some(checkpoint_release_evidence),
        };
        // The published model owns no checkpoint store or payload mapping.
        // The monolithic control retains the complete address window through
        // construction, then performs the same ordered release contract used
        // by the bounded-shard path.
        Ok(model)
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

fn validate_manifest_identity_catalog(
    native: &GptOssNativeCatalogMap,
    manifest: &GptOssExpertPlacementManifestV1,
) -> Result<()> {
    let config = native.config();
    if manifest.model.revision != native.revision()
        || manifest.model.config_sha256 != native.config_sha256()
        || manifest.model.index_sha256 != native.metadata_sha256()
        || manifest.model.mapping_sha256 != native.mapping_sha256()
        || usize::from(manifest.model.num_layers) != config.num_hidden_layers
        || usize::from(manifest.model.experts_per_layer) != config.num_experts
        || usize::from(manifest.model.hidden_size) != config.hidden_size
        || usize::from(manifest.model.intermediate_size) != config.intermediate_size
        || usize::from(manifest.model.top_k) != config.experts_per_token
    {
        return Err(LLMError::ModelError(
            "placement manifest does not identify the native shard catalog exactly".into(),
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
    upload_pinned_chunks_classified(stream, pinned, source, label).map_err(|failure| failure.error)
}

struct ConstructionUploadFailure {
    error: LLMError,
    terminal_unproven: bool,
}

impl ConstructionUploadFailure {
    fn before_enqueue(error: LLMError) -> Self {
        Self {
            error,
            terminal_unproven: false,
        }
    }

    fn after_enqueue(error: LLMError) -> Self {
        Self {
            error,
            terminal_unproven: true,
        }
    }
}

fn upload_pinned_chunks_classified(
    stream: &std::sync::Arc<CudaStream>,
    pinned: &mut TrackedPinnedHostSlice,
    source: &[u8],
    label: &str,
) -> std::result::Result<CudaSlice<u8>, ConstructionUploadFailure> {
    // SAFETY: the uninitialized device allocation is written completely by
    // the chunk loop before it becomes part of a published model.
    let mut destination = unsafe { stream.alloc::<u8>(source.len()) }
        .map_err(cuda_error("owner-selective dense allocation"))
        .map_err(ConstructionUploadFailure::before_enqueue)?;
    for (chunk_index, source_chunk) in source.chunks(pinned.allocation.len()).enumerate() {
        let start = chunk_index * pinned.allocation.len();
        let end = start + source_chunk.len();
        pinned
            .allocation
            .as_mut_slice()
            .map_err(cuda_error("owner-selective pinned write access"))
            .map_err(ConstructionUploadFailure::before_enqueue)?[..source_chunk.len()]
            .copy_from_slice(source_chunk);
        let mut target = destination.slice_mut(start..end);
        if source_chunk.len() == pinned.allocation.len() {
            stream
                .memcpy_htod(&pinned.allocation, &mut target)
                .map_err(cuda_error("owner-selective pinned H2D"))
                .map_err(ConstructionUploadFailure::after_enqueue)?;
        } else {
            // The backing address remains page-locked. cudarc cannot express a
            // subview of PinnedHostSlice, so the bounded tail uses a borrowed
            // slice and is synchronized before reuse/drop.
            let tail = &pinned
                .allocation
                .as_slice()
                .map_err(cuda_error("owner-selective pinned tail access"))
                .map_err(ConstructionUploadFailure::before_enqueue)?[..source_chunk.len()];
            stream
                .memcpy_htod(tail, &mut target)
                .map_err(cuda_error("owner-selective pinned tail H2D"))
                .map_err(ConstructionUploadFailure::after_enqueue)?;
        }
        stream
            .synchronize()
            .map_err(cuda_error("owner-selective dense upload drain"))
            .map_err(ConstructionUploadFailure::after_enqueue)?;
    }
    if destination.len() != source.len() {
        return Err(ConstructionUploadFailure::before_enqueue(
            LLMError::GpuError(format!("dense tensor {label} allocation length mismatch")),
        ));
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

fn verify_materialized_expert_weights(
    placement: &ResolvedExpertPlacement,
    layer_owner: &BTreeMap<GptOssExpertKey, Arc<CudaSelectedExpertWeights>>,
    remote: &BTreeMap<GptOssExpertKey, Arc<CudaSelectedExpertWeights>>,
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

/// Synthetic-only probe for the production router classifier and pair
/// accumulator. It is absent from normal runtime builds.
#[cfg(feature = "heterogeneous-test-faults")]
pub fn validate_router_publication_fixture_for_test(
    num_layers: usize,
    entries: &[(String, u8)],
) -> Result<Vec<(usize, u8, u8)>> {
    let mut pairs = RouterPairAccumulator::new(num_layers)?;
    for (name, value) in entries {
        let (layer, component) =
            classify_router_dense_tensor(name, num_layers)?.ok_or_else(|| {
                LLMError::ModelError(format!(
                    "synthetic publication entry {name} is not a router"
                ))
            })?;
        pairs.insert(layer, component, *value)?;
    }
    pairs.finish()
}

/// Synthetic-only probe for the payload-free metadata validator. It is absent
/// from normal runtime builds.
#[cfg(feature = "heterogeneous-test-faults")]
pub fn validate_native_metadata_fixture_for_test(
    config: GptOssNativeConfig,
    revision: String,
    config_sha256: String,
    metadata_sha256: String,
    mapping_sha256: String,
) -> Result<()> {
    OwnerSelectiveNativeMetadata {
        config,
        revision,
        config_sha256,
        metadata_sha256,
        mapping_sha256,
    }
    .validate()
}

/// Synthetic-only order probe for publication validation. It is absent from
/// normal runtime builds.
#[cfg(feature = "heterogeneous-test-faults")]
pub fn validate_router_publication_order_for_test(
    expected_layers: usize,
    observed_layers: &[usize],
) -> Result<()> {
    validate_router_publication_order(expected_layers, observed_layers.iter().copied())
}
