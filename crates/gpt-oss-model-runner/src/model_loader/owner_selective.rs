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
    CpuOwnerLayerRecord, CpuOwnerRepackCache, OWNER_EXPERT_BYTES, OWNER_REPACK_TEMP_BYTES_MAX,
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
use crate::heterogeneous::router::exact_router_owned_device_bytes;

use super::gpt_oss_native::GptOssCheckpointView;

pub const OWNER_SELECTIVE_PINNED_UPLOAD_BYTES: usize = 16 * 1024 * 1024;
pub const OWNER_SELECTIVE_TEMPORARY_CAP_BYTES: usize = 256 * 1024 * 1024;
pub const OWNER_SELECTIVE_GPU_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub const OWNER_SELECTIVE_PROOF_CONTEXT_CAP: usize = 4_096;
pub const OWNER_SELECTIVE_DECODE_MAX_ROWS: usize = 1;

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
        let execution_reserve_plan =
            ExecutionReservePlan::from_config(checkpoint.config(), remote_layers)?;
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
    layer_owner_dense: Vec<LayerOwnerDenseTensor>,
    remote_executor: Option<CudaSelectedExpertExecutor>,
    layer_owner_executor: Option<CudaSelectedExpertExecutor>,
    execution_quarantined: bool,
    checkpoint: GptOssCheckpointView,
    placement: ResolvedExpertPlacement,
    envelope: OwnerSelectiveEnvelope,
    ledger: ConstructionLedger,
}

pub(crate) struct OwnerSelectiveExecutionParts<'a> {
    pub cpu_layers: &'a BTreeMap<u16, CpuOwnerLayerRecord>,
    pub remote_gpu_experts: &'a BTreeMap<GptOssExpertKey, Arc<CudaSelectedExpertWeights>>,
    pub layer_owner_experts: &'a BTreeMap<GptOssExpertKey, Arc<CudaSelectedExpertWeights>>,
    pub remote_executor: &'a mut CudaSelectedExpertExecutor,
    pub layer_owner_executor: &'a mut CudaSelectedExpertExecutor,
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
        Ok(OwnerSelectiveModel {
            cpu_layers,
            remote_gpu_experts,
            layer_owner_experts,
            layer_owner_dense,
            remote_executor: Some(remote_executor),
            layer_owner_executor: Some(layer_owner_executor),
            execution_quarantined: false,
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
