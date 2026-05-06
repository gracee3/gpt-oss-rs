//! Non-executing CUDA resource skeletons for future layer sharding.
//!
//! The pure plan in this module is always available. CUDA resource construction
//! is feature-gated and deliberately stops at context/stream/cuBLAS/kernel-loader
//! ownership; it does not upload model tensors, allocate KV cache, or construct a
//! runner.

use crate::device_map::DeviceId;
use crate::fused_f16::{f16_scratch_element_counts, F16ScratchElementCounts};
use crate::model_loader::{ShardWeightStore, ShardWeightStorePlan};
use crate::shard_plan::{
    ShardTensorManifest, ShardedKvCachePlan, ShardedModelPlan, ShardedUploadManifest,
};
use std::collections::BTreeMap;
use std::str::FromStr;

pub const RUNTIME_METADATA_DEFERRED_REASON: &str =
    "request-shaped metadata packing buffers require batch/sequence inputs";
pub const FUSED_F16_DEFERRED_REASON: &str =
    "GpuModelRunner::fuse_weights assumes full-model runner-owned weight containers and full-runner layer indexing";
pub const FUSED_F16_CASTS_DEFERRED_REASON: &str =
    "fused QKV/gate-up buffers and f16 layernorm/postnorm/bias conversions allocated; final/embed conversions remain deferred by cast/helper boundary";
pub const F16_SCRATCH_DEFERRED_REASON: &str =
    "bench-only f16 scratch allocation requires the CUDA allocation pass; GpuModelRunner::F16LayerScratch remains private runner state";
pub const MOE_GPU_UPLOAD_DEFERRED_REASON: &str =
    "bench-only GPT-OSS MoE GPU upload is plan/status only; per-shard U8 host maps are counted but not retained/uploaded yet";
pub const LAYER_SKELETON_EXECUTABLE_DEFERRED_REASON: &str =
    "bench-only layer skeleton is non-executing; executable GpuTransformerLayer construction is deferred";

/// CUDA-free plan for one shard's resource island.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardResourcePlan {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub owns_embeddings: bool,
    pub owns_final_head: bool,
}

/// CUDA-free plan for all shard resource islands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedCudaResourcePlan {
    pub shards: Vec<CudaShardResourcePlan>,
}

/// Public status for a constructed resource island.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardResourceStatus {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub owns_embeddings: bool,
    pub owns_final_head: bool,
}

/// Public status for constructed sharded CUDA resources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedCudaResourceStatus {
    pub shards: Vec<CudaShardResourceStatus>,
}

/// CUDA-free configuration for shard-local runtime buffer planning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeRuntimeBufferConfig {
    pub head_dim: usize,
    pub max_position: usize,
    pub rope_theta: f32,
}

/// CUDA-free configuration for shard-local KV cache allocation planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheAllocationConfig {
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_gpu_blocks: usize,
    pub block_size: usize,
}

/// Synthetic metadata shape supported by the first split allocation smoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataMode {
    Decode,
}

/// CUDA-free configuration for synthetic request-shaped metadata allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataAllocationConfig {
    pub mode: MetadataMode,
    pub num_tokens: usize,
    pub num_seqs: usize,
    pub context_len: usize,
    pub block_size: usize,
    pub max_position: usize,
}

/// Metadata allocation state for the non-executing runtime buffer skeleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMetadataStatus {
    Allocated,
    Deferred,
    NotApplicable,
}

/// Allocation state for the non-executing fused f16/scratch skeleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedF16AllocationStatus {
    Allocated,
    AvailableFromUploadedF16,
    Deferred,
    NotApplicable,
}

/// Upload state for the non-executing GPT-OSS MoE GPU upload skeleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeGpuUploadStatus {
    Uploaded,
    Deferred,
    NotApplicable,
}

/// Readiness state for the non-executing layer-construction skeleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerConstructionReadinessStatus {
    SkeletonComplete,
    Allocated,
    Deferred,
    NotApplicable,
    NotRequested,
    NotConstructed,
    Blocked,
}

/// Non-executing blocker/deferred note for a future executable layer build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerConstructionBlocker {
    pub code: String,
    pub detail: String,
}

/// CUDA-free plan for one shard's RoPE/metadata runtime buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardRuntimeBufferPlan {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub owns_embeddings: bool,
    pub owns_final_head: bool,
    pub rope_cos_elements: usize,
    pub rope_sin_elements: usize,
    pub rope_total_bytes: usize,
    pub metadata_status: RuntimeMetadataStatus,
    pub metadata_deferred_reason: Option<String>,
}

/// CUDA-free plan for shard-local runtime buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedRuntimeBufferPlan {
    pub shards: Vec<CudaShardRuntimeBufferPlan>,
}

/// Public status for one shard's runtime buffer skeleton.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardRuntimeBufferStatus {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub owns_embeddings: bool,
    pub owns_final_head: bool,
    pub rope_allocated: bool,
    pub rope_cos_elements: usize,
    pub rope_sin_elements: usize,
    pub rope_total_bytes: usize,
    pub metadata_allocated: bool,
    pub metadata_status: RuntimeMetadataStatus,
    pub metadata_deferred_reason: Option<String>,
    pub runtime_buffer_error: Option<String>,
}

/// Public status for all shard-local runtime buffer skeletons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedRuntimeBufferStatus {
    pub shards: Vec<CudaShardRuntimeBufferStatus>,
}

/// CUDA-free plan for one shard-local layer KV cache entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaLayerKvCacheAllocationPlan {
    pub absolute_layer_idx: usize,
    pub local_cache_idx: usize,
    pub key_elements: usize,
    pub value_elements: usize,
    pub key_bytes: usize,
    pub value_bytes: usize,
}

/// CUDA-free plan for one shard's KV cache allocations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardKvCacheAllocationPlan {
    pub device_id: DeviceId,
    pub entries: Vec<CudaLayerKvCacheAllocationPlan>,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_gpu_blocks: usize,
    pub block_size: usize,
    pub key_total_bytes: usize,
    pub value_total_bytes: usize,
    pub total_bytes: usize,
}

/// CUDA-free plan for shard-local KV cache allocations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedKvCacheAllocationPlan {
    pub shards: Vec<CudaShardKvCacheAllocationPlan>,
}

/// Public status for one shard-local layer KV cache allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaLayerKvCacheAllocationStatus {
    pub absolute_layer_idx: usize,
    pub local_cache_idx: usize,
    pub key_bytes: usize,
    pub value_bytes: usize,
}

/// Public status for one shard's KV cache allocation skeleton.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardKvCacheAllocationStatus {
    pub device_id: DeviceId,
    pub kv_cache_allocated: bool,
    pub entries: Vec<CudaLayerKvCacheAllocationStatus>,
    pub key_total_bytes: usize,
    pub value_total_bytes: usize,
    pub total_bytes: usize,
    pub kv_cache_error: Option<String>,
}

/// Public status for all shard-local KV cache allocation skeletons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedKvCacheAllocationStatus {
    pub shards: Vec<CudaShardKvCacheAllocationStatus>,
}

/// CUDA-free plan for one shard's synthetic packed metadata buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardMetadataAllocationPlan {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub owns_embeddings: bool,
    pub owns_final_head: bool,
    pub mode: MetadataMode,
    pub num_tokens: usize,
    pub num_seqs: usize,
    pub context_len: usize,
    pub block_size: usize,
    pub graph_max_blocks: usize,
    pub max_context_len: usize,
    pub token_ids_len: usize,
    pub positions_len: usize,
    pub context_lens_len: usize,
    pub block_tables_len: usize,
    pub slot_mapping_len: usize,
    pub seq_start_pos_len: usize,
    pub packed_elements: usize,
    pub packed_bytes: usize,
}

/// CUDA-free plan for shard-local synthetic metadata allocations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedMetadataAllocationPlan {
    pub shards: Vec<CudaShardMetadataAllocationPlan>,
}

/// Public status for one shard's synthetic metadata allocation skeleton.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardMetadataAllocationStatus {
    pub device_id: DeviceId,
    pub metadata_allocated: bool,
    pub metadata_status: RuntimeMetadataStatus,
    pub mode: MetadataMode,
    pub num_tokens: usize,
    pub num_seqs: usize,
    pub graph_max_blocks: usize,
    pub packed_elements: usize,
    pub packed_bytes: usize,
    pub token_ids_len: usize,
    pub positions_len: usize,
    pub context_lens_len: usize,
    pub block_tables_len: usize,
    pub slot_mapping_len: usize,
    pub seq_start_pos_len: usize,
    pub max_context_len: usize,
    pub metadata_error: Option<String>,
}

/// Public status for all synthetic metadata allocation skeletons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedMetadataAllocationStatus {
    pub shards: Vec<CudaShardMetadataAllocationStatus>,
}

/// CUDA-free configuration for planned f16 scratch allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F16ScratchAllocationConfig {
    pub hidden_size: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub intermediate_size: usize,
    pub max_tokens: usize,
}

/// CUDA-free status for one f16 scratch buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F16ScratchBufferStatus {
    pub elements: usize,
    pub bytes: usize,
}

/// CUDA-free per-buffer status for the eight f16 scratch buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F16ScratchBufferStatuses {
    pub qkv: F16ScratchBufferStatus,
    pub attn_out: F16ScratchBufferStatus,
    pub o_proj: F16ScratchBufferStatus,
    pub normed: F16ScratchBufferStatus,
    pub residual: F16ScratchBufferStatus,
    pub gate_up: F16ScratchBufferStatus,
    pub silu_out: F16ScratchBufferStatus,
    pub down: F16ScratchBufferStatus,
}

/// CUDA-free plan for one layer's fused/preconverted f16 boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaLayerFusedF16AllocationPlan {
    pub absolute_layer_idx: usize,
    pub local_layer_idx: usize,
    pub fused_qkv_planned: bool,
    pub fused_gate_up_planned: bool,
    pub has_u8_expert_tensors: bool,
    pub f16_layernorm_planned: bool,
    pub f16_postnorm_planned: bool,
    pub f16_qkv_bias_planned: bool,
    pub f16_o_proj_bias_planned: bool,
}

/// Public status for one layer's fused/preconverted f16 boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaLayerFusedF16AllocationStatus {
    pub absolute_layer_idx: usize,
    pub local_layer_idx: usize,
    pub fused_qkv_allocated: bool,
    pub fused_qkv_status: FusedF16AllocationStatus,
    pub fused_qkv_bytes: usize,
    pub fused_gate_up_allocated: bool,
    pub fused_gate_up_status: FusedF16AllocationStatus,
    pub fused_gate_up_bytes: usize,
    pub layernorm_f16_status: FusedF16AllocationStatus,
    pub layernorm_f16_bytes: usize,
    pub postnorm_f16_status: FusedF16AllocationStatus,
    pub postnorm_f16_bytes: usize,
    pub qkv_bias_f16_status: FusedF16AllocationStatus,
    pub qkv_bias_f16_bytes: usize,
    pub o_proj_bias_f16_status: FusedF16AllocationStatus,
    pub o_proj_bias_f16_bytes: usize,
    pub layer_error: Option<String>,
}

/// CUDA-free plan for one shard's fused/preconverted f16 and scratch boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardFusedF16AllocationPlan {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub owns_embeddings: bool,
    pub owns_final_head: bool,
    pub fused_qkv_weight_count: usize,
    pub fused_gate_up_weight_count: usize,
    pub f16_layernorm_count: usize,
    pub f16_postnorm_count: usize,
    pub f16_qkv_bias_count: usize,
    pub f16_o_proj_bias_count: usize,
    pub embedding_f16_planned: bool,
    pub final_norm_f16_planned: bool,
    pub fused_qkv_total_bytes: usize,
    pub fused_gate_up_total_bytes: usize,
    pub f16_layernorm_total_bytes: usize,
    pub f16_postnorm_total_bytes: usize,
    pub f16_qkv_bias_total_bytes: usize,
    pub f16_o_proj_bias_total_bytes: usize,
    pub fused_layer_absolute_indices: Vec<usize>,
    pub fused_layer_plans: Vec<CudaLayerFusedF16AllocationPlan>,
    pub fused_total_bytes: usize,
    pub fused_status: FusedF16AllocationStatus,
    pub fused_deferred_reason: Option<String>,
    pub f16_scratch_status: FusedF16AllocationStatus,
    pub f16_scratch_max_tokens: Option<usize>,
    pub f16_scratch_total_elements: usize,
    pub f16_scratch_bytes: usize,
    pub f16_scratch_buffers: Option<F16ScratchBufferStatuses>,
    pub f16_scratch_deferred_reason: Option<String>,
}

/// CUDA-free plan for shard-local fused/preconverted f16 and scratch status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedFusedF16AllocationPlan {
    pub shards: Vec<CudaShardFusedF16AllocationPlan>,
}

/// Public status for one shard's fused/preconverted f16 and scratch skeleton.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardFusedF16AllocationStatus {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub owns_embeddings: bool,
    pub owns_final_head: bool,
    pub fused_f16_allocated: bool,
    pub fused_f16_status: FusedF16AllocationStatus,
    pub fused_qkv_weight_count: usize,
    pub fused_gate_up_weight_count: usize,
    pub f16_layernorm_count: usize,
    pub f16_postnorm_count: usize,
    pub f16_qkv_bias_count: usize,
    pub f16_o_proj_bias_count: usize,
    pub embedding_f16_allocated: bool,
    pub embedding_f16_status: FusedF16AllocationStatus,
    pub embedding_f16_bytes: usize,
    pub embedding_f16_source: Option<String>,
    pub final_norm_f16_allocated: bool,
    pub final_norm_f16_status: FusedF16AllocationStatus,
    pub final_norm_f16_bytes: usize,
    pub final_norm_f16_source: Option<String>,
    pub fused_qkv_total_bytes: usize,
    pub fused_gate_up_total_bytes: usize,
    pub f16_layernorm_total_bytes: usize,
    pub f16_postnorm_total_bytes: usize,
    pub f16_qkv_bias_total_bytes: usize,
    pub f16_o_proj_bias_total_bytes: usize,
    pub fused_total_bytes: usize,
    pub fused_layer_absolute_indices: Vec<usize>,
    pub fused_layer_statuses: Vec<CudaLayerFusedF16AllocationStatus>,
    pub fused_deferred_reason: Option<String>,
    pub fused_error: Option<String>,
    pub f16_scratch_allocated: bool,
    pub f16_scratch_status: FusedF16AllocationStatus,
    pub f16_scratch_total_elements: usize,
    pub f16_scratch_bytes: usize,
    pub f16_scratch_max_tokens: Option<usize>,
    pub f16_scratch_buffers: Option<F16ScratchBufferStatuses>,
    pub f16_scratch_deferred_reason: Option<String>,
    pub f16_scratch_error: Option<String>,
}

/// Public status for all shard-local fused/preconverted f16 and scratch skeletons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedFusedF16AllocationStatus {
    pub shards: Vec<CudaShardFusedF16AllocationStatus>,
}

/// CUDA-free plan for one GPT-OSS MoE layer's future GPU upload boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaLayerMoeGpuUploadPlan {
    pub absolute_layer_idx: usize,
    pub local_layer_idx: usize,
    pub gate_up_proj_blocks_planned: bool,
    pub gate_up_proj_scales_planned: bool,
    pub down_proj_blocks_planned: bool,
    pub down_proj_scales_planned: bool,
    pub gate_up_proj_blocks_bytes: usize,
    pub gate_up_proj_scales_bytes: usize,
    pub down_proj_blocks_bytes: usize,
    pub down_proj_scales_bytes: usize,
    pub router_planned: bool,
    pub expert_bias_planned: bool,
    pub partial_u8_payload: bool,
}

/// Public status for one GPT-OSS MoE layer's future GPU upload boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaLayerMoeGpuUploadStatus {
    pub absolute_layer_idx: usize,
    pub local_layer_idx: usize,
    pub gate_up_proj_blocks_status: MoeGpuUploadStatus,
    pub gate_up_proj_scales_status: MoeGpuUploadStatus,
    pub down_proj_blocks_status: MoeGpuUploadStatus,
    pub down_proj_scales_status: MoeGpuUploadStatus,
    pub gate_up_proj_blocks_bytes: usize,
    pub gate_up_proj_scales_bytes: usize,
    pub down_proj_blocks_bytes: usize,
    pub down_proj_scales_bytes: usize,
    pub router_status: MoeGpuUploadStatus,
    pub expert_bias_status: MoeGpuUploadStatus,
    pub supports_gpu_decode_status: String,
    pub layer_error: Option<String>,
}

/// CUDA-free plan for one shard's future GPT-OSS MoE GPU upload boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardMoeGpuUploadPlan {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub moe_layer_count: usize,
    pub moe_u8_host_tensor_count: usize,
    pub moe_u8_host_bytes: usize,
    pub moe_router_tensor_count: usize,
    pub moe_bias_tensor_count: usize,
    pub moe_layer_plans: Vec<CudaLayerMoeGpuUploadPlan>,
    pub moe_gpu_status: MoeGpuUploadStatus,
    pub moe_gpu_deferred_reason: Option<String>,
}

/// CUDA-free plan for future shard-local GPT-OSS MoE GPU upload status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedMoeGpuUploadPlan {
    pub shards: Vec<CudaShardMoeGpuUploadPlan>,
}

/// Public status for one shard's future GPT-OSS MoE GPU upload boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardMoeGpuUploadStatus {
    pub device_id: DeviceId,
    pub moe_gpu_uploaded: bool,
    pub moe_gpu_status: MoeGpuUploadStatus,
    pub moe_layer_count: usize,
    pub moe_u8_host_tensor_count: usize,
    pub moe_u8_gpu_tensor_count: usize,
    pub moe_u8_host_bytes: usize,
    pub moe_u8_gpu_bytes: usize,
    pub moe_router_tensor_count: usize,
    pub moe_bias_tensor_count: usize,
    pub moe_layer_statuses: Vec<CudaLayerMoeGpuUploadStatus>,
    pub moe_gpu_deferred_reason: Option<String>,
    pub moe_gpu_error: Option<String>,
}

/// Public status for all future shard-local GPT-OSS MoE GPU upload boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedMoeGpuUploadStatus {
    pub shards: Vec<CudaShardMoeGpuUploadStatus>,
}

/// CUDA-free plan for one shard-owned layer skeleton.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaLayerConstructionPlan {
    pub absolute_layer_idx: usize,
    pub local_layer_idx: usize,
    pub owns_layer: bool,
    pub required_f16_projection_tensor_names: Vec<String>,
    pub missing_required_f16_projection_tensor_names: Vec<String>,
    pub required_f32_norm_bias_tensor_names: Vec<String>,
    pub has_moe_u8_payload: bool,
}

/// CUDA-free plan for one shard's non-executing layer skeletons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardLayerConstructionPlan {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub layer_plans: Vec<CudaLayerConstructionPlan>,
}

/// CUDA-free plan for all shard-local non-executing layer skeletons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedLayerConstructionPlan {
    pub shards: Vec<CudaShardLayerConstructionPlan>,
}

/// Public status for one non-executing layer skeleton.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaLayerConstructionStatus {
    pub absolute_layer_idx: usize,
    pub local_layer_idx: usize,
    pub owns_layer: bool,
    pub layer_config_status: LayerConstructionReadinessStatus,
    pub required_f16_projection_status: LayerConstructionReadinessStatus,
    pub required_f32_norm_bias_status: LayerConstructionReadinessStatus,
    pub rope_status: LayerConstructionReadinessStatus,
    pub kv_cache_status: LayerConstructionReadinessStatus,
    pub metadata_status: LayerConstructionReadinessStatus,
    pub fused_qkv_status: LayerConstructionReadinessStatus,
    pub layernorm_f16_status: LayerConstructionReadinessStatus,
    pub postnorm_f16_status: LayerConstructionReadinessStatus,
    pub qkv_bias_f16_status: LayerConstructionReadinessStatus,
    pub o_proj_bias_f16_status: LayerConstructionReadinessStatus,
    pub f16_scratch_status: LayerConstructionReadinessStatus,
    pub moe_u8_upload_status: LayerConstructionReadinessStatus,
    pub moe_router_status: LayerConstructionReadinessStatus,
    pub moe_expert_bias_status: LayerConstructionReadinessStatus,
    pub supports_gpu_decode_status: String,
    pub executable_layer_status: LayerConstructionReadinessStatus,
    pub executable_layer_deferred_reason: Option<String>,
    pub blockers: Vec<LayerConstructionBlocker>,
}

/// Public status for one shard's non-executing layer skeletons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaShardLayerConstructionStatus {
    pub device_id: DeviceId,
    pub layer_skeleton_built: bool,
    pub layer_skeleton_status: LayerConstructionReadinessStatus,
    pub layer_skeleton_count: usize,
    pub layer_skeleton_ready_count: usize,
    pub layer_skeleton_blocked_count: usize,
    pub layer_skeleton_deferred_count: usize,
    pub layer_skeletons: Vec<CudaLayerConstructionStatus>,
    pub layer_skeleton_error: Option<String>,
}

/// Public status for all non-executing layer skeletons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedLayerConstructionStatus {
    pub shards: Vec<CudaShardLayerConstructionStatus>,
}

impl RopeRuntimeBufferConfig {
    pub fn new(head_dim: usize, max_position: usize, rope_theta: f32) -> Result<Self, String> {
        if head_dim == 0 {
            return Err("head_dim must be non-zero for RoPE runtime buffers".into());
        }
        if head_dim % 2 != 0 {
            return Err(format!(
                "head_dim must be even for RoPE runtime buffers, got {head_dim}"
            ));
        }
        if max_position == 0 {
            return Err("max_position must be non-zero for RoPE runtime buffers".into());
        }
        if !rope_theta.is_finite() || rope_theta <= 0.0 {
            return Err(format!(
                "rope_theta must be finite and positive for RoPE runtime buffers, got {rope_theta}"
            ));
        }

        Ok(Self {
            head_dim,
            max_position,
            rope_theta,
        })
    }

    pub fn runtime_max_position(&self) -> usize {
        self.max_position.min(8192)
    }

    pub fn rope_half_dim(&self) -> usize {
        self.head_dim / 2
    }

    pub fn rope_table_elements(&self) -> usize {
        self.runtime_max_position() * self.rope_half_dim()
    }

    pub fn rope_total_bytes(&self) -> usize {
        self.rope_table_elements() * 2 * std::mem::size_of::<f32>()
    }
}

impl KvCacheAllocationConfig {
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        num_gpu_blocks: usize,
        block_size: usize,
    ) -> Result<Self, String> {
        if num_kv_heads == 0 {
            return Err("num_kv_heads must be non-zero for KV cache allocation".into());
        }
        if head_dim == 0 {
            return Err("head_dim must be non-zero for KV cache allocation".into());
        }
        if num_gpu_blocks == 0 {
            return Err("num_gpu_blocks must be non-zero for KV cache allocation".into());
        }
        if block_size == 0 {
            return Err("block_size must be non-zero for KV cache allocation".into());
        }

        Ok(Self {
            num_kv_heads,
            head_dim,
            num_gpu_blocks,
            block_size,
        })
    }

    pub fn elements_per_layer_cache(&self) -> usize {
        self.num_gpu_blocks * self.block_size * self.num_kv_heads * self.head_dim
    }

    pub fn bytes_per_layer_cache(&self) -> usize {
        self.elements_per_layer_cache() * std::mem::size_of::<half::f16>()
    }
}

impl MetadataMode {
    pub fn as_str(self) -> &'static str {
        match self {
            MetadataMode::Decode => "decode",
        }
    }
}

impl FromStr for MetadataMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "decode" => Ok(Self::Decode),
            other => Err(format!(
                "unsupported metadata mode {other:?}; only decode is supported"
            )),
        }
    }
}

impl MetadataAllocationConfig {
    pub fn new_decode(
        num_tokens: usize,
        num_seqs: usize,
        context_len: usize,
        block_size: usize,
        max_position: usize,
        kv_cache_config: Option<&KvCacheAllocationConfig>,
    ) -> Result<Self, String> {
        if num_tokens == 0 {
            return Err("metadata-num-tokens must be non-zero for decode metadata".into());
        }
        if num_seqs == 0 {
            return Err("metadata-num-seqs must be non-zero for decode metadata".into());
        }
        if num_tokens != num_seqs {
            return Err(format!(
                "decode metadata requires metadata-num-tokens == metadata-num-seqs, got {num_tokens} and {num_seqs}"
            ));
        }
        if context_len == 0 {
            return Err("metadata-context-len must be non-zero for decode metadata".into());
        }
        if block_size == 0 {
            return Err("metadata-block-size must be non-zero for decode metadata".into());
        }
        if max_position == 0 {
            return Err("max_position must be non-zero for decode metadata".into());
        }

        let config = Self {
            mode: MetadataMode::Decode,
            num_tokens,
            num_seqs,
            context_len,
            block_size,
            max_position,
        };

        if config.required_blocks_per_seq() > config.graph_max_blocks() {
            return Err(format!(
                "metadata-context-len ({context_len}) requires {} block(s), exceeding graph_max_blocks ({}) derived from max_position ({max_position}) and metadata-block-size ({block_size})",
                config.required_blocks_per_seq(),
                config.graph_max_blocks()
            ));
        }

        if let Some(kv_cache_config) = kv_cache_config {
            if config.block_size != kv_cache_config.block_size {
                return Err(format!(
                    "metadata-block-size ({}) must match kv-block-size ({}) when both metadata and KV cache allocation are requested",
                    config.block_size, kv_cache_config.block_size
                ));
            }
            let required_blocks = config.required_blocks_per_seq();
            if required_blocks > kv_cache_config.num_gpu_blocks {
                return Err(format!(
                    "decode metadata requires {required_blocks} block(s) per sequence, exceeding kv-num-blocks ({})",
                    kv_cache_config.num_gpu_blocks
                ));
            }
        }

        Ok(config)
    }

    pub fn graph_max_blocks(&self) -> usize {
        self.max_position.div_ceil(self.block_size)
    }

    pub fn required_blocks_per_seq(&self) -> usize {
        self.context_len.div_ceil(self.block_size)
    }

    pub fn token_ids_len(&self) -> usize {
        self.num_tokens
    }

    pub fn positions_len(&self) -> usize {
        self.num_tokens
    }

    pub fn context_lens_len(&self) -> usize {
        self.num_seqs
    }

    pub fn block_tables_len(&self) -> usize {
        self.num_seqs * self.graph_max_blocks()
    }

    pub fn slot_mapping_len(&self) -> usize {
        self.num_tokens
    }

    pub fn seq_start_pos_len(&self) -> usize {
        self.num_seqs + 1
    }

    pub fn packed_elements(&self) -> usize {
        self.token_ids_len()
            + self.positions_len()
            + self.context_lens_len()
            + self.block_tables_len()
            + self.slot_mapping_len()
            + self.seq_start_pos_len()
    }

    pub fn packed_bytes(&self) -> usize {
        self.packed_elements() * std::mem::size_of::<i32>()
    }

    pub fn token_ids(&self) -> Vec<i32> {
        (0..self.num_tokens).map(|token| token as i32).collect()
    }

    pub fn positions(&self) -> Vec<i32> {
        vec![(self.context_len - 1) as i32; self.num_tokens]
    }

    pub fn context_lens(&self) -> Vec<i32> {
        vec![self.context_len as i32; self.num_seqs]
    }

    pub fn block_tables(&self) -> Vec<i32> {
        let graph_max_blocks = self.graph_max_blocks();
        let required_blocks = self.required_blocks_per_seq();
        let mut block_tables = vec![0i32; self.block_tables_len()];

        for seq_idx in 0..self.num_seqs {
            let row_start = seq_idx * graph_max_blocks;
            for block_idx in 0..required_blocks {
                block_tables[row_start + block_idx] = block_idx as i32;
            }
        }

        block_tables
    }

    pub fn slot_mapping(&self) -> Vec<i32> {
        let block_index = (self.context_len - 1) / self.block_size;
        let block_offset = (self.context_len - 1) % self.block_size;
        let slot = block_index * self.block_size + block_offset;
        vec![slot as i32; self.num_tokens]
    }

    pub fn seq_start_pos(&self) -> Vec<i32> {
        (0..=self.num_seqs).map(|pos| pos as i32).collect()
    }

    pub fn packed_metadata(&self) -> Vec<i32> {
        let mut packed = Vec::with_capacity(self.packed_elements());
        packed.extend(self.token_ids());
        packed.extend(self.positions());
        packed.extend(self.context_lens());
        packed.extend(self.block_tables());
        packed.extend(self.slot_mapping());
        packed.extend(self.seq_start_pos());
        packed
    }
}

impl RuntimeMetadataStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RuntimeMetadataStatus::Allocated => "allocated",
            RuntimeMetadataStatus::Deferred => "deferred",
            RuntimeMetadataStatus::NotApplicable => "not_applicable",
        }
    }
}

impl FusedF16AllocationStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            FusedF16AllocationStatus::Allocated => "allocated",
            FusedF16AllocationStatus::AvailableFromUploadedF16 => "available_from_uploaded_f16",
            FusedF16AllocationStatus::Deferred => "deferred",
            FusedF16AllocationStatus::NotApplicable => "not_applicable",
        }
    }
}

impl MoeGpuUploadStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            MoeGpuUploadStatus::Uploaded => "allocated",
            MoeGpuUploadStatus::Deferred => "deferred",
            MoeGpuUploadStatus::NotApplicable => "not_applicable",
        }
    }
}

impl LayerConstructionReadinessStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            LayerConstructionReadinessStatus::SkeletonComplete => "skeleton_complete",
            LayerConstructionReadinessStatus::Allocated => "allocated",
            LayerConstructionReadinessStatus::Deferred => "deferred",
            LayerConstructionReadinessStatus::NotApplicable => "not_applicable",
            LayerConstructionReadinessStatus::NotRequested => "not_requested",
            LayerConstructionReadinessStatus::NotConstructed => "not_constructed",
            LayerConstructionReadinessStatus::Blocked => "blocked",
        }
    }
}

impl F16ScratchAllocationConfig {
    pub fn new(
        hidden_size: usize,
        q_dim: usize,
        kv_dim: usize,
        intermediate_size: usize,
        max_tokens: usize,
    ) -> Result<Self, String> {
        if hidden_size == 0 {
            return Err("hidden_size must be non-zero for f16 scratch allocation".into());
        }
        if q_dim == 0 {
            return Err("q_dim must be non-zero for f16 scratch allocation".into());
        }
        if kv_dim == 0 {
            return Err("kv_dim must be non-zero for f16 scratch allocation".into());
        }
        if intermediate_size == 0 {
            return Err("intermediate_size must be non-zero for f16 scratch allocation".into());
        }
        if max_tokens == 0 {
            return Err("f16-scratch-max-tokens must be non-zero".into());
        }

        Ok(Self {
            hidden_size,
            q_dim,
            kv_dim,
            intermediate_size,
            max_tokens,
        })
    }

    pub fn element_counts(self) -> Result<F16ScratchElementCounts, String> {
        f16_scratch_element_counts(
            self.hidden_size,
            self.q_dim,
            self.kv_dim,
            self.intermediate_size,
            self.max_tokens,
        )
    }

    pub fn buffer_statuses(self) -> Result<F16ScratchBufferStatuses, String> {
        self.element_counts()
            .map(F16ScratchBufferStatuses::from_counts)
    }
}

impl F16ScratchBufferStatus {
    fn from_elements(elements: usize) -> Self {
        Self {
            elements,
            bytes: elements * std::mem::size_of::<half::f16>(),
        }
    }
}

impl F16ScratchBufferStatuses {
    pub fn from_counts(counts: F16ScratchElementCounts) -> Self {
        Self {
            qkv: F16ScratchBufferStatus::from_elements(counts.qkv),
            attn_out: F16ScratchBufferStatus::from_elements(counts.attn_out),
            o_proj: F16ScratchBufferStatus::from_elements(counts.o_proj),
            normed: F16ScratchBufferStatus::from_elements(counts.normed),
            residual: F16ScratchBufferStatus::from_elements(counts.residual),
            gate_up: F16ScratchBufferStatus::from_elements(counts.gate_up),
            silu_out: F16ScratchBufferStatus::from_elements(counts.silu_out),
            down: F16ScratchBufferStatus::from_elements(counts.down),
        }
    }

    pub fn total_elements(self) -> usize {
        self.qkv.elements
            + self.attn_out.elements
            + self.o_proj.elements
            + self.normed.elements
            + self.residual.elements
            + self.gate_up.elements
            + self.silu_out.elements
            + self.down.elements
    }

    pub fn total_bytes(self) -> usize {
        self.qkv.bytes
            + self.attn_out.bytes
            + self.o_proj.bytes
            + self.normed.bytes
            + self.residual.bytes
            + self.gate_up.bytes
            + self.silu_out.bytes
            + self.down.bytes
    }
}

impl ShardedCudaResourcePlan {
    /// Build the non-executing CUDA resource plan from a shard placement plan.
    pub fn from_model_plan(plan: &ShardedModelPlan) -> Self {
        Self {
            shards: plan
                .shards
                .iter()
                .map(|shard| CudaShardResourcePlan {
                    device_id: shard.device_id,
                    absolute_layers: shard.absolute_layers.clone(),
                    owns_embeddings: shard.owns_embeddings,
                    owns_final_head: shard.owns_final_head,
                })
                .collect(),
        }
    }

    /// Return true when every absolute layer appears in at most one shard.
    pub fn has_unique_absolute_layer_ownership(&self) -> bool {
        let mut seen = std::collections::BTreeSet::new();
        self.shards
            .iter()
            .flat_map(|shard| shard.absolute_layers.iter().copied())
            .all(|layer| seen.insert(layer))
    }
}

impl CudaShardResourcePlan {
    /// Metadata-only status matching the shape exposed by constructed resources.
    pub fn status(&self) -> CudaShardResourceStatus {
        CudaShardResourceStatus {
            device_id: self.device_id,
            absolute_layers: self.absolute_layers.clone(),
            owns_embeddings: self.owns_embeddings,
            owns_final_head: self.owns_final_head,
        }
    }
}

impl ShardedCudaResourceStatus {
    pub fn from_plan(plan: &ShardedCudaResourcePlan) -> Self {
        Self {
            shards: plan
                .shards
                .iter()
                .map(CudaShardResourcePlan::status)
                .collect(),
        }
    }
}

impl ShardedRuntimeBufferPlan {
    /// Build a metadata-only plan for shard-local RoPE tables and deferred
    /// request-shaped metadata buffers.
    pub fn from_model_plan(plan: &ShardedModelPlan, config: RopeRuntimeBufferConfig) -> Self {
        Self {
            shards: plan
                .shards
                .iter()
                .map(|shard| {
                    CudaShardRuntimeBufferPlan::from_parts(
                        shard.device_id,
                        shard.absolute_layers.clone(),
                        shard.owns_embeddings,
                        shard.owns_final_head,
                        config,
                    )
                })
                .collect(),
        }
    }
}

impl CudaShardRuntimeBufferPlan {
    fn from_parts(
        device_id: DeviceId,
        absolute_layers: Vec<usize>,
        owns_embeddings: bool,
        owns_final_head: bool,
        config: RopeRuntimeBufferConfig,
    ) -> Self {
        Self {
            device_id,
            absolute_layers,
            owns_embeddings,
            owns_final_head,
            rope_cos_elements: config.rope_table_elements(),
            rope_sin_elements: config.rope_table_elements(),
            rope_total_bytes: config.rope_total_bytes(),
            metadata_status: RuntimeMetadataStatus::Deferred,
            metadata_deferred_reason: Some(RUNTIME_METADATA_DEFERRED_REASON.into()),
        }
    }

    pub fn status(&self, rope_allocated: bool) -> CudaShardRuntimeBufferStatus {
        CudaShardRuntimeBufferStatus {
            device_id: self.device_id,
            absolute_layers: self.absolute_layers.clone(),
            owns_embeddings: self.owns_embeddings,
            owns_final_head: self.owns_final_head,
            rope_allocated,
            rope_cos_elements: self.rope_cos_elements,
            rope_sin_elements: self.rope_sin_elements,
            rope_total_bytes: self.rope_total_bytes,
            metadata_allocated: matches!(self.metadata_status, RuntimeMetadataStatus::Allocated),
            metadata_status: self.metadata_status,
            metadata_deferred_reason: self.metadata_deferred_reason.clone(),
            runtime_buffer_error: None,
        }
    }
}

impl ShardedRuntimeBufferStatus {
    pub fn from_plan(plan: &ShardedRuntimeBufferPlan, rope_allocated: bool) -> Self {
        Self {
            shards: plan
                .shards
                .iter()
                .map(|shard| shard.status(rope_allocated))
                .collect(),
        }
    }
}

impl ShardedKvCacheAllocationPlan {
    /// Build a metadata-only KV cache allocation plan from the existing
    /// absolute-layer keyed shard KV cache plan.
    pub fn from_model_plan(plan: &ShardedModelPlan, config: KvCacheAllocationConfig) -> Self {
        Self::from_kv_cache_plan(&plan.kv_cache_plan(), config)
    }

    pub fn from_kv_cache_plan(
        kv_cache_plan: &ShardedKvCachePlan,
        config: KvCacheAllocationConfig,
    ) -> Self {
        Self {
            shards: kv_cache_plan
                .shards
                .iter()
                .map(|shard| CudaShardKvCacheAllocationPlan::from_shard_plan(shard, config))
                .collect(),
        }
    }

    pub fn shard_for_device(&self, device_id: DeviceId) -> Option<&CudaShardKvCacheAllocationPlan> {
        self.shards
            .iter()
            .find(|shard| shard.device_id == device_id)
    }
}

impl CudaShardKvCacheAllocationPlan {
    fn from_shard_plan(
        shard: &crate::shard_plan::ShardKvCachePlan,
        config: KvCacheAllocationConfig,
    ) -> Self {
        let entries = shard
            .entries
            .iter()
            .map(|entry| CudaLayerKvCacheAllocationPlan {
                absolute_layer_idx: entry.absolute_layer_idx,
                local_cache_idx: entry.local_cache_idx,
                key_elements: config.elements_per_layer_cache(),
                value_elements: config.elements_per_layer_cache(),
                key_bytes: config.bytes_per_layer_cache(),
                value_bytes: config.bytes_per_layer_cache(),
            })
            .collect::<Vec<_>>();
        let key_total_bytes = entries.iter().map(|entry| entry.key_bytes).sum();
        let value_total_bytes = entries.iter().map(|entry| entry.value_bytes).sum();

        Self {
            device_id: shard.device_id,
            entries,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            num_gpu_blocks: config.num_gpu_blocks,
            block_size: config.block_size,
            key_total_bytes,
            value_total_bytes,
            total_bytes: key_total_bytes + value_total_bytes,
        }
    }

    pub fn status(&self, kv_cache_allocated: bool) -> CudaShardKvCacheAllocationStatus {
        CudaShardKvCacheAllocationStatus {
            device_id: self.device_id,
            kv_cache_allocated,
            entries: self
                .entries
                .iter()
                .map(CudaLayerKvCacheAllocationPlan::status)
                .collect(),
            key_total_bytes: self.key_total_bytes,
            value_total_bytes: self.value_total_bytes,
            total_bytes: self.total_bytes,
            kv_cache_error: None,
        }
    }
}

impl CudaLayerKvCacheAllocationPlan {
    pub fn status(&self) -> CudaLayerKvCacheAllocationStatus {
        CudaLayerKvCacheAllocationStatus {
            absolute_layer_idx: self.absolute_layer_idx,
            local_cache_idx: self.local_cache_idx,
            key_bytes: self.key_bytes,
            value_bytes: self.value_bytes,
        }
    }
}

impl ShardedKvCacheAllocationStatus {
    pub fn from_plan(plan: &ShardedKvCacheAllocationPlan, kv_cache_allocated: bool) -> Self {
        Self {
            shards: plan
                .shards
                .iter()
                .map(|shard| shard.status(kv_cache_allocated))
                .collect(),
        }
    }
}

impl ShardedMetadataAllocationPlan {
    /// Build a metadata-only allocation plan from a shard placement plan.
    ///
    /// The packed request metadata is intentionally duplicated to every shard
    /// that owns layers. It is not split by layer ownership.
    pub fn from_model_plan(plan: &ShardedModelPlan, config: MetadataAllocationConfig) -> Self {
        Self {
            shards: plan
                .shards
                .iter()
                .filter(|shard| !shard.absolute_layers.is_empty())
                .map(|shard| {
                    CudaShardMetadataAllocationPlan::from_parts(
                        shard.device_id,
                        shard.absolute_layers.clone(),
                        shard.owns_embeddings,
                        shard.owns_final_head,
                        config,
                    )
                })
                .collect(),
        }
    }

    pub fn shard_for_device(
        &self,
        device_id: DeviceId,
    ) -> Option<&CudaShardMetadataAllocationPlan> {
        self.shards
            .iter()
            .find(|shard| shard.device_id == device_id)
    }
}

impl CudaShardMetadataAllocationPlan {
    fn from_parts(
        device_id: DeviceId,
        absolute_layers: Vec<usize>,
        owns_embeddings: bool,
        owns_final_head: bool,
        config: MetadataAllocationConfig,
    ) -> Self {
        Self {
            device_id,
            absolute_layers,
            owns_embeddings,
            owns_final_head,
            mode: config.mode,
            num_tokens: config.num_tokens,
            num_seqs: config.num_seqs,
            context_len: config.context_len,
            block_size: config.block_size,
            graph_max_blocks: config.graph_max_blocks(),
            max_context_len: config.context_len,
            token_ids_len: config.token_ids_len(),
            positions_len: config.positions_len(),
            context_lens_len: config.context_lens_len(),
            block_tables_len: config.block_tables_len(),
            slot_mapping_len: config.slot_mapping_len(),
            seq_start_pos_len: config.seq_start_pos_len(),
            packed_elements: config.packed_elements(),
            packed_bytes: config.packed_bytes(),
        }
    }

    pub fn status(&self, metadata_allocated: bool) -> CudaShardMetadataAllocationStatus {
        CudaShardMetadataAllocationStatus {
            device_id: self.device_id,
            metadata_allocated,
            metadata_status: if metadata_allocated {
                RuntimeMetadataStatus::Allocated
            } else {
                RuntimeMetadataStatus::Deferred
            },
            mode: self.mode,
            num_tokens: self.num_tokens,
            num_seqs: self.num_seqs,
            graph_max_blocks: self.graph_max_blocks,
            packed_elements: self.packed_elements,
            packed_bytes: self.packed_bytes,
            token_ids_len: self.token_ids_len,
            positions_len: self.positions_len,
            context_lens_len: self.context_lens_len,
            block_tables_len: self.block_tables_len,
            slot_mapping_len: self.slot_mapping_len,
            seq_start_pos_len: self.seq_start_pos_len,
            max_context_len: self.max_context_len,
            metadata_error: None,
        }
    }

    pub fn packed_metadata(&self) -> Vec<i32> {
        MetadataAllocationConfig {
            mode: self.mode,
            num_tokens: self.num_tokens,
            num_seqs: self.num_seqs,
            context_len: self.context_len,
            block_size: self.block_size,
            max_position: self.graph_max_blocks * self.block_size,
        }
        .packed_metadata()
    }
}

impl ShardedMetadataAllocationStatus {
    pub fn from_plan(plan: &ShardedMetadataAllocationPlan, metadata_allocated: bool) -> Self {
        Self {
            shards: plan
                .shards
                .iter()
                .map(|shard| shard.status(metadata_allocated))
                .collect(),
        }
    }
}

impl ShardedFusedF16AllocationPlan {
    /// Build a non-executing fused/preconverted f16 and scratch status plan
    /// from the shard tensor manifest. This deliberately does not allocate
    /// fused buffers because the current runtime helper is coupled to
    /// `GpuModelRunner` and full-runner layer indexing.
    pub fn from_upload_manifest(
        manifest: &crate::shard_plan::ShardedUploadManifest,
        scratch_config: Option<F16ScratchAllocationConfig>,
    ) -> Self {
        Self {
            shards: manifest
                .shards
                .iter()
                .map(|shard| CudaShardFusedF16AllocationPlan::from_manifest(shard, scratch_config))
                .collect(),
        }
    }

    pub fn shard_for_device(
        &self,
        device_id: DeviceId,
    ) -> Option<&CudaShardFusedF16AllocationPlan> {
        self.shards
            .iter()
            .find(|shard| shard.device_id == device_id)
    }
}

impl CudaShardFusedF16AllocationPlan {
    fn from_manifest(
        manifest: &ShardTensorManifest,
        scratch_config: Option<F16ScratchAllocationConfig>,
    ) -> Self {
        let fused_layer_absolute_indices = manifest.absolute_layers.clone();
        let fused_layer_plans = fused_layer_absolute_indices
            .iter()
            .enumerate()
            .map(|(local_layer_idx, &absolute_layer_idx)| {
                CudaLayerFusedF16AllocationPlan::from_manifest(
                    manifest,
                    absolute_layer_idx,
                    local_layer_idx,
                )
            })
            .collect::<Vec<_>>();
        let fused_qkv_weight_count = fused_layer_plans
            .iter()
            .filter(|layer| layer.fused_qkv_planned)
            .count();
        let fused_gate_up_weight_count = fused_layer_plans
            .iter()
            .filter(|layer| layer.fused_gate_up_planned)
            .count();
        let f16_layernorm_count = fused_layer_plans
            .iter()
            .filter(|layer| layer.f16_layernorm_planned)
            .count();
        let f16_postnorm_count = fused_layer_plans
            .iter()
            .filter(|layer| layer.f16_postnorm_planned)
            .count();
        let f16_qkv_bias_count = fused_layer_plans
            .iter()
            .filter(|layer| layer.f16_qkv_bias_planned)
            .count();
        let f16_o_proj_bias_count = fused_layer_plans
            .iter()
            .filter(|layer| layer.f16_o_proj_bias_planned)
            .count();
        let owns_embeddings = manifest.should_load_required_tensor("model.embed_tokens.weight");
        let owns_final_head = manifest.should_load_required_tensor("model.norm.weight")
            || manifest.should_load_required_tensor("lm_head.weight");
        let embedding_f16_planned =
            owns_embeddings && manifest.should_load_required_tensor("model.embed_tokens.weight");
        let final_norm_f16_planned =
            owns_final_head && manifest.should_load_required_tensor("model.norm.weight");
        let fused_has_work = !fused_layer_absolute_indices.is_empty()
            || embedding_f16_planned
            || final_norm_f16_planned;
        let fused_status = if fused_has_work {
            FusedF16AllocationStatus::Deferred
        } else {
            FusedF16AllocationStatus::NotApplicable
        };
        let f16_scratch_buffers = scratch_config
            .filter(|_| !manifest.absolute_layers.is_empty())
            .and_then(|config| config.buffer_statuses().ok());
        let f16_scratch_status = if f16_scratch_buffers.is_some() {
            FusedF16AllocationStatus::Deferred
        } else {
            FusedF16AllocationStatus::NotApplicable
        };
        let f16_scratch_total_elements = f16_scratch_buffers
            .map(F16ScratchBufferStatuses::total_elements)
            .unwrap_or(0);
        let f16_scratch_bytes = f16_scratch_buffers
            .map(F16ScratchBufferStatuses::total_bytes)
            .unwrap_or(0);

        Self {
            device_id: manifest.device_id,
            absolute_layers: manifest.absolute_layers.clone(),
            owns_embeddings,
            owns_final_head,
            fused_qkv_weight_count,
            fused_gate_up_weight_count,
            f16_layernorm_count,
            f16_postnorm_count,
            f16_qkv_bias_count,
            f16_o_proj_bias_count,
            embedding_f16_planned,
            final_norm_f16_planned,
            fused_qkv_total_bytes: 0,
            fused_gate_up_total_bytes: 0,
            f16_layernorm_total_bytes: 0,
            f16_postnorm_total_bytes: 0,
            f16_qkv_bias_total_bytes: 0,
            f16_o_proj_bias_total_bytes: 0,
            fused_layer_absolute_indices,
            fused_layer_plans,
            fused_total_bytes: 0,
            fused_status,
            fused_deferred_reason: (fused_status == FusedF16AllocationStatus::Deferred)
                .then(|| FUSED_F16_DEFERRED_REASON.into()),
            f16_scratch_status,
            f16_scratch_max_tokens: scratch_config.map(|config| config.max_tokens),
            f16_scratch_total_elements,
            f16_scratch_bytes,
            f16_scratch_buffers,
            f16_scratch_deferred_reason: (f16_scratch_status == FusedF16AllocationStatus::Deferred)
                .then(|| F16_SCRATCH_DEFERRED_REASON.into()),
        }
    }

    pub fn status(
        &self,
        fused_allocated: bool,
        scratch_allocated: bool,
    ) -> CudaShardFusedF16AllocationStatus {
        let fused_f16_allocated =
            fused_allocated && self.fused_status != FusedF16AllocationStatus::NotApplicable;
        let f16_scratch_allocated =
            scratch_allocated && self.f16_scratch_status != FusedF16AllocationStatus::NotApplicable;
        let fused_f16_status = if fused_f16_allocated {
            FusedF16AllocationStatus::Allocated
        } else {
            self.fused_status
        };
        let f16_scratch_status = if f16_scratch_allocated {
            FusedF16AllocationStatus::Allocated
        } else {
            self.f16_scratch_status
        };

        CudaShardFusedF16AllocationStatus {
            device_id: self.device_id,
            absolute_layers: self.absolute_layers.clone(),
            owns_embeddings: self.owns_embeddings,
            owns_final_head: self.owns_final_head,
            fused_f16_allocated,
            fused_f16_status,
            fused_qkv_weight_count: self.fused_qkv_weight_count,
            fused_gate_up_weight_count: self.fused_gate_up_weight_count,
            f16_layernorm_count: self.f16_layernorm_count,
            f16_postnorm_count: self.f16_postnorm_count,
            f16_qkv_bias_count: self.f16_qkv_bias_count,
            f16_o_proj_bias_count: self.f16_o_proj_bias_count,
            embedding_f16_allocated: fused_f16_allocated && self.embedding_f16_planned,
            embedding_f16_status: allocation_status_for_planned(
                self.embedding_f16_planned,
                fused_f16_allocated && self.embedding_f16_planned,
            ),
            embedding_f16_bytes: 0,
            embedding_f16_source: None,
            final_norm_f16_allocated: fused_f16_allocated && self.final_norm_f16_planned,
            final_norm_f16_status: allocation_status_for_planned(
                self.final_norm_f16_planned,
                fused_f16_allocated && self.final_norm_f16_planned,
            ),
            final_norm_f16_bytes: 0,
            final_norm_f16_source: None,
            fused_qkv_total_bytes: if fused_f16_allocated {
                self.fused_qkv_total_bytes
            } else {
                0
            },
            fused_gate_up_total_bytes: if fused_f16_allocated {
                self.fused_gate_up_total_bytes
            } else {
                0
            },
            f16_layernorm_total_bytes: if fused_f16_allocated {
                self.f16_layernorm_total_bytes
            } else {
                0
            },
            f16_postnorm_total_bytes: if fused_f16_allocated {
                self.f16_postnorm_total_bytes
            } else {
                0
            },
            f16_qkv_bias_total_bytes: if fused_f16_allocated {
                self.f16_qkv_bias_total_bytes
            } else {
                0
            },
            f16_o_proj_bias_total_bytes: if fused_f16_allocated {
                self.f16_o_proj_bias_total_bytes
            } else {
                0
            },
            fused_total_bytes: if fused_f16_allocated {
                self.fused_total_bytes
            } else {
                0
            },
            fused_layer_absolute_indices: self.fused_layer_absolute_indices.clone(),
            fused_layer_statuses: self
                .fused_layer_plans
                .iter()
                .map(|layer| {
                    layer.status(
                        false, false, false, false, false, false, 0, 0, 0, 0, 0, 0, None,
                    )
                })
                .collect(),
            fused_deferred_reason: (!fused_f16_allocated)
                .then(|| self.fused_deferred_reason.clone())
                .flatten(),
            fused_error: None,
            f16_scratch_allocated,
            f16_scratch_status,
            f16_scratch_total_elements: if f16_scratch_allocated {
                self.f16_scratch_total_elements
            } else {
                0
            },
            f16_scratch_bytes: if f16_scratch_allocated {
                self.f16_scratch_bytes
            } else {
                0
            },
            f16_scratch_max_tokens: self.f16_scratch_max_tokens,
            f16_scratch_buffers: if f16_scratch_allocated {
                self.f16_scratch_buffers
            } else {
                None
            },
            f16_scratch_deferred_reason: (!f16_scratch_allocated)
                .then(|| self.f16_scratch_deferred_reason.clone())
                .flatten(),
            f16_scratch_error: None,
        }
    }
}

impl CudaLayerFusedF16AllocationPlan {
    fn from_manifest(
        manifest: &ShardTensorManifest,
        absolute_layer_idx: usize,
        local_layer_idx: usize,
    ) -> Self {
        Self {
            absolute_layer_idx,
            local_layer_idx,
            fused_qkv_planned: manifest_has_layer_tensors(
                manifest,
                absolute_layer_idx,
                &[
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
            ),
            fused_gate_up_planned: manifest_has_layer_tensors(
                manifest,
                absolute_layer_idx,
                &["mlp.gate_proj.weight", "mlp.up_proj.weight"],
            ),
            has_u8_expert_tensors: manifest_has_owned_u8_expert_tensors(
                manifest,
                absolute_layer_idx,
            ),
            f16_layernorm_planned: manifest_has_layer_tensor(
                manifest,
                absolute_layer_idx,
                "input_layernorm.weight",
            ),
            f16_postnorm_planned: manifest_has_layer_tensor(
                manifest,
                absolute_layer_idx,
                "post_attention_layernorm.weight",
            ),
            f16_qkv_bias_planned: manifest_has_any_layer_tensor(
                manifest,
                absolute_layer_idx,
                &[
                    "self_attn.q_proj.bias",
                    "self_attn.k_proj.bias",
                    "self_attn.v_proj.bias",
                ],
            ),
            f16_o_proj_bias_planned: manifest_has_layer_tensor(
                manifest,
                absolute_layer_idx,
                "self_attn.o_proj.bias",
            ),
        }
    }

    pub fn status(
        &self,
        fused_qkv_allocated: bool,
        fused_gate_up_allocated: bool,
        layernorm_f16_allocated: bool,
        postnorm_f16_allocated: bool,
        qkv_bias_f16_allocated: bool,
        o_proj_bias_f16_allocated: bool,
        fused_qkv_bytes: usize,
        fused_gate_up_bytes: usize,
        layernorm_f16_bytes: usize,
        postnorm_f16_bytes: usize,
        qkv_bias_f16_bytes: usize,
        o_proj_bias_f16_bytes: usize,
        layer_error: Option<String>,
    ) -> CudaLayerFusedF16AllocationStatus {
        CudaLayerFusedF16AllocationStatus {
            absolute_layer_idx: self.absolute_layer_idx,
            local_layer_idx: self.local_layer_idx,
            fused_qkv_allocated,
            fused_qkv_status: allocation_status_for_planned(
                self.fused_qkv_planned,
                fused_qkv_allocated,
            ),
            fused_qkv_bytes: if fused_qkv_allocated {
                fused_qkv_bytes
            } else {
                0
            },
            fused_gate_up_allocated,
            fused_gate_up_status: allocation_status_for_planned(
                self.fused_gate_up_planned,
                fused_gate_up_allocated,
            ),
            fused_gate_up_bytes: if fused_gate_up_allocated {
                fused_gate_up_bytes
            } else {
                0
            },
            layernorm_f16_status: allocation_status_for_planned(
                self.f16_layernorm_planned,
                layernorm_f16_allocated,
            ),
            layernorm_f16_bytes: if layernorm_f16_allocated {
                layernorm_f16_bytes
            } else {
                0
            },
            postnorm_f16_status: allocation_status_for_planned(
                self.f16_postnorm_planned,
                postnorm_f16_allocated,
            ),
            postnorm_f16_bytes: if postnorm_f16_allocated {
                postnorm_f16_bytes
            } else {
                0
            },
            qkv_bias_f16_status: allocation_status_for_planned(
                self.f16_qkv_bias_planned,
                qkv_bias_f16_allocated,
            ),
            qkv_bias_f16_bytes: if qkv_bias_f16_allocated {
                qkv_bias_f16_bytes
            } else {
                0
            },
            o_proj_bias_f16_status: allocation_status_for_planned(
                self.f16_o_proj_bias_planned,
                o_proj_bias_f16_allocated,
            ),
            o_proj_bias_f16_bytes: if o_proj_bias_f16_allocated {
                o_proj_bias_f16_bytes
            } else {
                0
            },
            layer_error,
        }
    }
}

fn allocation_status_for_planned(planned: bool, allocated: bool) -> FusedF16AllocationStatus {
    if allocated {
        FusedF16AllocationStatus::Allocated
    } else if planned {
        FusedF16AllocationStatus::Deferred
    } else {
        FusedF16AllocationStatus::NotApplicable
    }
}

impl ShardedFusedF16AllocationStatus {
    pub fn from_plan(
        plan: &ShardedFusedF16AllocationPlan,
        fused_allocated: bool,
        scratch_allocated: bool,
    ) -> Self {
        Self {
            shards: plan
                .shards
                .iter()
                .map(|shard| shard.status(fused_allocated, scratch_allocated))
                .collect(),
        }
    }
}

impl ShardedMoeGpuUploadPlan {
    /// Build a non-executing GPT-OSS MoE GPU upload plan from the shard tensor
    /// manifest. This only records shard-owned U8 expert payloads and f32
    /// router/bias readiness state; it does not retain host maps, upload U8
    /// payloads, construct layers, or evaluate executable graph-decode support.
    pub fn from_upload_manifest(
        manifest: &crate::shard_plan::ShardedUploadManifest,
        header_bytes: &BTreeMap<String, usize>,
    ) -> Self {
        Self {
            shards: manifest
                .shards
                .iter()
                .map(|shard| CudaShardMoeGpuUploadPlan::from_manifest(shard, header_bytes))
                .collect(),
        }
    }

    pub fn shard_for_device(&self, device_id: DeviceId) -> Option<&CudaShardMoeGpuUploadPlan> {
        self.shards
            .iter()
            .find(|shard| shard.device_id == device_id)
    }
}

impl CudaShardMoeGpuUploadPlan {
    fn from_manifest(
        manifest: &ShardTensorManifest,
        header_bytes: &BTreeMap<String, usize>,
    ) -> Self {
        let moe_layer_plans = manifest
            .absolute_layers
            .iter()
            .enumerate()
            .map(|(local_layer_idx, &absolute_layer_idx)| {
                CudaLayerMoeGpuUploadPlan::from_manifest(
                    manifest,
                    absolute_layer_idx,
                    local_layer_idx,
                    header_bytes,
                )
            })
            .collect::<Vec<_>>();
        let planned_layers = moe_layer_plans
            .iter()
            .filter(|layer| layer.has_any_u8_payload())
            .count();
        let moe_u8_host_tensor_count = moe_layer_plans
            .iter()
            .map(CudaLayerMoeGpuUploadPlan::u8_tensor_count)
            .sum();
        let moe_u8_host_bytes = moe_layer_plans
            .iter()
            .map(CudaLayerMoeGpuUploadPlan::u8_host_bytes)
            .sum();
        let moe_router_tensor_count = moe_layer_plans
            .iter()
            .filter(|layer| layer.router_planned)
            .count()
            * 2;
        let moe_bias_tensor_count = moe_layer_plans
            .iter()
            .filter(|layer| layer.expert_bias_planned)
            .count()
            * 2;
        let moe_gpu_status = if planned_layers > 0 {
            MoeGpuUploadStatus::Deferred
        } else {
            MoeGpuUploadStatus::NotApplicable
        };

        Self {
            device_id: manifest.device_id,
            absolute_layers: manifest.absolute_layers.clone(),
            moe_layer_count: planned_layers,
            moe_u8_host_tensor_count,
            moe_u8_host_bytes,
            moe_router_tensor_count,
            moe_bias_tensor_count,
            moe_layer_plans,
            moe_gpu_status,
            moe_gpu_deferred_reason: (moe_gpu_status == MoeGpuUploadStatus::Deferred)
                .then(|| MOE_GPU_UPLOAD_DEFERRED_REASON.into()),
        }
    }

    pub fn status(&self, moe_gpu_uploaded: bool) -> CudaShardMoeGpuUploadStatus {
        let all_applicable_layers_complete = self
            .moe_layer_plans
            .iter()
            .filter(|layer| layer.has_any_u8_payload())
            .all(CudaLayerMoeGpuUploadPlan::has_complete_u8_payload);
        let moe_gpu_uploaded = moe_gpu_uploaded
            && self.moe_gpu_status != MoeGpuUploadStatus::NotApplicable
            && all_applicable_layers_complete;
        let moe_gpu_status = if moe_gpu_uploaded {
            MoeGpuUploadStatus::Uploaded
        } else {
            self.moe_gpu_status
        };

        CudaShardMoeGpuUploadStatus {
            device_id: self.device_id,
            moe_gpu_uploaded,
            moe_gpu_status,
            moe_layer_count: self.moe_layer_count,
            moe_u8_host_tensor_count: self.moe_u8_host_tensor_count,
            moe_u8_gpu_tensor_count: if moe_gpu_uploaded {
                self.moe_u8_host_tensor_count
            } else {
                0
            },
            moe_u8_host_bytes: self.moe_u8_host_bytes,
            moe_u8_gpu_bytes: if moe_gpu_uploaded {
                self.moe_u8_host_bytes
            } else {
                0
            },
            moe_router_tensor_count: self.moe_router_tensor_count,
            moe_bias_tensor_count: self.moe_bias_tensor_count,
            moe_layer_statuses: self
                .moe_layer_plans
                .iter()
                .map(|layer| layer.status(moe_gpu_uploaded))
                .collect(),
            moe_gpu_deferred_reason: (!moe_gpu_uploaded)
                .then(|| self.moe_gpu_deferred_reason.clone())
                .flatten(),
            moe_gpu_error: None,
        }
    }
}

impl CudaLayerMoeGpuUploadPlan {
    fn from_manifest(
        manifest: &ShardTensorManifest,
        absolute_layer_idx: usize,
        local_layer_idx: usize,
        header_bytes: &BTreeMap<String, usize>,
    ) -> Self {
        let gate_up_proj_blocks_name =
            moe_layer_tensor_name(absolute_layer_idx, "experts.gate_up_proj_blocks");
        let gate_up_proj_scales_name =
            moe_layer_tensor_name(absolute_layer_idx, "experts.gate_up_proj_scales");
        let down_proj_blocks_name =
            moe_layer_tensor_name(absolute_layer_idx, "experts.down_proj_blocks");
        let down_proj_scales_name =
            moe_layer_tensor_name(absolute_layer_idx, "experts.down_proj_scales");
        let router_weight_name = moe_layer_tensor_name(absolute_layer_idx, "router.weight");
        let router_bias_name = moe_layer_tensor_name(absolute_layer_idx, "router.bias");
        let gate_up_bias_name =
            moe_layer_tensor_name(absolute_layer_idx, "experts.gate_up_proj_bias");
        let down_bias_name = moe_layer_tensor_name(absolute_layer_idx, "experts.down_proj_bias");

        let gate_up_proj_blocks_planned =
            manifest.should_load_host_u8_tensor(&gate_up_proj_blocks_name);
        let gate_up_proj_scales_planned =
            manifest.should_load_host_u8_tensor(&gate_up_proj_scales_name);
        let down_proj_blocks_planned = manifest.should_load_host_u8_tensor(&down_proj_blocks_name);
        let down_proj_scales_planned = manifest.should_load_host_u8_tensor(&down_proj_scales_name);
        let u8_count = [
            gate_up_proj_blocks_planned,
            gate_up_proj_scales_planned,
            down_proj_blocks_planned,
            down_proj_scales_planned,
        ]
        .into_iter()
        .filter(|planned| *planned)
        .count();
        let router_planned = manifest.should_load_required_tensor(&router_weight_name)
            && manifest.should_load_required_tensor(&router_bias_name);
        let expert_bias_planned = manifest.should_load_required_tensor(&gate_up_bias_name)
            && manifest.should_load_required_tensor(&down_bias_name);

        Self {
            absolute_layer_idx,
            local_layer_idx,
            gate_up_proj_blocks_planned,
            gate_up_proj_scales_planned,
            down_proj_blocks_planned,
            down_proj_scales_planned,
            gate_up_proj_blocks_bytes: header_bytes
                .get(&gate_up_proj_blocks_name)
                .copied()
                .unwrap_or(0),
            gate_up_proj_scales_bytes: header_bytes
                .get(&gate_up_proj_scales_name)
                .copied()
                .unwrap_or(0),
            down_proj_blocks_bytes: header_bytes
                .get(&down_proj_blocks_name)
                .copied()
                .unwrap_or(0),
            down_proj_scales_bytes: header_bytes
                .get(&down_proj_scales_name)
                .copied()
                .unwrap_or(0),
            router_planned,
            expert_bias_planned,
            partial_u8_payload: u8_count > 0 && u8_count < 4,
        }
    }

    fn status(&self, moe_gpu_uploaded: bool) -> CudaLayerMoeGpuUploadStatus {
        let has_any_u8_payload = self.has_any_u8_payload();
        let moe_gpu_uploaded = moe_gpu_uploaded && self.has_complete_u8_payload();
        let layer_error = self.partial_u8_payload.then(|| {
            format!(
                "partial GPT-OSS MoE U8 block/scale payload; upload requires all four U8 tensors; missing tensors: {}",
                self.missing_u8_tensor_names().join(", ")
            )
        });
        let supports_gpu_decode_status = if moe_gpu_uploaded {
            "gpu_u8_uploaded_but_not_evaluated_without_layer_construction".to_string()
        } else if has_any_u8_payload {
            "not_evaluated_without_layer_construction".to_string()
        } else {
            "not_applicable".to_string()
        };

        CudaLayerMoeGpuUploadStatus {
            absolute_layer_idx: self.absolute_layer_idx,
            local_layer_idx: self.local_layer_idx,
            gate_up_proj_blocks_status: moe_status_for_planned(
                self.gate_up_proj_blocks_planned,
                moe_gpu_uploaded,
            ),
            gate_up_proj_scales_status: moe_status_for_planned(
                self.gate_up_proj_scales_planned,
                moe_gpu_uploaded,
            ),
            down_proj_blocks_status: moe_status_for_planned(
                self.down_proj_blocks_planned,
                moe_gpu_uploaded,
            ),
            down_proj_scales_status: moe_status_for_planned(
                self.down_proj_scales_planned,
                moe_gpu_uploaded,
            ),
            gate_up_proj_blocks_bytes: self.gate_up_proj_blocks_bytes,
            gate_up_proj_scales_bytes: self.gate_up_proj_scales_bytes,
            down_proj_blocks_bytes: self.down_proj_blocks_bytes,
            down_proj_scales_bytes: self.down_proj_scales_bytes,
            router_status: moe_status_for_planned(self.router_planned, false),
            expert_bias_status: moe_status_for_planned(self.expert_bias_planned, false),
            supports_gpu_decode_status,
            layer_error,
        }
    }

    fn has_any_u8_payload(&self) -> bool {
        self.gate_up_proj_blocks_planned
            || self.gate_up_proj_scales_planned
            || self.down_proj_blocks_planned
            || self.down_proj_scales_planned
    }

    fn has_complete_u8_payload(&self) -> bool {
        self.gate_up_proj_blocks_planned
            && self.gate_up_proj_scales_planned
            && self.down_proj_blocks_planned
            && self.down_proj_scales_planned
    }

    fn u8_tensor_count(&self) -> usize {
        [
            self.gate_up_proj_blocks_planned,
            self.gate_up_proj_scales_planned,
            self.down_proj_blocks_planned,
            self.down_proj_scales_planned,
        ]
        .into_iter()
        .filter(|planned| *planned)
        .count()
    }

    fn u8_host_bytes(&self) -> usize {
        self.gate_up_proj_blocks_bytes
            + self.gate_up_proj_scales_bytes
            + self.down_proj_blocks_bytes
            + self.down_proj_scales_bytes
    }

    fn gate_up_proj_blocks_name(&self) -> String {
        moe_layer_tensor_name(self.absolute_layer_idx, "experts.gate_up_proj_blocks")
    }

    fn gate_up_proj_scales_name(&self) -> String {
        moe_layer_tensor_name(self.absolute_layer_idx, "experts.gate_up_proj_scales")
    }

    fn down_proj_blocks_name(&self) -> String {
        moe_layer_tensor_name(self.absolute_layer_idx, "experts.down_proj_blocks")
    }

    fn down_proj_scales_name(&self) -> String {
        moe_layer_tensor_name(self.absolute_layer_idx, "experts.down_proj_scales")
    }

    fn missing_u8_tensor_names(&self) -> Vec<String> {
        let mut missing = Vec::new();
        if !self.gate_up_proj_blocks_planned {
            missing.push(self.gate_up_proj_blocks_name());
        }
        if !self.gate_up_proj_scales_planned {
            missing.push(self.gate_up_proj_scales_name());
        }
        if !self.down_proj_blocks_planned {
            missing.push(self.down_proj_blocks_name());
        }
        if !self.down_proj_scales_planned {
            missing.push(self.down_proj_scales_name());
        }
        missing
    }
}

impl ShardedMoeGpuUploadStatus {
    pub fn from_plan(plan: &ShardedMoeGpuUploadPlan, moe_gpu_uploaded: bool) -> Self {
        Self {
            shards: plan
                .shards
                .iter()
                .map(|shard| shard.status(moe_gpu_uploaded))
                .collect(),
        }
    }
}

impl ShardedLayerConstructionPlan {
    /// Build a pure, non-executing layer-construction skeleton plan from the
    /// shard tensor manifest. It validates absolute-layer ownership and names
    /// only; it does not allocate buffers or instantiate runtime layer objects.
    pub fn from_upload_manifest(manifest: &ShardedUploadManifest) -> Self {
        Self {
            shards: manifest
                .shards
                .iter()
                .map(CudaShardLayerConstructionPlan::from_manifest)
                .collect(),
        }
    }
}

impl CudaShardLayerConstructionPlan {
    fn from_manifest(manifest: &ShardTensorManifest) -> Self {
        let store = layer_skeleton_store(manifest);
        let layer_plans = manifest
            .absolute_layers
            .iter()
            .enumerate()
            .map(|(local_layer_idx, &absolute_layer_idx)| {
                CudaLayerConstructionPlan::from_store(
                    &store,
                    manifest,
                    absolute_layer_idx,
                    local_layer_idx,
                )
            })
            .collect();

        Self {
            device_id: manifest.device_id,
            absolute_layers: manifest.absolute_layers.clone(),
            layer_plans,
        }
    }
}

impl CudaLayerConstructionPlan {
    fn from_store(
        store: &ShardWeightStore,
        manifest: &ShardTensorManifest,
        absolute_layer_idx: usize,
        local_layer_idx: usize,
    ) -> Self {
        let owns_layer = store.owns_layer(absolute_layer_idx);
        let mut required_f16_projection_tensor_names = Vec::new();
        let mut missing_required_f16_projection_tensor_names = Vec::new();

        for suffix in [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ] {
            let name = store
                .layer_tensor_name(absolute_layer_idx, suffix)
                .unwrap_or_else(|_| format!("model.layers.{absolute_layer_idx}.{suffix}"));
            if manifest.should_load_required_tensor(&name) {
                required_f16_projection_tensor_names.push(name);
            } else {
                missing_required_f16_projection_tensor_names.push(name);
            }
        }

        let dense_mlp_suffixes = [
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ];
        let dense_present_count = dense_mlp_suffixes
            .iter()
            .filter(|suffix| manifest_has_layer_tensor(manifest, absolute_layer_idx, suffix))
            .count();
        if dense_present_count > 0 {
            for suffix in dense_mlp_suffixes {
                let name = store
                    .layer_tensor_name(absolute_layer_idx, suffix)
                    .unwrap_or_else(|_| format!("model.layers.{absolute_layer_idx}.{suffix}"));
                if manifest.should_load_required_tensor(&name) {
                    required_f16_projection_tensor_names.push(name);
                } else {
                    missing_required_f16_projection_tensor_names.push(name);
                }
            }
        }

        let required_f32_norm_bias_tensor_names = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.bias",
            "self_attn.o_proj.bias",
        ]
        .into_iter()
        .filter_map(|suffix| {
            let name = store.layer_tensor_name(absolute_layer_idx, suffix).ok()?;
            manifest.should_load_required_tensor(&name).then_some(name)
        })
        .collect();

        Self {
            absolute_layer_idx,
            local_layer_idx,
            owns_layer,
            required_f16_projection_tensor_names,
            missing_required_f16_projection_tensor_names,
            required_f32_norm_bias_tensor_names,
            has_moe_u8_payload: manifest_has_owned_u8_expert_tensors(manifest, absolute_layer_idx),
        }
    }
}

impl ShardedLayerConstructionStatus {
    #[allow(clippy::too_many_arguments)]
    pub fn from_plan(
        plan: &ShardedLayerConstructionPlan,
        f16_projection_allocated: bool,
        runtime_buffer_status: Option<&ShardedRuntimeBufferStatus>,
        kv_cache_status: Option<&ShardedKvCacheAllocationStatus>,
        metadata_status: Option<&ShardedMetadataAllocationStatus>,
        fused_f16_status: Option<&ShardedFusedF16AllocationStatus>,
        moe_gpu_upload_status: Option<&ShardedMoeGpuUploadStatus>,
    ) -> Self {
        let runtime_by_device = runtime_buffer_status
            .map(|status| {
                status
                    .shards
                    .iter()
                    .map(|shard| (shard.device_id, shard))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let kv_by_device = kv_cache_status
            .map(|status| {
                status
                    .shards
                    .iter()
                    .map(|shard| (shard.device_id, shard))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let metadata_by_device = metadata_status
            .map(|status| {
                status
                    .shards
                    .iter()
                    .map(|shard| (shard.device_id, shard))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let fused_by_device = fused_f16_status
            .map(|status| {
                status
                    .shards
                    .iter()
                    .map(|shard| (shard.device_id, shard))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let moe_by_device = moe_gpu_upload_status
            .map(|status| {
                status
                    .shards
                    .iter()
                    .map(|shard| (shard.device_id, shard))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();

        Self {
            shards: plan
                .shards
                .iter()
                .map(|shard| {
                    shard.status(
                        f16_projection_allocated,
                        runtime_by_device.get(&shard.device_id).copied(),
                        kv_by_device.get(&shard.device_id).copied(),
                        metadata_by_device.get(&shard.device_id).copied(),
                        fused_by_device.get(&shard.device_id).copied(),
                        moe_by_device.get(&shard.device_id).copied(),
                    )
                })
                .collect(),
        }
    }
}

impl CudaShardLayerConstructionPlan {
    fn status(
        &self,
        f16_projection_allocated: bool,
        runtime_buffer: Option<&CudaShardRuntimeBufferStatus>,
        kv_cache: Option<&CudaShardKvCacheAllocationStatus>,
        metadata: Option<&CudaShardMetadataAllocationStatus>,
        fused_f16: Option<&CudaShardFusedF16AllocationStatus>,
        moe_gpu: Option<&CudaShardMoeGpuUploadStatus>,
    ) -> CudaShardLayerConstructionStatus {
        let layer_skeletons = self
            .layer_plans
            .iter()
            .map(|layer| {
                layer.status(
                    f16_projection_allocated,
                    runtime_buffer,
                    kv_cache,
                    metadata,
                    fused_f16,
                    moe_gpu,
                )
            })
            .collect::<Vec<_>>();
        let layer_skeleton_count = layer_skeletons.len();
        let layer_skeleton_ready_count = layer_skeletons
            .iter()
            .filter(|layer| {
                layer.layer_config_status == LayerConstructionReadinessStatus::SkeletonComplete
                    && layer.required_f16_projection_status
                        != LayerConstructionReadinessStatus::Blocked
            })
            .count();
        let layer_skeleton_blocked_count = layer_skeletons
            .iter()
            .filter(|layer| {
                layer.layer_config_status == LayerConstructionReadinessStatus::Blocked
                    || layer.required_f16_projection_status
                        == LayerConstructionReadinessStatus::Blocked
                    || layer
                        .blockers
                        .iter()
                        .any(|blocker| blocker.code.starts_with("blocking_"))
            })
            .count();
        let layer_skeleton_deferred_count = layer_skeletons
            .iter()
            .filter(|layer| {
                layer.executable_layer_status == LayerConstructionReadinessStatus::NotConstructed
            })
            .count();
        let layer_skeleton_status = if layer_skeleton_count == 0 {
            LayerConstructionReadinessStatus::NotApplicable
        } else if layer_skeleton_blocked_count > 0 {
            LayerConstructionReadinessStatus::Blocked
        } else {
            LayerConstructionReadinessStatus::SkeletonComplete
        };

        CudaShardLayerConstructionStatus {
            device_id: self.device_id,
            layer_skeleton_built: layer_skeleton_count > 0,
            layer_skeleton_status,
            layer_skeleton_count,
            layer_skeleton_ready_count,
            layer_skeleton_blocked_count,
            layer_skeleton_deferred_count,
            layer_skeletons,
            layer_skeleton_error: None,
        }
    }
}

impl CudaLayerConstructionPlan {
    fn status(
        &self,
        f16_projection_allocated: bool,
        runtime_buffer: Option<&CudaShardRuntimeBufferStatus>,
        kv_cache: Option<&CudaShardKvCacheAllocationStatus>,
        metadata: Option<&CudaShardMetadataAllocationStatus>,
        fused_f16: Option<&CudaShardFusedF16AllocationStatus>,
        moe_gpu: Option<&CudaShardMoeGpuUploadStatus>,
    ) -> CudaLayerConstructionStatus {
        let fused_layer = fused_f16.and_then(|shard| {
            shard
                .fused_layer_statuses
                .iter()
                .find(|layer| layer.absolute_layer_idx == self.absolute_layer_idx)
        });
        let moe_layer = moe_gpu.and_then(|shard| {
            shard
                .moe_layer_statuses
                .iter()
                .find(|layer| layer.absolute_layer_idx == self.absolute_layer_idx)
        });

        let mut blockers = Vec::new();
        let layer_config_status = if self.owns_layer {
            LayerConstructionReadinessStatus::SkeletonComplete
        } else {
            push_blocker(
                &mut blockers,
                "blocking_layer_not_owned",
                format!(
                    "absolute layer {} is not owned by this shard",
                    self.absolute_layer_idx
                ),
            );
            LayerConstructionReadinessStatus::Blocked
        };

        let required_f16_projection_status =
            if !self.missing_required_f16_projection_tensor_names.is_empty() {
                push_blocker(
                    &mut blockers,
                    "blocking_missing_required_f16_projection_tensor",
                    format!(
                        "missing projection tensors: {}",
                        self.missing_required_f16_projection_tensor_names.join(", ")
                    ),
                );
                LayerConstructionReadinessStatus::Blocked
            } else if f16_projection_allocated {
                LayerConstructionReadinessStatus::Allocated
            } else {
                LayerConstructionReadinessStatus::Deferred
            };

        let required_f32_norm_bias_status = required_f32_norm_bias_status_for_layer(
            &self.required_f32_norm_bias_tensor_names,
            fused_layer,
        );
        let rope_status = runtime_buffer
            .map(|status| {
                if status.rope_allocated {
                    LayerConstructionReadinessStatus::Allocated
                } else {
                    runtime_metadata_to_layer_status(status.metadata_status)
                }
            })
            .unwrap_or(LayerConstructionReadinessStatus::NotRequested);
        let kv_cache_status = kv_cache
            .map(|status| {
                let has_entry = status
                    .entries
                    .iter()
                    .any(|entry| entry.absolute_layer_idx == self.absolute_layer_idx);
                if status.kv_cache_allocated && has_entry {
                    LayerConstructionReadinessStatus::Allocated
                } else if has_entry {
                    LayerConstructionReadinessStatus::Deferred
                } else {
                    LayerConstructionReadinessStatus::NotApplicable
                }
            })
            .unwrap_or(LayerConstructionReadinessStatus::NotRequested);
        let metadata_status = metadata
            .map(|status| {
                if status.metadata_allocated {
                    LayerConstructionReadinessStatus::Allocated
                } else {
                    runtime_metadata_to_layer_status(status.metadata_status)
                }
            })
            .unwrap_or(LayerConstructionReadinessStatus::NotRequested);
        let fused_qkv_status = fused_layer
            .map(|layer| fused_to_layer_status(layer.fused_qkv_status))
            .unwrap_or(LayerConstructionReadinessStatus::NotRequested);
        let layernorm_f16_status = fused_layer
            .map(|layer| fused_to_layer_status(layer.layernorm_f16_status))
            .unwrap_or(LayerConstructionReadinessStatus::NotRequested);
        let postnorm_f16_status = fused_layer
            .map(|layer| fused_to_layer_status(layer.postnorm_f16_status))
            .unwrap_or(LayerConstructionReadinessStatus::NotRequested);
        let qkv_bias_f16_status = fused_layer
            .map(|layer| fused_to_layer_status(layer.qkv_bias_f16_status))
            .unwrap_or(LayerConstructionReadinessStatus::NotRequested);
        let o_proj_bias_f16_status = fused_layer
            .map(|layer| fused_to_layer_status(layer.o_proj_bias_f16_status))
            .unwrap_or(LayerConstructionReadinessStatus::NotRequested);
        let f16_scratch_status = fused_f16
            .map(|shard| fused_to_layer_status(shard.f16_scratch_status))
            .unwrap_or(LayerConstructionReadinessStatus::NotRequested);
        let moe_u8_upload_status = moe_layer.map(moe_u8_layer_status).unwrap_or_else(|| {
            if self.has_moe_u8_payload {
                LayerConstructionReadinessStatus::NotRequested
            } else {
                LayerConstructionReadinessStatus::NotApplicable
            }
        });
        let moe_router_status = moe_layer
            .map(|layer| moe_to_layer_status(layer.router_status))
            .unwrap_or_else(|| {
                if self.has_moe_u8_payload {
                    LayerConstructionReadinessStatus::NotRequested
                } else {
                    LayerConstructionReadinessStatus::NotApplicable
                }
            });
        let moe_expert_bias_status = moe_layer
            .map(|layer| moe_to_layer_status(layer.expert_bias_status))
            .unwrap_or_else(|| {
                if self.has_moe_u8_payload {
                    LayerConstructionReadinessStatus::NotRequested
                } else {
                    LayerConstructionReadinessStatus::NotApplicable
                }
            });
        let supports_gpu_decode_status = moe_layer
            .map(|layer| layer.supports_gpu_decode_status.clone())
            .unwrap_or_else(|| {
                if self.has_moe_u8_payload {
                    "not_requested".to_string()
                } else {
                    "not_applicable".to_string()
                }
            });

        if let Some(layer) = fused_layer {
            if let Some(error) = &layer.layer_error {
                push_blocker(
                    &mut blockers,
                    "blocking_fused_f16_layer_error",
                    error.clone(),
                );
            }
        }
        if let Some(layer) = moe_layer {
            if let Some(error) = &layer.layer_error {
                push_blocker(&mut blockers, "blocking_moe_layer_error", error.clone());
            }
            if moe_u8_upload_status == LayerConstructionReadinessStatus::Allocated
                && (moe_router_status == LayerConstructionReadinessStatus::Deferred
                    || moe_expert_bias_status == LayerConstructionReadinessStatus::Deferred)
            {
                push_blocker(
                    &mut blockers,
                    "moe_router_or_expert_bias_deferred",
                    "executable MoE readiness still requires router and expert-bias state".into(),
                );
            }
            if layer.supports_gpu_decode_status != "not_applicable" {
                push_blocker(
                    &mut blockers,
                    "supports_gpu_decode_not_evaluated",
                    "real supports_gpu_decode requires executable MoE layer construction".into(),
                );
            }
        }
        push_blocker(
            &mut blockers,
            "executable_layer_not_constructed",
            LAYER_SKELETON_EXECUTABLE_DEFERRED_REASON.into(),
        );

        CudaLayerConstructionStatus {
            absolute_layer_idx: self.absolute_layer_idx,
            local_layer_idx: self.local_layer_idx,
            owns_layer: self.owns_layer,
            layer_config_status,
            required_f16_projection_status,
            required_f32_norm_bias_status,
            rope_status,
            kv_cache_status,
            metadata_status,
            fused_qkv_status,
            layernorm_f16_status,
            postnorm_f16_status,
            qkv_bias_f16_status,
            o_proj_bias_f16_status,
            f16_scratch_status,
            moe_u8_upload_status,
            moe_router_status,
            moe_expert_bias_status,
            supports_gpu_decode_status,
            executable_layer_status: LayerConstructionReadinessStatus::NotConstructed,
            executable_layer_deferred_reason: Some(
                LAYER_SKELETON_EXECUTABLE_DEFERRED_REASON.into(),
            ),
            blockers,
        }
    }
}

fn layer_skeleton_store(manifest: &ShardTensorManifest) -> ShardWeightStore {
    let mut global_shape_names = manifest.required_tensor_filter_set();
    global_shape_names.extend(manifest.host_u8_tensor_filter_set());
    ShardWeightStore::new(ShardWeightStorePlan {
        device_id: manifest.device_id,
        absolute_layers: manifest.absolute_layers.clone(),
        owns_embeddings: manifest.should_load_required_tensor("model.embed_tokens.weight"),
        owns_final_head: manifest.should_load_required_tensor("model.norm.weight")
            || manifest.should_load_required_tensor("lm_head.weight"),
        required_tensor_names: manifest.required_tensor_filter_set(),
        host_u8_tensor_names: manifest.host_u8_tensor_filter_set(),
        global_shape_names,
        tied_lm_head_fallback_required: manifest.deferred_or_late_gpu_allocations.iter().any(
            |allocation| allocation == &crate::shard_plan::LateAllocationKind::TiedLmHeadFallback,
        ),
    })
}

fn push_blocker(blockers: &mut Vec<LayerConstructionBlocker>, code: &str, detail: String) {
    blockers.push(LayerConstructionBlocker {
        code: code.into(),
        detail,
    });
}

fn runtime_metadata_to_layer_status(
    status: RuntimeMetadataStatus,
) -> LayerConstructionReadinessStatus {
    match status {
        RuntimeMetadataStatus::Allocated => LayerConstructionReadinessStatus::Allocated,
        RuntimeMetadataStatus::Deferred => LayerConstructionReadinessStatus::Deferred,
        RuntimeMetadataStatus::NotApplicable => LayerConstructionReadinessStatus::NotApplicable,
    }
}

fn fused_to_layer_status(status: FusedF16AllocationStatus) -> LayerConstructionReadinessStatus {
    match status {
        FusedF16AllocationStatus::Allocated
        | FusedF16AllocationStatus::AvailableFromUploadedF16 => {
            LayerConstructionReadinessStatus::Allocated
        }
        FusedF16AllocationStatus::Deferred => LayerConstructionReadinessStatus::Deferred,
        FusedF16AllocationStatus::NotApplicable => LayerConstructionReadinessStatus::NotApplicable,
    }
}

fn moe_to_layer_status(status: MoeGpuUploadStatus) -> LayerConstructionReadinessStatus {
    match status {
        MoeGpuUploadStatus::Uploaded => LayerConstructionReadinessStatus::Allocated,
        MoeGpuUploadStatus::Deferred => LayerConstructionReadinessStatus::Deferred,
        MoeGpuUploadStatus::NotApplicable => LayerConstructionReadinessStatus::NotApplicable,
    }
}

fn required_f32_norm_bias_status_for_layer(
    names: &[String],
    fused_layer: Option<&CudaLayerFusedF16AllocationStatus>,
) -> LayerConstructionReadinessStatus {
    if names.is_empty() {
        return LayerConstructionReadinessStatus::NotApplicable;
    }
    let Some(layer) = fused_layer else {
        return LayerConstructionReadinessStatus::NotRequested;
    };
    if layer.layer_error.is_some() {
        return LayerConstructionReadinessStatus::Blocked;
    }
    let mut statuses = Vec::new();
    if names
        .iter()
        .any(|name| name.ends_with("input_layernorm.weight"))
    {
        statuses.push(layer.layernorm_f16_status);
    }
    if names
        .iter()
        .any(|name| name.ends_with("post_attention_layernorm.weight"))
    {
        statuses.push(layer.postnorm_f16_status);
    }
    if names.iter().any(|name| {
        name.ends_with("self_attn.q_proj.bias")
            || name.ends_with("self_attn.k_proj.bias")
            || name.ends_with("self_attn.v_proj.bias")
    }) {
        statuses.push(layer.qkv_bias_f16_status);
    }
    if names
        .iter()
        .any(|name| name.ends_with("self_attn.o_proj.bias"))
    {
        statuses.push(layer.o_proj_bias_f16_status);
    }
    if statuses
        .iter()
        .all(|status| *status == FusedF16AllocationStatus::Allocated)
    {
        LayerConstructionReadinessStatus::Allocated
    } else if statuses
        .iter()
        .any(|status| *status == FusedF16AllocationStatus::Deferred)
    {
        LayerConstructionReadinessStatus::Deferred
    } else {
        LayerConstructionReadinessStatus::NotApplicable
    }
}

fn moe_u8_layer_status(layer: &CudaLayerMoeGpuUploadStatus) -> LayerConstructionReadinessStatus {
    let statuses = [
        layer.gate_up_proj_blocks_status,
        layer.gate_up_proj_scales_status,
        layer.down_proj_blocks_status,
        layer.down_proj_scales_status,
    ];
    if layer.layer_error.is_some() {
        LayerConstructionReadinessStatus::Blocked
    } else if statuses
        .into_iter()
        .all(|status| status == MoeGpuUploadStatus::NotApplicable)
    {
        LayerConstructionReadinessStatus::NotApplicable
    } else if statuses
        .into_iter()
        .all(|status| status == MoeGpuUploadStatus::Uploaded)
    {
        LayerConstructionReadinessStatus::Allocated
    } else {
        LayerConstructionReadinessStatus::Deferred
    }
}

fn moe_status_for_planned(planned: bool, uploaded: bool) -> MoeGpuUploadStatus {
    if uploaded && planned {
        MoeGpuUploadStatus::Uploaded
    } else if planned {
        MoeGpuUploadStatus::Deferred
    } else {
        MoeGpuUploadStatus::NotApplicable
    }
}

fn moe_layer_tensor_name(layer_idx: usize, suffix: &str) -> String {
    format!("model.layers.{layer_idx}.mlp.{suffix}")
}

fn manifest_has_layer_tensors(
    manifest: &ShardTensorManifest,
    layer_idx: usize,
    suffixes: &[&str],
) -> bool {
    suffixes
        .iter()
        .all(|suffix| manifest_has_layer_tensor(manifest, layer_idx, suffix))
}

fn manifest_has_any_layer_tensor(
    manifest: &ShardTensorManifest,
    layer_idx: usize,
    suffixes: &[&str],
) -> bool {
    suffixes
        .iter()
        .any(|suffix| manifest_has_layer_tensor(manifest, layer_idx, suffix))
}

fn manifest_has_layer_tensor(
    manifest: &ShardTensorManifest,
    layer_idx: usize,
    suffix: &str,
) -> bool {
    manifest.should_load_required_tensor(&format!("model.layers.{layer_idx}.{suffix}"))
}

fn manifest_has_owned_u8_expert_tensors(manifest: &ShardTensorManifest, layer_idx: usize) -> bool {
    let prefix = format!("model.layers.{layer_idx}.mlp.experts.");
    manifest
        .host_u8_tensor_names
        .iter()
        .any(|name| name.starts_with(&prefix))
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::path::Path;
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream};
    use gpt_oss_core::prelude::{LLMError, Result};
    use gpt_oss_gpu::cublas::CublasHandle;
    use gpt_oss_gpu::kernel_loader::KernelLoader;
    use half::f16;

    use super::{
        CudaLayerFusedF16AllocationPlan, CudaLayerFusedF16AllocationStatus,
        CudaLayerKvCacheAllocationPlan, CudaLayerMoeGpuUploadPlan, CudaLayerMoeGpuUploadStatus,
        CudaShardFusedF16AllocationPlan, CudaShardFusedF16AllocationStatus,
        CudaShardKvCacheAllocationPlan, CudaShardKvCacheAllocationStatus,
        CudaShardMetadataAllocationPlan, CudaShardMetadataAllocationStatus,
        CudaShardMoeGpuUploadPlan, CudaShardMoeGpuUploadStatus, CudaShardResourcePlan,
        CudaShardResourceStatus, CudaShardRuntimeBufferPlan, CudaShardRuntimeBufferStatus,
        F16ScratchAllocationConfig, F16ScratchBufferStatuses, FusedF16AllocationStatus,
        MetadataAllocationConfig, MoeGpuUploadStatus, RopeRuntimeBufferConfig,
        ShardedCudaResourcePlan, ShardedCudaResourceStatus, ShardedFusedF16AllocationStatus,
        ShardedKvCacheAllocationPlan, ShardedKvCacheAllocationStatus,
        ShardedMetadataAllocationStatus, ShardedMoeGpuUploadPlan, ShardedMoeGpuUploadStatus,
        ShardedRuntimeBufferStatus, FUSED_F16_CASTS_DEFERRED_REASON,
    };
    use crate::device_map::DeviceId;
    use crate::fused_f16::{
        cast_f32_tensor_to_f16, fused_gate_up_num_elements, fused_qkv_bias_num_elements,
        fused_qkv_num_elements, get_or_load_cast_f32_to_f16_kernel,
    };
    use crate::model_loader::{ShardWeightStore, ShardWeightStorePlan};
    use crate::rope_validation::build_runtime_rope_tables;
    use crate::shard_plan::{
        LateAllocationKind, ShardTensorManifest, ShardedModelPlan, ShardedUploadManifest,
    };

    /// One non-executing CUDA ownership island for a future layer shard.
    pub struct CudaShardResources {
        pub device_id: DeviceId,
        pub absolute_layers: Vec<usize>,
        pub owns_embeddings: bool,
        pub owns_final_head: bool,
        pub context: Arc<CudaContext>,
        pub stream: Arc<CudaStream>,
        pub blas: CublasHandle,
        pub loader: Arc<KernelLoader>,
    }

    /// Non-executing collection of per-shard CUDA ownership islands.
    pub struct ShardedCudaResources {
        pub shards: Vec<CudaShardResources>,
    }

    /// One shard's non-executing runtime buffers.
    pub struct CudaShardRuntimeBuffers {
        pub device_id: DeviceId,
        pub absolute_layers: Vec<usize>,
        pub owns_embeddings: bool,
        pub owns_final_head: bool,
        pub rope_cos: CudaSlice<f32>,
        pub rope_sin: CudaSlice<f32>,
        pub plan: CudaShardRuntimeBufferPlan,
    }

    /// Shard-local runtime buffers allocated from existing resource islands.
    pub struct ShardedRuntimeBuffers {
        pub shards: Vec<CudaShardRuntimeBuffers>,
    }

    /// One layer's non-executing shard-local KV cache buffers.
    pub struct CudaLayerKvCacheBuffers {
        pub absolute_layer_idx: usize,
        pub local_cache_idx: usize,
        pub key_cache: CudaSlice<f16>,
        pub value_cache: CudaSlice<f16>,
        pub plan: CudaLayerKvCacheAllocationPlan,
    }

    /// One shard's non-executing KV cache buffers.
    pub struct CudaShardKvCacheBuffers {
        pub device_id: DeviceId,
        pub entries: Vec<CudaLayerKvCacheBuffers>,
        pub plan: CudaShardKvCacheAllocationPlan,
    }

    /// Shard-local KV cache buffers allocated from existing resource islands.
    pub struct ShardedKvCacheBuffers {
        pub shards: Vec<CudaShardKvCacheBuffers>,
    }

    /// One shard's non-executing synthetic packed metadata buffer.
    pub struct CudaShardMetadataBuffers {
        pub device_id: DeviceId,
        pub packed_metadata: CudaSlice<i32>,
        pub plan: CudaShardMetadataAllocationPlan,
    }

    /// Shard-local synthetic metadata buffers allocated from resource islands.
    pub struct ShardedMetadataBuffers {
        pub shards: Vec<CudaShardMetadataBuffers>,
    }

    /// One layer's non-executing fused f16 buffers.
    pub struct CudaLayerFusedF16Buffers {
        pub absolute_layer_idx: usize,
        pub local_layer_idx: usize,
        pub fused_qkv: Option<CudaSlice<f16>>,
        pub fused_gate_up: Option<CudaSlice<f16>>,
        pub input_layernorm: Option<CudaSlice<f16>>,
        pub post_attention_layernorm: Option<CudaSlice<f16>>,
        pub fused_qkv_bias: Option<CudaSlice<f16>>,
        pub o_proj_bias: Option<CudaSlice<f16>>,
        pub status: CudaLayerFusedF16AllocationStatus,
    }

    /// One shard's non-executing fused f16 buffers.
    pub struct CudaShardFusedF16Buffers {
        pub device_id: DeviceId,
        pub layers: Vec<CudaLayerFusedF16Buffers>,
        pub embedding_f16: Option<CudaSlice<f16>>,
        pub final_norm_f16: Option<CudaSlice<f16>>,
        pub f16_scratch: Option<CudaShardF16ScratchBuffers>,
        pub status: CudaShardFusedF16AllocationStatus,
    }

    /// Shard-local fused f16 buffers allocated from resource islands.
    pub struct ShardedFusedF16Buffers {
        pub shards: Vec<CudaShardFusedF16Buffers>,
    }

    /// One shard's non-executing f16 scratch buffers.
    pub struct CudaShardF16ScratchBuffers {
        pub device_id: DeviceId,
        pub qkv: CudaSlice<f16>,
        pub attn_out: CudaSlice<f16>,
        pub o_proj: CudaSlice<f16>,
        pub normed: CudaSlice<f16>,
        pub residual: CudaSlice<f16>,
        pub gate_up: CudaSlice<f16>,
        pub silu_out: CudaSlice<f16>,
        pub down: CudaSlice<f16>,
        pub buffers: F16ScratchBufferStatuses,
    }

    /// Shard-local f16 scratch buffers allocated from resource islands.
    pub struct ShardedF16ScratchBuffers {
        pub shards: Vec<CudaShardF16ScratchBuffers>,
        pub status: ShardedFusedF16AllocationStatus,
    }

    /// One layer's non-executing GPT-OSS MoE U8 GPU upload buffers.
    pub struct CudaLayerMoeGpuUploadBuffers {
        pub absolute_layer_idx: usize,
        pub local_layer_idx: usize,
        pub gate_up_proj_blocks: Option<CudaSlice<u8>>,
        pub gate_up_proj_scales: Option<CudaSlice<u8>>,
        pub down_proj_blocks: Option<CudaSlice<u8>>,
        pub down_proj_scales: Option<CudaSlice<u8>>,
        pub status: CudaLayerMoeGpuUploadStatus,
    }

    /// One shard's non-executing GPT-OSS MoE U8 GPU upload buffers.
    pub struct CudaShardMoeGpuUploadBuffers {
        pub device_id: DeviceId,
        pub layers: Vec<CudaLayerMoeGpuUploadBuffers>,
        pub status: CudaShardMoeGpuUploadStatus,
    }

    /// Shard-local GPT-OSS MoE U8 GPU upload buffers allocated from resource islands.
    pub struct ShardedMoeGpuUploadBuffers {
        pub shards: Vec<CudaShardMoeGpuUploadBuffers>,
        pub status: ShardedMoeGpuUploadStatus,
    }

    impl ShardedCudaResources {
        /// Construct contexts, streams, cuBLAS handles, and kernel loaders only.
        ///
        /// This is intentionally not an execution-ready runtime path: it does
        /// not upload weights, allocate KV cache, create layers, or build a
        /// `GpuModelRunner`.
        pub fn create_for_plan(plan: &ShardedModelPlan) -> Result<Self> {
            Self::create_for_plan_with_kernel_dir(plan, Path::new("/nonexistent"))
        }

        /// Construct resource islands and pass `kernel_dir` to each loader.
        pub fn create_for_plan_with_kernel_dir(
            plan: &ShardedModelPlan,
            kernel_dir: &Path,
        ) -> Result<Self> {
            let resource_plan = ShardedCudaResourcePlan::from_model_plan(plan);
            let mut shards = Vec::with_capacity(resource_plan.shards.len());

            for shard_plan in &resource_plan.shards {
                shards.push(CudaShardResources::create_from_plan(
                    shard_plan, kernel_dir,
                )?);
            }

            Ok(Self { shards })
        }

        pub fn status(&self) -> ShardedCudaResourceStatus {
            ShardedCudaResourceStatus {
                shards: self.shards.iter().map(CudaShardResources::status).collect(),
            }
        }
    }

    impl CudaShardResources {
        fn create_from_plan(plan: &CudaShardResourcePlan, kernel_dir: &Path) -> Result<Self> {
            let context = CudaContext::new(plan.device_id.0).map_err(|e| {
                LLMError::GpuError(format!(
                    "CUDA context init failed for sharded resource device {}: {e}",
                    plan.device_id
                ))
            })?;
            let stream = context.new_stream().map_err(|e| {
                LLMError::GpuError(format!(
                    "CUDA stream init failed for sharded resource device {}: {e}",
                    plan.device_id
                ))
            })?;
            let blas = CublasHandle::new(stream.clone()).map_err(|e| {
                LLMError::GpuError(format!(
                    "cuBLAS init failed for sharded resource device {}: {e}",
                    plan.device_id
                ))
            })?;
            let loader =
                KernelLoader::new(context.clone(), stream.clone(), kernel_dir).map_err(|e| {
                    LLMError::GpuError(format!(
                        "kernel loader init failed for sharded resource device {}: {e}",
                        plan.device_id
                    ))
                })?;

            Ok(Self {
                device_id: plan.device_id,
                absolute_layers: plan.absolute_layers.clone(),
                owns_embeddings: plan.owns_embeddings,
                owns_final_head: plan.owns_final_head,
                context,
                stream,
                blas,
                loader: Arc::new(loader),
            })
        }

        pub fn status(&self) -> CudaShardResourceStatus {
            CudaShardResourceStatus {
                device_id: self.device_id,
                absolute_layers: self.absolute_layers.clone(),
                owns_embeddings: self.owns_embeddings,
                owns_final_head: self.owns_final_head,
            }
        }
    }

    impl ShardedRuntimeBuffers {
        /// Allocate RoPE tables on each shard's stream. Request-shaped metadata
        /// buffers remain explicitly deferred.
        pub fn create_for_resources(
            resources: &ShardedCudaResources,
            config: RopeRuntimeBufferConfig,
        ) -> Result<Self> {
            let mut shards = Vec::with_capacity(resources.shards.len());
            for resource in &resources.shards {
                shards.push(CudaShardRuntimeBuffers::create_for_resource(
                    resource, config,
                )?);
            }
            Ok(Self { shards })
        }

        pub fn status(&self) -> ShardedRuntimeBufferStatus {
            ShardedRuntimeBufferStatus {
                shards: self
                    .shards
                    .iter()
                    .map(CudaShardRuntimeBuffers::status)
                    .collect(),
            }
        }
    }

    impl CudaShardRuntimeBuffers {
        fn create_for_resource(
            resource: &CudaShardResources,
            config: RopeRuntimeBufferConfig,
        ) -> Result<Self> {
            let (cos_table, sin_table) = build_runtime_rope_tables(
                config.head_dim,
                config.runtime_max_position(),
                config.rope_theta,
            );
            let rope_cos = resource.stream.clone_htod(&cos_table).map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} rope cos HtoD failed: {e}",
                    resource.device_id
                ))
            })?;
            let rope_sin = resource.stream.clone_htod(&sin_table).map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} rope sin HtoD failed: {e}",
                    resource.device_id
                ))
            })?;
            let plan = CudaShardRuntimeBufferPlan::from_parts(
                resource.device_id,
                resource.absolute_layers.clone(),
                resource.owns_embeddings,
                resource.owns_final_head,
                config,
            );

            Ok(Self {
                device_id: resource.device_id,
                absolute_layers: resource.absolute_layers.clone(),
                owns_embeddings: resource.owns_embeddings,
                owns_final_head: resource.owns_final_head,
                rope_cos,
                rope_sin,
                plan,
            })
        }

        pub fn status(&self) -> CudaShardRuntimeBufferStatus {
            self.plan.status(true)
        }
    }

    impl ShardedKvCacheBuffers {
        /// Allocate f16 KV cache key/value buffers for each shard-owned absolute
        /// layer. The buffers are not attached to a runner or execution path.
        pub fn create_for_resources(
            resources: &ShardedCudaResources,
            plan: &ShardedKvCacheAllocationPlan,
        ) -> Result<Self> {
            let mut shards = Vec::with_capacity(resources.shards.len());
            for resource in &resources.shards {
                let shard_plan = plan.shard_for_device(resource.device_id).ok_or_else(|| {
                    LLMError::GpuError(format!(
                        "missing KV cache allocation plan for device {}",
                        resource.device_id
                    ))
                })?;
                shards.push(CudaShardKvCacheBuffers::create_for_resource(
                    resource, shard_plan,
                )?);
            }
            Ok(Self { shards })
        }

        pub fn status(&self) -> ShardedKvCacheAllocationStatus {
            ShardedKvCacheAllocationStatus {
                shards: self
                    .shards
                    .iter()
                    .map(CudaShardKvCacheBuffers::status)
                    .collect(),
            }
        }
    }

    impl CudaShardKvCacheBuffers {
        fn create_for_resource(
            resource: &CudaShardResources,
            plan: &CudaShardKvCacheAllocationPlan,
        ) -> Result<Self> {
            let mut entries = Vec::with_capacity(plan.entries.len());

            for entry_plan in &plan.entries {
                let key_cache = resource
                    .stream
                    .alloc_zeros::<f16>(entry_plan.key_elements)
                    .map_err(|e| {
                        LLMError::GpuError(format!(
                            "shard {} KV key alloc failed absolute layer {}: {e}",
                            resource.device_id, entry_plan.absolute_layer_idx
                        ))
                    })?;
                let value_cache = resource
                    .stream
                    .alloc_zeros::<f16>(entry_plan.value_elements)
                    .map_err(|e| {
                        LLMError::GpuError(format!(
                            "shard {} KV value alloc failed absolute layer {}: {e}",
                            resource.device_id, entry_plan.absolute_layer_idx
                        ))
                    })?;

                entries.push(CudaLayerKvCacheBuffers {
                    absolute_layer_idx: entry_plan.absolute_layer_idx,
                    local_cache_idx: entry_plan.local_cache_idx,
                    key_cache,
                    value_cache,
                    plan: entry_plan.clone(),
                });
            }

            Ok(Self {
                device_id: resource.device_id,
                entries,
                plan: plan.clone(),
            })
        }

        pub fn status(&self) -> CudaShardKvCacheAllocationStatus {
            self.plan.status(true)
        }
    }

    impl ShardedMetadataBuffers {
        /// Allocate and copy synthetic packed request metadata to each shard's
        /// stream. The buffer is not attached to a runner or execution path.
        pub fn create_for_resources(
            resources: &ShardedCudaResources,
            config: MetadataAllocationConfig,
        ) -> Result<Self> {
            let mut shards = Vec::with_capacity(resources.shards.len());
            for resource in &resources.shards {
                let shard_plan = CudaShardMetadataAllocationPlan::from_parts(
                    resource.device_id,
                    resource.absolute_layers.clone(),
                    resource.owns_embeddings,
                    resource.owns_final_head,
                    config,
                );
                shards.push(CudaShardMetadataBuffers::create_for_resource(
                    resource,
                    &shard_plan,
                )?);
            }
            Ok(Self { shards })
        }

        pub fn status(&self) -> ShardedMetadataAllocationStatus {
            ShardedMetadataAllocationStatus {
                shards: self
                    .shards
                    .iter()
                    .map(CudaShardMetadataBuffers::status)
                    .collect(),
            }
        }
    }

    impl CudaShardMetadataBuffers {
        fn create_for_resource(
            resource: &CudaShardResources,
            plan: &CudaShardMetadataAllocationPlan,
        ) -> Result<Self> {
            let packed_values = plan.packed_metadata();
            let packed_metadata = resource.stream.clone_htod(&packed_values).map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} metadata HtoD failed: {e}",
                    resource.device_id
                ))
            })?;

            Ok(Self {
                device_id: resource.device_id,
                packed_metadata,
                plan: plan.clone(),
            })
        }

        pub fn status(&self) -> CudaShardMetadataAllocationStatus {
            self.plan.status(true)
        }
    }

    impl ShardedFusedF16Buffers {
        /// Allocate non-executing fused QKV and optional dense gate/up buffers
        /// for each shard-owned absolute layer. Buffers are not attached to a
        /// layer, runner, graph, or execution path.
        pub fn create_for_resources(
            resources: &ShardedCudaResources,
            upload_manifest: &crate::shard_plan::ShardedUploadManifest,
            f16_weights_by_device: &BTreeMap<DeviceId, HashMap<String, CudaSlice<f16>>>,
            f16_shapes_by_device: &BTreeMap<DeviceId, HashMap<String, Vec<usize>>>,
            f32_weights_by_device: &BTreeMap<DeviceId, HashMap<String, CudaSlice<f32>>>,
            f32_shapes_by_device: &BTreeMap<DeviceId, HashMap<String, Vec<usize>>>,
            scratch_config: Option<F16ScratchAllocationConfig>,
        ) -> Result<Self> {
            let mut shards = Vec::with_capacity(resources.shards.len());
            for resource in &resources.shards {
                let manifest = upload_manifest
                    .shard_for_device(resource.device_id)
                    .ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing fused f16 upload manifest for device {}",
                            resource.device_id
                        ))
                    })?;
                let weights = f16_weights_by_device
                    .get(&resource.device_id)
                    .ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f16 uploaded weights for fused allocation device {}",
                            resource.device_id
                        ))
                    })?;
                let shapes = f16_shapes_by_device
                    .get(&resource.device_id)
                    .ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f16 uploaded shapes for fused allocation device {}",
                            resource.device_id
                        ))
                    })?;
                let plan = CudaShardFusedF16AllocationPlan::from_manifest(manifest, scratch_config);
                shards.push(CudaShardFusedF16Buffers::create_for_resource(
                    resource,
                    manifest,
                    &plan,
                    weights,
                    shapes,
                    f32_weights_by_device.get(&resource.device_id),
                    f32_shapes_by_device.get(&resource.device_id),
                )?);
            }

            Ok(Self { shards })
        }

        pub fn status(&self) -> ShardedFusedF16AllocationStatus {
            ShardedFusedF16AllocationStatus {
                shards: self
                    .shards
                    .iter()
                    .map(|shard| shard.status.clone())
                    .collect(),
            }
        }
    }

    impl ShardedF16ScratchBuffers {
        /// Allocate non-executing f16 scratch buffers for each layer-owning
        /// shard. Buffers are not attached to a layer, runner, graph, or
        /// execution path.
        pub fn create_for_resources(
            resources: &ShardedCudaResources,
            upload_manifest: &crate::shard_plan::ShardedUploadManifest,
            config: F16ScratchAllocationConfig,
        ) -> Result<Self> {
            let plan = super::ShardedFusedF16AllocationPlan::from_upload_manifest(
                upload_manifest,
                Some(config),
            );
            let mut shards = Vec::with_capacity(resources.shards.len());
            let mut statuses = Vec::with_capacity(resources.shards.len());
            for resource in &resources.shards {
                let shard_plan = plan.shard_for_device(resource.device_id).ok_or_else(|| {
                    LLMError::GpuError(format!(
                        "missing f16 scratch allocation plan for device {}",
                        resource.device_id
                    ))
                })?;
                if shard_plan.f16_scratch_status == FusedF16AllocationStatus::NotApplicable {
                    statuses.push(shard_plan.status(false, false));
                    continue;
                }
                shards.push(CudaShardF16ScratchBuffers::create_for_resource(
                    resource, shard_plan,
                )?);
                statuses.push(shard_plan.status(false, true));
            }

            Ok(Self {
                shards,
                status: ShardedFusedF16AllocationStatus { shards: statuses },
            })
        }

        pub fn status(&self) -> ShardedFusedF16AllocationStatus {
            self.status.clone()
        }
    }

    impl CudaShardF16ScratchBuffers {
        fn create_for_resource(
            resource: &CudaShardResources,
            plan: &CudaShardFusedF16AllocationPlan,
        ) -> Result<Self> {
            let buffers = plan.f16_scratch_buffers.ok_or_else(|| {
                LLMError::GpuError(format!(
                    "missing f16 scratch sizing for shard {}",
                    resource.device_id
                ))
            })?;
            let alloc = |name: &str, elements: usize| -> Result<CudaSlice<f16>> {
                resource.stream.alloc_zeros::<f16>(elements).map_err(|e| {
                    LLMError::GpuError(format!(
                        "shard {} f16 scratch {name} alloc failed ({elements} elems): {e}",
                        resource.device_id
                    ))
                })
            };

            Ok(Self {
                device_id: resource.device_id,
                qkv: alloc("qkv", buffers.qkv.elements)?,
                attn_out: alloc("attn_out", buffers.attn_out.elements)?,
                o_proj: alloc("o_proj", buffers.o_proj.elements)?,
                normed: alloc("normed", buffers.normed.elements)?,
                residual: alloc("residual", buffers.residual.elements)?,
                gate_up: alloc("gate_up", buffers.gate_up.elements)?,
                silu_out: alloc("silu_out", buffers.silu_out.elements)?,
                down: alloc("down", buffers.down.elements)?,
                buffers,
            })
        }
    }

    impl ShardedMoeGpuUploadBuffers {
        /// Upload shard-owned GPT-OSS MoE U8 expert block/scale payloads to each
        /// shard's stream. Buffers are not wrapped in `GptOssMoeLayerWeights`,
        /// attached to layers, or used for graph decode/execution.
        pub fn create_for_resources(
            resources: &ShardedCudaResources,
            upload_manifest: &ShardedUploadManifest,
            header_bytes: &BTreeMap<String, usize>,
            u8_host_by_device: &BTreeMap<DeviceId, HashMap<String, Vec<u8>>>,
        ) -> Result<Self> {
            let plan = ShardedMoeGpuUploadPlan::from_upload_manifest(upload_manifest, header_bytes);
            let mut shards = Vec::with_capacity(resources.shards.len());
            let mut statuses = Vec::with_capacity(resources.shards.len());

            for resource in &resources.shards {
                let shard_plan = plan.shard_for_device(resource.device_id).ok_or_else(|| {
                    LLMError::GpuError(format!(
                        "missing GPT-OSS MoE upload plan for device {}",
                        resource.device_id
                    ))
                })?;
                if shard_plan.moe_gpu_status == MoeGpuUploadStatus::NotApplicable {
                    statuses.push(shard_plan.status(false));
                    continue;
                }
                let host_u8 = u8_host_by_device.get(&resource.device_id).ok_or_else(|| {
                    LLMError::GpuError(format!(
                        "missing retained U8 host map for GPT-OSS MoE upload device {}",
                        resource.device_id
                    ))
                })?;
                let shard = CudaShardMoeGpuUploadBuffers::create_for_resource(
                    resource, shard_plan, host_u8,
                )?;
                statuses.push(shard.status.clone());
                shards.push(shard);
            }

            Ok(Self {
                shards,
                status: ShardedMoeGpuUploadStatus { shards: statuses },
            })
        }

        pub fn status(&self) -> ShardedMoeGpuUploadStatus {
            self.status.clone()
        }
    }

    impl CudaShardMoeGpuUploadBuffers {
        fn create_for_resource(
            resource: &CudaShardResources,
            plan: &CudaShardMoeGpuUploadPlan,
            host_u8: &HashMap<String, Vec<u8>>,
        ) -> Result<Self> {
            let mut layers = Vec::with_capacity(plan.moe_layer_plans.len());
            let mut layer_statuses = Vec::with_capacity(plan.moe_layer_plans.len());

            for layer_plan in &plan.moe_layer_plans {
                if !layer_plan.has_any_u8_payload() {
                    layer_statuses.push(layer_plan.status(false));
                    continue;
                }
                if layer_plan.partial_u8_payload {
                    return Err(LLMError::GpuError(format!(
                        "shard {} partial GPT-OSS MoE U8 payload absolute layer {} missing tensors: {}",
                        resource.device_id,
                        layer_plan.absolute_layer_idx,
                        layer_plan.missing_u8_tensor_names().join(", ")
                    )));
                }

                let layer =
                    CudaLayerMoeGpuUploadBuffers::create_for_layer(resource, layer_plan, host_u8)?;
                layer_statuses.push(layer.status.clone());
                layers.push(layer);
            }

            let moe_gpu_uploaded = plan.moe_layer_count > 0;
            let moe_u8_gpu_tensor_count = layer_statuses
                .iter()
                .map(|layer| {
                    [
                        layer.gate_up_proj_blocks_status,
                        layer.gate_up_proj_scales_status,
                        layer.down_proj_blocks_status,
                        layer.down_proj_scales_status,
                    ]
                    .into_iter()
                    .filter(|status| status == &MoeGpuUploadStatus::Uploaded)
                    .count()
                })
                .sum();
            let moe_u8_gpu_bytes = layer_statuses
                .iter()
                .map(|layer| {
                    let mut bytes = 0;
                    if layer.gate_up_proj_blocks_status == MoeGpuUploadStatus::Uploaded {
                        bytes += layer.gate_up_proj_blocks_bytes;
                    }
                    if layer.gate_up_proj_scales_status == MoeGpuUploadStatus::Uploaded {
                        bytes += layer.gate_up_proj_scales_bytes;
                    }
                    if layer.down_proj_blocks_status == MoeGpuUploadStatus::Uploaded {
                        bytes += layer.down_proj_blocks_bytes;
                    }
                    if layer.down_proj_scales_status == MoeGpuUploadStatus::Uploaded {
                        bytes += layer.down_proj_scales_bytes;
                    }
                    bytes
                })
                .sum();

            let status = CudaShardMoeGpuUploadStatus {
                device_id: resource.device_id,
                moe_gpu_uploaded,
                moe_gpu_status: if moe_gpu_uploaded {
                    MoeGpuUploadStatus::Uploaded
                } else {
                    MoeGpuUploadStatus::NotApplicable
                },
                moe_layer_count: plan.moe_layer_count,
                moe_u8_host_tensor_count: plan.moe_u8_host_tensor_count,
                moe_u8_gpu_tensor_count,
                moe_u8_host_bytes: plan.moe_u8_host_bytes,
                moe_u8_gpu_bytes,
                moe_router_tensor_count: plan.moe_router_tensor_count,
                moe_bias_tensor_count: plan.moe_bias_tensor_count,
                moe_layer_statuses: layer_statuses,
                moe_gpu_deferred_reason: None,
                moe_gpu_error: None,
            };

            Ok(Self {
                device_id: resource.device_id,
                layers,
                status,
            })
        }
    }

    impl CudaLayerMoeGpuUploadBuffers {
        fn create_for_layer(
            resource: &CudaShardResources,
            plan: &CudaLayerMoeGpuUploadPlan,
            host_u8: &HashMap<String, Vec<u8>>,
        ) -> Result<Self> {
            let (gate_up_proj_blocks, gate_up_proj_blocks_bytes) =
                upload_moe_u8_tensor(resource, plan, host_u8, &plan.gate_up_proj_blocks_name())?;
            let (gate_up_proj_scales, gate_up_proj_scales_bytes) =
                upload_moe_u8_tensor(resource, plan, host_u8, &plan.gate_up_proj_scales_name())?;
            let (down_proj_blocks, down_proj_blocks_bytes) =
                upload_moe_u8_tensor(resource, plan, host_u8, &plan.down_proj_blocks_name())?;
            let (down_proj_scales, down_proj_scales_bytes) =
                upload_moe_u8_tensor(resource, plan, host_u8, &plan.down_proj_scales_name())?;

            let mut status = plan.status(true);
            status.gate_up_proj_blocks_bytes = gate_up_proj_blocks_bytes;
            status.gate_up_proj_scales_bytes = gate_up_proj_scales_bytes;
            status.down_proj_blocks_bytes = down_proj_blocks_bytes;
            status.down_proj_scales_bytes = down_proj_scales_bytes;

            Ok(Self {
                absolute_layer_idx: plan.absolute_layer_idx,
                local_layer_idx: plan.local_layer_idx,
                gate_up_proj_blocks: Some(gate_up_proj_blocks),
                gate_up_proj_scales: Some(gate_up_proj_scales),
                down_proj_blocks: Some(down_proj_blocks),
                down_proj_scales: Some(down_proj_scales),
                status,
            })
        }
    }

    fn upload_moe_u8_tensor(
        resource: &CudaShardResources,
        plan: &CudaLayerMoeGpuUploadPlan,
        host_u8: &HashMap<String, Vec<u8>>,
        name: &str,
    ) -> Result<(CudaSlice<u8>, usize)> {
        let data = host_u8.get(name).ok_or_else(|| {
            LLMError::GpuError(format!(
                "shard {} GPT-OSS MoE U8 host payload missing absolute layer {} tensor {name}",
                resource.device_id, plan.absolute_layer_idx
            ))
        })?;
        let buffer = resource.stream.clone_htod(data).map_err(|e| {
            LLMError::GpuError(format!(
                "shard {} GPT-OSS MoE U8 upload failed absolute layer {} tensor {name}: {e}",
                resource.device_id, plan.absolute_layer_idx
            ))
        })?;
        Ok((buffer, data.len()))
    }

    impl CudaShardFusedF16Buffers {
        fn create_for_resource(
            resource: &CudaShardResources,
            manifest: &ShardTensorManifest,
            plan: &CudaShardFusedF16AllocationPlan,
            weights: &HashMap<String, CudaSlice<f16>>,
            shapes: &HashMap<String, Vec<usize>>,
            f32_weights: Option<&HashMap<String, CudaSlice<f32>>>,
            f32_shapes: Option<&HashMap<String, Vec<usize>>>,
        ) -> Result<Self> {
            let mut global_shape_names = shapes.keys().cloned().collect::<BTreeSet<_>>();
            if let Some(f32_shapes) = f32_shapes {
                global_shape_names.extend(f32_shapes.keys().cloned());
            }
            let store = ShardWeightStore::new(ShardWeightStorePlan {
                device_id: manifest.device_id,
                absolute_layers: manifest.absolute_layers.clone(),
                owns_embeddings: resource.owns_embeddings,
                owns_final_head: resource.owns_final_head,
                required_tensor_names: manifest.required_tensor_filter_set(),
                host_u8_tensor_names: manifest.host_u8_tensor_filter_set(),
                global_shape_names,
                tied_lm_head_fallback_required: manifest
                    .deferred_or_late_gpu_allocations
                    .iter()
                    .any(|allocation| allocation == &LateAllocationKind::TiedLmHeadFallback),
            });
            let needs_f32_casts = plan.f16_layernorm_count > 0
                || plan.f16_postnorm_count > 0
                || plan.f16_qkv_bias_count > 0
                || plan.f16_o_proj_bias_count > 0
                || plan.final_norm_f16_planned;
            let cast_kernel = if needs_f32_casts {
                Some(
                    get_or_load_cast_f32_to_f16_kernel(
                        &resource.loader,
                        &resource.context,
                        &resource.stream,
                    )
                    .map_err(|e| {
                        LLMError::GpuError(format!(
                            "shard {} f16 cast kernel load failed: {e}",
                            resource.device_id
                        ))
                    })?,
                )
            } else {
                None
            };

            let mut layers = Vec::with_capacity(plan.fused_layer_plans.len());
            for layer_plan in &plan.fused_layer_plans {
                layers.push(CudaLayerFusedF16Buffers::create_for_layer(
                    resource,
                    &store,
                    layer_plan,
                    weights,
                    shapes,
                    f32_weights,
                    f32_shapes,
                    cast_kernel.as_ref(),
                )?);
            }

            let embedding_f16 = if plan.embedding_f16_planned {
                embedding_f16_available_from_uploaded(resource, &store, weights)
            } else {
                GlobalF16SideBuffer::not_applicable()
            };
            let final_norm_f16 = if plan.final_norm_f16_planned {
                allocate_final_norm_f16(
                    resource,
                    &store,
                    f32_weights.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f32 final norm weights for shard {}",
                            resource.device_id
                        ))
                    })?,
                    f32_shapes.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f32 final norm shapes for shard {}",
                            resource.device_id
                        ))
                    })?,
                    cast_kernel.as_ref().ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f16 final norm cast kernel for shard {}",
                            resource.device_id
                        ))
                    })?,
                )?
            } else {
                GlobalF16SideBuffer::not_applicable()
            };

            let fused_qkv_weight_count = layers
                .iter()
                .filter(|layer| layer.status.fused_qkv_allocated)
                .count();
            let fused_gate_up_weight_count = layers
                .iter()
                .filter(|layer| layer.status.fused_gate_up_allocated)
                .count();
            let fused_qkv_total_bytes = layers
                .iter()
                .map(|layer| layer.status.fused_qkv_bytes)
                .sum();
            let fused_gate_up_total_bytes = layers
                .iter()
                .map(|layer| layer.status.fused_gate_up_bytes)
                .sum();
            let f16_layernorm_count = layers
                .iter()
                .filter(|layer| {
                    layer.status.layernorm_f16_status == FusedF16AllocationStatus::Allocated
                })
                .count();
            let f16_postnorm_count = layers
                .iter()
                .filter(|layer| {
                    layer.status.postnorm_f16_status == FusedF16AllocationStatus::Allocated
                })
                .count();
            let f16_layernorm_total_bytes = layers
                .iter()
                .map(|layer| layer.status.layernorm_f16_bytes)
                .sum();
            let f16_postnorm_total_bytes = layers
                .iter()
                .map(|layer| layer.status.postnorm_f16_bytes)
                .sum();
            let f16_qkv_bias_count = layers
                .iter()
                .filter(|layer| {
                    layer.status.qkv_bias_f16_status == FusedF16AllocationStatus::Allocated
                })
                .count();
            let f16_o_proj_bias_count = layers
                .iter()
                .filter(|layer| {
                    layer.status.o_proj_bias_f16_status == FusedF16AllocationStatus::Allocated
                })
                .count();
            let f16_qkv_bias_total_bytes = layers
                .iter()
                .map(|layer| layer.status.qkv_bias_f16_bytes)
                .sum();
            let f16_o_proj_bias_total_bytes = layers
                .iter()
                .map(|layer| layer.status.o_proj_bias_f16_bytes)
                .sum();
            let fused_total_bytes = fused_qkv_total_bytes
                + fused_gate_up_total_bytes
                + f16_layernorm_total_bytes
                + f16_postnorm_total_bytes
                + f16_qkv_bias_total_bytes
                + f16_o_proj_bias_total_bytes
                + final_norm_f16.bytes;
            let fused_f16_allocated = fused_qkv_weight_count > 0
                || fused_gate_up_weight_count > 0
                || f16_layernorm_count > 0
                || f16_postnorm_count > 0
                || f16_qkv_bias_count > 0
                || f16_o_proj_bias_count > 0
                || final_norm_f16.buffer.is_some()
                || embedding_f16.status == FusedF16AllocationStatus::AvailableFromUploadedF16;
            let fused_f16_status = if fused_f16_allocated {
                FusedF16AllocationStatus::Allocated
            } else {
                plan.fused_status
            };
            let conversion_work_deferred = embedding_f16.status
                == FusedF16AllocationStatus::Deferred
                || final_norm_f16.status == FusedF16AllocationStatus::Deferred;
            let f16_scratch = if plan.f16_scratch_status != FusedF16AllocationStatus::NotApplicable
            {
                Some(CudaShardF16ScratchBuffers::create_for_resource(
                    resource, plan,
                )?)
            } else {
                None
            };
            let f16_scratch_allocated = f16_scratch.is_some();

            let status = CudaShardFusedF16AllocationStatus {
                device_id: resource.device_id,
                absolute_layers: resource.absolute_layers.clone(),
                owns_embeddings: resource.owns_embeddings,
                owns_final_head: resource.owns_final_head,
                fused_f16_allocated,
                fused_f16_status,
                fused_qkv_weight_count,
                fused_gate_up_weight_count,
                f16_layernorm_count,
                f16_postnorm_count,
                f16_qkv_bias_count,
                f16_o_proj_bias_count,
                embedding_f16_allocated: false,
                embedding_f16_status: embedding_f16.status,
                embedding_f16_bytes: embedding_f16.bytes,
                embedding_f16_source: embedding_f16.source.clone(),
                final_norm_f16_allocated: final_norm_f16.buffer.is_some(),
                final_norm_f16_status: final_norm_f16.status,
                final_norm_f16_bytes: final_norm_f16.bytes,
                final_norm_f16_source: final_norm_f16.source.clone(),
                fused_qkv_total_bytes,
                fused_gate_up_total_bytes,
                f16_layernorm_total_bytes,
                f16_postnorm_total_bytes,
                f16_qkv_bias_total_bytes,
                f16_o_proj_bias_total_bytes,
                fused_total_bytes,
                fused_layer_absolute_indices: plan.fused_layer_absolute_indices.clone(),
                fused_layer_statuses: layers.iter().map(|layer| layer.status.clone()).collect(),
                fused_deferred_reason: conversion_work_deferred
                    .then(|| FUSED_F16_CASTS_DEFERRED_REASON.into())
                    .or_else(|| {
                        (!fused_f16_allocated)
                            .then(|| plan.fused_deferred_reason.clone())
                            .flatten()
                    }),
                fused_error: None,
                f16_scratch_allocated,
                f16_scratch_status: if f16_scratch_allocated {
                    FusedF16AllocationStatus::Allocated
                } else {
                    plan.f16_scratch_status
                },
                f16_scratch_total_elements: if f16_scratch_allocated {
                    plan.f16_scratch_total_elements
                } else {
                    0
                },
                f16_scratch_bytes: if f16_scratch_allocated {
                    plan.f16_scratch_bytes
                } else {
                    0
                },
                f16_scratch_max_tokens: plan.f16_scratch_max_tokens,
                f16_scratch_buffers: if f16_scratch_allocated {
                    plan.f16_scratch_buffers
                } else {
                    None
                },
                f16_scratch_deferred_reason: (!f16_scratch_allocated)
                    .then(|| plan.f16_scratch_deferred_reason.clone())
                    .flatten(),
                f16_scratch_error: None,
            };

            Ok(Self {
                device_id: resource.device_id,
                layers,
                embedding_f16: embedding_f16.buffer,
                final_norm_f16: final_norm_f16.buffer,
                f16_scratch,
                status,
            })
        }
    }

    impl CudaLayerFusedF16Buffers {
        fn create_for_layer(
            resource: &CudaShardResources,
            store: &ShardWeightStore,
            plan: &CudaLayerFusedF16AllocationPlan,
            weights: &HashMap<String, CudaSlice<f16>>,
            shapes: &HashMap<String, Vec<usize>>,
            f32_weights: Option<&HashMap<String, CudaSlice<f32>>>,
            f32_shapes: Option<&HashMap<String, Vec<usize>>>,
            cast_kernel: Option<&CudaFunction>,
        ) -> Result<Self> {
            let (fused_qkv, fused_qkv_bytes) = if plan.fused_qkv_planned {
                let fused_qkv =
                    allocate_fused_qkv(resource, store, plan.absolute_layer_idx, weights, shapes)?;
                let bytes = fused_qkv.len() * std::mem::size_of::<f16>();
                (Some(fused_qkv), bytes)
            } else {
                (None, 0)
            };

            let (fused_gate_up, fused_gate_up_bytes) = if plan.fused_gate_up_planned {
                let fused_gate_up = allocate_fused_gate_up(
                    resource,
                    store,
                    plan.absolute_layer_idx,
                    weights,
                    shapes,
                )?;
                let bytes = fused_gate_up.len() * std::mem::size_of::<f16>();
                (Some(fused_gate_up), bytes)
            } else {
                (None, 0)
            };

            let (input_layernorm, layernorm_f16_bytes) = if plan.f16_layernorm_planned {
                let input_layernorm = allocate_norm_f16(
                    resource,
                    store,
                    plan.absolute_layer_idx,
                    "input_layernorm.weight",
                    f32_weights.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f32 norm weights for shard {} absolute layer {}",
                            resource.device_id, plan.absolute_layer_idx
                        ))
                    })?,
                    f32_shapes.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f32 norm shapes for shard {} absolute layer {}",
                            resource.device_id, plan.absolute_layer_idx
                        ))
                    })?,
                    cast_kernel.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f16 norm cast kernel for shard {} absolute layer {}",
                            resource.device_id, plan.absolute_layer_idx
                        ))
                    })?,
                )?;
                let bytes = input_layernorm.len() * std::mem::size_of::<f16>();
                (Some(input_layernorm), bytes)
            } else {
                (None, 0)
            };

            let (post_attention_layernorm, postnorm_f16_bytes) = if plan.f16_postnorm_planned {
                let post_attention_layernorm = allocate_norm_f16(
                    resource,
                    store,
                    plan.absolute_layer_idx,
                    "post_attention_layernorm.weight",
                    f32_weights.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f32 norm weights for shard {} absolute layer {}",
                            resource.device_id, plan.absolute_layer_idx
                        ))
                    })?,
                    f32_shapes.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f32 norm shapes for shard {} absolute layer {}",
                            resource.device_id, plan.absolute_layer_idx
                        ))
                    })?,
                    cast_kernel.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f16 norm cast kernel for shard {} absolute layer {}",
                            resource.device_id, plan.absolute_layer_idx
                        ))
                    })?,
                )?;
                let bytes = post_attention_layernorm.len() * std::mem::size_of::<f16>();
                (Some(post_attention_layernorm), bytes)
            } else {
                (None, 0)
            };

            let (fused_qkv_bias, qkv_bias_f16_bytes) = if plan.f16_qkv_bias_planned {
                let fused_qkv_bias = allocate_fused_qkv_bias_f16(
                    resource,
                    store,
                    plan.absolute_layer_idx,
                    f32_weights.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f32 bias weights for shard {} absolute layer {}",
                            resource.device_id, plan.absolute_layer_idx
                        ))
                    })?,
                    f32_shapes.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f32 bias shapes for shard {} absolute layer {}",
                            resource.device_id, plan.absolute_layer_idx
                        ))
                    })?,
                    cast_kernel.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f16 bias cast kernel for shard {} absolute layer {}",
                            resource.device_id, plan.absolute_layer_idx
                        ))
                    })?,
                )?;
                let bytes = fused_qkv_bias
                    .as_ref()
                    .map(|bias| bias.len() * std::mem::size_of::<f16>())
                    .unwrap_or(0);
                (fused_qkv_bias, bytes)
            } else {
                (None, 0)
            };

            let (o_proj_bias, o_proj_bias_f16_bytes) = if plan.f16_o_proj_bias_planned {
                let o_proj_bias = allocate_o_proj_bias_f16(
                    resource,
                    store,
                    plan.absolute_layer_idx,
                    f32_weights.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f32 bias weights for shard {} absolute layer {}",
                            resource.device_id, plan.absolute_layer_idx
                        ))
                    })?,
                    f32_shapes.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f32 bias shapes for shard {} absolute layer {}",
                            resource.device_id, plan.absolute_layer_idx
                        ))
                    })?,
                    cast_kernel.ok_or_else(|| {
                        LLMError::GpuError(format!(
                            "missing f16 bias cast kernel for shard {} absolute layer {}",
                            resource.device_id, plan.absolute_layer_idx
                        ))
                    })?,
                )?;
                let bytes = o_proj_bias
                    .as_ref()
                    .map(|bias| bias.len() * std::mem::size_of::<f16>())
                    .unwrap_or(0);
                (o_proj_bias, bytes)
            } else {
                (None, 0)
            };

            let status = plan.status(
                fused_qkv.is_some(),
                fused_gate_up.is_some(),
                input_layernorm.is_some(),
                post_attention_layernorm.is_some(),
                fused_qkv_bias.is_some(),
                o_proj_bias.is_some(),
                fused_qkv_bytes,
                fused_gate_up_bytes,
                layernorm_f16_bytes,
                postnorm_f16_bytes,
                qkv_bias_f16_bytes,
                o_proj_bias_f16_bytes,
                None,
            );

            Ok(Self {
                absolute_layer_idx: plan.absolute_layer_idx,
                local_layer_idx: plan.local_layer_idx,
                fused_qkv,
                fused_gate_up,
                input_layernorm,
                post_attention_layernorm,
                fused_qkv_bias,
                o_proj_bias,
                status,
            })
        }
    }

    struct GlobalF16SideBuffer {
        buffer: Option<CudaSlice<f16>>,
        status: FusedF16AllocationStatus,
        bytes: usize,
        source: Option<String>,
    }

    impl GlobalF16SideBuffer {
        fn not_applicable() -> Self {
            Self {
                buffer: None,
                status: FusedF16AllocationStatus::NotApplicable,
                bytes: 0,
                source: None,
            }
        }
    }

    fn embedding_f16_available_from_uploaded(
        resource: &CudaShardResources,
        store: &ShardWeightStore,
        weights: &HashMap<String, CudaSlice<f16>>,
    ) -> GlobalF16SideBuffer {
        let name = match store.embedding_tensor_name_if_owned() {
            Ok(Some(name)) => name,
            Ok(None) => return GlobalF16SideBuffer::not_applicable(),
            Err(error) => {
                return GlobalF16SideBuffer {
                    buffer: None,
                    status: FusedF16AllocationStatus::Deferred,
                    bytes: 0,
                    source: Some(format!("embedding ownership deferred: {error:?}")),
                };
            }
        };

        if let Some(weight) = weights.get(&name) {
            GlobalF16SideBuffer {
                buffer: None,
                status: FusedF16AllocationStatus::AvailableFromUploadedF16,
                bytes: weight.len() * std::mem::size_of::<f16>(),
                source: Some("uploaded_f16".into()),
            }
        } else {
            GlobalF16SideBuffer {
                buffer: None,
                status: FusedF16AllocationStatus::Deferred,
                bytes: 0,
                source: Some(format!(
                    "deferred_by_embedding_f32_fallback_boundary on shard {}",
                    resource.device_id
                )),
            }
        }
    }

    fn allocate_final_norm_f16(
        resource: &CudaShardResources,
        store: &ShardWeightStore,
        f32_weights: &HashMap<String, CudaSlice<f32>>,
        f32_shapes: &HashMap<String, Vec<usize>>,
        cast_kernel: &CudaFunction,
    ) -> Result<GlobalF16SideBuffer> {
        let Some(name) = store
            .final_norm_tensor_name_if_owned()
            .map_err(|error| LLMError::GpuError(format!("f16 final norm ownership: {error:?}")))?
        else {
            return Ok(GlobalF16SideBuffer::not_applicable());
        };
        let weight = require_f32_weight(f32_weights, &name)?;
        let elements = require_vector_shape(f32_shapes, &name)?;
        require_f32_slice_len(weight, elements, &name)?;

        let final_norm = cast_f32_tensor_to_f16(&resource.stream, weight, elements, cast_kernel)
            .map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} f16 final norm cast failed tensor {}: {e}",
                    resource.device_id, name
                ))
            })?;
        let bytes = final_norm.len() * std::mem::size_of::<f16>();
        Ok(GlobalF16SideBuffer {
            buffer: Some(final_norm),
            status: FusedF16AllocationStatus::Allocated,
            bytes,
            source: Some("f32_cast".into()),
        })
    }

    fn allocate_fused_qkv(
        resource: &CudaShardResources,
        store: &ShardWeightStore,
        absolute_layer_idx: usize,
        weights: &HashMap<String, CudaSlice<f16>>,
        shapes: &HashMap<String, Vec<usize>>,
    ) -> Result<CudaSlice<f16>> {
        let q_name = store
            .require_owned_layer_tensor_name(absolute_layer_idx, "self_attn.q_proj.weight")
            .map_err(|error| LLMError::GpuError(format!("fused QKV q ownership: {error:?}")))?;
        let k_name = store
            .require_owned_layer_tensor_name(absolute_layer_idx, "self_attn.k_proj.weight")
            .map_err(|error| LLMError::GpuError(format!("fused QKV k ownership: {error:?}")))?;
        let v_name = store
            .require_owned_layer_tensor_name(absolute_layer_idx, "self_attn.v_proj.weight")
            .map_err(|error| LLMError::GpuError(format!("fused QKV v ownership: {error:?}")))?;

        let q = require_f16_weight(weights, &q_name)?;
        let k = require_f16_weight(weights, &k_name)?;
        let v = require_f16_weight(weights, &v_name)?;
        let (q_dim, hidden) = require_matrix_shape(shapes, &q_name)?;
        let (k_dim, k_hidden) = require_matrix_shape(shapes, &k_name)?;
        let (v_dim, v_hidden) = require_matrix_shape(shapes, &v_name)?;
        if hidden != k_hidden || hidden != v_hidden {
            return Err(LLMError::GpuError(format!(
                "fused QKV hidden mismatch for layer {absolute_layer_idx}: q={hidden}, k={k_hidden}, v={v_hidden}"
            )));
        }
        if k_dim != v_dim {
            return Err(LLMError::GpuError(format!(
                "fused QKV kv dim mismatch for layer {absolute_layer_idx}: k={k_dim}, v={v_dim}"
            )));
        }

        let q_elements = checked_matrix_elements(q_dim, hidden, &q_name)?;
        let k_elements = checked_matrix_elements(k_dim, hidden, &k_name)?;
        let v_elements = checked_matrix_elements(v_dim, hidden, &v_name)?;
        require_slice_len(q, q_elements, &q_name)?;
        require_slice_len(k, k_elements, &k_name)?;
        require_slice_len(v, v_elements, &v_name)?;

        let total_elements = fused_qkv_num_elements(q_dim, k_dim, hidden);
        let mut fused = resource
            .stream
            .alloc_zeros::<f16>(total_elements)
            .map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} fused QKV alloc failed absolute layer {}: {e}",
                    resource.device_id, absolute_layer_idx
                ))
            })?;
        resource
            .stream
            .memcpy_dtod(q, &mut fused.slice_mut(..q_elements))
            .map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} fused QKV q copy failed absolute layer {}: {e}",
                    resource.device_id, absolute_layer_idx
                ))
            })?;
        resource
            .stream
            .memcpy_dtod(k, &mut fused.slice_mut(q_elements..q_elements + k_elements))
            .map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} fused QKV k copy failed absolute layer {}: {e}",
                    resource.device_id, absolute_layer_idx
                ))
            })?;
        resource
            .stream
            .memcpy_dtod(
                v,
                &mut fused.slice_mut(q_elements + k_elements..total_elements),
            )
            .map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} fused QKV v copy failed absolute layer {}: {e}",
                    resource.device_id, absolute_layer_idx
                ))
            })?;

        Ok(fused)
    }

    fn allocate_norm_f16(
        resource: &CudaShardResources,
        store: &ShardWeightStore,
        absolute_layer_idx: usize,
        suffix: &str,
        f32_weights: &HashMap<String, CudaSlice<f32>>,
        f32_shapes: &HashMap<String, Vec<usize>>,
        cast_kernel: &CudaFunction,
    ) -> Result<CudaSlice<f16>> {
        let name = store
            .require_owned_layer_tensor_name(absolute_layer_idx, suffix)
            .map_err(|error| LLMError::GpuError(format!("f16 norm ownership: {error:?}")))?;
        let weight = require_f32_weight(f32_weights, &name)?;
        let elements = require_vector_shape(f32_shapes, &name)?;
        require_f32_slice_len(weight, elements, &name)?;

        cast_f32_tensor_to_f16(&resource.stream, weight, elements, cast_kernel).map_err(|e| {
            LLMError::GpuError(format!(
                "shard {} f16 norm cast failed absolute layer {} tensor {}: {e}",
                resource.device_id, absolute_layer_idx, name
            ))
        })
    }

    fn allocate_fused_qkv_bias_f16(
        resource: &CudaShardResources,
        store: &ShardWeightStore,
        absolute_layer_idx: usize,
        f32_weights: &HashMap<String, CudaSlice<f32>>,
        f32_shapes: &HashMap<String, Vec<usize>>,
        cast_kernel: &CudaFunction,
    ) -> Result<Option<CudaSlice<f16>>> {
        let q_name = store
            .require_owned_layer_tensor_name(absolute_layer_idx, "self_attn.q_proj.bias")
            .map_err(|error| LLMError::GpuError(format!("f16 bias q ownership: {error:?}")))?;
        let k_name = store
            .require_owned_layer_tensor_name(absolute_layer_idx, "self_attn.k_proj.bias")
            .map_err(|error| LLMError::GpuError(format!("f16 bias k ownership: {error:?}")))?;
        let v_name = store
            .require_owned_layer_tensor_name(absolute_layer_idx, "self_attn.v_proj.bias")
            .map_err(|error| LLMError::GpuError(format!("f16 bias v ownership: {error:?}")))?;

        let present_count = [&q_name, &k_name, &v_name]
            .iter()
            .filter(|name| f32_weights.contains_key((*name).as_str()))
            .count();
        if present_count == 0 {
            return Ok(None);
        }
        if present_count != 3 {
            let missing_names = [&q_name, &k_name, &v_name]
                .iter()
                .filter(|name| !f32_weights.contains_key((*name).as_str()))
                .map(|name| name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(LLMError::GpuError(format!(
                "shard {} f16 bias partial QKV bias set absolute layer {} missing tensors: {}",
                resource.device_id, absolute_layer_idx, missing_names
            )));
        }

        let q = require_f32_weight(f32_weights, &q_name)?;
        let k = require_f32_weight(f32_weights, &k_name)?;
        let v = require_f32_weight(f32_weights, &v_name)?;
        let q_dim = require_vector_shape(f32_shapes, &q_name)
            .map_err(|e| LLMError::GpuError(format!("f16 bias q shape failed: {e}")))?;
        let k_dim = require_vector_shape(f32_shapes, &k_name)
            .map_err(|e| LLMError::GpuError(format!("f16 bias k shape failed: {e}")))?;
        let v_dim = require_vector_shape(f32_shapes, &v_name)
            .map_err(|e| LLMError::GpuError(format!("f16 bias v shape failed: {e}")))?;
        if k_dim != v_dim {
            return Err(LLMError::GpuError(format!(
                "shard {} f16 bias QKV kv dim mismatch absolute layer {}: k={}, v={}",
                resource.device_id, absolute_layer_idx, k_dim, v_dim
            )));
        }
        require_f32_slice_len(q, q_dim, &q_name)
            .map_err(|e| LLMError::GpuError(format!("f16 bias q length failed: {e}")))?;
        require_f32_slice_len(k, k_dim, &k_name)
            .map_err(|e| LLMError::GpuError(format!("f16 bias k length failed: {e}")))?;
        require_f32_slice_len(v, v_dim, &v_name)
            .map_err(|e| LLMError::GpuError(format!("f16 bias v length failed: {e}")))?;

        let total_elements = fused_qkv_bias_num_elements(q_dim, k_dim);
        let mut fused_bias_f32 =
            resource
                .stream
                .alloc_zeros::<f32>(total_elements)
                .map_err(|e| {
                    LLMError::GpuError(format!(
                        "shard {} f16 bias fused QKV f32 alloc failed absolute layer {}: {e}",
                        resource.device_id, absolute_layer_idx
                    ))
                })?;
        resource
            .stream
            .memcpy_dtod(q, &mut fused_bias_f32.slice_mut(..q_dim))
            .map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} f16 bias fused QKV q copy failed absolute layer {}: {e}",
                    resource.device_id, absolute_layer_idx
                ))
            })?;
        resource
            .stream
            .memcpy_dtod(k, &mut fused_bias_f32.slice_mut(q_dim..q_dim + k_dim))
            .map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} f16 bias fused QKV k copy failed absolute layer {}: {e}",
                    resource.device_id, absolute_layer_idx
                ))
            })?;
        resource
            .stream
            .memcpy_dtod(
                v,
                &mut fused_bias_f32.slice_mut(q_dim + k_dim..total_elements),
            )
            .map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} f16 bias fused QKV v copy failed absolute layer {}: {e}",
                    resource.device_id, absolute_layer_idx
                ))
            })?;

        cast_f32_tensor_to_f16(
            &resource.stream,
            &fused_bias_f32,
            total_elements,
            cast_kernel,
        )
        .map(Some)
        .map_err(|e| {
            LLMError::GpuError(format!(
                "shard {} f16 bias fused QKV cast failed absolute layer {}: {e}",
                resource.device_id, absolute_layer_idx
            ))
        })
    }

    fn allocate_o_proj_bias_f16(
        resource: &CudaShardResources,
        store: &ShardWeightStore,
        absolute_layer_idx: usize,
        f32_weights: &HashMap<String, CudaSlice<f32>>,
        f32_shapes: &HashMap<String, Vec<usize>>,
        cast_kernel: &CudaFunction,
    ) -> Result<Option<CudaSlice<f16>>> {
        let name = store
            .require_owned_layer_tensor_name(absolute_layer_idx, "self_attn.o_proj.bias")
            .map_err(|error| LLMError::GpuError(format!("f16 O bias ownership: {error:?}")))?;
        let Some(weight) = f32_weights.get(&name) else {
            return Ok(None);
        };
        let elements = require_vector_shape(f32_shapes, &name)
            .map_err(|e| LLMError::GpuError(format!("f16 O bias shape failed: {e}")))?;
        require_f32_slice_len(weight, elements, &name)
            .map_err(|e| LLMError::GpuError(format!("f16 O bias length failed: {e}")))?;

        cast_f32_tensor_to_f16(&resource.stream, weight, elements, cast_kernel)
            .map(Some)
            .map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} f16 O bias cast failed absolute layer {} tensor {}: {e}",
                    resource.device_id, absolute_layer_idx, name
                ))
            })
    }

    fn allocate_fused_gate_up(
        resource: &CudaShardResources,
        store: &ShardWeightStore,
        absolute_layer_idx: usize,
        weights: &HashMap<String, CudaSlice<f16>>,
        shapes: &HashMap<String, Vec<usize>>,
    ) -> Result<CudaSlice<f16>> {
        let gate_name = store
            .require_owned_layer_tensor_name(absolute_layer_idx, "mlp.gate_proj.weight")
            .map_err(|error| {
                LLMError::GpuError(format!("fused gate/up gate ownership: {error:?}"))
            })?;
        let up_name = store
            .require_owned_layer_tensor_name(absolute_layer_idx, "mlp.up_proj.weight")
            .map_err(|error| {
                LLMError::GpuError(format!("fused gate/up up ownership: {error:?}"))
            })?;
        let gate = require_f16_weight(weights, &gate_name)?;
        let up = require_f16_weight(weights, &up_name)?;
        let (intermediate, hidden) = require_matrix_shape(shapes, &gate_name)?;
        let (up_intermediate, up_hidden) = require_matrix_shape(shapes, &up_name)?;
        if intermediate != up_intermediate || hidden != up_hidden {
            return Err(LLMError::GpuError(format!(
                "fused gate/up shape mismatch for layer {absolute_layer_idx}: gate=[{intermediate},{hidden}], up=[{up_intermediate},{up_hidden}]"
            )));
        }

        let gate_elements = checked_matrix_elements(intermediate, hidden, &gate_name)?;
        let up_elements = checked_matrix_elements(up_intermediate, up_hidden, &up_name)?;
        require_slice_len(gate, gate_elements, &gate_name)?;
        require_slice_len(up, up_elements, &up_name)?;

        let total_elements = fused_gate_up_num_elements(intermediate, hidden);
        let mut fused = resource
            .stream
            .alloc_zeros::<f16>(total_elements)
            .map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} fused gate/up alloc failed absolute layer {}: {e}",
                    resource.device_id, absolute_layer_idx
                ))
            })?;
        resource
            .stream
            .memcpy_dtod(gate, &mut fused.slice_mut(..gate_elements))
            .map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} fused gate copy failed absolute layer {}: {e}",
                    resource.device_id, absolute_layer_idx
                ))
            })?;
        resource
            .stream
            .memcpy_dtod(up, &mut fused.slice_mut(gate_elements..total_elements))
            .map_err(|e| {
                LLMError::GpuError(format!(
                    "shard {} fused up copy failed absolute layer {}: {e}",
                    resource.device_id, absolute_layer_idx
                ))
            })?;

        Ok(fused)
    }

    fn require_f16_weight<'a>(
        weights: &'a HashMap<String, CudaSlice<f16>>,
        name: &str,
    ) -> Result<&'a CudaSlice<f16>> {
        weights
            .get(name)
            .ok_or_else(|| LLMError::GpuError(format!("missing uploaded f16 tensor {name}")))
    }

    fn require_f32_weight<'a>(
        weights: &'a HashMap<String, CudaSlice<f32>>,
        name: &str,
    ) -> Result<&'a CudaSlice<f32>> {
        weights
            .get(name)
            .ok_or_else(|| LLMError::GpuError(format!("missing uploaded f32 tensor {name}")))
    }

    fn require_matrix_shape(
        shapes: &HashMap<String, Vec<usize>>,
        name: &str,
    ) -> Result<(usize, usize)> {
        let shape = shapes
            .get(name)
            .ok_or_else(|| LLMError::GpuError(format!("missing f16 shape for tensor {name}")))?;
        match shape.as_slice() {
            [rows, cols] => Ok((*rows, *cols)),
            other => Err(LLMError::GpuError(format!(
                "expected 2D f16 tensor shape for {name}, got {other:?}"
            ))),
        }
    }

    fn require_vector_shape(shapes: &HashMap<String, Vec<usize>>, name: &str) -> Result<usize> {
        let shape = shapes
            .get(name)
            .ok_or_else(|| LLMError::GpuError(format!("missing f32 shape for tensor {name}")))?;
        match shape.as_slice() {
            [elements] => Ok(*elements),
            other => Err(LLMError::GpuError(format!(
                "expected 1D f32 tensor shape for {name}, got {other:?}"
            ))),
        }
    }

    fn checked_matrix_elements(rows: usize, cols: usize, name: &str) -> Result<usize> {
        rows.checked_mul(cols).ok_or_else(|| {
            LLMError::GpuError(format!(
                "matrix element count overflow for tensor {name}: {rows} x {cols}"
            ))
        })
    }

    fn require_slice_len(slice: &CudaSlice<f16>, expected: usize, name: &str) -> Result<()> {
        if slice.len() == expected {
            Ok(())
        } else {
            Err(LLMError::GpuError(format!(
                "f16 tensor {name} length mismatch: got {}, expected {expected}",
                slice.len()
            )))
        }
    }

    fn require_f32_slice_len(slice: &CudaSlice<f32>, expected: usize, name: &str) -> Result<()> {
        if slice.len() == expected {
            Ok(())
        } else {
            Err(LLMError::GpuError(format!(
                "f32 tensor {name} length mismatch: got {}, expected {expected}",
                slice.len()
            )))
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::{
    CudaLayerFusedF16Buffers, CudaLayerKvCacheBuffers, CudaLayerMoeGpuUploadBuffers,
    CudaShardF16ScratchBuffers, CudaShardFusedF16Buffers, CudaShardKvCacheBuffers,
    CudaShardMetadataBuffers, CudaShardMoeGpuUploadBuffers, CudaShardResources,
    CudaShardRuntimeBuffers, ShardedCudaResources, ShardedF16ScratchBuffers,
    ShardedFusedF16Buffers, ShardedKvCacheBuffers, ShardedMetadataBuffers,
    ShardedMoeGpuUploadBuffers, ShardedRuntimeBuffers,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_map::{DeviceId, DeviceMap};
    use crate::shard_plan::{ShardedModelPlan, UploadManifestOptions};
    #[cfg(feature = "cuda")]
    use cudarc::driver::CudaContext;

    fn single_plan() -> ShardedModelPlan {
        let device_map = DeviceMap::single(24, DeviceId(0)).unwrap();
        ShardedModelPlan::from_device_map(device_map, 24).unwrap()
    }

    fn split_plan() -> ShardedModelPlan {
        let device_map = DeviceMap::parse("split:0-11@0,12-23@1", 24, DeviceId(0)).unwrap();
        ShardedModelPlan::from_device_map(device_map, 24).unwrap()
    }

    fn runtime_buffer_config() -> RopeRuntimeBufferConfig {
        RopeRuntimeBufferConfig::new(4, 16, 10000.0).unwrap()
    }

    fn kv_cache_allocation_config() -> KvCacheAllocationConfig {
        KvCacheAllocationConfig::new(2, 4, 3, 5).unwrap()
    }

    fn metadata_allocation_config() -> MetadataAllocationConfig {
        MetadataAllocationConfig::new_decode(2, 2, 17, 16, 64, None).unwrap()
    }

    fn fused_manifest_tensor_names() -> Vec<&'static str> {
        vec![
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.o_proj.bias",
            "model.layers.11.self_attn.q_proj.weight",
            "model.layers.11.self_attn.k_proj.weight",
            "model.layers.11.self_attn.v_proj.weight",
            "model.layers.11.mlp.gate_proj.weight",
            "model.layers.11.mlp.up_proj.weight",
            "model.layers.11.input_layernorm.weight",
            "model.layers.11.post_attention_layernorm.weight",
            "model.layers.12.self_attn.q_proj.weight",
            "model.layers.12.self_attn.k_proj.weight",
            "model.layers.12.self_attn.v_proj.weight",
            "model.layers.12.mlp.gate_proj.weight",
            "model.layers.12.mlp.up_proj.weight",
            "model.layers.12.input_layernorm.weight",
            "model.layers.12.post_attention_layernorm.weight",
            "model.layers.23.self_attn.q_proj.weight",
            "model.layers.23.self_attn.k_proj.weight",
            "model.layers.23.self_attn.v_proj.weight",
            "model.layers.23.mlp.gate_proj.weight",
            "model.layers.23.mlp.up_proj.weight",
            "model.layers.23.input_layernorm.weight",
            "model.layers.23.post_attention_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
    }

    fn split_fused_upload_manifest() -> crate::shard_plan::ShardedUploadManifest {
        split_plan()
            .upload_manifest_for_tensor_names(
                fused_manifest_tensor_names(),
                UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap()
    }

    fn split_gpt_oss_upload_manifest() -> crate::shard_plan::ShardedUploadManifest {
        split_plan()
            .upload_manifest_for_tensor_names(
                vec![
                    "model.embed_tokens.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.k_proj.weight",
                    "model.layers.0.self_attn.v_proj.weight",
                    "model.layers.0.input_layernorm.weight",
                    "model.layers.0.post_attention_layernorm.weight",
                    "model.layers.0.mlp.experts.gate_up_proj_blocks",
                    "model.layers.11.mlp.experts.gate_up_proj_scales",
                    "model.layers.12.self_attn.q_proj.weight",
                    "model.layers.12.self_attn.k_proj.weight",
                    "model.layers.12.self_attn.v_proj.weight",
                    "model.layers.12.input_layernorm.weight",
                    "model.layers.12.post_attention_layernorm.weight",
                    "model.layers.23.mlp.experts.down_proj_blocks",
                    "model.norm.weight",
                ],
                UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap()
    }

    fn split_full_norm_upload_manifest() -> crate::shard_plan::ShardedUploadManifest {
        let mut names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        for layer_idx in 0..24 {
            names.push(format!("model.layers.{layer_idx}.input_layernorm.weight"));
            names.push(format!(
                "model.layers.{layer_idx}.post_attention_layernorm.weight"
            ));
        }

        split_plan()
            .upload_manifest_for_tensor_names(
                names.iter().map(String::as_str),
                UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap()
    }

    fn split_full_bias_upload_manifest() -> crate::shard_plan::ShardedUploadManifest {
        let mut names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        for layer_idx in 0..24 {
            names.push(format!("model.layers.{layer_idx}.self_attn.q_proj.bias"));
            names.push(format!("model.layers.{layer_idx}.self_attn.k_proj.bias"));
            names.push(format!("model.layers.{layer_idx}.self_attn.v_proj.bias"));
            names.push(format!("model.layers.{layer_idx}.self_attn.o_proj.bias"));
        }

        split_plan()
            .upload_manifest_for_tensor_names(
                names.iter().map(String::as_str),
                UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap()
    }

    fn split_partial_qkv_bias_upload_manifest() -> crate::shard_plan::ShardedUploadManifest {
        split_plan()
            .upload_manifest_for_tensor_names(
                [
                    "model.embed_tokens.weight",
                    "model.layers.12.self_attn.q_proj.bias",
                    "model.norm.weight",
                    "lm_head.weight",
                ],
                UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap()
    }

    fn split_moe_upload_manifest() -> crate::shard_plan::ShardedUploadManifest {
        split_plan()
            .upload_manifest_for_tensor_names(
                [
                    "model.embed_tokens.weight",
                    "model.norm.weight",
                    "lm_head.weight",
                    "model.layers.0.mlp.experts.gate_up_proj_blocks",
                    "model.layers.0.mlp.experts.gate_up_proj_scales",
                    "model.layers.0.mlp.experts.down_proj_blocks",
                    "model.layers.0.mlp.experts.down_proj_scales",
                    "model.layers.0.mlp.router.weight",
                    "model.layers.0.mlp.router.bias",
                    "model.layers.0.mlp.experts.gate_up_proj_bias",
                    "model.layers.0.mlp.experts.down_proj_bias",
                    "model.layers.12.mlp.experts.gate_up_proj_blocks",
                    "model.layers.12.mlp.experts.gate_up_proj_scales",
                    "model.layers.12.mlp.experts.down_proj_blocks",
                    "model.layers.12.mlp.experts.down_proj_scales",
                    "model.layers.12.mlp.router.weight",
                    "model.layers.12.mlp.router.bias",
                    "model.layers.12.mlp.experts.gate_up_proj_bias",
                    "model.layers.12.mlp.experts.down_proj_bias",
                ],
                UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap()
    }

    fn split_layer_skeleton_upload_manifest() -> crate::shard_plan::ShardedUploadManifest {
        let mut names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        for layer_idx in 0..24 {
            for suffix in [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.bias",
                "self_attn.o_proj.bias",
            ] {
                names.push(format!("model.layers.{layer_idx}.{suffix}"));
            }
        }

        split_plan()
            .upload_manifest_for_tensor_names(
                names.iter().map(String::as_str),
                UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap()
    }

    fn split_layer_skeleton_moe_upload_manifest() -> crate::shard_plan::ShardedUploadManifest {
        let mut names = split_layer_skeleton_tensor_names();
        for layer_idx in [0, 12] {
            for suffix in [
                "mlp.experts.gate_up_proj_blocks",
                "mlp.experts.gate_up_proj_scales",
                "mlp.experts.down_proj_blocks",
                "mlp.experts.down_proj_scales",
                "mlp.router.weight",
                "mlp.router.bias",
                "mlp.experts.gate_up_proj_bias",
                "mlp.experts.down_proj_bias",
            ] {
                names.push(format!("model.layers.{layer_idx}.{suffix}"));
            }
        }

        split_plan()
            .upload_manifest_for_tensor_names(
                names.iter().map(String::as_str),
                UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap()
    }

    fn split_layer_skeleton_tensor_names() -> Vec<String> {
        let mut names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.norm.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        for layer_idx in 0..24 {
            for suffix in [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.bias",
                "self_attn.o_proj.bias",
            ] {
                names.push(format!("model.layers.{layer_idx}.{suffix}"));
            }
        }
        names
    }

    fn moe_header_bytes() -> BTreeMap<String, usize> {
        [
            "model.layers.0.mlp.experts.gate_up_proj_blocks",
            "model.layers.0.mlp.experts.gate_up_proj_scales",
            "model.layers.0.mlp.experts.down_proj_blocks",
            "model.layers.0.mlp.experts.down_proj_scales",
            "model.layers.12.mlp.experts.gate_up_proj_blocks",
            "model.layers.12.mlp.experts.gate_up_proj_scales",
            "model.layers.12.mlp.experts.down_proj_blocks",
            "model.layers.12.mlp.experts.down_proj_scales",
        ]
        .into_iter()
        .map(|name| (name.to_string(), 4))
        .collect()
    }

    #[test]
    fn resource_plan_single_has_one_shard() {
        let plan = ShardedCudaResourcePlan::from_model_plan(&single_plan());

        assert_eq!(plan.shards.len(), 1);
        assert_eq!(plan.shards[0].device_id, DeviceId(0));
        assert_eq!(plan.shards[0].absolute_layers, (0..24).collect::<Vec<_>>());
        assert!(plan.shards[0].owns_embeddings);
        assert!(plan.shards[0].owns_final_head);
    }

    #[test]
    fn resource_plan_split_has_two_shards() {
        let plan = ShardedCudaResourcePlan::from_model_plan(&split_plan());

        assert_eq!(plan.shards.len(), 2);
        assert_eq!(plan.shards[0].device_id, DeviceId(0));
        assert_eq!(plan.shards[1].device_id, DeviceId(1));
    }

    #[test]
    fn resource_plan_gpu0_owns_embeddings_and_layers_0_through_11() {
        let plan = ShardedCudaResourcePlan::from_model_plan(&split_plan());
        let gpu0 = &plan.shards[0];

        assert_eq!(gpu0.absolute_layers, (0..12).collect::<Vec<_>>());
        assert!(gpu0.owns_embeddings);
        assert!(!gpu0.owns_final_head);
    }

    #[test]
    fn resource_plan_gpu1_owns_layers_12_through_23_and_final_head() {
        let plan = ShardedCudaResourcePlan::from_model_plan(&split_plan());
        let gpu1 = &plan.shards[1];

        assert_eq!(gpu1.absolute_layers, (12..24).collect::<Vec<_>>());
        assert!(!gpu1.owns_embeddings);
        assert!(gpu1.owns_final_head);
    }

    #[test]
    fn resource_plan_preserves_absolute_layer_ids() {
        let plan = ShardedCudaResourcePlan::from_model_plan(&split_plan());

        assert_eq!(plan.shards[0].absolute_layers.first(), Some(&0));
        assert_eq!(plan.shards[0].absolute_layers.last(), Some(&11));
        assert_eq!(plan.shards[1].absolute_layers.first(), Some(&12));
        assert_eq!(plan.shards[1].absolute_layers.last(), Some(&23));
    }

    #[test]
    fn resource_plan_has_no_duplicate_absolute_layer_ownership() {
        let plan = ShardedCudaResourcePlan::from_model_plan(&split_plan());

        assert!(plan.has_unique_absolute_layer_ownership());
    }

    #[test]
    fn resource_status_matches_plan_without_cuda() {
        let plan = ShardedCudaResourcePlan::from_model_plan(&split_plan());
        let status = ShardedCudaResourceStatus::from_plan(&plan);

        assert_eq!(
            status.shards,
            plan.shards.iter().map(|s| s.status()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn moe_gpu_upload_plan_preserves_absolute_layer_ownership() {
        let manifest = split_moe_upload_manifest();
        let plan = ShardedMoeGpuUploadPlan::from_upload_manifest(&manifest, &moe_header_bytes());

        let gpu0 = plan.shard_for_device(DeviceId(0)).unwrap();
        assert_eq!(gpu0.absolute_layers, (0..12).collect::<Vec<_>>());
        assert_eq!(gpu0.moe_gpu_status, MoeGpuUploadStatus::Deferred);
        assert_eq!(gpu0.moe_layer_count, 1);
        assert_eq!(gpu0.moe_u8_host_tensor_count, 4);
        assert_eq!(gpu0.moe_u8_host_bytes, 16);
        assert_eq!(gpu0.moe_router_tensor_count, 2);
        assert_eq!(gpu0.moe_bias_tensor_count, 2);
        let gpu0_layer0 = gpu0
            .moe_layer_plans
            .iter()
            .find(|layer| layer.absolute_layer_idx == 0)
            .unwrap();
        assert_eq!(gpu0_layer0.local_layer_idx, 0);
        assert!(gpu0_layer0.gate_up_proj_blocks_planned);
        assert!(gpu0_layer0.gate_up_proj_scales_planned);
        assert!(gpu0_layer0.down_proj_blocks_planned);
        assert!(gpu0_layer0.down_proj_scales_planned);
        assert!(gpu0_layer0.router_planned);
        assert!(gpu0_layer0.expert_bias_planned);
        assert!(!gpu0_layer0.partial_u8_payload);

        let gpu1 = plan.shard_for_device(DeviceId(1)).unwrap();
        assert_eq!(gpu1.absolute_layers, (12..24).collect::<Vec<_>>());
        assert_eq!(gpu1.moe_gpu_status, MoeGpuUploadStatus::Deferred);
        assert_eq!(gpu1.moe_layer_count, 1);
        assert_eq!(gpu1.moe_u8_host_tensor_count, 4);
        assert_eq!(gpu1.moe_u8_host_bytes, 16);
        assert!(gpu1
            .moe_layer_plans
            .iter()
            .all(|layer| layer.absolute_layer_idx >= 12));
        let gpu1_layer12 = gpu1
            .moe_layer_plans
            .iter()
            .find(|layer| layer.absolute_layer_idx == 12)
            .unwrap();
        assert_eq!(gpu1_layer12.local_layer_idx, 0);
        assert!(gpu1_layer12.gate_up_proj_blocks_planned);
    }

    #[test]
    fn moe_gpu_upload_status_is_deferred_until_explicit_upload() {
        let manifest = split_moe_upload_manifest();
        let plan = ShardedMoeGpuUploadPlan::from_upload_manifest(&manifest, &moe_header_bytes());
        let status = ShardedMoeGpuUploadStatus::from_plan(&plan, false);

        let gpu0 = status
            .shards
            .iter()
            .find(|shard| shard.device_id == DeviceId(0))
            .unwrap();
        assert!(!gpu0.moe_gpu_uploaded);
        assert_eq!(gpu0.moe_gpu_status, MoeGpuUploadStatus::Deferred);
        assert_eq!(gpu0.moe_u8_gpu_tensor_count, 0);
        assert_eq!(gpu0.moe_u8_gpu_bytes, 0);
        assert!(gpu0.moe_gpu_deferred_reason.is_some());
        let layer0 = gpu0
            .moe_layer_statuses
            .iter()
            .find(|layer| layer.absolute_layer_idx == 0)
            .unwrap();
        assert_eq!(
            layer0.gate_up_proj_blocks_status,
            MoeGpuUploadStatus::Deferred
        );
        assert_eq!(layer0.router_status, MoeGpuUploadStatus::Deferred);
        assert_eq!(layer0.expert_bias_status, MoeGpuUploadStatus::Deferred);
        assert_eq!(
            layer0.supports_gpu_decode_status,
            "not_evaluated_without_layer_construction"
        );
    }

    #[test]
    fn moe_gpu_upload_status_marks_uploaded_u8_without_decode_readiness() {
        let manifest = split_moe_upload_manifest();
        let plan = ShardedMoeGpuUploadPlan::from_upload_manifest(&manifest, &moe_header_bytes());
        let status = ShardedMoeGpuUploadStatus::from_plan(&plan, true);

        let gpu1 = status
            .shards
            .iter()
            .find(|shard| shard.device_id == DeviceId(1))
            .unwrap();
        assert!(gpu1.moe_gpu_uploaded);
        assert_eq!(gpu1.moe_gpu_status, MoeGpuUploadStatus::Uploaded);
        assert_eq!(gpu1.moe_u8_gpu_tensor_count, 4);
        assert_eq!(gpu1.moe_u8_gpu_bytes, 16);
        assert!(gpu1.moe_gpu_deferred_reason.is_none());
        let layer12 = gpu1
            .moe_layer_statuses
            .iter()
            .find(|layer| layer.absolute_layer_idx == 12)
            .unwrap();
        assert_eq!(layer12.local_layer_idx, 0);
        assert_eq!(
            layer12.gate_up_proj_blocks_status,
            MoeGpuUploadStatus::Uploaded
        );
        assert_eq!(layer12.router_status, MoeGpuUploadStatus::Deferred);
        assert_eq!(layer12.expert_bias_status, MoeGpuUploadStatus::Deferred);
        assert_eq!(
            layer12.supports_gpu_decode_status,
            "gpu_u8_uploaded_but_not_evaluated_without_layer_construction"
        );
    }

    #[test]
    fn layer_construction_plan_preserves_split_absolute_and_local_indices() {
        let manifest = split_layer_skeleton_upload_manifest();
        let plan = ShardedLayerConstructionPlan::from_upload_manifest(&manifest);

        assert_eq!(plan.shards.len(), 2);
        assert_eq!(plan.shards[0].device_id, DeviceId(0));
        assert_eq!(plan.shards[0].absolute_layers, (0..12).collect::<Vec<_>>());
        assert_eq!(plan.shards[0].layer_plans.len(), 12);
        assert_eq!(plan.shards[0].layer_plans[0].absolute_layer_idx, 0);
        assert_eq!(plan.shards[0].layer_plans[0].local_layer_idx, 0);
        assert!(plan.shards[0].layer_plans[0].owns_layer);

        let gpu1 = &plan.shards[1];
        assert_eq!(gpu1.device_id, DeviceId(1));
        assert_eq!(gpu1.absolute_layers, (12..24).collect::<Vec<_>>());
        assert_eq!(gpu1.layer_plans.len(), 12);
        let layer12 = &gpu1.layer_plans[0];
        assert_eq!(layer12.absolute_layer_idx, 12);
        assert_eq!(layer12.local_layer_idx, 0);
        assert!(layer12.owns_layer);
        assert!(layer12
            .required_f16_projection_tensor_names
            .contains(&"model.layers.12.self_attn.q_proj.weight".to_string()));
        assert!(layer12
            .missing_required_f16_projection_tensor_names
            .is_empty());
    }

    #[test]
    fn layer_construction_status_is_nonexecuting_without_allocations() {
        let manifest = split_layer_skeleton_upload_manifest();
        let plan = ShardedLayerConstructionPlan::from_upload_manifest(&manifest);
        let status =
            ShardedLayerConstructionStatus::from_plan(&plan, false, None, None, None, None, None);

        assert_eq!(status.shards.len(), 2);
        for shard in &status.shards {
            assert!(shard.layer_skeleton_built);
            assert_eq!(
                shard.layer_skeleton_status,
                LayerConstructionReadinessStatus::SkeletonComplete
            );
            assert_eq!(shard.layer_skeleton_count, 12);
            assert_eq!(shard.layer_skeleton_ready_count, 12);
            assert_eq!(shard.layer_skeleton_blocked_count, 0);
            assert_eq!(shard.layer_skeleton_deferred_count, 12);
            for layer in &shard.layer_skeletons {
                assert_eq!(
                    layer.layer_config_status,
                    LayerConstructionReadinessStatus::SkeletonComplete
                );
                assert_eq!(
                    layer.required_f16_projection_status,
                    LayerConstructionReadinessStatus::Deferred
                );
                assert_eq!(
                    layer.kv_cache_status,
                    LayerConstructionReadinessStatus::NotRequested
                );
                assert_eq!(
                    layer.metadata_status,
                    LayerConstructionReadinessStatus::NotRequested
                );
                assert_eq!(
                    layer.fused_qkv_status,
                    LayerConstructionReadinessStatus::NotRequested
                );
                assert_eq!(
                    layer.f16_scratch_status,
                    LayerConstructionReadinessStatus::NotRequested
                );
                assert_eq!(
                    layer.executable_layer_status,
                    LayerConstructionReadinessStatus::NotConstructed
                );
                assert_ne!(layer.supports_gpu_decode_status, "true");
                assert!(layer
                    .blockers
                    .iter()
                    .any(|blocker| blocker.code == "executable_layer_not_constructed"));
            }
        }
    }

    #[test]
    fn layer_construction_status_surfaces_requested_kv_metadata_fused_and_scratch() {
        let manifest = split_layer_skeleton_upload_manifest();
        let layer_plan = ShardedLayerConstructionPlan::from_upload_manifest(&manifest);
        let runtime_plan =
            ShardedRuntimeBufferPlan::from_model_plan(&split_plan(), runtime_buffer_config());
        let runtime_status = ShardedRuntimeBufferStatus::from_plan(&runtime_plan, true);
        let kv_plan = ShardedKvCacheAllocationPlan::from_model_plan(
            &split_plan(),
            kv_cache_allocation_config(),
        );
        let kv_status = ShardedKvCacheAllocationStatus::from_plan(&kv_plan, true);
        let metadata_plan = ShardedMetadataAllocationPlan::from_model_plan(
            &split_plan(),
            metadata_allocation_config(),
        );
        let metadata_status = ShardedMetadataAllocationStatus::from_plan(&metadata_plan, true);
        let scratch_config = F16ScratchAllocationConfig::new(8, 8, 8, 16, 1).unwrap();
        let fused_plan =
            ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, Some(scratch_config));
        let fused_status = ShardedFusedF16AllocationStatus::from_plan(&fused_plan, true, true);

        let status = ShardedLayerConstructionStatus::from_plan(
            &layer_plan,
            true,
            Some(&runtime_status),
            Some(&kv_status),
            Some(&metadata_status),
            Some(&fused_status),
            None,
        );

        let layer12 = status.shards[1]
            .layer_skeletons
            .iter()
            .find(|layer| layer.absolute_layer_idx == 12)
            .unwrap();
        assert_eq!(layer12.local_layer_idx, 0);
        assert_eq!(
            layer12.required_f16_projection_status,
            LayerConstructionReadinessStatus::Allocated
        );
        assert_eq!(
            layer12.required_f32_norm_bias_status,
            LayerConstructionReadinessStatus::Deferred
        );
        assert_eq!(
            layer12.rope_status,
            LayerConstructionReadinessStatus::Allocated
        );
        assert_eq!(
            layer12.kv_cache_status,
            LayerConstructionReadinessStatus::Allocated
        );
        assert_eq!(
            layer12.metadata_status,
            LayerConstructionReadinessStatus::Allocated
        );
        assert_eq!(
            layer12.fused_qkv_status,
            LayerConstructionReadinessStatus::Deferred
        );
        assert_eq!(
            layer12.layernorm_f16_status,
            LayerConstructionReadinessStatus::Deferred
        );
        assert_eq!(
            layer12.postnorm_f16_status,
            LayerConstructionReadinessStatus::Deferred
        );
        assert_eq!(
            layer12.qkv_bias_f16_status,
            LayerConstructionReadinessStatus::Deferred
        );
        assert_eq!(
            layer12.o_proj_bias_f16_status,
            LayerConstructionReadinessStatus::Deferred
        );
        assert_eq!(
            layer12.f16_scratch_status,
            LayerConstructionReadinessStatus::Allocated
        );
        assert_eq!(
            layer12.executable_layer_status,
            LayerConstructionReadinessStatus::NotConstructed
        );
    }

    #[test]
    fn layer_construction_status_surfaces_moe_upload_without_decode_readiness() {
        let manifest = split_layer_skeleton_moe_upload_manifest();
        let layer_plan = ShardedLayerConstructionPlan::from_upload_manifest(&manifest);
        let moe_plan =
            ShardedMoeGpuUploadPlan::from_upload_manifest(&manifest, &moe_header_bytes());
        let moe_status = ShardedMoeGpuUploadStatus::from_plan(&moe_plan, true);
        let status = ShardedLayerConstructionStatus::from_plan(
            &layer_plan,
            true,
            None,
            None,
            None,
            None,
            Some(&moe_status),
        );

        let layer12 = status.shards[1]
            .layer_skeletons
            .iter()
            .find(|layer| layer.absolute_layer_idx == 12)
            .unwrap();
        assert_eq!(layer12.local_layer_idx, 0);
        assert_eq!(
            layer12.moe_u8_upload_status,
            LayerConstructionReadinessStatus::Allocated
        );
        assert_eq!(
            layer12.moe_router_status,
            LayerConstructionReadinessStatus::Deferred
        );
        assert_eq!(
            layer12.moe_expert_bias_status,
            LayerConstructionReadinessStatus::Deferred
        );
        assert_eq!(
            layer12.supports_gpu_decode_status,
            "gpu_u8_uploaded_but_not_evaluated_without_layer_construction"
        );
        assert_ne!(layer12.supports_gpu_decode_status, "true");
        assert_eq!(
            layer12.executable_layer_status,
            LayerConstructionReadinessStatus::NotConstructed
        );
        assert!(layer12
            .blockers
            .iter()
            .any(|blocker| blocker.code == "moe_router_or_expert_bias_deferred"));
        assert!(layer12
            .blockers
            .iter()
            .any(|blocker| blocker.code == "supports_gpu_decode_not_evaluated"));
    }

    #[test]
    fn runtime_buffer_plan_single_has_one_shard_with_rope_tables() {
        let plan =
            ShardedRuntimeBufferPlan::from_model_plan(&single_plan(), runtime_buffer_config());

        assert_eq!(plan.shards.len(), 1);
        assert_eq!(plan.shards[0].device_id, DeviceId(0));
        assert_eq!(plan.shards[0].absolute_layers, (0..24).collect::<Vec<_>>());
        assert_eq!(plan.shards[0].rope_cos_elements, 32);
        assert_eq!(plan.shards[0].rope_sin_elements, 32);
        assert_eq!(plan.shards[0].rope_total_bytes, 256);
    }

    #[test]
    fn runtime_buffer_plan_split_preserves_shard_ownership() {
        let plan =
            ShardedRuntimeBufferPlan::from_model_plan(&split_plan(), runtime_buffer_config());

        assert_eq!(plan.shards.len(), 2);
        assert_eq!(plan.shards[0].device_id, DeviceId(0));
        assert_eq!(plan.shards[0].absolute_layers, (0..12).collect::<Vec<_>>());
        assert!(plan.shards[0].owns_embeddings);
        assert!(!plan.shards[0].owns_final_head);
        assert_eq!(plan.shards[1].device_id, DeviceId(1));
        assert_eq!(plan.shards[1].absolute_layers, (12..24).collect::<Vec<_>>());
        assert!(!plan.shards[1].owns_embeddings);
        assert!(plan.shards[1].owns_final_head);
    }

    #[test]
    fn runtime_buffer_plan_defers_request_shaped_metadata() {
        let plan =
            ShardedRuntimeBufferPlan::from_model_plan(&split_plan(), runtime_buffer_config());

        for shard in &plan.shards {
            assert_eq!(shard.metadata_status, RuntimeMetadataStatus::Deferred);
            assert!(shard
                .metadata_deferred_reason
                .as_deref()
                .unwrap()
                .contains("request-shaped metadata"));
        }
    }

    #[test]
    fn runtime_buffer_status_reports_allocated_rope_flag() {
        let plan =
            ShardedRuntimeBufferPlan::from_model_plan(&split_plan(), runtime_buffer_config());
        let status = ShardedRuntimeBufferStatus::from_plan(&plan, true);

        assert_eq!(status.shards.len(), 2);
        for shard in &status.shards {
            assert!(shard.rope_allocated);
            assert_eq!(shard.rope_cos_elements, 32);
            assert_eq!(shard.rope_sin_elements, 32);
            assert_eq!(shard.rope_total_bytes, 256);
            assert!(!shard.metadata_allocated);
            assert_eq!(shard.metadata_status, RuntimeMetadataStatus::Deferred);
            assert!(shard.runtime_buffer_error.is_none());
        }
    }

    #[test]
    fn kv_cache_allocation_plan_single_has_all_absolute_layers() {
        let plan = ShardedKvCacheAllocationPlan::from_model_plan(
            &single_plan(),
            kv_cache_allocation_config(),
        );

        assert_eq!(plan.shards.len(), 1);
        assert_eq!(plan.shards[0].device_id, DeviceId(0));
        assert_eq!(
            plan.shards[0]
                .entries
                .iter()
                .map(|entry| entry.absolute_layer_idx)
                .collect::<Vec<_>>(),
            (0..24).collect::<Vec<_>>()
        );
    }

    #[test]
    fn kv_cache_allocation_plan_split_preserves_absolute_and_local_indices() {
        let plan = ShardedKvCacheAllocationPlan::from_model_plan(
            &split_plan(),
            kv_cache_allocation_config(),
        );

        assert_eq!(plan.shards.len(), 2);
        assert_eq!(plan.shards[0].device_id, DeviceId(0));
        assert_eq!(plan.shards[1].device_id, DeviceId(1));
        assert_eq!(plan.shards[0].entries[0].absolute_layer_idx, 0);
        assert_eq!(plan.shards[0].entries[0].local_cache_idx, 0);
        assert_eq!(plan.shards[0].entries[11].absolute_layer_idx, 11);
        assert_eq!(plan.shards[0].entries[11].local_cache_idx, 11);
        assert_eq!(plan.shards[1].entries[0].absolute_layer_idx, 12);
        assert_eq!(plan.shards[1].entries[0].local_cache_idx, 0);
        assert_eq!(plan.shards[1].entries[11].absolute_layer_idx, 23);
        assert_eq!(plan.shards[1].entries[11].local_cache_idx, 11);
    }

    #[test]
    fn kv_cache_allocation_plan_reports_bytes_per_shard() {
        let config = kv_cache_allocation_config();
        let plan = ShardedKvCacheAllocationPlan::from_model_plan(&split_plan(), config);
        let per_cache = config.bytes_per_layer_cache();

        for shard in &plan.shards {
            assert_eq!(shard.entries.len(), 12);
            assert_eq!(shard.key_total_bytes, per_cache * 12);
            assert_eq!(shard.value_total_bytes, per_cache * 12);
            assert_eq!(shard.total_bytes, per_cache * 24);
            for entry in &shard.entries {
                assert_eq!(entry.key_bytes, per_cache);
                assert_eq!(entry.value_bytes, per_cache);
            }
        }
    }

    #[test]
    fn kv_cache_allocation_status_reports_allocated_flag() {
        let plan = ShardedKvCacheAllocationPlan::from_model_plan(
            &split_plan(),
            kv_cache_allocation_config(),
        );
        let status = ShardedKvCacheAllocationStatus::from_plan(&plan, true);

        assert_eq!(status.shards.len(), 2);
        for shard in &status.shards {
            assert!(shard.kv_cache_allocated);
            assert_eq!(shard.entries.len(), 12);
            assert!(shard.key_total_bytes > 0);
            assert!(shard.value_total_bytes > 0);
            assert!(shard.kv_cache_error.is_none());
        }
    }

    #[test]
    fn decode_metadata_requires_tokens_equal_sequences() {
        let err = MetadataAllocationConfig::new_decode(2, 1, 1, 16, 64, None).unwrap_err();

        assert!(err.contains("metadata-num-tokens == metadata-num-seqs"));
    }

    #[test]
    fn decode_metadata_rejects_zero_context_len() {
        let err = MetadataAllocationConfig::new_decode(1, 1, 0, 16, 64, None).unwrap_err();

        assert!(err.contains("metadata-context-len"));
    }

    #[test]
    fn decode_metadata_rejects_zero_block_size() {
        let err = MetadataAllocationConfig::new_decode(1, 1, 1, 0, 64, None).unwrap_err();

        assert!(err.contains("metadata-block-size"));
    }

    #[test]
    fn metadata_mode_rejects_unsupported_values() {
        let err = "prefill".parse::<MetadataMode>().unwrap_err();

        assert!(err.contains("only decode is supported"));
    }

    #[test]
    fn decode_metadata_packed_element_count_matches_layout_formula() {
        let config = metadata_allocation_config();

        assert_eq!(config.graph_max_blocks(), 4);
        assert_eq!(config.token_ids_len(), 2);
        assert_eq!(config.positions_len(), 2);
        assert_eq!(config.context_lens_len(), 2);
        assert_eq!(config.block_tables_len(), 8);
        assert_eq!(config.slot_mapping_len(), 2);
        assert_eq!(config.seq_start_pos_len(), 3);
        assert_eq!(config.packed_elements(), 19);
        assert_eq!(config.packed_bytes(), 19 * std::mem::size_of::<i32>());
    }

    #[test]
    fn decode_metadata_generates_expected_seq_start_positions() {
        let config = metadata_allocation_config();

        assert_eq!(config.seq_start_pos(), vec![0, 1, 2]);
    }

    #[test]
    fn decode_metadata_positions_are_context_len_minus_one() {
        let config = metadata_allocation_config();

        assert_eq!(config.positions(), vec![16, 16]);
    }

    #[test]
    fn decode_metadata_slot_mapping_uses_final_token_slot() {
        let config = metadata_allocation_config();

        assert_eq!(config.slot_mapping(), vec![16, 16]);
    }

    #[test]
    fn decode_metadata_block_tables_are_padded_to_graph_max_blocks() {
        let config = metadata_allocation_config();

        assert_eq!(config.block_tables(), vec![0, 1, 0, 0, 0, 1, 0, 0]);
    }

    #[test]
    fn decode_metadata_rejects_required_blocks_beyond_kv_blocks() {
        let kv_config = KvCacheAllocationConfig::new(2, 4, 2, 16).unwrap();
        let err =
            MetadataAllocationConfig::new_decode(1, 1, 33, 16, 64, Some(&kv_config)).unwrap_err();

        assert!(err.contains("exceeding kv-num-blocks"));
    }

    #[test]
    fn decode_metadata_rejects_kv_block_size_mismatch() {
        let kv_config = KvCacheAllocationConfig::new(2, 4, 3, 8).unwrap();
        let err =
            MetadataAllocationConfig::new_decode(1, 1, 17, 16, 64, Some(&kv_config)).unwrap_err();

        assert!(err.contains("metadata-block-size"));
        assert!(err.contains("kv-block-size"));
    }

    #[test]
    fn metadata_allocation_plan_split_reports_per_shard_shapes() {
        let plan = ShardedMetadataAllocationPlan::from_model_plan(
            &split_plan(),
            metadata_allocation_config(),
        );
        let status = ShardedMetadataAllocationStatus::from_plan(&plan, false);

        assert_eq!(plan.shards.len(), 2);
        assert_eq!(status.shards.len(), 2);
        assert_eq!(status.shards[0].device_id, DeviceId(0));
        assert!(!status.shards[0].metadata_allocated);
        assert_eq!(
            status.shards[0].metadata_status,
            RuntimeMetadataStatus::Deferred
        );
        assert_eq!(status.shards[0].num_tokens, 2);
        assert_eq!(status.shards[0].num_seqs, 2);
        assert_eq!(status.shards[0].graph_max_blocks, 4);
        assert_eq!(status.shards[0].packed_elements, 19);
        assert_eq!(status.shards[1].device_id, DeviceId(1));
        assert_eq!(status.shards[1].packed_bytes, status.shards[0].packed_bytes);
    }

    #[test]
    fn fused_f16_plan_split_has_two_shards() {
        let manifest = split_fused_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);

        assert_eq!(plan.shards.len(), 2);
        assert_eq!(plan.shards[0].device_id, DeviceId(0));
        assert_eq!(plan.shards[1].device_id, DeviceId(1));
    }

    #[test]
    fn fused_f16_plan_preserves_split_absolute_layers() {
        let manifest = split_fused_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);

        assert_eq!(plan.shards[0].absolute_layers, (0..12).collect::<Vec<_>>());
        assert_eq!(
            plan.shards[0].fused_layer_absolute_indices,
            (0..12).collect::<Vec<_>>()
        );
        assert_eq!(plan.shards[1].absolute_layers, (12..24).collect::<Vec<_>>());
        assert_eq!(
            plan.shards[1].fused_layer_absolute_indices,
            (12..24).collect::<Vec<_>>()
        );
    }

    #[test]
    fn fused_f16_plan_places_embedding_and_final_norm_on_owning_shards() {
        let manifest = split_fused_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);

        assert!(plan.shards[0].owns_embeddings);
        assert!(plan.shards[0].embedding_f16_planned);
        assert!(!plan.shards[0].final_norm_f16_planned);

        assert!(plan.shards[1].owns_final_head);
        assert!(!plan.shards[1].embedding_f16_planned);
        assert!(plan.shards[1].final_norm_f16_planned);
    }

    #[test]
    fn fused_f16_plan_status_reports_global_side_buffers_deferred_until_allocation() {
        let manifest = split_fused_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);
        let status = ShardedFusedF16AllocationStatus::from_plan(&plan, false, false);

        let gpu0 = &status.shards[0];
        assert_eq!(
            gpu0.embedding_f16_status,
            FusedF16AllocationStatus::Deferred
        );
        assert_eq!(
            gpu0.final_norm_f16_status,
            FusedF16AllocationStatus::NotApplicable
        );
        assert_eq!(gpu0.embedding_f16_bytes, 0);
        assert_eq!(gpu0.embedding_f16_source, None);
        assert!(!gpu0.embedding_f16_allocated);

        let gpu1 = &status.shards[1];
        assert_eq!(
            gpu1.embedding_f16_status,
            FusedF16AllocationStatus::NotApplicable
        );
        assert_eq!(
            gpu1.final_norm_f16_status,
            FusedF16AllocationStatus::Deferred
        );
        assert_eq!(gpu1.final_norm_f16_bytes, 0);
        assert_eq!(gpu1.final_norm_f16_source, None);
        assert!(!gpu1.final_norm_f16_allocated);
    }

    #[test]
    fn fused_f16_plan_counts_only_present_layer_fused_inputs() {
        let manifest = split_fused_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);

        assert_eq!(plan.shards[0].fused_qkv_weight_count, 2);
        assert_eq!(plan.shards[0].fused_gate_up_weight_count, 2);
        assert_eq!(plan.shards[0].f16_layernorm_count, 2);
        assert_eq!(plan.shards[0].f16_postnorm_count, 2);
        assert_eq!(plan.shards[0].f16_qkv_bias_count, 1);
        assert_eq!(plan.shards[0].f16_o_proj_bias_count, 1);

        assert_eq!(plan.shards[1].fused_qkv_weight_count, 2);
        assert_eq!(plan.shards[1].fused_gate_up_weight_count, 2);
        assert_eq!(plan.shards[1].f16_layernorm_count, 2);
        assert_eq!(plan.shards[1].f16_postnorm_count, 2);
        assert_eq!(plan.shards[1].f16_qkv_bias_count, 0);
        assert_eq!(plan.shards[1].f16_o_proj_bias_count, 0);
    }

    #[test]
    fn fused_f16_layer_statuses_preserve_absolute_and_local_indices() {
        let manifest = split_fused_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);
        let status = ShardedFusedF16AllocationStatus::from_plan(&plan, false, false);
        let gpu1 = &status.shards[1];

        let layer12 = gpu1
            .fused_layer_statuses
            .iter()
            .find(|layer| layer.absolute_layer_idx == 12)
            .unwrap();

        assert_eq!(layer12.local_layer_idx, 0);
        assert!(!layer12.fused_qkv_allocated);
        assert_eq!(layer12.fused_qkv_status, FusedF16AllocationStatus::Deferred);
        assert_eq!(layer12.fused_qkv_bytes, 0);
    }

    #[test]
    fn fused_f16_norm_plan_covers_split_absolute_layers() {
        let manifest = split_full_norm_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);

        assert_eq!(plan.shards[0].f16_layernorm_count, 12);
        assert_eq!(plan.shards[0].f16_postnorm_count, 12);
        assert_eq!(plan.shards[1].f16_layernorm_count, 12);
        assert_eq!(plan.shards[1].f16_postnorm_count, 12);
        assert!(plan.shards[0].fused_layer_plans[..12]
            .iter()
            .all(|layer| layer.f16_layernorm_planned && layer.f16_postnorm_planned));
        assert_eq!(plan.shards[1].fused_layer_plans[0].absolute_layer_idx, 12);
        assert!(plan.shards[1].fused_layer_plans[0].f16_layernorm_planned);
        assert!(plan.shards[1].fused_layer_plans[0].f16_postnorm_planned);
    }

    #[test]
    fn fused_f16_norm_plan_status_defers_bytes_until_allocation() {
        let manifest = split_full_norm_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);
        let status = ShardedFusedF16AllocationStatus::from_plan(&plan, false, false);

        let gpu1_layer12 = status.shards[1]
            .fused_layer_statuses
            .iter()
            .find(|layer| layer.absolute_layer_idx == 12)
            .unwrap();
        assert_eq!(gpu1_layer12.local_layer_idx, 0);
        assert_eq!(
            gpu1_layer12.layernorm_f16_status,
            FusedF16AllocationStatus::Deferred
        );
        assert_eq!(
            gpu1_layer12.postnorm_f16_status,
            FusedF16AllocationStatus::Deferred
        );
        assert_eq!(gpu1_layer12.layernorm_f16_bytes, 0);
        assert_eq!(gpu1_layer12.postnorm_f16_bytes, 0);
        assert_eq!(status.shards[1].f16_layernorm_total_bytes, 0);
        assert_eq!(status.shards[1].f16_postnorm_total_bytes, 0);
    }

    #[test]
    fn fused_f16_bias_plan_covers_split_absolute_layers() {
        let manifest = split_full_bias_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);

        assert_eq!(plan.shards[0].f16_qkv_bias_count, 12);
        assert_eq!(plan.shards[0].f16_o_proj_bias_count, 12);
        assert_eq!(plan.shards[1].f16_qkv_bias_count, 12);
        assert_eq!(plan.shards[1].f16_o_proj_bias_count, 12);
        assert!(plan.shards[0].fused_layer_plans[..12]
            .iter()
            .all(|layer| layer.f16_qkv_bias_planned && layer.f16_o_proj_bias_planned));
        assert_eq!(plan.shards[1].fused_layer_plans[0].absolute_layer_idx, 12);
        assert!(plan.shards[1].fused_layer_plans[0].f16_qkv_bias_planned);
        assert!(plan.shards[1].fused_layer_plans[0].f16_o_proj_bias_planned);
    }

    #[test]
    fn fused_f16_bias_plan_status_defers_bytes_until_allocation() {
        let manifest = split_full_bias_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);
        let status = ShardedFusedF16AllocationStatus::from_plan(&plan, false, false);

        let gpu1_layer12 = status.shards[1]
            .fused_layer_statuses
            .iter()
            .find(|layer| layer.absolute_layer_idx == 12)
            .unwrap();
        assert_eq!(gpu1_layer12.local_layer_idx, 0);
        assert_eq!(
            gpu1_layer12.qkv_bias_f16_status,
            FusedF16AllocationStatus::Deferred
        );
        assert_eq!(
            gpu1_layer12.o_proj_bias_f16_status,
            FusedF16AllocationStatus::Deferred
        );
        assert_eq!(gpu1_layer12.qkv_bias_f16_bytes, 0);
        assert_eq!(gpu1_layer12.o_proj_bias_f16_bytes, 0);
        assert_eq!(status.shards[1].f16_qkv_bias_total_bytes, 0);
        assert_eq!(status.shards[1].f16_o_proj_bias_total_bytes, 0);
    }

    #[test]
    fn fused_f16_missing_all_qkv_biases_reports_not_applicable() {
        let manifest = split_gpt_oss_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);
        let status = ShardedFusedF16AllocationStatus::from_plan(&plan, false, false);

        let gpu1_layer12 = status.shards[1]
            .fused_layer_statuses
            .iter()
            .find(|layer| layer.absolute_layer_idx == 12)
            .unwrap();
        assert_eq!(
            gpu1_layer12.qkv_bias_f16_status,
            FusedF16AllocationStatus::NotApplicable
        );
        assert_eq!(gpu1_layer12.qkv_bias_f16_bytes, 0);
    }

    #[test]
    fn fused_f16_partial_qkv_bias_presence_is_planned_for_all_or_error_boundary() {
        let manifest = split_partial_qkv_bias_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);
        let status = ShardedFusedF16AllocationStatus::from_plan(&plan, false, false);

        assert_eq!(plan.shards[0].f16_qkv_bias_count, 0);
        assert_eq!(plan.shards[1].f16_qkv_bias_count, 1);
        let gpu1_layer12 = status.shards[1]
            .fused_layer_statuses
            .iter()
            .find(|layer| layer.absolute_layer_idx == 12)
            .unwrap();
        assert_eq!(
            gpu1_layer12.qkv_bias_f16_status,
            FusedF16AllocationStatus::Deferred
        );
    }

    #[test]
    fn fused_f16_plan_does_not_require_dense_gate_up_for_gpt_oss_u8_experts() {
        let manifest = split_gpt_oss_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);
        let status = ShardedFusedF16AllocationStatus::from_plan(&plan, false, false);

        assert_eq!(plan.shards[0].fused_gate_up_weight_count, 0);
        assert_eq!(plan.shards[1].fused_gate_up_weight_count, 0);
        assert_eq!(plan.shards[0].fused_qkv_weight_count, 1);
        assert_eq!(plan.shards[1].fused_qkv_weight_count, 1);

        let gpu0_layer0 = status.shards[0]
            .fused_layer_statuses
            .iter()
            .find(|layer| layer.absolute_layer_idx == 0)
            .unwrap();
        assert_eq!(
            gpu0_layer0.fused_qkv_status,
            FusedF16AllocationStatus::Deferred
        );
        assert_eq!(
            gpu0_layer0.fused_gate_up_status,
            FusedF16AllocationStatus::NotApplicable
        );
        assert!(plan.shards[0].fused_layer_plans[0].has_u8_expert_tensors);
    }

    #[test]
    fn fused_f16_status_reports_deferred_runner_coupling_reason() {
        let manifest = split_fused_upload_manifest();
        let plan = ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, None);
        let status = ShardedFusedF16AllocationStatus::from_plan(&plan, false, false);

        for shard in &status.shards {
            assert!(!shard.fused_f16_allocated);
            assert_eq!(shard.fused_f16_status, FusedF16AllocationStatus::Deferred);
            assert!(shard
                .fused_deferred_reason
                .as_deref()
                .unwrap()
                .contains("GpuModelRunner::fuse_weights"));
            assert!(!shard.embedding_f16_allocated);
            assert!(!shard.final_norm_f16_allocated);
        }
    }

    #[test]
    fn f16_scratch_status_requires_explicit_config_and_reports_deferred_reason() {
        let manifest = split_fused_upload_manifest();
        let scratch_config = F16ScratchAllocationConfig::new(8, 8, 8, 16, 1).unwrap();
        let plan =
            ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, Some(scratch_config));
        let status = ShardedFusedF16AllocationStatus::from_plan(&plan, false, false);

        for shard in &status.shards {
            assert!(!shard.f16_scratch_allocated);
            assert_eq!(shard.f16_scratch_status, FusedF16AllocationStatus::Deferred);
            assert_eq!(shard.f16_scratch_max_tokens, Some(1));
            assert_eq!(shard.f16_scratch_total_elements, 0);
            assert!(shard.f16_scratch_buffers.is_none());
            assert!(shard
                .f16_scratch_deferred_reason
                .as_deref()
                .unwrap()
                .contains("CUDA allocation pass"));
        }
    }

    #[test]
    fn f16_scratch_allocated_status_reports_per_buffer_counts() {
        let manifest = split_fused_upload_manifest();
        let scratch_config = F16ScratchAllocationConfig::new(8, 8, 8, 16, 1).unwrap();
        let expected = scratch_config.buffer_statuses().unwrap();
        let plan =
            ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, Some(scratch_config));
        let status = ShardedFusedF16AllocationStatus::from_plan(&plan, false, true);

        for shard in &status.shards {
            assert!(shard.f16_scratch_allocated);
            assert_eq!(
                shard.f16_scratch_status,
                FusedF16AllocationStatus::Allocated
            );
            assert_eq!(shard.f16_scratch_max_tokens, Some(1));
            assert_eq!(shard.f16_scratch_total_elements, expected.total_elements());
            assert_eq!(shard.f16_scratch_bytes, expected.total_bytes());
            assert_eq!(shard.f16_scratch_buffers, Some(expected));
            assert_eq!(shard.f16_scratch_deferred_reason, None);
        }
    }

    #[test]
    fn f16_scratch_config_rejects_zero_max_tokens() {
        let err = F16ScratchAllocationConfig::new(8, 8, 8, 16, 0).unwrap_err();

        assert!(err.contains("f16-scratch-max-tokens"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires two visible CUDA devices and constructs context/stream/cuBLAS/kernel-loader islands"]
    fn ignored_two_gpu_sharded_cuda_resource_constructor_smoke() -> gpt_oss_core::prelude::Result<()>
    {
        let visible_devices = CudaContext::device_count()
            .map_err(|e| gpt_oss_core::prelude::LLMError::GpuError(format!("{e}")))?;
        if visible_devices < 2 {
            eprintln!(
                "skipping two-GPU sharded resource smoke: only {visible_devices} CUDA device(s) visible"
            );
            return Ok(());
        }

        let resources = ShardedCudaResources::create_for_plan(&split_plan())?;
        let status = resources.status();

        assert_eq!(status.shards.len(), 2);
        assert_eq!(status.shards[0].device_id, DeviceId(0));
        assert_eq!(
            status.shards[0].absolute_layers,
            (0..12).collect::<Vec<_>>()
        );
        assert_eq!(status.shards[1].device_id, DeviceId(1));
        assert_eq!(
            status.shards[1].absolute_layers,
            (12..24).collect::<Vec<_>>()
        );
        Ok(())
    }
}
