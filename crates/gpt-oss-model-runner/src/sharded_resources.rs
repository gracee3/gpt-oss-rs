//! Non-executing CUDA resource skeletons for future layer sharding.
//!
//! The pure plan in this module is always available. CUDA resource construction
//! is feature-gated and deliberately stops at context/stream/cuBLAS/kernel-loader
//! ownership; it does not upload model tensors, allocate KV cache, or construct a
//! runner.

use crate::device_map::DeviceId;
use crate::shard_plan::{ShardTensorManifest, ShardedKvCachePlan, ShardedModelPlan};
use std::str::FromStr;

pub const RUNTIME_METADATA_DEFERRED_REASON: &str =
    "request-shaped metadata packing buffers require batch/sequence inputs";
pub const FUSED_F16_DEFERRED_REASON: &str =
    "GpuModelRunner::fuse_weights assumes full-model runner-owned weight containers and full-runner layer indexing";
pub const FUSED_F16_CASTS_DEFERRED_REASON: &str =
    "fused QKV/gate-up buffers and f16 layernorm/postnorm conversions allocated; f16 bias/final/embed conversions remain deferred by cast/helper boundary";
pub const F16_SCRATCH_DEFERRED_REASON: &str =
    "F16LayerScratch is private runner state and tied to GpuModelRunner allocation";

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
    Deferred,
    NotApplicable,
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
    pub max_tokens: usize,
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
    pub o_proj_bias_f16_status: FusedF16AllocationStatus,
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
    pub fused_layer_absolute_indices: Vec<usize>,
    pub fused_layer_plans: Vec<CudaLayerFusedF16AllocationPlan>,
    pub fused_total_bytes: usize,
    pub fused_status: FusedF16AllocationStatus,
    pub fused_deferred_reason: Option<String>,
    pub f16_scratch_status: FusedF16AllocationStatus,
    pub f16_scratch_max_tokens: Option<usize>,
    pub f16_scratch_bytes: usize,
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
    pub final_norm_f16_allocated: bool,
    pub fused_qkv_total_bytes: usize,
    pub fused_gate_up_total_bytes: usize,
    pub f16_layernorm_total_bytes: usize,
    pub f16_postnorm_total_bytes: usize,
    pub fused_total_bytes: usize,
    pub fused_layer_absolute_indices: Vec<usize>,
    pub fused_layer_statuses: Vec<CudaLayerFusedF16AllocationStatus>,
    pub fused_deferred_reason: Option<String>,
    pub fused_error: Option<String>,
    pub f16_scratch_allocated: bool,
    pub f16_scratch_status: FusedF16AllocationStatus,
    pub f16_scratch_bytes: usize,
    pub f16_scratch_max_tokens: Option<usize>,
    pub f16_scratch_deferred_reason: Option<String>,
    pub f16_scratch_error: Option<String>,
}

/// Public status for all shard-local fused/preconverted f16 and scratch skeletons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedFusedF16AllocationStatus {
    pub shards: Vec<CudaShardFusedF16AllocationStatus>,
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
            FusedF16AllocationStatus::Deferred => "deferred",
            FusedF16AllocationStatus::NotApplicable => "not_applicable",
        }
    }
}

impl F16ScratchAllocationConfig {
    pub fn new(max_tokens: usize) -> Result<Self, String> {
        if max_tokens == 0 {
            return Err("f16-scratch-max-tokens must be non-zero".into());
        }

        Ok(Self { max_tokens })
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
        let f16_scratch_status = if scratch_config.is_some() && !manifest.absolute_layers.is_empty()
        {
            FusedF16AllocationStatus::Deferred
        } else {
            FusedF16AllocationStatus::NotApplicable
        };

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
            fused_layer_absolute_indices,
            fused_layer_plans,
            fused_total_bytes: 0,
            fused_status,
            fused_deferred_reason: (fused_status == FusedF16AllocationStatus::Deferred)
                .then(|| FUSED_F16_DEFERRED_REASON.into()),
            f16_scratch_status,
            f16_scratch_max_tokens: scratch_config.map(|config| config.max_tokens),
            f16_scratch_bytes: 0,
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
            final_norm_f16_allocated: fused_f16_allocated && self.final_norm_f16_planned,
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
            fused_total_bytes: if fused_f16_allocated {
                self.fused_total_bytes
            } else {
                0
            },
            fused_layer_absolute_indices: self.fused_layer_absolute_indices.clone(),
            fused_layer_statuses: self
                .fused_layer_plans
                .iter()
                .map(|layer| layer.status(false, false, false, false, 0, 0, 0, 0, None))
                .collect(),
            fused_deferred_reason: (!fused_f16_allocated)
                .then(|| self.fused_deferred_reason.clone())
                .flatten(),
            fused_error: None,
            f16_scratch_allocated,
            f16_scratch_status,
            f16_scratch_bytes: if f16_scratch_allocated {
                self.f16_scratch_bytes
            } else {
                0
            },
            f16_scratch_max_tokens: self.f16_scratch_max_tokens,
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
            f16_qkv_bias_planned: manifest_has_layer_tensors(
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
        fused_qkv_bytes: usize,
        fused_gate_up_bytes: usize,
        layernorm_f16_bytes: usize,
        postnorm_f16_bytes: usize,
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
            qkv_bias_f16_status: deferred_status_for_planned(self.f16_qkv_bias_planned),
            o_proj_bias_f16_status: deferred_status_for_planned(self.f16_o_proj_bias_planned),
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

fn deferred_status_for_planned(planned: bool) -> FusedF16AllocationStatus {
    if planned {
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

fn manifest_has_layer_tensors(
    manifest: &ShardTensorManifest,
    layer_idx: usize,
    suffixes: &[&str],
) -> bool {
    suffixes
        .iter()
        .all(|suffix| manifest_has_layer_tensor(manifest, layer_idx, suffix))
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
        CudaLayerKvCacheAllocationPlan, CudaShardFusedF16AllocationPlan,
        CudaShardFusedF16AllocationStatus, CudaShardKvCacheAllocationPlan,
        CudaShardKvCacheAllocationStatus, CudaShardMetadataAllocationPlan,
        CudaShardMetadataAllocationStatus, CudaShardResourcePlan, CudaShardResourceStatus,
        CudaShardRuntimeBufferPlan, CudaShardRuntimeBufferStatus, F16ScratchAllocationConfig,
        FusedF16AllocationStatus, MetadataAllocationConfig, RopeRuntimeBufferConfig,
        ShardedCudaResourcePlan, ShardedCudaResourceStatus, ShardedFusedF16AllocationStatus,
        ShardedKvCacheAllocationPlan, ShardedKvCacheAllocationStatus,
        ShardedMetadataAllocationStatus, ShardedRuntimeBufferStatus,
        FUSED_F16_CASTS_DEFERRED_REASON,
    };
    use crate::device_map::DeviceId;
    use crate::fused_f16::{
        cast_f32_tensor_to_f16, fused_gate_up_num_elements, fused_qkv_num_elements,
        get_or_load_cast_f32_to_f16_kernel,
    };
    use crate::model_loader::{ShardWeightStore, ShardWeightStorePlan};
    use crate::rope_validation::build_runtime_rope_tables;
    use crate::shard_plan::{LateAllocationKind, ShardTensorManifest, ShardedModelPlan};

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
        pub status: CudaLayerFusedF16AllocationStatus,
    }

    /// One shard's non-executing fused f16 buffers.
    pub struct CudaShardFusedF16Buffers {
        pub device_id: DeviceId,
        pub layers: Vec<CudaLayerFusedF16Buffers>,
        pub status: CudaShardFusedF16AllocationStatus,
    }

    /// Shard-local fused f16 buffers allocated from resource islands.
    pub struct ShardedFusedF16Buffers {
        pub shards: Vec<CudaShardFusedF16Buffers>,
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
            let needs_norm_casts = plan.f16_layernorm_count > 0 || plan.f16_postnorm_count > 0;
            let cast_kernel = if needs_norm_casts {
                Some(
                    get_or_load_cast_f32_to_f16_kernel(
                        &resource.loader,
                        &resource.context,
                        &resource.stream,
                    )
                    .map_err(|e| {
                        LLMError::GpuError(format!(
                            "shard {} f16 norm cast kernel load failed: {e}",
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
            let fused_total_bytes = fused_qkv_total_bytes
                + fused_gate_up_total_bytes
                + f16_layernorm_total_bytes
                + f16_postnorm_total_bytes;
            let fused_f16_allocated = fused_qkv_weight_count > 0
                || fused_gate_up_weight_count > 0
                || f16_layernorm_count > 0
                || f16_postnorm_count > 0;
            let fused_f16_status = if fused_f16_allocated {
                FusedF16AllocationStatus::Allocated
            } else {
                plan.fused_status
            };
            let conversion_work_deferred = plan.f16_qkv_bias_count > 0
                || plan.f16_o_proj_bias_count > 0
                || plan.embedding_f16_planned
                || plan.final_norm_f16_planned;

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
                f16_qkv_bias_count: plan.f16_qkv_bias_count,
                f16_o_proj_bias_count: plan.f16_o_proj_bias_count,
                embedding_f16_allocated: false,
                final_norm_f16_allocated: false,
                fused_qkv_total_bytes,
                fused_gate_up_total_bytes,
                f16_layernorm_total_bytes,
                f16_postnorm_total_bytes,
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
                f16_scratch_allocated: false,
                f16_scratch_status: plan.f16_scratch_status,
                f16_scratch_bytes: 0,
                f16_scratch_max_tokens: plan.f16_scratch_max_tokens,
                f16_scratch_deferred_reason: plan.f16_scratch_deferred_reason.clone(),
                f16_scratch_error: None,
            };

            Ok(Self {
                device_id: resource.device_id,
                layers,
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

            let status = plan.status(
                fused_qkv.is_some(),
                fused_gate_up.is_some(),
                input_layernorm.is_some(),
                post_attention_layernorm.is_some(),
                fused_qkv_bytes,
                fused_gate_up_bytes,
                layernorm_f16_bytes,
                postnorm_f16_bytes,
                None,
            );

            Ok(Self {
                absolute_layer_idx: plan.absolute_layer_idx,
                local_layer_idx: plan.local_layer_idx,
                fused_qkv,
                fused_gate_up,
                input_layernorm,
                post_attention_layernorm,
                status,
            })
        }
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
    CudaLayerFusedF16Buffers, CudaLayerKvCacheBuffers, CudaShardFusedF16Buffers,
    CudaShardKvCacheBuffers, CudaShardMetadataBuffers, CudaShardResources, CudaShardRuntimeBuffers,
    ShardedCudaResources, ShardedFusedF16Buffers, ShardedKvCacheBuffers, ShardedMetadataBuffers,
    ShardedRuntimeBuffers,
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
        let scratch_config = F16ScratchAllocationConfig::new(8).unwrap();
        let plan =
            ShardedFusedF16AllocationPlan::from_upload_manifest(&manifest, Some(scratch_config));
        let status = ShardedFusedF16AllocationStatus::from_plan(&plan, false, false);

        for shard in &status.shards {
            assert!(!shard.f16_scratch_allocated);
            assert_eq!(shard.f16_scratch_status, FusedF16AllocationStatus::Deferred);
            assert_eq!(shard.f16_scratch_max_tokens, Some(8));
            assert!(shard
                .f16_scratch_deferred_reason
                .as_deref()
                .unwrap()
                .contains("F16LayerScratch"));
        }
    }

    #[test]
    fn f16_scratch_config_rejects_zero_max_tokens() {
        let err = F16ScratchAllocationConfig::new(0).unwrap_err();

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
