#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
//! Transformer forward pass for gpt-oss-rs.
//!
//! Provides `ModelRunner` which orchestrates the forward pass through
//! embedding, transformer layers, and the final LM head. Layer primitives
//! (norms, linear, rotary, activations, MLP) are naive CPU mock
//! implementations; real GPU kernels dispatch to CUDA.
extern crate self as gpt_oss_model_runner;

pub mod architectures;
pub mod attention;
pub mod bridge;
pub mod device_map;
pub mod fused_f16;
pub mod input;
pub mod kv_cache;
pub mod layers;
pub mod model_loader;
pub mod quant;
pub mod runner;
pub mod sampling;
pub mod shard_plan;
pub mod sharded_resources;
#[cfg(feature = "cuda")]
pub mod tensor_parallel;

// GPU forward-pass modules (CUDA-only)
#[cfg(feature = "cuda")]
pub mod gpu_layer;
#[cfg(feature = "cuda")]
pub mod gpu_runner;
#[cfg(feature = "cuda")]
pub mod mxfp4_validation;
#[cfg(feature = "cuda")]
pub mod rope_validation;

/// Type alias for cublasLt handle. Compiles to a usable type with the
/// `cublaslt` feature, or a zero-size dummy without it. This lets
/// function signatures reference it unconditionally.
#[cfg(feature = "cublaslt")]
pub type CublasLtRef = gpt_oss_gpu::cublaslt_ops::CublasLtOps;
#[cfg(not(feature = "cublaslt"))]
pub type CublasLtRef = ();

pub use architectures::{create_model, Architecture};
pub use attention::{
    select_backend, select_backend_with_options, select_decode_backend, AttentionBackend,
    AttentionMetadata, FlashAttention2, FlashAttention2Config, FlashAttentionPaged, GpuBuffer,
    MockAttentionBackend, PagedAttentionV2, SlidingWindowAttention, SlidingWindowConfig,
    SplitKvAttention,
};
pub use device_map::{DeviceId, DeviceMap, DeviceMapError};
#[cfg(feature = "cuda")]
pub use fused_f16::cast_f32_tensor_to_f16;
pub use fused_f16::{
    f16_scratch_element_counts, fused_gate_up_dim, fused_gate_up_num_elements, fused_gate_up_shape,
    fused_qkv_dim, fused_qkv_num_elements, fused_qkv_shape, F16ScratchElementCounts,
};
pub use input::ModelInput;
pub use kv_cache::{reshape_and_cache, CacheConfig, CacheEngine, KVCache};
pub use model_loader::{
    detect_format, load_model_weights, ModelFormat, SafetensorHeaderManifest,
    SafetensorHeaderMergePolicy, SafetensorTensorInfo, ShardTensorAvailability,
    ShardWeightLookupError, ShardWeightStore, ShardWeightStorePlan, ShardWeightStoreStatus,
};
pub use quant::{detect_quant_method, QuantConfig, QuantMethod, QuantizedLinear, QuantizedWeight};
pub use runner::{ModelRunner, ModelRunnerConfig};
pub use sampling::{sample_batch, sample_batch_parallel, Sampler, SamplerOutput};
pub use shard_plan::{
    GpuShardPlan, LateAllocationKind, LayerKvCachePlan, ShardAllocationReport, ShardKvCachePlan,
    ShardPlanError, ShardTensorManifest, ShardedKvCachePlan, ShardedModelPlan,
    ShardedUploadManifest, SplitAllocationReport, TensorPlacement, TensorPlacementReason,
    UploadManifestOptions,
};
pub use sharded_resources::{
    CudaLayerFusedF16AllocationPlan, CudaLayerFusedF16AllocationStatus,
    CudaLayerKvCacheAllocationPlan, CudaLayerKvCacheAllocationStatus,
    CudaShardFusedF16AllocationPlan, CudaShardFusedF16AllocationStatus,
    CudaShardKvCacheAllocationPlan, CudaShardKvCacheAllocationStatus,
    CudaShardMetadataAllocationPlan, CudaShardMetadataAllocationStatus, CudaShardResourcePlan,
    CudaShardResourceStatus, CudaShardRuntimeBufferPlan, CudaShardRuntimeBufferStatus,
    F16ScratchAllocationConfig, FusedF16AllocationStatus, KvCacheAllocationConfig,
    MetadataAllocationConfig, MetadataMode, RopeRuntimeBufferConfig, RuntimeMetadataStatus,
    ShardedCudaResourcePlan, ShardedCudaResourceStatus, ShardedFusedF16AllocationPlan,
    ShardedFusedF16AllocationStatus, ShardedKvCacheAllocationPlan, ShardedKvCacheAllocationStatus,
    ShardedMetadataAllocationPlan, ShardedMetadataAllocationStatus, ShardedRuntimeBufferPlan,
    ShardedRuntimeBufferStatus,
};
#[cfg(feature = "cuda")]
pub use sharded_resources::{
    CudaLayerFusedF16Buffers, CudaLayerKvCacheBuffers, CudaShardFusedF16Buffers,
    CudaShardKvCacheBuffers, CudaShardMetadataBuffers, CudaShardResources, CudaShardRuntimeBuffers,
    ShardedCudaResources, ShardedFusedF16Buffers, ShardedKvCacheBuffers, ShardedMetadataBuffers,
    ShardedRuntimeBuffers,
};
