//! Non-executing CUDA resource skeletons for future layer sharding.
//!
//! The pure plan in this module is always available. CUDA resource construction
//! is feature-gated and deliberately stops at context/stream/cuBLAS/kernel-loader
//! ownership; it does not upload model tensors, allocate KV cache, or construct a
//! runner.

use crate::device_map::DeviceId;
use crate::shard_plan::ShardedModelPlan;

pub const RUNTIME_METADATA_DEFERRED_REASON: &str =
    "request-shaped metadata packing buffers require batch/sequence inputs";

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

/// Metadata allocation state for the non-executing runtime buffer skeleton.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMetadataStatus {
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

impl RuntimeMetadataStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RuntimeMetadataStatus::Allocated => "allocated",
            RuntimeMetadataStatus::Deferred => "deferred",
            RuntimeMetadataStatus::NotApplicable => "not_applicable",
        }
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

#[cfg(feature = "cuda")]
mod cuda {
    use std::path::Path;
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
    use gpt_oss_core::prelude::{LLMError, Result};
    use gpt_oss_gpu::cublas::CublasHandle;
    use gpt_oss_gpu::kernel_loader::KernelLoader;

    use super::{
        CudaShardResourcePlan, CudaShardResourceStatus, CudaShardRuntimeBufferPlan,
        CudaShardRuntimeBufferStatus, RopeRuntimeBufferConfig, ShardedCudaResourcePlan,
        ShardedCudaResourceStatus, ShardedRuntimeBufferStatus,
    };
    use crate::device_map::DeviceId;
    use crate::rope_validation::build_runtime_rope_tables;
    use crate::shard_plan::ShardedModelPlan;

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
}

#[cfg(feature = "cuda")]
pub use cuda::{
    CudaShardResources, CudaShardRuntimeBuffers, ShardedCudaResources, ShardedRuntimeBuffers,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_map::{DeviceId, DeviceMap};
    use crate::shard_plan::ShardedModelPlan;
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
