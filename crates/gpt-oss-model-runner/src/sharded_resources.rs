//! Non-executing CUDA resource skeletons for future layer sharding.
//!
//! The pure plan in this module is always available. CUDA resource construction
//! is feature-gated and deliberately stops at context/stream/cuBLAS/kernel-loader
//! ownership; it does not upload model tensors, allocate KV cache, or construct a
//! runner.

use crate::device_map::DeviceId;
use crate::shard_plan::ShardedModelPlan;

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

#[cfg(feature = "cuda")]
mod cuda {
    use std::path::Path;
    use std::sync::Arc;

    use cudarc::driver::{CudaContext, CudaStream};
    use gpt_oss_core::prelude::{LLMError, Result};
    use gpt_oss_gpu::cublas::CublasHandle;
    use gpt_oss_gpu::kernel_loader::KernelLoader;

    use super::{
        CudaShardResourcePlan, CudaShardResourceStatus, ShardedCudaResourcePlan,
        ShardedCudaResourceStatus,
    };
    use crate::device_map::DeviceId;
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
}

#[cfg(feature = "cuda")]
pub use cuda::{CudaShardResources, ShardedCudaResources};

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
