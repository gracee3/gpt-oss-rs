//! CUDA-free shard-local weight access planning.
//!
//! This module does not own CUDA buffers and does not call loaders. It records
//! the ownership and shape-visibility boundary a future shard-local
//! `GpuModelWeights` wrapper will need before fused allocation or layer
//! construction can operate without a full `GpuModelRunner`.

use std::collections::BTreeSet;

use crate::device_map::DeviceId;
use crate::shard_plan::{GpuShardPlan, LateAllocationKind, ShardTensorManifest};

/// Pure plan for one shard's weight access boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardWeightStorePlan {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub owns_embeddings: bool,
    pub owns_final_head: bool,
    pub required_tensor_names: BTreeSet<String>,
    pub host_u8_tensor_names: BTreeSet<String>,
    pub global_shape_names: BTreeSet<String>,
    pub tied_lm_head_fallback_required: bool,
}

/// Public status for one shard's weight access boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardWeightStoreStatus {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub owns_embeddings: bool,
    pub owns_final_head: bool,
    pub required_tensor_count: usize,
    pub host_u8_tensor_count: usize,
    pub global_shape_count: usize,
    pub tied_lm_head_fallback_required: bool,
    pub missing_required_tensor_names: Vec<String>,
}

/// Classification of a tensor name from one shard's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardTensorAvailability {
    Required,
    HostU8,
    ShapeOnly,
    NotVisible,
}

/// Ownership-aware lookup errors for shard-local weight access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardWeightLookupError {
    LayerNotOwned { absolute_layer_idx: usize },
    TensorNotOwned { name: String },
    RequiredTensorMissing { name: String },
    ShapeMissing { name: String },
    TiedLmHeadFallbackDeferred,
}

/// CUDA-free shard-local weight access wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardWeightStore {
    plan: ShardWeightStorePlan,
}

impl ShardWeightStorePlan {
    /// Build from existing pure shard and upload-manifest plans.
    pub fn from_shard_manifest<I, S>(
        shard_plan: &GpuShardPlan,
        shard_manifest: &ShardTensorManifest,
        global_shapes: I,
        has_lm_head_weight: bool,
        tie_word_embeddings: bool,
    ) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let tied_lm_head_fallback_required =
            shard_plan.owns_final_head && tie_word_embeddings && !has_lm_head_weight
                || shard_manifest
                    .deferred_or_late_gpu_allocations
                    .iter()
                    .any(|allocation| allocation == &LateAllocationKind::TiedLmHeadFallback);

        Self {
            device_id: shard_plan.device_id,
            absolute_layers: shard_plan.absolute_layers.clone(),
            owns_embeddings: shard_plan.owns_embeddings,
            owns_final_head: shard_plan.owns_final_head,
            required_tensor_names: shard_manifest.required_tensor_filter_set(),
            host_u8_tensor_names: shard_manifest.host_u8_tensor_filter_set(),
            global_shape_names: global_shapes.into_iter().map(Into::into).collect(),
            tied_lm_head_fallback_required,
        }
    }

    pub fn into_store(self) -> ShardWeightStore {
        ShardWeightStore::new(self)
    }

    pub fn status(&self) -> ShardWeightStoreStatus {
        let missing_required_tensor_names = self
            .required_tensor_names
            .iter()
            .filter(|name| !self.global_shape_names.contains(*name))
            .cloned()
            .collect();

        ShardWeightStoreStatus {
            device_id: self.device_id,
            absolute_layers: self.absolute_layers.clone(),
            owns_embeddings: self.owns_embeddings,
            owns_final_head: self.owns_final_head,
            required_tensor_count: self.required_tensor_names.len(),
            host_u8_tensor_count: self.host_u8_tensor_names.len(),
            global_shape_count: self.global_shape_names.len(),
            tied_lm_head_fallback_required: self.tied_lm_head_fallback_required,
            missing_required_tensor_names,
        }
    }
}

impl ShardWeightStore {
    pub fn new(plan: ShardWeightStorePlan) -> Self {
        Self { plan }
    }

    pub fn plan(&self) -> &ShardWeightStorePlan {
        &self.plan
    }

    pub fn status(&self) -> ShardWeightStoreStatus {
        self.plan.status()
    }

    pub fn owns_layer(&self, absolute_layer_idx: usize) -> bool {
        self.plan
            .absolute_layers
            .iter()
            .any(|&owned_layer| owned_layer == absolute_layer_idx)
    }

    pub fn local_layer_idx(&self, absolute_layer_idx: usize) -> Option<usize> {
        self.plan
            .absolute_layers
            .iter()
            .position(|&owned_layer| owned_layer == absolute_layer_idx)
    }

    pub fn layer_tensor_name(
        &self,
        absolute_layer_idx: usize,
        suffix: &str,
    ) -> Result<String, ShardWeightLookupError> {
        self.ensure_layer_owned(absolute_layer_idx)?;
        Ok(format!("model.layers.{absolute_layer_idx}.{suffix}"))
    }

    pub fn require_owned_layer_tensor_name(
        &self,
        absolute_layer_idx: usize,
        suffix: &str,
    ) -> Result<String, ShardWeightLookupError> {
        let name = self.layer_tensor_name(absolute_layer_idx, suffix)?;
        if self.should_load_required_tensor(&name) {
            Ok(name)
        } else {
            Err(ShardWeightLookupError::RequiredTensorMissing { name })
        }
    }

    pub fn optional_owned_layer_tensor_name(
        &self,
        absolute_layer_idx: usize,
        suffix: &str,
    ) -> Result<Option<String>, ShardWeightLookupError> {
        let name = self.layer_tensor_name(absolute_layer_idx, suffix)?;
        Ok(self.should_load_required_tensor(&name).then_some(name))
    }

    pub fn u8_owned_layer_tensor_name(
        &self,
        absolute_layer_idx: usize,
        suffix: &str,
    ) -> Result<String, ShardWeightLookupError> {
        let name = self.layer_tensor_name(absolute_layer_idx, suffix)?;
        if self.should_load_host_u8_tensor(&name) {
            Ok(name)
        } else {
            Err(ShardWeightLookupError::RequiredTensorMissing { name })
        }
    }

    pub fn require_required_tensor_name(
        &self,
        name: &str,
    ) -> Result<String, ShardWeightLookupError> {
        if self.should_load_required_tensor(name) {
            Ok(name.to_string())
        } else if self.shape_visible(name) || self.should_load_host_u8_tensor(name) {
            Err(ShardWeightLookupError::TensorNotOwned {
                name: name.to_string(),
            })
        } else {
            Err(ShardWeightLookupError::RequiredTensorMissing {
                name: name.to_string(),
            })
        }
    }

    pub fn should_load_required_tensor(&self, name: &str) -> bool {
        self.plan.required_tensor_names.contains(name)
    }

    pub fn should_load_host_u8_tensor(&self, name: &str) -> bool {
        self.plan.host_u8_tensor_names.contains(name)
    }

    pub fn has_global_shape(&self, name: &str) -> bool {
        self.plan.global_shape_names.contains(name)
    }

    pub fn shape_visible(&self, name: &str) -> bool {
        self.has_global_shape(name)
    }

    pub fn require_shape_visible(&self, name: &str) -> Result<String, ShardWeightLookupError> {
        if self.shape_visible(name) {
            Ok(name.to_string())
        } else {
            Err(ShardWeightLookupError::ShapeMissing {
                name: name.to_string(),
            })
        }
    }

    pub fn tensor_availability(&self, name: &str) -> ShardTensorAvailability {
        if self.should_load_required_tensor(name) {
            ShardTensorAvailability::Required
        } else if self.should_load_host_u8_tensor(name) {
            ShardTensorAvailability::HostU8
        } else if self.shape_visible(name) {
            ShardTensorAvailability::ShapeOnly
        } else {
            ShardTensorAvailability::NotVisible
        }
    }

    pub fn embedding_tensor_name_if_owned(&self) -> Result<Option<String>, ShardWeightLookupError> {
        let name = "model.embed_tokens.weight";
        if !self.plan.owns_embeddings {
            return Ok(None);
        }
        if self.should_load_required_tensor(name) {
            Ok(Some(name.to_string()))
        } else {
            Err(ShardWeightLookupError::RequiredTensorMissing {
                name: name.to_string(),
            })
        }
    }

    pub fn final_norm_tensor_name_if_owned(
        &self,
    ) -> Result<Option<String>, ShardWeightLookupError> {
        let name = "model.norm.weight";
        if !self.plan.owns_final_head {
            return Ok(None);
        }
        if self.should_load_required_tensor(name) {
            Ok(Some(name.to_string()))
        } else {
            Err(ShardWeightLookupError::RequiredTensorMissing {
                name: name.to_string(),
            })
        }
    }

    pub fn lm_head_tensor_name_if_owned_or_deferred(
        &self,
    ) -> Result<Option<String>, ShardWeightLookupError> {
        let name = "lm_head.weight";
        if !self.plan.owns_final_head {
            return Ok(None);
        }
        if self.should_load_required_tensor(name) {
            Ok(Some(name.to_string()))
        } else if self.plan.tied_lm_head_fallback_required {
            Err(ShardWeightLookupError::TiedLmHeadFallbackDeferred)
        } else {
            Ok(None)
        }
    }

    fn ensure_layer_owned(&self, absolute_layer_idx: usize) -> Result<(), ShardWeightLookupError> {
        if self.owns_layer(absolute_layer_idx) {
            Ok(())
        } else {
            Err(ShardWeightLookupError::LayerNotOwned { absolute_layer_idx })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_map::{DeviceId, DeviceMap};
    use crate::shard_plan::{ShardedModelPlan, ShardedUploadManifest, UploadManifestOptions};

    fn split_plan() -> ShardedModelPlan {
        let map = DeviceMap::parse("split:0-11@0,12-23@1", 24, DeviceId(0)).unwrap();
        ShardedModelPlan::from_device_map(map, 24).unwrap()
    }

    fn tensor_names_with_lm_head() -> Vec<&'static str> {
        vec![
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.mlp.experts.gate_up_proj_blocks",
            "model.layers.0.mlp.experts.gate_up_proj_scales",
            "model.layers.11.self_attn.q_proj.weight",
            "model.layers.11.mlp.experts.down_proj_blocks",
            "model.layers.12.input_layernorm.weight",
            "model.layers.12.self_attn.q_proj.weight",
            "model.layers.12.self_attn.q_proj.bias",
            "model.layers.12.mlp.experts.gate_up_proj_blocks",
            "model.layers.12.mlp.experts.down_proj_blocks",
            "model.layers.23.self_attn.q_proj.weight",
            "model.layers.23.mlp.experts.down_proj_scales",
            "model.norm.weight",
            "lm_head.weight",
        ]
    }

    fn tensor_names_without_lm_head() -> Vec<&'static str> {
        tensor_names_with_lm_head()
            .into_iter()
            .filter(|name| *name != "lm_head.weight")
            .collect()
    }

    fn manifest(names: &[&'static str], tie_word_embeddings: bool) -> ShardedUploadManifest {
        split_plan()
            .upload_manifest_for_tensor_names(
                names.iter().copied(),
                UploadManifestOptions {
                    tie_word_embeddings,
                },
            )
            .unwrap()
    }

    fn store_for_device(
        device_id: usize,
        names: &[&'static str],
        global_shape_names: Vec<&'static str>,
        has_lm_head_weight: bool,
        tie_word_embeddings: bool,
    ) -> ShardWeightStore {
        let plan = split_plan();
        let manifest = manifest(names, tie_word_embeddings);
        let shard_plan = plan.shard_for_device(DeviceId(device_id)).unwrap();
        let shard_manifest = manifest.shard_for_device(DeviceId(device_id)).unwrap();
        ShardWeightStorePlan::from_shard_manifest(
            shard_plan,
            shard_manifest,
            global_shape_names.into_iter().map(String::from),
            has_lm_head_weight,
            tie_word_embeddings,
        )
        .into_store()
    }

    fn gpu0_store() -> ShardWeightStore {
        let names = tensor_names_with_lm_head();
        store_for_device(0, &names, names.clone(), true, true)
    }

    fn gpu1_store() -> ShardWeightStore {
        let names = tensor_names_with_lm_head();
        store_for_device(1, &names, names.clone(), true, true)
    }

    #[test]
    fn split_stores_preserve_absolute_layer_ownership() {
        let gpu0 = gpu0_store();
        let gpu1 = gpu1_store();

        assert!(gpu0.owns_layer(0));
        assert!(gpu0.owns_layer(11));
        assert!(!gpu0.owns_layer(12));
        assert_eq!(gpu0.local_layer_idx(11), Some(11));

        assert!(gpu1.owns_layer(12));
        assert!(gpu1.owns_layer(23));
        assert!(!gpu1.owns_layer(11));
        assert_eq!(gpu1.local_layer_idx(12), Some(0));
    }

    #[test]
    fn gpu1_layer_12_uses_absolute_tensor_name() {
        let gpu1 = gpu1_store();

        let name = gpu1
            .layer_tensor_name(12, "self_attn.q_proj.weight")
            .unwrap();

        assert_eq!(name, "model.layers.12.self_attn.q_proj.weight");
        assert!(!name.contains("model.layers.0."));
    }

    #[test]
    fn non_owned_layer_is_rejected_before_lookup() {
        let gpu0 = gpu0_store();
        let err = gpu0
            .require_owned_layer_tensor_name(12, "self_attn.q_proj.weight")
            .unwrap_err();

        assert_eq!(
            err,
            ShardWeightLookupError::LayerNotOwned {
                absolute_layer_idx: 12
            }
        );
    }

    #[test]
    fn required_lookup_accepts_owned_layer_tensors_and_rejects_unowned() {
        let gpu0 = gpu0_store();
        let gpu1 = gpu1_store();

        assert_eq!(
            gpu1.require_owned_layer_tensor_name(12, "self_attn.q_proj.weight")
                .unwrap(),
            "model.layers.12.self_attn.q_proj.weight"
        );
        assert!(gpu1.should_load_required_tensor("model.layers.12.self_attn.q_proj.weight"));
        assert!(!gpu1.should_load_required_tensor("model.layers.11.self_attn.q_proj.weight"));

        assert!(matches!(
            gpu0.require_required_tensor_name("model.layers.12.self_attn.q_proj.weight"),
            Err(ShardWeightLookupError::TensorNotOwned { .. })
        ));
    }

    #[test]
    fn u8_lookup_accepts_only_owned_layer_expert_tensors() {
        let gpu0 = gpu0_store();
        let gpu1 = gpu1_store();

        assert_eq!(
            gpu0.u8_owned_layer_tensor_name(0, "mlp.experts.gate_up_proj_blocks")
                .unwrap(),
            "model.layers.0.mlp.experts.gate_up_proj_blocks"
        );
        assert!(gpu0.should_load_host_u8_tensor("model.layers.0.mlp.experts.gate_up_proj_blocks"));
        assert!(!gpu0.should_load_host_u8_tensor("model.layers.12.mlp.experts.gate_up_proj_blocks"));

        assert_eq!(
            gpu1.u8_owned_layer_tensor_name(12, "mlp.experts.down_proj_blocks")
                .unwrap(),
            "model.layers.12.mlp.experts.down_proj_blocks"
        );
        assert!(matches!(
            gpu1.u8_owned_layer_tensor_name(0, "mlp.experts.gate_up_proj_blocks"),
            Err(ShardWeightLookupError::LayerNotOwned {
                absolute_layer_idx: 0
            })
        ));
    }

    #[test]
    fn embedding_final_norm_and_lm_head_follow_shard_ownership() {
        let gpu0 = gpu0_store();
        let gpu1 = gpu1_store();

        assert_eq!(
            gpu0.embedding_tensor_name_if_owned().unwrap(),
            Some("model.embed_tokens.weight".to_string())
        );
        assert_eq!(gpu1.embedding_tensor_name_if_owned().unwrap(), None);

        assert_eq!(gpu0.final_norm_tensor_name_if_owned().unwrap(), None);
        assert_eq!(
            gpu1.final_norm_tensor_name_if_owned().unwrap(),
            Some("model.norm.weight".to_string())
        );

        assert_eq!(
            gpu0.lm_head_tensor_name_if_owned_or_deferred().unwrap(),
            None
        );
        assert_eq!(
            gpu1.lm_head_tensor_name_if_owned_or_deferred().unwrap(),
            Some("lm_head.weight".to_string())
        );
    }

    #[test]
    fn tied_lm_head_fallback_is_deferred_without_embedding_loadability_on_final_shard() {
        let names = tensor_names_without_lm_head();
        let gpu1 = store_for_device(1, &names, tensor_names_with_lm_head(), false, true);

        assert_eq!(
            gpu1.lm_head_tensor_name_if_owned_or_deferred().unwrap_err(),
            ShardWeightLookupError::TiedLmHeadFallbackDeferred
        );
        assert_eq!(gpu1.embedding_tensor_name_if_owned().unwrap(), None);
        assert!(gpu1.shape_visible("model.embed_tokens.weight"));
        assert!(!gpu1.should_load_required_tensor("model.embed_tokens.weight"));
        assert_eq!(
            gpu1.tensor_availability("model.embed_tokens.weight"),
            ShardTensorAvailability::ShapeOnly
        );
    }

    #[test]
    fn global_shape_visibility_does_not_imply_loadability() {
        let gpu1 = gpu1_store();

        assert!(gpu1.shape_visible("model.layers.0.self_attn.q_proj.weight"));
        assert_eq!(
            gpu1.tensor_availability("model.layers.0.self_attn.q_proj.weight"),
            ShardTensorAvailability::ShapeOnly
        );
        assert!(matches!(
            gpu1.require_required_tensor_name("model.layers.0.self_attn.q_proj.weight"),
            Err(ShardWeightLookupError::TensorNotOwned { .. })
        ));
    }

    #[test]
    fn missing_owned_required_tensor_is_distinct_from_unowned_tensor() {
        let names = tensor_names_with_lm_head();
        let gpu0 = gpu0_store();

        assert_eq!(
            gpu0.require_owned_layer_tensor_name(0, "self_attn.k_proj.weight")
                .unwrap_err(),
            ShardWeightLookupError::RequiredTensorMissing {
                name: "model.layers.0.self_attn.k_proj.weight".to_string()
            }
        );
        assert_eq!(
            gpu0.require_owned_layer_tensor_name(12, "self_attn.q_proj.weight")
                .unwrap_err(),
            ShardWeightLookupError::LayerNotOwned {
                absolute_layer_idx: 12
            }
        );

        let store = store_for_device(
            0,
            &names,
            names
                .iter()
                .copied()
                .filter(|name| *name != "model.layers.0.self_attn.q_proj.weight")
                .collect(),
            true,
            true,
        );
        let status = store.status();
        assert_eq!(
            status.missing_required_tensor_names,
            vec!["model.layers.0.self_attn.q_proj.weight".to_string()]
        );
    }

    #[test]
    fn shape_lookup_reports_missing_shape_separately() {
        let gpu0 = gpu0_store();

        assert_eq!(
            gpu0.require_shape_visible("missing.tensor").unwrap_err(),
            ShardWeightLookupError::ShapeMissing {
                name: "missing.tensor".to_string()
            }
        );
    }
}
