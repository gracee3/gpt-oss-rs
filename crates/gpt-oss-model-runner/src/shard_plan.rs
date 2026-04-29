//! CUDA-free sharded model planning for future layer placement.
//!
//! This module turns an inert [`DeviceMap`] into explicit per-device ownership
//! metadata. It does not allocate CUDA resources, upload tensors, or make split
//! maps executable.

use std::fmt;

use crate::device_map::{DeviceId, DeviceMap};

/// Non-executing placement plan for a sharded model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedModelPlan {
    pub device_map: DeviceMap,
    pub shards: Vec<GpuShardPlan>,
}

/// Pure metadata describing one device's intended whole-layer ownership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuShardPlan {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub first_layer: Option<usize>,
    pub last_layer: Option<usize>,
    pub owns_embeddings: bool,
    pub owns_final_head: bool,
}

/// Intended owner for a model tensor name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorPlacement {
    pub device_id: DeviceId,
    pub reason: TensorPlacementReason,
}

/// Why a tensor belongs to a shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorPlacementReason {
    Embeddings,
    Layer { layer_idx: usize },
    FinalNorm,
    LmHead,
    TiedLmHeadFallback,
}

/// Shard-planning error with user-facing detail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardPlanError {
    InvalidPlan(String),
}

impl fmt::Display for ShardPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShardPlanError::InvalidPlan(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ShardPlanError {}

impl ShardedModelPlan {
    /// Build a non-executing shard plan from an already parsed device map.
    pub fn from_device_map(
        device_map: DeviceMap,
        num_layers: usize,
    ) -> Result<Self, ShardPlanError> {
        if num_layers == 0 {
            return Err(ShardPlanError::InvalidPlan(
                "num_layers must be greater than 0".into(),
            ));
        }
        if device_map.layer_device.len() != num_layers {
            return Err(ShardPlanError::InvalidPlan(format!(
                "device map layer count {} does not match num_layers={num_layers}",
                device_map.layer_device.len()
            )));
        }

        let mut shards = Vec::new();
        for (layer_idx, &device_id) in device_map.layer_device.iter().enumerate() {
            let shard_idx = find_or_insert_shard(&mut shards, device_id, &device_map);
            shards[shard_idx].absolute_layers.push(layer_idx);
        }

        ensure_shard_for_device(&mut shards, device_map.embedding_device, &device_map);
        ensure_shard_for_device(&mut shards, device_map.final_device, &device_map);

        for shard in &mut shards {
            shard.first_layer = shard.absolute_layers.first().copied();
            shard.last_layer = shard.absolute_layers.last().copied();
        }

        Ok(Self { device_map, shards })
    }

    /// Return the plan shard for a device.
    pub fn shard_for_device(&self, device_id: DeviceId) -> Option<&GpuShardPlan> {
        self.shards
            .iter()
            .find(|shard| shard.device_id == device_id)
    }

    /// Return the planned owner for an absolute layer.
    pub fn device_for_layer(&self, layer_idx: usize) -> Result<DeviceId, ShardPlanError> {
        self.device_map
            .layer_device
            .get(layer_idx)
            .copied()
            .ok_or_else(|| {
                ShardPlanError::InvalidPlan(format!(
                    "layer {layer_idx} is out of bounds for num_layers={}",
                    self.device_map.layer_device.len()
                ))
            })
    }

    /// Classify a tensor name without loading or uploading it.
    pub fn tensor_placement(
        &self,
        tensor_name: &str,
    ) -> Result<Option<TensorPlacement>, ShardPlanError> {
        match classify_tensor_name(tensor_name)? {
            Some(TensorPlacementReason::Embeddings) => Ok(Some(TensorPlacement {
                device_id: self.device_map.embedding_device,
                reason: TensorPlacementReason::Embeddings,
            })),
            Some(TensorPlacementReason::Layer { layer_idx }) => Ok(Some(TensorPlacement {
                device_id: self.device_for_layer(layer_idx)?,
                reason: TensorPlacementReason::Layer { layer_idx },
            })),
            Some(TensorPlacementReason::FinalNorm) => Ok(Some(TensorPlacement {
                device_id: self.device_map.final_device,
                reason: TensorPlacementReason::FinalNorm,
            })),
            Some(TensorPlacementReason::LmHead) => Ok(Some(TensorPlacement {
                device_id: self.device_map.final_device,
                reason: TensorPlacementReason::LmHead,
            })),
            Some(TensorPlacementReason::TiedLmHeadFallback) => Ok(Some(TensorPlacement {
                device_id: self.device_map.final_device,
                reason: TensorPlacementReason::TiedLmHeadFallback,
            })),
            None => Ok(None),
        }
    }

    /// Planned owner for a future tied LM-head fallback copy.
    pub fn tied_lm_head_fallback_placement(&self) -> TensorPlacement {
        TensorPlacement {
            device_id: self.device_map.final_device,
            reason: TensorPlacementReason::TiedLmHeadFallback,
        }
    }
}

fn find_or_insert_shard(
    shards: &mut Vec<GpuShardPlan>,
    device_id: DeviceId,
    device_map: &DeviceMap,
) -> usize {
    if let Some(idx) = shards.iter().position(|shard| shard.device_id == device_id) {
        return idx;
    }

    shards.push(new_shard(device_id, device_map));
    shards.len() - 1
}

fn ensure_shard_for_device(
    shards: &mut Vec<GpuShardPlan>,
    device_id: DeviceId,
    device_map: &DeviceMap,
) {
    if shards.iter().any(|shard| shard.device_id == device_id) {
        return;
    }

    shards.push(new_shard(device_id, device_map));
}

fn new_shard(device_id: DeviceId, device_map: &DeviceMap) -> GpuShardPlan {
    GpuShardPlan {
        device_id,
        absolute_layers: Vec::new(),
        first_layer: None,
        last_layer: None,
        owns_embeddings: device_map.embedding_device == device_id,
        owns_final_head: device_map.final_device == device_id,
    }
}

fn classify_tensor_name(
    tensor_name: &str,
) -> Result<Option<TensorPlacementReason>, ShardPlanError> {
    if tensor_name == "model.embed_tokens.weight" {
        return Ok(Some(TensorPlacementReason::Embeddings));
    }
    if tensor_name == "model.norm.weight" {
        return Ok(Some(TensorPlacementReason::FinalNorm));
    }
    if tensor_name == "lm_head.weight" {
        return Ok(Some(TensorPlacementReason::LmHead));
    }

    let Some(rest) = tensor_name.strip_prefix("model.layers.") else {
        return Ok(None);
    };
    let Some((layer_spec, suffix)) = rest.split_once('.') else {
        return Ok(None);
    };
    if suffix.is_empty() {
        return Ok(None);
    }

    let layer_idx = layer_spec.parse::<usize>().map_err(|_| {
        ShardPlanError::InvalidPlan(format!(
            "malformed layer tensor name '{tensor_name}': layer id must be numeric"
        ))
    })?;
    Ok(Some(TensorPlacementReason::Layer { layer_idx }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_plan() -> ShardedModelPlan {
        let map = DeviceMap::single(24, DeviceId(3)).unwrap();
        ShardedModelPlan::from_device_map(map, 24).unwrap()
    }

    fn split_plan() -> ShardedModelPlan {
        let map = DeviceMap::parse("split:0-11@0,12-23@1", 24, DeviceId(0)).unwrap();
        ShardedModelPlan::from_device_map(map, 24).unwrap()
    }

    fn assert_tensor_device(plan: &ShardedModelPlan, name: &str, device_id: usize) {
        let placement = plan
            .tensor_placement(name)
            .unwrap()
            .unwrap_or_else(|| panic!("{name} should be assigned"));
        assert_eq!(placement.device_id, DeviceId(device_id));
    }

    #[test]
    fn shard_plan_single_builds_one_shard_with_all_layers() {
        let plan = single_plan();

        assert_eq!(plan.shards.len(), 1);
        let shard = &plan.shards[0];
        assert_eq!(shard.device_id, DeviceId(3));
        assert_eq!(shard.absolute_layers, (0..24).collect::<Vec<_>>());
        assert_eq!(shard.first_layer, Some(0));
        assert_eq!(shard.last_layer, Some(23));
        assert!(shard.owns_embeddings);
        assert!(shard.owns_final_head);
    }

    #[test]
    fn shard_plan_split_builds_expected_shards() {
        let plan = split_plan();

        assert_eq!(plan.shards.len(), 2);
        assert_eq!(plan.shards[0].device_id, DeviceId(0));
        assert_eq!(plan.shards[0].absolute_layers, (0..=11).collect::<Vec<_>>());
        assert_eq!(plan.shards[0].first_layer, Some(0));
        assert_eq!(plan.shards[0].last_layer, Some(11));
        assert!(plan.shards[0].owns_embeddings);
        assert!(!plan.shards[0].owns_final_head);

        assert_eq!(plan.shards[1].device_id, DeviceId(1));
        assert_eq!(
            plan.shards[1].absolute_layers,
            (12..=23).collect::<Vec<_>>()
        );
        assert_eq!(plan.shards[1].first_layer, Some(12));
        assert_eq!(plan.shards[1].last_layer, Some(23));
        assert!(!plan.shards[1].owns_embeddings);
        assert!(plan.shards[1].owns_final_head);
    }

    #[test]
    fn shard_plan_split_preserves_absolute_layer_ids() {
        let plan = split_plan();

        assert_eq!(plan.device_for_layer(0).unwrap(), DeviceId(0));
        assert_eq!(plan.device_for_layer(11).unwrap(), DeviceId(0));
        assert_eq!(plan.device_for_layer(12).unwrap(), DeviceId(1));
        assert_eq!(plan.device_for_layer(23).unwrap(), DeviceId(1));
    }

    #[test]
    fn shard_plan_maps_embedding_tensor_to_first_shard() {
        assert_tensor_device(&split_plan(), "model.embed_tokens.weight", 0);
    }

    #[test]
    fn shard_plan_maps_layer_0_tensor_to_first_shard() {
        assert_tensor_device(&split_plan(), "model.layers.0.self_attn.q_proj.weight", 0);
    }

    #[test]
    fn shard_plan_maps_layer_11_tensor_to_first_shard() {
        assert_tensor_device(&split_plan(), "model.layers.11.mlp.down_proj.weight", 0);
    }

    #[test]
    fn shard_plan_maps_layer_12_tensor_to_second_shard() {
        assert_tensor_device(&split_plan(), "model.layers.12.self_attn.k_proj.weight", 1);
    }

    #[test]
    fn shard_plan_maps_layer_23_tensor_to_second_shard() {
        assert_tensor_device(
            &split_plan(),
            "model.layers.23.post_attention_layernorm.weight",
            1,
        );
    }

    #[test]
    fn shard_plan_maps_final_norm_to_final_shard() {
        let placement = split_plan()
            .tensor_placement("model.norm.weight")
            .unwrap()
            .unwrap();

        assert_eq!(placement.device_id, DeviceId(1));
        assert_eq!(placement.reason, TensorPlacementReason::FinalNorm);
    }

    #[test]
    fn shard_plan_maps_lm_head_to_final_shard() {
        let placement = split_plan()
            .tensor_placement("lm_head.weight")
            .unwrap()
            .unwrap();

        assert_eq!(placement.device_id, DeviceId(1));
        assert_eq!(placement.reason, TensorPlacementReason::LmHead);
    }

    #[test]
    fn shard_plan_reports_tied_lm_head_fallback_on_final_shard() {
        let placement = split_plan().tied_lm_head_fallback_placement();

        assert_eq!(placement.device_id, DeviceId(1));
        assert_eq!(placement.reason, TensorPlacementReason::TiedLmHeadFallback);
    }

    #[test]
    fn shard_plan_leaves_unknown_tensor_unassigned() {
        let placement = split_plan()
            .tensor_placement("model.rotary_emb.inv_freq")
            .unwrap();

        assert_eq!(placement, None);
    }

    #[test]
    fn shard_plan_rejects_out_of_range_layer_tensor() {
        let err = split_plan()
            .tensor_placement("model.layers.24.self_attn.q_proj.weight")
            .unwrap_err();

        assert!(
            err.to_string().contains("layer 24 is out of bounds"),
            "got: {err}"
        );
    }

    #[test]
    fn shard_plan_rejects_malformed_layer_tensor_id() {
        let err = split_plan()
            .tensor_placement("model.layers.next.self_attn.q_proj.weight")
            .unwrap_err();

        assert!(
            err.to_string().contains("layer id must be numeric"),
            "got: {err}"
        );
    }
}
