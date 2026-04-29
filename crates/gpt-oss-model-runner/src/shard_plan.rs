//! CUDA-free sharded model planning for future layer placement.
//!
//! This module turns an inert [`DeviceMap`] into explicit per-device ownership
//! metadata. It does not allocate CUDA resources, upload tensors, or make split
//! maps executable.

use std::collections::BTreeSet;
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

/// Pure upload manifest for future split allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedUploadManifest {
    pub shards: Vec<ShardTensorManifest>,
    pub unassigned_tensor_names: Vec<String>,
    pub invalid_tensor_names: Vec<String>,
}

/// Pure per-shard tensor ownership metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardTensorManifest {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub required_tensor_names: Vec<String>,
    pub optional_tensor_names: Vec<String>,
    pub host_u8_tensor_names: Vec<String>,
    pub deferred_or_late_gpu_allocations: Vec<LateAllocationKind>,
}

/// CUDA-free KV cache ownership plan for future split allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardedKvCachePlan {
    pub shards: Vec<ShardKvCachePlan>,
}

/// Per-shard KV cache metadata keyed by absolute layer id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardKvCachePlan {
    pub device_id: DeviceId,
    pub entries: Vec<LayerKvCachePlan>,
}

/// Mapping between an absolute model layer and a shard-local KV cache slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerKvCachePlan {
    pub absolute_layer_idx: usize,
    pub local_cache_idx: usize,
}

/// Pure status report for a future split allocation smoke.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitAllocationReport {
    pub devices: Vec<DeviceId>,
    pub embedding_device: DeviceId,
    pub final_device: DeviceId,
    pub shards: Vec<ShardAllocationReport>,
    pub unassigned_tensor_count: usize,
    pub invalid_tensor_count: usize,
    pub unassigned_tensor_names: Vec<String>,
    pub invalid_tensor_names: Vec<String>,
}

/// Pure per-shard status report combining layer, tensor, and KV plans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardAllocationReport {
    pub device_id: DeviceId,
    pub absolute_layers: Vec<usize>,
    pub owns_embeddings: bool,
    pub owns_final_head: bool,
    pub required_tensor_count: usize,
    pub optional_tensor_count: usize,
    pub host_u8_tensor_count: usize,
    pub late_allocation_count: usize,
    pub kv_cache_entry_count: usize,
    pub required_tensor_names: Vec<String>,
    pub host_u8_tensor_names: Vec<String>,
    pub late_allocations: Vec<LateAllocationKind>,
    pub kv_cache_layers: Vec<usize>,
}

/// Late shard-local allocations that are not initial safetensor uploads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LateAllocationKind {
    RopeTables,
    MetadataBuffers,
    KvCache { absolute_layers: Vec<usize> },
    F16FusedLayerWeights { layer_idx: usize },
    F16ConvertedEmbedding,
    F16ConvertedFinalNorm,
    GptOssMoeGpuUpload { layer_idx: usize },
    TiedLmHeadFallback,
}

/// Options for pure upload manifest construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UploadManifestOptions {
    pub tie_word_embeddings: bool,
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

    /// Build a CUDA-free tensor upload manifest from discovered safetensor names.
    pub fn upload_manifest_for_tensor_names<'a>(
        &self,
        tensor_names: impl IntoIterator<Item = &'a str>,
        options: UploadManifestOptions,
    ) -> Result<ShardedUploadManifest, ShardPlanError> {
        let discovered = tensor_names.into_iter().collect::<Vec<_>>();
        let has_lm_head_weight = discovered.iter().any(|name| *name == "lm_head.weight");

        let mut manifest = ShardedUploadManifest {
            shards: self
                .shards
                .iter()
                .map(ShardTensorManifest::from_shard_plan)
                .collect(),
            unassigned_tensor_names: Vec::new(),
            invalid_tensor_names: Vec::new(),
        };

        for tensor_name in discovered {
            match self.tensor_placement(tensor_name) {
                Ok(Some(placement)) => {
                    let shard = manifest
                        .shard_for_device_mut(placement.device_id)
                        .ok_or_else(|| {
                            ShardPlanError::InvalidPlan(format!(
                                "tensor '{tensor_name}' maps to device {} with no shard manifest",
                                placement.device_id
                            ))
                        })?;

                    if is_gpt_oss_u8_expert_tensor(tensor_name) {
                        push_unique(&mut shard.host_u8_tensor_names, tensor_name);
                    } else {
                        push_unique(&mut shard.required_tensor_names, tensor_name);
                    }
                }
                Ok(None) => {
                    push_unique(&mut manifest.unassigned_tensor_names, tensor_name);
                }
                Err(_) => {
                    push_unique(&mut manifest.invalid_tensor_names, tensor_name);
                }
            }
        }

        if options.tie_word_embeddings && !has_lm_head_weight {
            let final_device = self.device_map.final_device;
            let shard = manifest.shard_for_device_mut(final_device).ok_or_else(|| {
                ShardPlanError::InvalidPlan(format!(
                    "tied LM-head fallback maps to device {final_device} with no shard manifest"
                ))
            })?;
            push_late_unique(
                &mut shard.deferred_or_late_gpu_allocations,
                LateAllocationKind::TiedLmHeadFallback,
            );
        }

        Ok(manifest)
    }

    /// Build a CUDA-free KV cache plan with absolute-layer lookup metadata.
    pub fn kv_cache_plan(&self) -> ShardedKvCachePlan {
        ShardedKvCachePlan {
            shards: self
                .shards
                .iter()
                .map(ShardKvCachePlan::from_shard_plan)
                .collect(),
        }
    }

    /// Build a pure split-allocation status report from existing plans.
    pub fn split_allocation_report(
        &self,
        upload_manifest: &ShardedUploadManifest,
        kv_cache_plan: &ShardedKvCachePlan,
    ) -> Result<SplitAllocationReport, ShardPlanError> {
        let mut shards = Vec::with_capacity(self.shards.len());

        for model_shard in &self.shards {
            let upload_shard = upload_manifest
                .shard_for_device(model_shard.device_id)
                .ok_or_else(|| {
                    ShardPlanError::InvalidPlan(format!(
                        "missing upload manifest for device {}",
                        model_shard.device_id
                    ))
                })?;
            let kv_shard = kv_cache_plan
                .shard_for_device(model_shard.device_id)
                .ok_or_else(|| {
                    ShardPlanError::InvalidPlan(format!(
                        "missing KV cache plan for device {}",
                        model_shard.device_id
                    ))
                })?;

            shards.push(ShardAllocationReport {
                device_id: model_shard.device_id,
                absolute_layers: model_shard.absolute_layers.clone(),
                owns_embeddings: model_shard.owns_embeddings,
                owns_final_head: model_shard.owns_final_head,
                required_tensor_count: upload_shard.required_tensor_names.len(),
                optional_tensor_count: upload_shard.optional_tensor_names.len(),
                host_u8_tensor_count: upload_shard.host_u8_tensor_names.len(),
                late_allocation_count: upload_shard.deferred_or_late_gpu_allocations.len(),
                kv_cache_entry_count: kv_shard.entries.len(),
                required_tensor_names: upload_shard.required_tensor_names.clone(),
                host_u8_tensor_names: upload_shard.host_u8_tensor_names.clone(),
                late_allocations: upload_shard.deferred_or_late_gpu_allocations.clone(),
                kv_cache_layers: kv_shard
                    .entries
                    .iter()
                    .map(|entry| entry.absolute_layer_idx)
                    .collect(),
            });
        }

        Ok(SplitAllocationReport {
            devices: self.device_map.devices.clone(),
            embedding_device: self.device_map.embedding_device,
            final_device: self.device_map.final_device,
            shards,
            unassigned_tensor_count: upload_manifest.unassigned_tensor_names.len(),
            invalid_tensor_count: upload_manifest.invalid_tensor_names.len(),
            unassigned_tensor_names: upload_manifest.unassigned_tensor_names.clone(),
            invalid_tensor_names: upload_manifest.invalid_tensor_names.clone(),
        })
    }
}

impl SplitAllocationReport {
    /// Return the allocation report shard for a device.
    pub fn shard_for_device(&self, device_id: DeviceId) -> Option<&ShardAllocationReport> {
        self.shards
            .iter()
            .find(|shard| shard.device_id == device_id)
    }
}

impl ShardedUploadManifest {
    /// Return the manifest shard for a device.
    pub fn shard_for_device(&self, device_id: DeviceId) -> Option<&ShardTensorManifest> {
        self.shards
            .iter()
            .find(|shard| shard.device_id == device_id)
    }

    fn shard_for_device_mut(&mut self, device_id: DeviceId) -> Option<&mut ShardTensorManifest> {
        self.shards
            .iter_mut()
            .find(|shard| shard.device_id == device_id)
    }
}

impl ShardTensorManifest {
    /// Deterministic set for f32/f16 GPU loader filters.
    pub fn required_tensor_filter_set(&self) -> BTreeSet<String> {
        self.required_tensor_names.iter().cloned().collect()
    }

    /// Deterministic set for U8 host loader filters.
    pub fn host_u8_tensor_filter_set(&self) -> BTreeSet<String> {
        self.host_u8_tensor_names.iter().cloned().collect()
    }

    /// Return true when a tensor should be loaded by f32/f16 GPU loaders.
    pub fn should_load_required_tensor(&self, name: &str) -> bool {
        self.required_tensor_names
            .iter()
            .any(|tensor_name| tensor_name == name)
    }

    /// Return true when a tensor should be loaded by the U8 host loader.
    pub fn should_load_host_u8_tensor(&self, name: &str) -> bool {
        self.host_u8_tensor_names
            .iter()
            .any(|tensor_name| tensor_name == name)
    }

    fn from_shard_plan(shard: &GpuShardPlan) -> Self {
        let mut deferred_or_late_gpu_allocations = vec![
            LateAllocationKind::RopeTables,
            LateAllocationKind::MetadataBuffers,
            LateAllocationKind::KvCache {
                absolute_layers: shard.absolute_layers.clone(),
            },
        ];

        if shard.owns_embeddings {
            deferred_or_late_gpu_allocations.push(LateAllocationKind::F16ConvertedEmbedding);
        }
        if shard.owns_final_head {
            deferred_or_late_gpu_allocations.push(LateAllocationKind::F16ConvertedFinalNorm);
        }
        for &layer_idx in &shard.absolute_layers {
            deferred_or_late_gpu_allocations
                .push(LateAllocationKind::F16FusedLayerWeights { layer_idx });
            deferred_or_late_gpu_allocations
                .push(LateAllocationKind::GptOssMoeGpuUpload { layer_idx });
        }

        Self {
            device_id: shard.device_id,
            absolute_layers: shard.absolute_layers.clone(),
            required_tensor_names: Vec::new(),
            optional_tensor_names: Vec::new(),
            host_u8_tensor_names: Vec::new(),
            deferred_or_late_gpu_allocations,
        }
    }
}

impl ShardedKvCachePlan {
    /// Return the KV cache plan shard for a device.
    pub fn shard_for_device(&self, device_id: DeviceId) -> Option<&ShardKvCachePlan> {
        self.shards
            .iter()
            .find(|shard| shard.device_id == device_id)
    }

    /// Return the planned cache entry for an absolute layer across all shards.
    pub fn entry_for_absolute_layer(
        &self,
        layer_idx: usize,
    ) -> Option<(&ShardKvCachePlan, &LayerKvCachePlan)> {
        self.shards.iter().find_map(|shard| {
            shard
                .entry_for_absolute_layer(layer_idx)
                .map(|entry| (shard, entry))
        })
    }
}

impl ShardKvCachePlan {
    fn from_shard_plan(shard: &GpuShardPlan) -> Self {
        Self {
            device_id: shard.device_id,
            entries: shard
                .absolute_layers
                .iter()
                .enumerate()
                .map(|(local_cache_idx, &absolute_layer_idx)| LayerKvCachePlan {
                    absolute_layer_idx,
                    local_cache_idx,
                })
                .collect(),
        }
    }

    /// Return this shard's local cache entry for an absolute layer id.
    pub fn entry_for_absolute_layer(&self, layer_idx: usize) -> Option<&LayerKvCachePlan> {
        self.entries
            .iter()
            .find(|entry| entry.absolute_layer_idx == layer_idx)
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

fn is_gpt_oss_u8_expert_tensor(tensor_name: &str) -> bool {
    let Some((_prefix, suffix)) = tensor_name.split_once(".mlp.experts.") else {
        return false;
    };

    matches!(
        suffix,
        "gate_up_proj_blocks" | "gate_up_proj_scales" | "down_proj_blocks" | "down_proj_scales"
    )
}

fn push_unique(values: &mut Vec<String>, value: &str) {
    if !values.iter().any(|existing| existing == value) {
        values.push(value.to_string());
    }
}

fn push_late_unique(values: &mut Vec<LateAllocationKind>, value: LateAllocationKind) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
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

    fn sample_tensor_names() -> Vec<&'static str> {
        vec![
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.11.mlp.down_proj.weight",
            "model.layers.12.self_attn.k_proj.weight",
            "model.layers.23.post_attention_layernorm.weight",
            "model.layers.0.mlp.experts.gate_up_proj_blocks",
            "model.layers.11.mlp.experts.gate_up_proj_scales",
            "model.layers.12.mlp.experts.down_proj_blocks",
            "model.layers.23.mlp.experts.down_proj_scales",
            "model.norm.weight",
            "lm_head.weight",
        ]
    }

    fn split_manifest() -> ShardedUploadManifest {
        split_plan()
            .upload_manifest_for_tensor_names(
                sample_tensor_names(),
                UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap()
    }

    fn manifest_shard(manifest: &ShardedUploadManifest, device_id: usize) -> &ShardTensorManifest {
        manifest
            .shard_for_device(DeviceId(device_id))
            .unwrap_or_else(|| panic!("missing shard for device {device_id}"))
    }

    fn kv_shard(plan: &ShardedKvCachePlan, device_id: usize) -> &ShardKvCachePlan {
        plan.shard_for_device(DeviceId(device_id))
            .unwrap_or_else(|| panic!("missing KV shard for device {device_id}"))
    }

    fn allocation_report(
        plan: &ShardedModelPlan,
        names: Vec<&'static str>,
        tie_word_embeddings: bool,
    ) -> SplitAllocationReport {
        let manifest = plan
            .upload_manifest_for_tensor_names(
                names,
                UploadManifestOptions {
                    tie_word_embeddings,
                },
            )
            .unwrap();
        let kv_plan = plan.kv_cache_plan();
        plan.split_allocation_report(&manifest, &kv_plan).unwrap()
    }

    fn split_report() -> SplitAllocationReport {
        allocation_report(&split_plan(), sample_tensor_names(), true)
    }

    fn report_shard(report: &SplitAllocationReport, device_id: usize) -> &ShardAllocationReport {
        report
            .shard_for_device(DeviceId(device_id))
            .unwrap_or_else(|| panic!("missing report shard for device {device_id}"))
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

    #[test]
    fn upload_manifest_single_puts_known_tensors_on_selected_device() {
        let manifest = single_plan()
            .upload_manifest_for_tensor_names(
                sample_tensor_names(),
                UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap();

        assert_eq!(manifest.shards.len(), 1);
        assert!(manifest.unassigned_tensor_names.is_empty());
        assert!(manifest.invalid_tensor_names.is_empty());

        let shard = manifest_shard(&manifest, 3);
        assert!(shard
            .required_tensor_names
            .contains(&"model.embed_tokens.weight".to_string()));
        assert!(shard
            .required_tensor_names
            .contains(&"model.layers.23.post_attention_layernorm.weight".to_string()));
        assert!(shard
            .required_tensor_names
            .contains(&"model.norm.weight".to_string()));
        assert!(shard
            .required_tensor_names
            .contains(&"lm_head.weight".to_string()));
        assert!(shard
            .host_u8_tensor_names
            .contains(&"model.layers.0.mlp.experts.gate_up_proj_blocks".to_string()));
        assert!(shard
            .host_u8_tensor_names
            .contains(&"model.layers.23.mlp.experts.down_proj_scales".to_string()));
    }

    #[test]
    fn upload_manifest_split_sends_embedding_to_gpu0() {
        let manifest = split_manifest();
        let shard = manifest_shard(&manifest, 0);

        assert!(shard
            .required_tensor_names
            .contains(&"model.embed_tokens.weight".to_string()));
    }

    #[test]
    fn upload_manifest_split_sends_layer_0_and_11_to_gpu0() {
        let manifest = split_manifest();
        let shard = manifest_shard(&manifest, 0);

        assert!(shard
            .required_tensor_names
            .contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(shard
            .required_tensor_names
            .contains(&"model.layers.11.mlp.down_proj.weight".to_string()));
    }

    #[test]
    fn upload_manifest_split_sends_layer_12_and_23_to_gpu1() {
        let manifest = split_manifest();
        let shard = manifest_shard(&manifest, 1);

        assert!(shard
            .required_tensor_names
            .contains(&"model.layers.12.self_attn.k_proj.weight".to_string()));
        assert!(shard
            .required_tensor_names
            .contains(&"model.layers.23.post_attention_layernorm.weight".to_string()));
    }

    #[test]
    fn upload_manifest_split_sends_final_tensors_to_gpu1() {
        let manifest = split_manifest();
        let shard = manifest_shard(&manifest, 1);

        assert!(shard
            .required_tensor_names
            .contains(&"model.norm.weight".to_string()));
        assert!(shard
            .required_tensor_names
            .contains(&"lm_head.weight".to_string()));
    }

    #[test]
    fn upload_manifest_maps_gpt_oss_u8_expert_tensors_to_host_u8_lists() {
        let manifest = split_manifest();
        let gpu0 = manifest_shard(&manifest, 0);
        let gpu1 = manifest_shard(&manifest, 1);

        assert!(gpu0
            .host_u8_tensor_names
            .contains(&"model.layers.0.mlp.experts.gate_up_proj_blocks".to_string()));
        assert!(gpu0
            .host_u8_tensor_names
            .contains(&"model.layers.11.mlp.experts.gate_up_proj_scales".to_string()));
        assert!(gpu1
            .host_u8_tensor_names
            .contains(&"model.layers.12.mlp.experts.down_proj_blocks".to_string()));
        assert!(gpu1
            .host_u8_tensor_names
            .contains(&"model.layers.23.mlp.experts.down_proj_scales".to_string()));
        assert!(!gpu0
            .required_tensor_names
            .contains(&"model.layers.0.mlp.experts.gate_up_proj_blocks".to_string()));
        assert!(!gpu0
            .required_tensor_names
            .contains(&"model.layers.11.mlp.experts.gate_up_proj_scales".to_string()));
        assert!(!gpu1
            .required_tensor_names
            .contains(&"model.layers.12.mlp.experts.down_proj_blocks".to_string()));
        assert!(!gpu1
            .required_tensor_names
            .contains(&"model.layers.23.mlp.experts.down_proj_scales".to_string()));
    }

    #[test]
    fn manifest_filter_sets_are_deterministic() {
        let manifest = split_manifest();
        let gpu0 = manifest_shard(&manifest, 0);
        let gpu1 = manifest_shard(&manifest, 1);

        assert_eq!(
            gpu0.required_tensor_filter_set()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![
                "model.embed_tokens.weight".to_string(),
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                "model.layers.11.mlp.down_proj.weight".to_string(),
            ]
        );
        assert_eq!(
            gpu1.host_u8_tensor_filter_set()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![
                "model.layers.12.mlp.experts.down_proj_blocks".to_string(),
                "model.layers.23.mlp.experts.down_proj_scales".to_string(),
            ]
        );
    }

    #[test]
    fn gpu0_manifest_required_filter_accepts_only_gpu0_required_tensors() {
        let manifest = split_manifest();
        let gpu0 = manifest_shard(&manifest, 0);

        assert!(gpu0.should_load_required_tensor("model.embed_tokens.weight"));
        assert!(gpu0.should_load_required_tensor("model.layers.0.self_attn.q_proj.weight"));
        assert!(gpu0.should_load_required_tensor("model.layers.11.mlp.down_proj.weight"));

        assert!(!gpu0.should_load_required_tensor("model.layers.12.self_attn.k_proj.weight"));
        assert!(!gpu0.should_load_required_tensor("model.norm.weight"));
        assert!(!gpu0.should_load_required_tensor("lm_head.weight"));
        assert!(!gpu0.should_load_required_tensor("model.layers.0.mlp.experts.gate_up_proj_blocks"));
        assert!(!gpu0.should_load_required_tensor("model.rotary_emb.inv_freq"));
        assert!(!gpu0.should_load_required_tensor("model.layers.24.self_attn.q_proj.weight"));
    }

    #[test]
    fn gpu0_manifest_host_u8_filter_accepts_only_gpu0_u8_tensors() {
        let manifest = split_manifest();
        let gpu0 = manifest_shard(&manifest, 0);

        assert!(gpu0.should_load_host_u8_tensor("model.layers.0.mlp.experts.gate_up_proj_blocks"));
        assert!(gpu0.should_load_host_u8_tensor("model.layers.11.mlp.experts.gate_up_proj_scales"));

        assert!(!gpu0.should_load_host_u8_tensor("model.layers.12.mlp.experts.down_proj_blocks"));
        assert!(!gpu0.should_load_host_u8_tensor("model.layers.23.mlp.experts.down_proj_scales"));
        assert!(!gpu0.should_load_host_u8_tensor("model.embed_tokens.weight"));
        assert!(!gpu0.should_load_host_u8_tensor("model.layers.0.self_attn.q_proj.weight"));
        assert!(!gpu0.should_load_host_u8_tensor("model.rotary_emb.inv_freq"));
        assert!(!gpu0.should_load_host_u8_tensor("model.layers.24.mlp.experts.down_proj_blocks"));
    }

    #[test]
    fn gpu1_manifest_required_filter_accepts_only_gpu1_required_tensors() {
        let manifest = split_manifest();
        let gpu1 = manifest_shard(&manifest, 1);

        assert!(gpu1.should_load_required_tensor("model.layers.12.self_attn.k_proj.weight"));
        assert!(gpu1.should_load_required_tensor("model.layers.23.post_attention_layernorm.weight"));
        assert!(gpu1.should_load_required_tensor("model.norm.weight"));
        assert!(gpu1.should_load_required_tensor("lm_head.weight"));

        assert!(!gpu1.should_load_required_tensor("model.embed_tokens.weight"));
        assert!(!gpu1.should_load_required_tensor("model.layers.11.mlp.down_proj.weight"));
        assert!(!gpu1.should_load_required_tensor("model.layers.12.mlp.experts.down_proj_blocks"));
        assert!(!gpu1.should_load_required_tensor("model.rotary_emb.inv_freq"));
        assert!(!gpu1.should_load_required_tensor("model.layers.24.self_attn.q_proj.weight"));
    }

    #[test]
    fn gpu1_manifest_host_u8_filter_accepts_only_gpu1_u8_tensors() {
        let manifest = split_manifest();
        let gpu1 = manifest_shard(&manifest, 1);

        assert!(gpu1.should_load_host_u8_tensor("model.layers.12.mlp.experts.down_proj_blocks"));
        assert!(gpu1.should_load_host_u8_tensor("model.layers.23.mlp.experts.down_proj_scales"));

        assert!(!gpu1.should_load_host_u8_tensor("model.layers.0.mlp.experts.gate_up_proj_blocks"));
        assert!(!gpu1.should_load_host_u8_tensor("model.layers.11.mlp.experts.gate_up_proj_scales"));
        assert!(!gpu1.should_load_host_u8_tensor("model.norm.weight"));
        assert!(!gpu1.should_load_host_u8_tensor("lm_head.weight"));
        assert!(!gpu1.should_load_host_u8_tensor("model.rotary_emb.inv_freq"));
        assert!(!gpu1.should_load_host_u8_tensor("model.layers.24.mlp.experts.down_proj_blocks"));
    }

    #[test]
    fn tied_lm_head_fallback_does_not_add_embedding_to_final_required_filter() {
        let manifest = split_plan()
            .upload_manifest_for_tensor_names(
                [
                    "model.embed_tokens.weight",
                    "model.layers.23.post_attention_layernorm.weight",
                    "model.norm.weight",
                ],
                UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap();

        let final_shard = manifest_shard(&manifest, 1);

        assert!(final_shard
            .deferred_or_late_gpu_allocations
            .contains(&LateAllocationKind::TiedLmHeadFallback));
        assert!(!final_shard.should_load_required_tensor("model.embed_tokens.weight"));
        assert!(!final_shard
            .required_tensor_filter_set()
            .contains("model.embed_tokens.weight"));
    }

    #[test]
    fn upload_manifest_leaves_unknown_tensor_unassigned() {
        let manifest = split_plan()
            .upload_manifest_for_tensor_names(
                ["model.embed_tokens.weight", "model.rotary_emb.inv_freq"],
                UploadManifestOptions::default(),
            )
            .unwrap();

        assert_eq!(
            manifest.unassigned_tensor_names,
            vec!["model.rotary_emb.inv_freq".to_string()]
        );
        assert!(manifest.shards.iter().all(|shard| !shard
            .required_tensor_names
            .contains(&"model.rotary_emb.inv_freq".to_string())));
    }

    #[test]
    fn upload_manifest_marks_out_of_range_layer_tensor_invalid() {
        let manifest = split_plan()
            .upload_manifest_for_tensor_names(
                ["model.layers.24.self_attn.q_proj.weight"],
                UploadManifestOptions::default(),
            )
            .unwrap();

        assert_eq!(
            manifest.invalid_tensor_names,
            vec!["model.layers.24.self_attn.q_proj.weight".to_string()]
        );
    }

    #[test]
    fn upload_manifest_marks_malformed_layer_tensor_invalid() {
        let manifest = split_plan()
            .upload_manifest_for_tensor_names(
                ["model.layers.next.self_attn.q_proj.weight"],
                UploadManifestOptions::default(),
            )
            .unwrap();

        assert_eq!(
            manifest.invalid_tensor_names,
            vec!["model.layers.next.self_attn.q_proj.weight".to_string()]
        );
    }

    #[test]
    fn upload_manifest_adds_tied_lm_head_fallback_when_lm_head_absent() {
        let manifest = split_plan()
            .upload_manifest_for_tensor_names(
                [
                    "model.embed_tokens.weight",
                    "model.layers.23.post_attention_layernorm.weight",
                    "model.norm.weight",
                ],
                UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap();

        let final_shard = manifest_shard(&manifest, 1);
        assert!(final_shard
            .deferred_or_late_gpu_allocations
            .contains(&LateAllocationKind::TiedLmHeadFallback));
    }

    #[test]
    fn upload_manifest_skips_tied_lm_head_fallback_when_lm_head_exists() {
        let manifest = split_manifest();

        let final_shard = manifest_shard(&manifest, 1);
        assert!(!final_shard
            .deferred_or_late_gpu_allocations
            .contains(&LateAllocationKind::TiedLmHeadFallback));
    }

    #[test]
    fn upload_manifest_records_late_allocations_per_shard() {
        let manifest = split_manifest();
        let gpu0 = manifest_shard(&manifest, 0);
        let gpu1 = manifest_shard(&manifest, 1);

        assert!(gpu0
            .deferred_or_late_gpu_allocations
            .contains(&LateAllocationKind::RopeTables));
        assert!(gpu0
            .deferred_or_late_gpu_allocations
            .contains(&LateAllocationKind::MetadataBuffers));
        assert!(gpu0
            .deferred_or_late_gpu_allocations
            .contains(&LateAllocationKind::KvCache {
                absolute_layers: (0..=11).collect()
            }));
        assert!(gpu0
            .deferred_or_late_gpu_allocations
            .contains(&LateAllocationKind::F16ConvertedEmbedding));
        assert!(gpu0
            .deferred_or_late_gpu_allocations
            .contains(&LateAllocationKind::F16FusedLayerWeights { layer_idx: 11 }));

        assert!(gpu1
            .deferred_or_late_gpu_allocations
            .contains(&LateAllocationKind::KvCache {
                absolute_layers: (12..=23).collect()
            }));
        assert!(gpu1
            .deferred_or_late_gpu_allocations
            .contains(&LateAllocationKind::F16ConvertedFinalNorm));
        assert!(gpu1
            .deferred_or_late_gpu_allocations
            .contains(&LateAllocationKind::GptOssMoeGpuUpload { layer_idx: 23 }));
    }

    #[test]
    fn kv_cache_plan_single_has_one_shard_with_all_absolute_layers() {
        let plan = single_plan().kv_cache_plan();

        assert_eq!(plan.shards.len(), 1);
        let shard = kv_shard(&plan, 3);
        assert_eq!(shard.entries.len(), 24);
        assert_eq!(
            shard
                .entries
                .iter()
                .map(|entry| entry.absolute_layer_idx)
                .collect::<Vec<_>>(),
            (0..24).collect::<Vec<_>>()
        );
    }

    #[test]
    fn kv_cache_plan_single_local_indices_match_absolute_layers() {
        let plan = single_plan().kv_cache_plan();
        let shard = kv_shard(&plan, 3);

        for layer_idx in 0..24 {
            let entry = shard.entry_for_absolute_layer(layer_idx).unwrap();
            assert_eq!(entry.local_cache_idx, layer_idx);
        }
    }

    #[test]
    fn kv_cache_plan_split_has_two_shards() {
        let plan = split_plan().kv_cache_plan();

        assert_eq!(plan.shards.len(), 2);
        assert!(plan.shard_for_device(DeviceId(0)).is_some());
        assert!(plan.shard_for_device(DeviceId(1)).is_some());
    }

    #[test]
    fn kv_cache_plan_split_gpu0_owns_layers_0_through_11_only() {
        let plan = split_plan().kv_cache_plan();
        let shard = kv_shard(&plan, 0);

        assert_eq!(
            shard
                .entries
                .iter()
                .map(|entry| entry.absolute_layer_idx)
                .collect::<Vec<_>>(),
            (0..=11).collect::<Vec<_>>()
        );
        assert!(shard.entry_for_absolute_layer(12).is_none());
    }

    #[test]
    fn kv_cache_plan_split_gpu1_owns_layers_12_through_23_only() {
        let plan = split_plan().kv_cache_plan();
        let shard = kv_shard(&plan, 1);

        assert_eq!(
            shard
                .entries
                .iter()
                .map(|entry| entry.absolute_layer_idx)
                .collect::<Vec<_>>(),
            (12..=23).collect::<Vec<_>>()
        );
        assert!(shard.entry_for_absolute_layer(11).is_none());
    }

    #[test]
    fn kv_cache_plan_split_gpu1_local_index_starts_at_zero() {
        let plan = split_plan().kv_cache_plan();
        let shard = kv_shard(&plan, 1);

        assert_eq!(
            shard.entry_for_absolute_layer(12),
            Some(&LayerKvCachePlan {
                absolute_layer_idx: 12,
                local_cache_idx: 0,
            })
        );
        assert_eq!(
            shard.entry_for_absolute_layer(23),
            Some(&LayerKvCachePlan {
                absolute_layer_idx: 23,
                local_cache_idx: 11,
            })
        );
    }

    #[test]
    fn kv_cache_plan_absolute_lookup_is_shard_specific() {
        let plan = split_plan().kv_cache_plan();
        let gpu0 = kv_shard(&plan, 0);
        let gpu1 = kv_shard(&plan, 1);

        assert_eq!(
            gpu0.entry_for_absolute_layer(11),
            Some(&LayerKvCachePlan {
                absolute_layer_idx: 11,
                local_cache_idx: 11,
            })
        );
        assert!(gpu1.entry_for_absolute_layer(11).is_none());

        assert_eq!(
            gpu1.entry_for_absolute_layer(12),
            Some(&LayerKvCachePlan {
                absolute_layer_idx: 12,
                local_cache_idx: 0,
            })
        );
        assert!(gpu0.entry_for_absolute_layer(12).is_none());
    }

    #[test]
    fn kv_cache_plan_global_absolute_lookup_returns_owner_shard() {
        let plan = split_plan().kv_cache_plan();

        let (shard, entry) = plan.entry_for_absolute_layer(12).unwrap();
        assert_eq!(shard.device_id, DeviceId(1));
        assert_eq!(entry.local_cache_idx, 0);
        assert_eq!(entry.absolute_layer_idx, 12);
        assert!(plan.entry_for_absolute_layer(24).is_none());
    }

    #[test]
    fn kv_cache_plan_has_no_duplicate_absolute_layers() {
        let plan = split_plan().kv_cache_plan();
        let mut layers = plan
            .shards
            .iter()
            .flat_map(|shard| shard.entries.iter().map(|entry| entry.absolute_layer_idx))
            .collect::<Vec<_>>();
        let original_len = layers.len();
        layers.sort_unstable();
        layers.dedup();

        assert_eq!(layers.len(), original_len);
    }

    #[test]
    fn kv_cache_plan_covers_every_model_layer_once() {
        let plan = split_plan().kv_cache_plan();
        let mut layers = plan
            .shards
            .iter()
            .flat_map(|shard| shard.entries.iter().map(|entry| entry.absolute_layer_idx))
            .collect::<Vec<_>>();
        layers.sort_unstable();

        assert_eq!(layers, (0..24).collect::<Vec<_>>());
    }

    #[test]
    fn split_allocation_report_single_has_one_shard_with_all_layers_and_kv_entries() {
        let report = allocation_report(&single_plan(), sample_tensor_names(), true);

        assert_eq!(report.devices, vec![DeviceId(3)]);
        assert_eq!(report.shards.len(), 1);
        let shard = report_shard(&report, 3);
        assert_eq!(shard.absolute_layers, (0..24).collect::<Vec<_>>());
        assert_eq!(shard.kv_cache_layers, (0..24).collect::<Vec<_>>());
        assert_eq!(shard.kv_cache_entry_count, 24);
        assert!(shard.owns_embeddings);
        assert!(shard.owns_final_head);
        assert!(shard
            .required_tensor_names
            .contains(&"model.embed_tokens.weight".to_string()));
        assert!(shard
            .required_tensor_names
            .contains(&"lm_head.weight".to_string()));
    }

    #[test]
    fn split_allocation_report_split_has_two_shards() {
        let report = split_report();

        assert_eq!(report.devices, vec![DeviceId(0), DeviceId(1)]);
        assert_eq!(report.embedding_device, DeviceId(0));
        assert_eq!(report.final_device, DeviceId(1));
        assert_eq!(report.shards.len(), 2);
    }

    #[test]
    fn split_allocation_report_gpu0_owns_embeddings_and_layers_0_through_11() {
        let report = split_report();
        let shard = report_shard(&report, 0);

        assert!(shard.owns_embeddings);
        assert!(!shard.owns_final_head);
        assert_eq!(shard.absolute_layers, (0..=11).collect::<Vec<_>>());
    }

    #[test]
    fn split_allocation_report_gpu1_owns_layers_12_through_23_and_final_head() {
        let report = split_report();
        let shard = report_shard(&report, 1);

        assert!(!shard.owns_embeddings);
        assert!(shard.owns_final_head);
        assert_eq!(shard.absolute_layers, (12..=23).collect::<Vec<_>>());
    }

    #[test]
    fn split_allocation_report_includes_kv_layers_for_each_shard() {
        let report = split_report();
        let gpu0 = report_shard(&report, 0);
        let gpu1 = report_shard(&report, 1);

        assert_eq!(gpu0.kv_cache_layers, (0..=11).collect::<Vec<_>>());
        assert_eq!(gpu0.kv_cache_entry_count, 12);
        assert_eq!(gpu1.kv_cache_layers, (12..=23).collect::<Vec<_>>());
        assert_eq!(gpu1.kv_cache_entry_count, 12);
    }

    #[test]
    fn split_allocation_report_places_embedding_tensor_on_gpu0() {
        let report = split_report();
        let shard = report_shard(&report, 0);

        assert!(shard
            .required_tensor_names
            .contains(&"model.embed_tokens.weight".to_string()));
    }

    #[test]
    fn split_allocation_report_places_final_tensors_on_gpu1() {
        let report = split_report();
        let shard = report_shard(&report, 1);

        assert!(shard
            .required_tensor_names
            .contains(&"model.norm.weight".to_string()));
        assert!(shard
            .required_tensor_names
            .contains(&"lm_head.weight".to_string()));
    }

    #[test]
    fn split_allocation_report_places_gpt_oss_u8_tensors_on_owning_shards() {
        let report = split_report();
        let gpu0 = report_shard(&report, 0);
        let gpu1 = report_shard(&report, 1);

        assert!(gpu0
            .host_u8_tensor_names
            .contains(&"model.layers.0.mlp.experts.gate_up_proj_blocks".to_string()));
        assert!(gpu0
            .host_u8_tensor_names
            .contains(&"model.layers.11.mlp.experts.gate_up_proj_scales".to_string()));
        assert_eq!(gpu0.host_u8_tensor_count, 2);
        assert!(gpu1
            .host_u8_tensor_names
            .contains(&"model.layers.12.mlp.experts.down_proj_blocks".to_string()));
        assert!(gpu1
            .host_u8_tensor_names
            .contains(&"model.layers.23.mlp.experts.down_proj_scales".to_string()));
        assert_eq!(gpu1.host_u8_tensor_count, 2);
    }

    #[test]
    fn split_allocation_report_surfaces_unassigned_tensors_globally() {
        let report = allocation_report(
            &split_plan(),
            vec!["model.embed_tokens.weight", "model.rotary_emb.inv_freq"],
            false,
        );

        assert_eq!(
            report.unassigned_tensor_names,
            vec!["model.rotary_emb.inv_freq".to_string()]
        );
        assert_eq!(report.unassigned_tensor_count, 1);
        assert!(report.shards.iter().all(|shard| !shard
            .required_tensor_names
            .contains(&"model.rotary_emb.inv_freq".to_string())));
    }

    #[test]
    fn split_allocation_report_surfaces_invalid_tensors_globally() {
        let report = allocation_report(
            &split_plan(),
            vec!["model.layers.24.self_attn.q_proj.weight"],
            false,
        );

        assert_eq!(
            report.invalid_tensor_names,
            vec!["model.layers.24.self_attn.q_proj.weight".to_string()]
        );
        assert_eq!(report.invalid_tensor_count, 1);
    }

    #[test]
    fn split_allocation_report_includes_tied_lm_head_fallback_when_needed() {
        let report = allocation_report(
            &split_plan(),
            vec![
                "model.embed_tokens.weight",
                "model.layers.23.post_attention_layernorm.weight",
                "model.norm.weight",
            ],
            true,
        );
        let final_shard = report_shard(&report, 1);

        assert!(final_shard
            .late_allocations
            .contains(&LateAllocationKind::TiedLmHeadFallback));
    }

    #[test]
    fn split_allocation_report_omits_tied_lm_head_fallback_when_lm_head_exists() {
        let report = split_report();
        let final_shard = report_shard(&report, 1);

        assert!(!final_shard
            .late_allocations
            .contains(&LateAllocationKind::TiedLmHeadFallback));
    }

    #[test]
    fn split_allocation_report_counts_required_host_u8_and_late_allocations() {
        let report = split_report();
        let gpu0 = report_shard(&report, 0);
        let gpu1 = report_shard(&report, 1);

        assert_eq!(gpu0.required_tensor_count, 3);
        assert_eq!(gpu0.host_u8_tensor_count, 2);
        assert_eq!(gpu0.optional_tensor_count, 0);
        assert_eq!(gpu0.late_allocation_count, gpu0.late_allocations.len());

        assert_eq!(gpu1.required_tensor_count, 4);
        assert_eq!(gpu1.host_u8_tensor_count, 2);
        assert_eq!(gpu1.optional_tensor_count, 0);
        assert_eq!(gpu1.late_allocation_count, gpu1.late_allocations.len());
    }
}
