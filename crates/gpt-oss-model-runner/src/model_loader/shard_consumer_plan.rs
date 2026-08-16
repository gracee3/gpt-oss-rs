//! Deterministic metadata-only per-shard consumption planning.
//!
//! This module binds the bounded shard catalog, native-to-runtime mapping, and
//! static expert placement before any tensor payload is mapped. It does not
//! execute construction or authorize a shard mapping lifetime.

use gpt_oss_core::error::{LLMError, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::heterogeneous::placement::{
    ExpertOwner, GptOssExpertKey, GptOssExpertPlacementManifestV1,
};

use super::gpt_oss_native::{GptOssNativeCatalogMap, GptOssTensorViewSpec};
use super::shard_catalog::{SafeTensorFileIdentity, SafeTensorShardCatalog};

pub const GPT_OSS_SHARD_CONSUMER_PLAN_SCHEMA_V1: &str = "gpt-oss-rs.shard-consumer-plan/v1";
pub const MAX_GPT_OSS_SHARD_CONSUMER_ACTIONS: usize = 100_000;

#[derive(Clone)]
struct PlanTensorMetadata {
    shard_index: usize,
    byte_len: u64,
    absolute_range: [u64; 2],
}

trait ShardPlanCatalogSource {
    fn catalog_sha256(&self) -> &str;
    fn shards_for_plan(&self) -> Result<Vec<SafeTensorFileIdentity>>;
    fn tensor_for_plan(&self, name: &str) -> Result<PlanTensorMetadata>;
    fn total_payload_bytes_for_plan(&self) -> u64;
}

impl ShardPlanCatalogSource for SafeTensorShardCatalog {
    fn catalog_sha256(&self) -> &str {
        self.metadata_sha256()
    }

    fn shards_for_plan(&self) -> Result<Vec<SafeTensorFileIdentity>> {
        Ok(self
            .shards()
            .iter()
            .map(|shard| shard.identity.clone())
            .collect())
    }

    fn tensor_for_plan(&self, name: &str) -> Result<PlanTensorMetadata> {
        let tensor = self.tensor(name)?;
        Ok(PlanTensorMetadata {
            shard_index: tensor.shard_index,
            byte_len: tensor.byte_len(),
            absolute_range: tensor.absolute_range,
        })
    }

    fn total_payload_bytes_for_plan(&self) -> u64 {
        self.total_payload_bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GptOssExpertSurface {
    GateUpBias,
    GateUpBlocks,
    GateUpScales,
    DownBias,
    DownBlocks,
    DownScales,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GptOssShardConsumer {
    LayerOwnerDense {
        runtime_tensor: String,
    },
    OwnedExpert {
        key: GptOssExpertKey,
        owner: ExpertOwner,
        surface: GptOssExpertSurface,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GptOssShardConsumerAction {
    pub native_tensor: String,
    pub native_tensor_range: [u64; 2],
    pub shard_absolute_range: [u64; 2],
    pub consumer: GptOssShardConsumer,
}

impl GptOssShardConsumerAction {
    pub fn byte_len(&self) -> Result<u64> {
        self.shard_absolute_range[1]
            .checked_sub(self.shard_absolute_range[0])
            .ok_or_else(|| model_error("consumer action has a reversed absolute range"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GptOssShardConsumption {
    pub shard: SafeTensorFileIdentity,
    pub actions: Vec<GptOssShardConsumerAction>,
    pub planned_payload_bytes: u64,
}

#[derive(Serialize)]
pub struct GptOssShardConsumerPlan {
    schema: String,
    catalog_sha256: String,
    compatibility_metadata_sha256: String,
    mapping_sha256: String,
    placement_sha256: String,
    placement_epoch: u64,
    shards: Vec<GptOssShardConsumption>,
    total_actions: usize,
    total_planned_payload_bytes: u64,
    plan_sha256: String,
}

impl GptOssShardConsumerPlan {
    pub fn build(
        catalog: &SafeTensorShardCatalog,
        native: &GptOssNativeCatalogMap,
        manifest: &GptOssExpertPlacementManifestV1,
    ) -> Result<Self> {
        Self::build_from_source(catalog, native, manifest)
    }

    fn build_from_source(
        catalog: &impl ShardPlanCatalogSource,
        native: &GptOssNativeCatalogMap,
        manifest: &GptOssExpertPlacementManifestV1,
    ) -> Result<Self> {
        if native.catalog_sha256() != catalog.catalog_sha256() {
            return Err(model_error(
                "native catalog mapping identifies a different shard catalog",
            ));
        }
        validate_manifest_identity(native, manifest)?;
        let placement = manifest
            .validate_static()
            .map_err(|error| model_error(format!("static placement manifest: {error}")))?;

        let catalog_shards = catalog.shards_for_plan()?;
        let mut actions = vec![Vec::new(); catalog_shards.len()];
        let mut total_actions = 0_usize;
        for mapping in native.mappings() {
            let tensor = catalog.tensor_for_plan(&mapping.native)?;
            let shard = catalog_shards.get(tensor.shard_index).ok_or_else(|| {
                model_error(format!(
                    "mapped tensor {} references missing shard {}",
                    mapping.native, tensor.shard_index
                ))
            })?;
            let tensor_range_length = tensor.absolute_range[1]
                .checked_sub(tensor.absolute_range[0])
                .ok_or_else(|| {
                    model_error(format!(
                        "catalog tensor {} has a reversed absolute range",
                        mapping.native
                    ))
                })?;
            if tensor_range_length != tensor.byte_len
                || tensor.absolute_range[0] < shard.data_start
                || tensor.absolute_range[1] > shard.file_length
            {
                return Err(model_error(format!(
                    "catalog tensor {} has an invalid shard range",
                    mapping.native
                )));
            }
            if mapping.native_shard != shard.file_name {
                return Err(model_error(format!(
                    "runtime view {} shard identity differs from catalog",
                    mapping.runtime
                )));
            }
            let mapped_end = u64_value(mapping.native_slice[1], "mapping end")?;
            let mapped_length = mapping.native_slice[1]
                .checked_sub(mapping.native_slice[0])
                .ok_or_else(|| {
                    model_error(format!(
                        "runtime view {} has a reversed native slice",
                        mapping.runtime
                    ))
                })?;
            if mapped_length != mapping.bytes {
                return Err(model_error(format!(
                    "runtime view {} native slice length differs from mapped bytes",
                    mapping.runtime
                )));
            }
            if mapped_end > tensor.byte_len {
                return Err(model_error(format!(
                    "runtime view {} range exceeds catalog tensor {}",
                    mapping.runtime, mapping.native
                )));
            }

            if let Some((layer, surface)) = expert_mapping(mapping)? {
                let experts = native.config().num_experts;
                if mapping.runtime_shape.first() != Some(&experts) || mapping.bytes % experts != 0 {
                    return Err(model_error(format!(
                        "expert runtime view {} does not split evenly across {} experts",
                        mapping.runtime, experts
                    )));
                }
                let stride = mapping.bytes / experts;
                for expert in 0..experts {
                    let key = GptOssExpertKey {
                        layer: u16_value(layer, "expert layer")?,
                        expert: u16_value(expert, "expert index")?,
                    };
                    let owner = placement.owner(key).ok_or_else(|| {
                        model_error(format!("static placement omits expert ({layer},{expert})"))
                    })?;
                    let start = mapping.native_slice[0]
                        .checked_add(
                            expert
                                .checked_mul(stride)
                                .ok_or_else(|| model_error("expert slice offset overflows"))?,
                        )
                        .ok_or_else(|| model_error("expert slice start overflows"))?;
                    let end = start
                        .checked_add(stride)
                        .ok_or_else(|| model_error("expert slice end overflows"))?;
                    push_action(
                        &mut actions[tensor.shard_index],
                        &mut total_actions,
                        action(
                            tensor.absolute_range[0],
                            mapping,
                            [start, end],
                            GptOssShardConsumer::OwnedExpert {
                                key,
                                owner: owner.clone(),
                                surface,
                            },
                        )?,
                    )?;
                }
            } else {
                push_action(
                    &mut actions[tensor.shard_index],
                    &mut total_actions,
                    action(
                        tensor.absolute_range[0],
                        mapping,
                        mapping.native_slice,
                        GptOssShardConsumer::LayerOwnerDense {
                            runtime_tensor: mapping.runtime.clone(),
                        },
                    )?,
                )?;
            }
        }

        let mut total_planned_payload_bytes = 0_u64;
        let mut shards = Vec::with_capacity(catalog_shards.len());
        for (identity, mut shard_actions) in catalog_shards.into_iter().zip(actions) {
            shard_actions.sort_by(|left, right| {
                left.shard_absolute_range
                    .cmp(&right.shard_absolute_range)
                    .then_with(|| left.native_tensor.cmp(&right.native_tensor))
            });
            let mut next = identity.data_start;
            let mut planned_payload_bytes = 0_u64;
            for action in &shard_actions {
                if action.shard_absolute_range[0] != next {
                    return Err(model_error(format!(
                        "shard {} consumer ranges overlap or leave a gap at byte {next}",
                        identity.file_name
                    )));
                }
                next = action.shard_absolute_range[1];
                planned_payload_bytes = planned_payload_bytes
                    .checked_add(action.byte_len()?)
                    .ok_or_else(|| model_error("shard planned bytes overflow"))?;
            }
            if next != identity.file_length || planned_payload_bytes != identity.payload_length {
                return Err(model_error(format!(
                    "shard {} consumer plan does not cover its payload exactly",
                    identity.file_name
                )));
            }
            total_planned_payload_bytes = total_planned_payload_bytes
                .checked_add(planned_payload_bytes)
                .ok_or_else(|| model_error("consumer plan payload bytes overflow"))?;
            shards.push(GptOssShardConsumption {
                shard: identity,
                actions: shard_actions,
                planned_payload_bytes,
            });
        }
        if total_planned_payload_bytes != catalog.total_payload_bytes_for_plan() {
            return Err(model_error(
                "consumer plan total differs from catalog payload bytes",
            ));
        }

        let placement_sha256 = placement.manifest_hash().to_owned();
        let plan_sha256 = plan_identity(
            native.catalog_sha256(),
            native.metadata_sha256(),
            native.mapping_sha256(),
            &placement_sha256,
            placement.placement_epoch(),
            &shards,
            total_actions,
            total_planned_payload_bytes,
        )?;
        Ok(Self {
            schema: GPT_OSS_SHARD_CONSUMER_PLAN_SCHEMA_V1.to_owned(),
            catalog_sha256: native.catalog_sha256().to_owned(),
            compatibility_metadata_sha256: native.metadata_sha256().to_owned(),
            mapping_sha256: native.mapping_sha256().to_owned(),
            placement_sha256,
            placement_epoch: placement.placement_epoch(),
            shards,
            total_actions,
            total_planned_payload_bytes,
            plan_sha256,
        })
    }

    pub fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn compatibility_metadata_sha256(&self) -> &str {
        &self.compatibility_metadata_sha256
    }

    pub fn mapping_sha256(&self) -> &str {
        &self.mapping_sha256
    }

    pub fn placement_sha256(&self) -> &str {
        &self.placement_sha256
    }

    pub const fn placement_epoch(&self) -> u64 {
        self.placement_epoch
    }

    pub fn shards(&self) -> &[GptOssShardConsumption] {
        &self.shards
    }

    pub const fn total_actions(&self) -> usize {
        self.total_actions
    }

    pub const fn total_planned_payload_bytes(&self) -> u64 {
        self.total_planned_payload_bytes
    }

    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }

    /// Recompute all framed plan totals and the v1 plan identity.
    ///
    /// Construction currently receives plans produced in-process, but the
    /// scoped shard transaction revalidates this immutable authority before
    /// it admits any payload mapping.
    pub fn validate_identity(&self) -> Result<()> {
        if self.schema != GPT_OSS_SHARD_CONSUMER_PLAN_SCHEMA_V1 {
            return Err(model_error("shard consumer plan schema is unsupported"));
        }
        let mut total_actions = 0_usize;
        let mut total_planned_payload_bytes = 0_u64;
        for shard in &self.shards {
            total_actions = total_actions
                .checked_add(shard.actions.len())
                .ok_or_else(|| model_error("shard consumer action total overflows"))?;
            let mut shard_bytes = 0_u64;
            for action in &shard.actions {
                shard_bytes = shard_bytes
                    .checked_add(action.byte_len()?)
                    .ok_or_else(|| model_error("shard consumer byte total overflows"))?;
            }
            if shard_bytes != shard.planned_payload_bytes {
                return Err(model_error(format!(
                    "shard {} action bytes differ from the recorded plan",
                    shard.shard.file_name
                )));
            }
            total_planned_payload_bytes = total_planned_payload_bytes
                .checked_add(shard_bytes)
                .ok_or_else(|| model_error("shard consumer payload total overflows"))?;
        }
        if total_actions != self.total_actions
            || total_planned_payload_bytes != self.total_planned_payload_bytes
        {
            return Err(model_error(
                "shard consumer plan totals differ from their framed identity",
            ));
        }
        let identity = plan_identity(
            &self.catalog_sha256,
            &self.compatibility_metadata_sha256,
            &self.mapping_sha256,
            &self.placement_sha256,
            self.placement_epoch,
            &self.shards,
            self.total_actions,
            self.total_planned_payload_bytes,
        )?;
        if identity != self.plan_sha256 {
            return Err(model_error(
                "shard consumer plan identity failed recomputation",
            ));
        }
        Ok(())
    }
}

fn validate_manifest_identity(
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
        return Err(model_error(
            "placement manifest does not identify the native catalog mapping exactly",
        ));
    }
    Ok(())
}

fn expert_mapping(mapping: &GptOssTensorViewSpec) -> Result<Option<(usize, GptOssExpertSurface)>> {
    let Some(rest) = mapping.runtime.strip_prefix("model.layers.") else {
        return Ok(None);
    };
    let Some((layer, suffix)) = rest.split_once(".mlp.experts.") else {
        return Ok(None);
    };
    let layer = layer.parse::<usize>().map_err(|_| {
        model_error(format!(
            "expert runtime view {} has an invalid layer",
            mapping.runtime
        ))
    })?;
    let surface = match suffix {
        "gate_up_proj_bias" => GptOssExpertSurface::GateUpBias,
        "gate_up_proj_blocks" => GptOssExpertSurface::GateUpBlocks,
        "gate_up_proj_scales" => GptOssExpertSurface::GateUpScales,
        "down_proj_bias" => GptOssExpertSurface::DownBias,
        "down_proj_blocks" => GptOssExpertSurface::DownBlocks,
        "down_proj_scales" => GptOssExpertSurface::DownScales,
        _ => {
            return Err(model_error(format!(
                "unsupported expert runtime surface {}",
                mapping.runtime
            )))
        }
    };
    Ok(Some((layer, surface)))
}

fn action(
    tensor_absolute_start: u64,
    mapping: &GptOssTensorViewSpec,
    native_tensor_range: [usize; 2],
    consumer: GptOssShardConsumer,
) -> Result<GptOssShardConsumerAction> {
    let start = u64_value(native_tensor_range[0], "consumer range start")?;
    let end = u64_value(native_tensor_range[1], "consumer range end")?;
    if start >= end {
        return Err(model_error(format!(
            "runtime view {} has an empty consumer range",
            mapping.runtime
        )));
    }
    Ok(GptOssShardConsumerAction {
        native_tensor: mapping.native.clone(),
        native_tensor_range: [start, end],
        shard_absolute_range: [
            tensor_absolute_start
                .checked_add(start)
                .ok_or_else(|| model_error("consumer absolute start overflows"))?,
            tensor_absolute_start
                .checked_add(end)
                .ok_or_else(|| model_error("consumer absolute end overflows"))?,
        ],
        consumer,
    })
}

fn push_action(
    actions: &mut Vec<GptOssShardConsumerAction>,
    total: &mut usize,
    action: GptOssShardConsumerAction,
) -> Result<()> {
    *total = total
        .checked_add(1)
        .ok_or_else(|| model_error("consumer action count overflows"))?;
    if *total > MAX_GPT_OSS_SHARD_CONSUMER_ACTIONS {
        return Err(model_error("consumer action count exceeds bound"));
    }
    actions.push(action);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn plan_identity(
    catalog_sha256: &str,
    compatibility_metadata_sha256: &str,
    mapping_sha256: &str,
    placement_sha256: &str,
    placement_epoch: u64,
    shards: &[GptOssShardConsumption],
    total_actions: usize,
    total_planned_payload_bytes: u64,
) -> Result<String> {
    // The schema-tagged JSON object provides structural field, sequence, and
    // enum framing. It deliberately contains no filesystem path or process-
    // local inode/device guard from the catalog.
    #[derive(Serialize)]
    struct Identity<'a> {
        schema: &'static str,
        catalog_sha256: &'a str,
        compatibility_metadata_sha256: &'a str,
        mapping_sha256: &'a str,
        placement_sha256: &'a str,
        placement_epoch: u64,
        shards: &'a [GptOssShardConsumption],
        total_actions: u64,
        total_planned_payload_bytes: u64,
    }
    let bytes = serde_json::to_vec(&Identity {
        schema: GPT_OSS_SHARD_CONSUMER_PLAN_SCHEMA_V1,
        catalog_sha256,
        compatibility_metadata_sha256,
        mapping_sha256,
        placement_sha256,
        placement_epoch,
        shards,
        total_actions: u64_value(total_actions, "consumer action identity count")?,
        total_planned_payload_bytes,
    })
    .map_err(|error| model_error(format!("serialize shard consumer plan: {error}")))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn u64_value(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| model_error(format!("{label} exceeds u64")))
}

fn u16_value(value: usize, label: &str) -> Result<u16> {
    u16::try_from(value).map_err(|_| model_error(format!("{label} exceeds u16")))
}

fn model_error(message: impl Into<String>) -> LLMError {
    LLMError::ModelError(message.into())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs::File;
    use std::io::Write;

    use gpt_oss_gpu::device::{PciBusId, StableCudaDeviceId};
    use serde::Serialize;
    use tempfile::tempdir;

    use crate::heterogeneous::placement::{
        CpuPoolId, ExpertAssignment, GptOssPlacementModel, PlacementBudgets, PlacementPolicyClass,
        HETEROGENEOUS_PLACEMENT_SCHEMA_V1,
    };
    use crate::model_loader::dtype::DType;
    use crate::model_loader::gpt_oss_native::{NativeTensorMetadata, NativeTensorMetadataSource};

    use super::*;

    #[derive(Clone)]
    struct FakeTensor {
        metadata: NativeTensorMetadata,
        plan: PlanTensorMetadata,
    }

    #[derive(Clone)]
    struct FakeCatalog {
        catalog_sha256: String,
        tensors: BTreeMap<String, FakeTensor>,
        shards: Vec<SafeTensorFileIdentity>,
        total_payload_bytes: u64,
    }

    impl NativeTensorMetadataSource for FakeCatalog {
        fn tensor_names(&self) -> Result<BTreeSet<String>> {
            Ok(self.tensors.keys().cloned().collect())
        }

        fn tensor_metadata(&self, name: &str) -> Result<NativeTensorMetadata> {
            self.tensors
                .get(name)
                .map(|tensor| tensor.metadata.clone())
                .ok_or_else(|| model_error(format!("missing fake tensor {name}")))
        }

        fn shard_files(&self) -> Result<Vec<(String, u64)>> {
            Ok(self
                .shards
                .iter()
                .map(|shard| (shard.file_name.clone(), shard.file_length))
                .collect())
        }
    }

    impl ShardPlanCatalogSource for FakeCatalog {
        fn catalog_sha256(&self) -> &str {
            &self.catalog_sha256
        }

        fn shards_for_plan(&self) -> Result<Vec<SafeTensorFileIdentity>> {
            Ok(self.shards.clone())
        }

        fn tensor_for_plan(&self, name: &str) -> Result<PlanTensorMetadata> {
            self.tensors
                .get(name)
                .map(|tensor| tensor.plan.clone())
                .ok_or_else(|| model_error(format!("missing fake tensor {name}")))
        }

        fn total_payload_bytes_for_plan(&self) -> u64 {
            self.total_payload_bytes
        }
    }

    fn config_bytes(layers: usize, experts: usize) -> Vec<u8> {
        #[derive(Serialize)]
        struct ConfigFixture {
            num_hidden_layers: usize,
            num_experts: usize,
            experts_per_token: usize,
            vocab_size: usize,
            hidden_size: usize,
            intermediate_size: usize,
            head_dim: usize,
            num_attention_heads: usize,
            num_key_value_heads: usize,
        }
        serde_json::to_vec(&ConfigFixture {
            num_hidden_layers: layers,
            num_experts: experts,
            experts_per_token: 4,
            vocab_size: 201_088,
            hidden_size: 2_880,
            intermediate_size: 2_880,
            head_dim: 64,
            num_attention_heads: 64,
            num_key_value_heads: 8,
        })
        .unwrap()
    }

    fn fake_catalog(layers: usize, experts: usize) -> FakeCatalog {
        let hidden = 2_880;
        let intermediate = 2_880;
        let heads = 64;
        let q_rows = 4_096;
        let kv_rows = 512;
        let qkv_rows = q_rows + 2 * kv_rows;
        let blocks = hidden / 32;
        let mut specs = BTreeMap::<String, (DType, Vec<usize>)>::new();
        specs.insert(
            "embedding.weight".into(),
            (DType::BF16, vec![201_088, hidden]),
        );
        specs.insert("norm.scale".into(), (DType::BF16, vec![hidden]));
        specs.insert(
            "unembedding.weight".into(),
            (DType::BF16, vec![201_088, hidden]),
        );
        for layer in 0..layers {
            let prefix = format!("block.{layer}");
            for (suffix, dtype, shape) in [
                ("attn.norm.scale", DType::BF16, vec![hidden]),
                ("attn.out.bias", DType::BF16, vec![hidden]),
                ("attn.out.weight", DType::BF16, vec![hidden, q_rows]),
                ("attn.qkv.bias", DType::BF16, vec![qkv_rows]),
                ("attn.qkv.weight", DType::BF16, vec![qkv_rows, hidden]),
                ("attn.sinks", DType::BF16, vec![heads]),
                ("mlp.gate.bias", DType::BF16, vec![experts]),
                ("mlp.gate.weight", DType::BF16, vec![experts, hidden]),
                (
                    "mlp.mlp1_bias",
                    DType::BF16,
                    vec![experts, intermediate * 2],
                ),
                (
                    "mlp.mlp1_weight.blocks",
                    DType::U8,
                    vec![experts, intermediate * 2, blocks, 16],
                ),
                (
                    "mlp.mlp1_weight.scales",
                    DType::U8,
                    vec![experts, intermediate * 2, blocks],
                ),
                ("mlp.mlp2_bias", DType::BF16, vec![experts, hidden]),
                (
                    "mlp.mlp2_weight.blocks",
                    DType::U8,
                    vec![experts, hidden, blocks, 16],
                ),
                (
                    "mlp.mlp2_weight.scales",
                    DType::U8,
                    vec![experts, hidden, blocks],
                ),
                ("mlp.norm.scale", DType::BF16, vec![hidden]),
            ] {
                specs.insert(format!("{prefix}.{suffix}"), (dtype, shape));
            }
        }

        let shard_name = "synthetic-metadata-only.safetensors".to_owned();
        let data_start = 8_u64;
        let mut next = data_start;
        let mut tensors = BTreeMap::new();
        for (name, (dtype, shape)) in specs {
            let bytes = shape
                .iter()
                .try_fold(dtype.size_of(), |bytes, dimension| {
                    bytes.checked_mul(*dimension)
                })
                .unwrap();
            let bytes_u64 = u64::try_from(bytes).unwrap();
            let end = next.checked_add(bytes_u64).unwrap();
            tensors.insert(
                name,
                FakeTensor {
                    metadata: NativeTensorMetadata {
                        dtype,
                        shape,
                        bytes,
                        shard_name: shard_name.clone(),
                    },
                    plan: PlanTensorMetadata {
                        shard_index: 0,
                        byte_len: bytes_u64,
                        absolute_range: [next, end],
                    },
                },
            );
            next = end;
        }
        let payload = next - data_start;
        FakeCatalog {
            catalog_sha256: "a".repeat(64),
            tensors,
            shards: vec![SafeTensorFileIdentity {
                file_name: shard_name,
                file_length: next,
                header_sha256: "b".repeat(64),
                data_start,
                payload_length: payload,
                device: 0,
                inode: 0,
            }],
            total_payload_bytes: payload,
        }
    }

    fn repartition(mut catalog: FakeCatalog, shard_count: usize) -> FakeCatalog {
        assert!(shard_count > 0);
        let tensor_count = catalog.tensors.len();
        let mut next = vec![8_u64; shard_count];
        let names = catalog.tensors.keys().cloned().collect::<Vec<_>>();
        for (index, name) in names.into_iter().enumerate() {
            let shard_index = index * shard_count / tensor_count;
            let shard_name = format!(
                "synthetic-{:05}-of-{:05}.safetensors",
                shard_index + 1,
                shard_count
            );
            let tensor = catalog.tensors.get_mut(&name).unwrap();
            let start = next[shard_index];
            let end = start.checked_add(tensor.plan.byte_len).unwrap();
            tensor.metadata.shard_name = shard_name;
            tensor.plan.shard_index = shard_index;
            tensor.plan.absolute_range = [start, end];
            next[shard_index] = end;
        }
        catalog.shards = next
            .into_iter()
            .enumerate()
            .map(|(index, file_length)| SafeTensorFileIdentity {
                file_name: format!(
                    "synthetic-{:05}-of-{:05}.safetensors",
                    index + 1,
                    shard_count
                ),
                file_length,
                header_sha256: "b".repeat(64),
                data_start: 8,
                payload_length: file_length - 8,
                device: u64::try_from(index + 1).unwrap(),
                inode: u64::try_from(index + 11).unwrap(),
            })
            .collect();
        catalog.total_payload_bytes = catalog
            .shards
            .iter()
            .map(|shard| shard.payload_length)
            .sum();
        catalog
    }

    fn stable(pci: &str) -> StableCudaDeviceId {
        StableCudaDeviceId {
            pci_bus_id: pci.parse::<PciBusId>().unwrap(),
            expected_name: "NVIDIA GeForce RTX 3090".into(),
            compute_capability: (8, 6),
            minimum_memory: 24 * 1024 * 1024 * 1024,
        }
    }

    fn manifest(native: &GptOssNativeCatalogMap, reverse: bool) -> GptOssExpertPlacementManifestV1 {
        let layer_owner = stable("0000:19:00.0");
        let remote_worker = stable("0000:65:00.0");
        let mut assignments = (0..native.config().num_hidden_layers)
            .flat_map(|layer| {
                let layer_owner = layer_owner.clone();
                let remote_worker = remote_worker.clone();
                (0..native.config().num_experts).map(move |expert| ExpertAssignment {
                    key: GptOssExpertKey {
                        layer: u16::try_from(layer).unwrap(),
                        expert: u16::try_from(expert).unwrap(),
                    },
                    owner: match (layer + expert) % 3 {
                        0 => ExpertOwner::Cpu { pool: CpuPoolId(0) },
                        1 => ExpertOwner::LayerOwnerGpu {
                            device: layer_owner.clone(),
                        },
                        _ => ExpertOwner::RemoteGpu {
                            device: remote_worker.clone(),
                        },
                    },
                })
            })
            .collect::<Vec<_>>();
        if reverse {
            assignments.reverse();
        }
        GptOssExpertPlacementManifestV1 {
            schema: HETEROGENEOUS_PLACEMENT_SCHEMA_V1.into(),
            model: GptOssPlacementModel {
                revision: native.revision().into(),
                config_sha256: native.config_sha256().into(),
                index_sha256: native.metadata_sha256().into(),
                mapping_sha256: native.mapping_sha256().into(),
                num_layers: u16::try_from(native.config().num_hidden_layers).unwrap(),
                experts_per_layer: u16::try_from(native.config().num_experts).unwrap(),
                hidden_size: 2_880,
                intermediate_size: 2_880,
                top_k: 4,
            },
            layer_owner,
            remote_worker,
            policy: PlacementPolicyClass::Proof,
            policy_seed: 17,
            placement_epoch: 3,
            budgets: PlacementBudgets {
                max_cpu_experts: u32::MAX,
                max_layer_owner_experts: u32::MAX,
                max_remote_gpu_experts: u32::MAX,
                max_host_owner_bytes: u64::MAX,
                max_layer_owner_bytes: u64::MAX,
                max_remote_gpu_bytes: u64::MAX,
            },
            assignments,
        }
    }

    fn native_map(config: &[u8], catalog: &FakeCatalog) -> GptOssNativeCatalogMap {
        GptOssNativeCatalogMap::from_metadata_source(
            config,
            "synthetic-revision",
            catalog.catalog_sha256(),
            catalog,
        )
        .unwrap()
    }

    #[test]
    fn plans_exact_20b_and_120b_payloads_without_payload_storage() {
        for (
            layers,
            experts,
            expected_actions,
            expected_metadata_identity,
            expected_mapping_identity,
            expected_plan_identity,
        ) in [
            (
                24,
                32,
                4_923_usize,
                "6205a6c690ef4168328d61fcc0896998778df81d367ba33357e692718ed2b04d",
                "bd6b537ca72ade7c71c37a4a9447820d2bcbde66590a03196f709c8e38c0c79c",
                "7ae911f978d42957d0f35740c23c04fcd31b5bfde627b44f996a8fc16b8859a3",
            ),
            (
                36,
                128,
                28_119_usize,
                "ff89ee14ea3c70b1114d86c9a3a38366c73dccdcb9d1e5c58eed772cd377ef00",
                "7fccaf4f0bb43abd321111f301ce30265d8f4fc2fbb309626b7885aa1b739240",
                "085f20acca3343358c3d5a5105d853e7696f49a3f3f193c4b4998a759c76afc6",
            ),
        ] {
            let config = config_bytes(layers, experts);
            let shard_count = if experts == 32 { 3 } else { 7 };
            let catalog = repartition(fake_catalog(layers, experts), shard_count);
            let native = native_map(&config, &catalog);
            assert_eq!(native.mappings().count(), 3 + layers * 19);
            assert_eq!(native.metadata_sha256(), expected_metadata_identity);
            assert_eq!(native.mapping_sha256(), expected_mapping_identity);
            let first_manifest = manifest(&native, false);
            let reversed_manifest = manifest(&native, true);
            let first =
                GptOssShardConsumerPlan::build_from_source(&catalog, &native, &first_manifest)
                    .unwrap();
            let reversed =
                GptOssShardConsumerPlan::build_from_source(&catalog, &native, &reversed_manifest)
                    .unwrap();

            assert_eq!(first.total_actions(), expected_actions);
            assert_eq!(
                first.total_planned_payload_bytes(),
                catalog.total_payload_bytes
            );
            assert_eq!(first.shards().len(), shard_count);
            assert_eq!(
                first
                    .shards()
                    .iter()
                    .map(|shard| shard.actions.len())
                    .sum::<usize>(),
                expected_actions
            );
            assert_eq!(first.plan_sha256(), reversed.plan_sha256());
            assert_eq!(first.placement_sha256(), reversed.placement_sha256());
            assert_eq!(first.plan_sha256(), expected_plan_identity);
            first.validate_identity().unwrap();
            reversed.validate_identity().unwrap();
            assert!(first
                .shards()
                .iter()
                .all(|shard| shard.actions.windows(2).all(
                    |pair| pair[0].shard_absolute_range[1] == pair[1].shard_absolute_range[0]
                )));

            let mut tampered = first;
            tampered.total_actions = tampered.total_actions.checked_add(1).unwrap();
            assert!(tampered.validate_identity().is_err());
        }
    }

    #[test]
    fn every_expert_surface_retains_exact_owner_and_key() {
        let config = config_bytes(24, 32);
        let catalog = fake_catalog(24, 32);
        let native = native_map(&config, &catalog);
        let manifest = manifest(&native, false);
        let placement = manifest.validate_static().unwrap();
        let plan =
            GptOssShardConsumerPlan::build_from_source(&catalog, &native, &manifest).unwrap();
        let mut per_expert = BTreeMap::<GptOssExpertKey, BTreeSet<GptOssExpertSurface>>::new();
        for action in &plan.shards()[0].actions {
            if let GptOssShardConsumer::OwnedExpert {
                key,
                owner,
                surface,
            } = &action.consumer
            {
                assert_eq!(Some(owner), placement.owner(*key));
                per_expert.entry(*key).or_default().insert(*surface);
            }
        }
        assert_eq!(per_expert.len(), 24 * 32);
        assert!(per_expert.values().all(|surfaces| surfaces.len() == 6));
    }

    #[test]
    fn plan_rejects_identity_mismatch_and_range_gap() {
        let config = config_bytes(24, 32);
        let mut catalog = fake_catalog(24, 32);
        let native = native_map(&config, &catalog);
        let mut bad_manifest = manifest(&native, false);
        bad_manifest.model.mapping_sha256 = "f".repeat(64);
        assert!(
            GptOssShardConsumerPlan::build_from_source(&catalog, &native, &bad_manifest).is_err()
        );

        let manifest = manifest(&native, false);
        catalog.catalog_sha256 = "c".repeat(64);
        assert!(GptOssShardConsumerPlan::build_from_source(&catalog, &native, &manifest).is_err());
        catalog.catalog_sha256 = native.catalog_sha256().into();
        let first = catalog.tensors.values_mut().next().unwrap();
        first.plan.absolute_range[0] += 1;
        assert!(GptOssShardConsumerPlan::build_from_source(&catalog, &native, &manifest).is_err());
    }

    #[test]
    fn native_and_plan_fail_closed_on_set_shape_shard_range_and_duplicates() {
        let config = config_bytes(24, 32);
        let catalog = fake_catalog(24, 32);

        let mut missing = catalog.clone();
        missing.tensors.pop_first();
        assert!(GptOssNativeCatalogMap::from_metadata_source(
            &config,
            "synthetic-revision",
            missing.catalog_sha256(),
            &missing,
        )
        .is_err());

        let mut extra = catalog.clone();
        let (_, template) = extra.tensors.first_key_value().unwrap();
        extra
            .tensors
            .insert("unexpected.tensor".into(), template.clone());
        assert!(GptOssNativeCatalogMap::from_metadata_source(
            &config,
            "synthetic-revision",
            extra.catalog_sha256(),
            &extra,
        )
        .is_err());

        let mut wrong_shape = catalog.clone();
        wrong_shape
            .tensors
            .get_mut("embedding.weight")
            .unwrap()
            .metadata
            .shape[0] -= 1;
        assert!(GptOssNativeCatalogMap::from_metadata_source(
            &config,
            "synthetic-revision",
            wrong_shape.catalog_sha256(),
            &wrong_shape,
        )
        .is_err());

        let native = native_map(&config, &catalog);
        let manifest = manifest(&native, false);

        let mut missing_shard = catalog.clone();
        missing_shard
            .tensors
            .values_mut()
            .next()
            .unwrap()
            .plan
            .shard_index = 1;
        assert!(
            GptOssShardConsumerPlan::build_from_source(&missing_shard, &native, &manifest).is_err()
        );

        let mut wrong_shard = catalog.clone();
        wrong_shard.shards[0].file_name = "wrong.safetensors".into();
        assert!(
            GptOssShardConsumerPlan::build_from_source(&wrong_shard, &native, &manifest).is_err()
        );

        let mut overflowing_range = catalog.clone();
        overflowing_range
            .tensors
            .values_mut()
            .next()
            .unwrap()
            .plan
            .absolute_range = [u64::MAX, 0];
        assert!(
            GptOssShardConsumerPlan::build_from_source(&overflowing_range, &native, &manifest,)
                .is_err()
        );

        let mut duplicate = manifest.clone();
        duplicate.assignments.push(duplicate.assignments[0].clone());
        assert!(GptOssShardConsumerPlan::build_from_source(&catalog, &native, &duplicate).is_err());
    }

    #[test]
    fn plan_identity_excludes_process_local_file_identity() {
        let config = config_bytes(24, 32);
        let catalog = repartition(fake_catalog(24, 32), 3);
        let native = native_map(&config, &catalog);
        let manifest = manifest(&native, false);
        let first =
            GptOssShardConsumerPlan::build_from_source(&catalog, &native, &manifest).unwrap();
        let mut changed_guards = catalog.clone();
        for shard in &mut changed_guards.shards {
            shard.device = shard.device.checked_add(100).unwrap();
            shard.inode = shard.inode.checked_add(200).unwrap();
        }
        let second =
            GptOssShardConsumerPlan::build_from_source(&changed_guards, &native, &manifest)
                .unwrap();
        assert_eq!(first.plan_sha256(), second.plan_sha256());

        let serialized = serde_json::to_value(&first).unwrap();
        assert_eq!(serialized["schema"], GPT_OSS_SHARD_CONSUMER_PLAN_SCHEMA_V1);
        let serialized_text = serde_json::to_string(&serialized).unwrap();
        assert!(!serialized_text.contains("\"inode\""));
        assert!(!serialized_text.contains("\"path\""));
        for shard in serialized["shards"].as_array().unwrap() {
            let identity = shard["shard"].as_object().unwrap();
            assert_eq!(
                identity.keys().cloned().collect::<BTreeSet<_>>(),
                [
                    "data_start".to_owned(),
                    "file_length".to_owned(),
                    "file_name".to_owned(),
                    "header_sha256".to_owned(),
                    "payload_length".to_owned(),
                ]
                .into_iter()
                .collect()
            );
            assert!(!identity.contains_key("device"));
            assert!(!identity.contains_key("inode"));
            assert!(!identity.contains_key("path"));
        }
    }

    #[test]
    fn real_catalog_metadata_adapter_uses_only_a_tiny_synthetic_header() {
        let root = tempdir().unwrap();
        let path = root.path().join("model.safetensors");
        let header = br#"{"tiny":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}"#;
        let mut file = File::create(&path).unwrap();
        file.write_all(&u64::try_from(header.len()).unwrap().to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[7, 8, 9]).unwrap();
        drop(file);

        let catalog = SafeTensorShardCatalog::open(root.path()).unwrap();
        let names = NativeTensorMetadataSource::tensor_names(&catalog).unwrap();
        assert_eq!(names, ["tiny".to_owned()].into_iter().collect());
        let tensor = NativeTensorMetadataSource::tensor_metadata(&catalog, "tiny").unwrap();
        assert_eq!(tensor.dtype, DType::U8);
        assert_eq!(tensor.shape, [3]);
        assert_eq!(tensor.bytes, 3);
        assert_eq!(tensor.shard_name, "model.safetensors");
        let shards = NativeTensorMetadataSource::shard_files(&catalog).unwrap();
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0].0, "model.safetensors");
        assert_eq!(shards[0].1, 8 + u64::try_from(header.len()).unwrap() + 3);
    }

    #[test]
    fn scoped_transaction_public_adapter_uses_a_real_tiny_catalog_and_plan() {
        let root = tempdir().unwrap();
        let path = root.path().join("model.safetensors");
        let header = br#"{"tiny":{"dtype":"U8","shape":[3],"data_offsets":[0,3]}}"#;
        let mut file = File::create(&path).unwrap();
        file.write_all(&u64::try_from(header.len()).unwrap().to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[7, 8, 9]).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let catalog = SafeTensorShardCatalog::open(root.path()).unwrap();
        let tensor = catalog.tensor("tiny").unwrap();
        let shards = vec![GptOssShardConsumption {
            shard: catalog.shards()[0].identity.clone(),
            actions: vec![GptOssShardConsumerAction {
                native_tensor: "tiny".into(),
                native_tensor_range: [0, 3],
                shard_absolute_range: tensor.absolute_range,
                consumer: GptOssShardConsumer::LayerOwnerDense {
                    runtime_tensor: "runtime.tiny".into(),
                },
            }],
            planned_payload_bytes: 3,
        }];
        let plan_sha256 = plan_identity(
            catalog.metadata_sha256(),
            "synthetic-compatibility",
            "synthetic-mapping",
            "synthetic-placement",
            1,
            &shards,
            1,
            3,
        )
        .unwrap();
        let plan = GptOssShardConsumerPlan {
            schema: GPT_OSS_SHARD_CONSUMER_PLAN_SCHEMA_V1.into(),
            catalog_sha256: catalog.metadata_sha256().into(),
            compatibility_metadata_sha256: "synthetic-compatibility".into(),
            mapping_sha256: "synthetic-mapping".into(),
            placement_sha256: "synthetic-placement".into(),
            placement_epoch: 1,
            shards,
            total_actions: 1,
            total_planned_payload_bytes: 3,
            plan_sha256,
        };
        let copied = catalog
            .with_scoped_shard_transaction(&plan, 0, |transaction| {
                transaction.with_synchronous_action(0, |action| Ok(action.bytes().to_vec()))
            })
            .unwrap();
        assert_eq!(copied, [7, 8, 9]);
        assert_eq!(catalog.mapping_activity().current, 0);
        assert_eq!(catalog.mapping_activity().high_water, 1);
    }
}
