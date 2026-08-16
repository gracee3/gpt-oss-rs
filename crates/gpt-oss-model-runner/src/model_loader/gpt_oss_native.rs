//! Read-only native GPT-OSS checkpoint aliases and Q/K/V slices.
//!
//! This view owns the shard mappings through [`CpuTensorStore`]. Runtime names
//! borrow immutable byte ranges; opening the view never transforms or copies a
//! tensor payload.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use gpt_oss_core::error::{LLMError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cpu_tensor_store::CpuTensorStore;

use super::dtype::DType;
use super::shard_catalog::SafeTensorShardCatalog;

pub const GPT_OSS_NATIVE_MAPPING_SCHEMA_V1: &str = "gpt-oss-rs.native-checkpoint-map/v1";

#[derive(Debug, Clone, Deserialize)]
pub struct GptOssNativeConfig {
    pub num_hidden_layers: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
}

impl GptOssNativeConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if !matches!(
            (self.num_hidden_layers, self.num_experts),
            (24, 32) | (36, 128)
        ) || self.experts_per_token != 4
            || self.vocab_size != 201_088
            || self.hidden_size != 2_880
            || self.intermediate_size != 2_880
            || self.head_dim != 64
            || self.num_attention_heads != 64
            || self.num_key_value_heads != 8
        {
            return Err(LLMError::ModelError(format!(
                "unsupported native GPT-OSS dimensions: layers={} experts={} top_k={} vocab={} hidden={} intermediate={} heads={} kv_heads={} head_dim={}",
                self.num_hidden_layers,
                self.num_experts,
                self.experts_per_token,
                self.vocab_size,
                self.hidden_size,
                self.intermediate_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim
            )));
        }
        Ok(())
    }

    pub const fn native_tensor_count(&self) -> usize {
        3 + self.num_hidden_layers * 15
    }

    pub const fn runtime_tensor_count(&self) -> usize {
        3 + self.num_hidden_layers * 19
    }
}

#[derive(Debug, Clone)]
pub(super) struct NativeTensorMetadata {
    pub(super) dtype: DType,
    pub(super) shape: Vec<usize>,
    pub(super) bytes: usize,
    pub(super) shard_name: String,
}

pub(super) trait NativeTensorMetadataSource {
    fn tensor_names(&self) -> Result<BTreeSet<String>>;
    fn tensor_metadata(&self, name: &str) -> Result<NativeTensorMetadata>;
    fn shard_files(&self) -> Result<Vec<(String, u64)>>;
}

impl NativeTensorMetadataSource for CpuTensorStore {
    fn tensor_names(&self) -> Result<BTreeSet<String>> {
        Ok(self.names().map(str::to_owned).collect())
    }

    fn tensor_metadata(&self, name: &str) -> Result<NativeTensorMetadata> {
        let tensor = self.tensor(name)?;
        let shard_name = tensor
            .shard_path()
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("shard")
            .to_owned();
        Ok(NativeTensorMetadata {
            dtype: tensor.dtype(),
            shape: tensor.shape().to_vec(),
            bytes: tensor.bytes().len(),
            shard_name,
        })
    }

    fn shard_files(&self) -> Result<Vec<(String, u64)>> {
        self.shard_paths()
            .iter()
            .map(|path| {
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("shard")
                    .to_owned();
                Ok((name, std::fs::metadata(path)?.len()))
            })
            .collect()
    }
}

impl NativeTensorMetadataSource for SafeTensorShardCatalog {
    fn tensor_names(&self) -> Result<BTreeSet<String>> {
        Ok(self.tensors().map(|tensor| tensor.name.clone()).collect())
    }

    fn tensor_metadata(&self, name: &str) -> Result<NativeTensorMetadata> {
        let tensor = self.tensor(name)?;
        let shard = self.shards().get(tensor.shard_index).ok_or_else(|| {
            LLMError::ModelError(format!(
                "catalog tensor {name} references missing shard {}",
                tensor.shard_index
            ))
        })?;
        let dtype = DType::from_safetensors_str(&tensor.dtype).ok_or_else(|| {
            LLMError::ModelError(format!(
                "catalog tensor {name} has unsupported dtype {}",
                tensor.dtype
            ))
        })?;
        Ok(NativeTensorMetadata {
            dtype,
            shape: tensor.shape.clone(),
            bytes: usize::try_from(tensor.byte_len()).map_err(|_| {
                LLMError::ModelError(format!("catalog tensor {name} bytes exceed usize"))
            })?,
            shard_name: shard.identity.file_name.clone(),
        })
    }

    fn shard_files(&self) -> Result<Vec<(String, u64)>> {
        Ok(self
            .shards()
            .iter()
            .map(|shard| (shard.identity.file_name.clone(), shard.identity.file_length))
            .collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GptOssTensorViewSpec {
    pub native: String,
    pub runtime: String,
    pub native_shard: String,
    pub native_slice: [usize; 2],
    pub dtype: String,
    pub native_shape: Vec<usize>,
    pub runtime_shape: Vec<usize>,
    pub bytes: usize,
}

pub struct GptOssCheckpointView {
    source_root: PathBuf,
    config: GptOssNativeConfig,
    store: CpuTensorStore,
    mappings: BTreeMap<String, GptOssTensorViewSpec>,
    config_sha256: String,
    metadata_sha256: String,
    mapping_sha256: String,
    revision: String,
    mapped_payload_bytes: u64,
    expert_payload_bytes: u64,
    non_expert_payload_bytes: u64,
}

/// Payload-free native GPT-OSS mapping validated from a shard catalog.
///
/// This carries the compatibility metadata and runtime-view identities used by
/// the existing checkpoint loader without retaining or exposing tensor bytes.
pub struct GptOssNativeCatalogMap {
    config: GptOssNativeConfig,
    mappings: BTreeMap<String, GptOssTensorViewSpec>,
    revision: String,
    config_sha256: String,
    catalog_sha256: String,
    metadata_sha256: String,
    mapping_sha256: String,
}

impl GptOssNativeCatalogMap {
    pub fn from_config_bytes(
        config_bytes: &[u8],
        revision: impl Into<String>,
        catalog: &SafeTensorShardCatalog,
    ) -> Result<Self> {
        Self::from_metadata_source(config_bytes, revision, catalog.metadata_sha256(), catalog)
    }

    pub(super) fn from_metadata_source(
        config_bytes: &[u8],
        revision: impl Into<String>,
        catalog_sha256: &str,
        source: &impl NativeTensorMetadataSource,
    ) -> Result<Self> {
        let config: GptOssNativeConfig = serde_json::from_slice(config_bytes).map_err(|error| {
            LLMError::ModelError(format!("invalid native GPT-OSS config metadata: {error}"))
        })?;
        let revision = revision.into();
        if revision.trim().is_empty() {
            return Err(LLMError::ModelError(
                "native GPT-OSS catalog revision must not be empty".into(),
            ));
        }
        config.validate()?;
        validate_expected_names(source, &config)?;
        let mappings = build_mappings(source, &config)?;
        if mappings.len() != config.runtime_tensor_count() {
            return Err(LLMError::ModelError(format!(
                "native GPT-OSS catalog mapping produced {} runtime views, expected {}",
                mappings.len(),
                config.runtime_tensor_count()
            )));
        }
        Ok(Self {
            config,
            revision,
            config_sha256: hash_bytes(config_bytes),
            metadata_sha256: metadata_hash(source)?,
            mapping_sha256: mapping_hash(&mappings)?,
            catalog_sha256: catalog_sha256.to_owned(),
            mappings,
        })
    }

    pub const fn config(&self) -> &GptOssNativeConfig {
        &self.config
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    pub fn mappings(&self) -> impl Iterator<Item = &GptOssTensorViewSpec> {
        self.mappings.values()
    }

    pub fn spec(&self, runtime_name: &str) -> Result<&GptOssTensorViewSpec> {
        self.mappings.get(runtime_name).ok_or_else(|| {
            LLMError::ModelError(format!(
                "missing native GPT-OSS catalog view {runtime_name}"
            ))
        })
    }

    pub fn catalog_sha256(&self) -> &str {
        &self.catalog_sha256
    }

    pub fn metadata_sha256(&self) -> &str {
        &self.metadata_sha256
    }

    pub fn mapping_sha256(&self) -> &str {
        &self.mapping_sha256
    }
}

impl GptOssCheckpointView {
    pub fn open(source_root: impl AsRef<Path>) -> Result<Self> {
        let source_root = source_root.as_ref();
        let config_path = source_root.join("config.json");
        let config_bytes = std::fs::read(&config_path)?;
        let config: GptOssNativeConfig =
            serde_json::from_slice(&config_bytes).map_err(|error| {
                LLMError::ModelError(format!(
                    "invalid native GPT-OSS config {}: {error}",
                    config_path.display()
                ))
            })?;
        config.validate()?;
        let store = CpuTensorStore::open(source_root)?;
        validate_expected_names(&store, &config)?;

        let mappings = build_mappings(&store, &config)?;
        if mappings.len() != config.runtime_tensor_count() {
            return Err(LLMError::ModelError(format!(
                "native GPT-OSS mapping produced {} runtime views, expected {}",
                mappings.len(),
                config.runtime_tensor_count()
            )));
        }
        let config_sha256 = hash_bytes(&config_bytes);
        let metadata_sha256 = metadata_hash(&store)?;
        let mapping_sha256 = mapping_hash(&mappings)?;
        let revision =
            read_revision(source_root).unwrap_or_else(|| format!("metadata:{metadata_sha256}"));
        let mapped_payload_bytes = mappings
            .values()
            .map(|mapping| mapping.bytes as u64)
            .sum::<u64>();
        let expert_payload_bytes = mappings
            .values()
            .filter(|mapping| mapping.runtime.contains(".mlp.experts."))
            .map(|mapping| mapping.bytes as u64)
            .sum::<u64>();
        let non_expert_payload_bytes = mapped_payload_bytes - expert_payload_bytes;
        Ok(Self {
            source_root: source_root.to_path_buf(),
            config,
            store,
            mappings,
            config_sha256,
            metadata_sha256,
            mapping_sha256,
            revision,
            mapped_payload_bytes,
            expert_payload_bytes,
            non_expert_payload_bytes,
        })
    }

    pub fn source_root(&self) -> &Path {
        &self.source_root
    }

    pub const fn config(&self) -> &GptOssNativeConfig {
        &self.config
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    pub fn metadata_sha256(&self) -> &str {
        &self.metadata_sha256
    }

    pub fn mapping_sha256(&self) -> &str {
        &self.mapping_sha256
    }

    pub const fn mapped_payload_bytes(&self) -> u64 {
        self.mapped_payload_bytes
    }

    pub const fn expert_payload_bytes(&self) -> u64 {
        self.expert_payload_bytes
    }

    pub const fn non_expert_payload_bytes(&self) -> u64 {
        self.non_expert_payload_bytes
    }

    pub fn mappings(&self) -> impl Iterator<Item = &GptOssTensorViewSpec> {
        self.mappings.values()
    }

    pub fn spec(&self, runtime_name: &str) -> Result<&GptOssTensorViewSpec> {
        self.mappings.get(runtime_name).ok_or_else(|| {
            LLMError::ModelError(format!(
                "missing native GPT-OSS runtime view {runtime_name}"
            ))
        })
    }

    pub fn tensor(&self, runtime_name: &str) -> Result<GptOssTensorView<'_>> {
        let spec = self.spec(runtime_name)?;
        let native = self.store.tensor(&spec.native)?;
        let [start, end] = spec.native_slice;
        Ok(GptOssTensorView {
            spec,
            bytes: &native.bytes()[start..end],
        })
    }

    pub fn mapping_json(&self) -> Result<Vec<u8>> {
        #[derive(Serialize)]
        struct MappingDocument<'a> {
            schema: &'static str,
            native_tensor_count: usize,
            runtime_tensor_count: usize,
            mapping_count: usize,
            config_sha256: &'a str,
            metadata_sha256: &'a str,
            mappings: Vec<&'a GptOssTensorViewSpec>,
        }
        let document = MappingDocument {
            schema: GPT_OSS_NATIVE_MAPPING_SCHEMA_V1,
            native_tensor_count: self.store.len(),
            runtime_tensor_count: self.mappings.len(),
            mapping_count: self.mappings.len(),
            config_sha256: &self.config_sha256,
            metadata_sha256: &self.metadata_sha256,
            mappings: self.mappings.values().collect(),
        };
        let mut bytes = serde_json::to_vec_pretty(&document)
            .map_err(|error| LLMError::ModelError(format!("serialize checkpoint map: {error}")))?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

pub struct GptOssTensorView<'a> {
    spec: &'a GptOssTensorViewSpec,
    bytes: &'a [u8],
}

impl<'a> GptOssTensorView<'a> {
    pub const fn spec(&self) -> &'a GptOssTensorViewSpec {
        self.spec
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

fn expected_native_names(config: &GptOssNativeConfig) -> BTreeSet<String> {
    let mut names = ["embedding.weight", "norm.scale", "unembedding.weight"]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    let suffixes = [
        "attn.norm.scale",
        "attn.out.bias",
        "attn.out.weight",
        "attn.qkv.bias",
        "attn.qkv.weight",
        "attn.sinks",
        "mlp.gate.bias",
        "mlp.gate.weight",
        "mlp.mlp1_bias",
        "mlp.mlp1_weight.blocks",
        "mlp.mlp1_weight.scales",
        "mlp.mlp2_bias",
        "mlp.mlp2_weight.blocks",
        "mlp.mlp2_weight.scales",
        "mlp.norm.scale",
    ];
    for layer in 0..config.num_hidden_layers {
        for suffix in suffixes {
            names.insert(format!("block.{layer}.{suffix}"));
        }
    }
    names
}

fn validate_expected_names(
    source: &impl NativeTensorMetadataSource,
    config: &GptOssNativeConfig,
) -> Result<()> {
    let expected_names = expected_native_names(config);
    let observed_names = source.tensor_names()?;
    if observed_names != expected_names {
        let missing = expected_names
            .difference(&observed_names)
            .take(4)
            .cloned()
            .collect::<Vec<_>>();
        let extra = observed_names
            .difference(&expected_names)
            .take(4)
            .cloned()
            .collect::<Vec<_>>();
        return Err(LLMError::ModelError(format!(
            "native GPT-OSS tensor set mismatch: expected={} observed={} missing={missing:?} extra={extra:?}",
            expected_names.len(),
            observed_names.len()
        )));
    }
    Ok(())
}

fn build_mappings(
    store: &impl NativeTensorMetadataSource,
    config: &GptOssNativeConfig,
) -> Result<BTreeMap<String, GptOssTensorViewSpec>> {
    let mut mappings = BTreeMap::new();
    insert_alias(
        &mut mappings,
        store,
        "embedding.weight",
        "model.embed_tokens.weight",
        DType::BF16,
        &[config.vocab_size, config.hidden_size],
    )?;
    insert_alias(
        &mut mappings,
        store,
        "norm.scale",
        "model.norm.weight",
        DType::BF16,
        &[config.hidden_size],
    )?;
    insert_alias(
        &mut mappings,
        store,
        "unembedding.weight",
        "lm_head.weight",
        DType::BF16,
        &[config.vocab_size, config.hidden_size],
    )?;

    let q_rows = config.num_attention_heads * config.head_dim;
    let kv_rows = config.num_key_value_heads * config.head_dim;
    let qkv_rows = q_rows + 2 * kv_rows;
    let blocks = config.hidden_size / 32;
    for layer in 0..config.num_hidden_layers {
        let native = |suffix: &str| format!("block.{layer}.{suffix}");
        let runtime = |suffix: &str| format!("model.layers.{layer}.{suffix}");
        for (source, target, dtype, shape) in [
            (
                native("attn.norm.scale"),
                runtime("input_layernorm.weight"),
                DType::BF16,
                vec![config.hidden_size],
            ),
            (
                native("attn.out.bias"),
                runtime("self_attn.o_proj.bias"),
                DType::BF16,
                vec![config.hidden_size],
            ),
            (
                native("attn.out.weight"),
                runtime("self_attn.o_proj.weight"),
                DType::BF16,
                vec![config.hidden_size, q_rows],
            ),
            (
                native("attn.sinks"),
                runtime("self_attn.sinks"),
                DType::BF16,
                vec![config.num_attention_heads],
            ),
            (
                native("mlp.norm.scale"),
                runtime("post_attention_layernorm.weight"),
                DType::BF16,
                vec![config.hidden_size],
            ),
            (
                native("mlp.gate.bias"),
                runtime("mlp.router.bias"),
                DType::BF16,
                vec![config.num_experts],
            ),
            (
                native("mlp.gate.weight"),
                runtime("mlp.router.weight"),
                DType::BF16,
                vec![config.num_experts, config.hidden_size],
            ),
            (
                native("mlp.mlp1_bias"),
                runtime("mlp.experts.gate_up_proj_bias"),
                DType::BF16,
                vec![config.num_experts, config.intermediate_size * 2],
            ),
            (
                native("mlp.mlp1_weight.blocks"),
                runtime("mlp.experts.gate_up_proj_blocks"),
                DType::U8,
                vec![config.num_experts, config.intermediate_size * 2, blocks, 16],
            ),
            (
                native("mlp.mlp1_weight.scales"),
                runtime("mlp.experts.gate_up_proj_scales"),
                DType::U8,
                vec![config.num_experts, config.intermediate_size * 2, blocks],
            ),
            (
                native("mlp.mlp2_bias"),
                runtime("mlp.experts.down_proj_bias"),
                DType::BF16,
                vec![config.num_experts, config.hidden_size],
            ),
            (
                native("mlp.mlp2_weight.blocks"),
                runtime("mlp.experts.down_proj_blocks"),
                DType::U8,
                vec![config.num_experts, config.hidden_size, blocks, 16],
            ),
            (
                native("mlp.mlp2_weight.scales"),
                runtime("mlp.experts.down_proj_scales"),
                DType::U8,
                vec![config.num_experts, config.hidden_size, blocks],
            ),
        ] {
            insert_alias(&mut mappings, store, &source, &target, dtype, &shape)?;
        }

        for (projection, start_row, rows) in [
            ("q", 0, q_rows),
            ("k", q_rows, kv_rows),
            ("v", q_rows + kv_rows, kv_rows),
        ] {
            insert_slice(
                &mut mappings,
                store,
                &native("attn.qkv.weight"),
                &runtime(&format!("self_attn.{projection}_proj.weight")),
                DType::BF16,
                &[qkv_rows, config.hidden_size],
                &[rows, config.hidden_size],
                start_row * config.hidden_size * 2,
            )?;
            insert_slice(
                &mut mappings,
                store,
                &native("attn.qkv.bias"),
                &runtime(&format!("self_attn.{projection}_proj.bias")),
                DType::BF16,
                &[qkv_rows],
                &[rows],
                start_row * 2,
            )?;
        }
    }
    Ok(mappings)
}

fn insert_alias(
    mappings: &mut BTreeMap<String, GptOssTensorViewSpec>,
    store: &impl NativeTensorMetadataSource,
    native: &str,
    runtime: &str,
    dtype: DType,
    shape: &[usize],
) -> Result<()> {
    insert_slice(mappings, store, native, runtime, dtype, shape, shape, 0)
}

#[allow(clippy::too_many_arguments)]
fn insert_slice(
    mappings: &mut BTreeMap<String, GptOssTensorViewSpec>,
    store: &impl NativeTensorMetadataSource,
    native_name: &str,
    runtime_name: &str,
    dtype: DType,
    native_shape: &[usize],
    runtime_shape: &[usize],
    start: usize,
) -> Result<()> {
    let tensor = store.tensor_metadata(native_name)?;
    if tensor.dtype != dtype || tensor.shape != native_shape {
        return Err(LLMError::ModelError(format!(
            "native tensor {native_name} metadata mismatch: dtype={} shape={:?}, expected {dtype} {native_shape:?}",
            tensor.dtype,
            tensor.shape
        )));
    }
    let bytes = runtime_shape
        .iter()
        .try_fold(dtype.size_of(), |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| LLMError::ModelError(format!("runtime view {runtime_name} overflows")))?;
    let end = start
        .checked_add(bytes)
        .ok_or_else(|| LLMError::ModelError(format!("runtime view {runtime_name} overflows")))?;
    if end > tensor.bytes {
        return Err(LLMError::ModelError(format!(
            "runtime view {runtime_name} range [{start},{end}) exceeds native tensor {} bytes",
            tensor.bytes
        )));
    }
    let spec = GptOssTensorViewSpec {
        native: native_name.to_owned(),
        runtime: runtime_name.to_owned(),
        native_shard: tensor.shard_name,
        native_slice: [start, end],
        dtype: dtype.to_string(),
        native_shape: native_shape.to_vec(),
        runtime_shape: runtime_shape.to_vec(),
        bytes,
    };
    if mappings.insert(runtime_name.to_owned(), spec).is_some() {
        return Err(LLMError::ModelError(format!(
            "duplicate runtime checkpoint view {runtime_name}"
        )));
    }
    Ok(())
}

fn metadata_hash(store: &impl NativeTensorMetadataSource) -> Result<String> {
    let names = store.tensor_names()?;
    let mut digest = Sha256::new();
    digest.update(b"gpt-oss-rs-native-metadata-v1");
    for name in names {
        let tensor = store.tensor_metadata(&name)?;
        digest.update(name.as_bytes());
        digest.update(tensor.dtype.to_string().as_bytes());
        for dimension in &tensor.shape {
            digest.update(dimension.to_le_bytes());
        }
        digest.update(tensor.bytes.to_le_bytes());
        digest.update(tensor.shard_name.as_bytes());
    }
    for (name, length) in store.shard_files()? {
        digest.update(name.as_bytes());
        digest.update(length.to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn mapping_hash(mappings: &BTreeMap<String, GptOssTensorViewSpec>) -> Result<String> {
    let bytes = serde_json::to_vec(mappings)
        .map_err(|error| LLMError::ModelError(format!("serialize checkpoint map: {error}")))?;
    Ok(hash_bytes(&bytes))
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn read_revision(source_root: &Path) -> Option<String> {
    [
        source_root.join("REVISION"),
        source_root.parent()?.join("REVISION"),
    ]
    .into_iter()
    .find_map(|path| {
        std::fs::read_to_string(path)
            .ok()
            .map(|revision| revision.trim().to_owned())
            .filter(|revision| !revision.is_empty())
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::fs::File;
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;

    fn write_tiny_u8_shard(path: &Path) {
        let header = br#"{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut file = File::create(path).unwrap();
        file.write_all(&u64::try_from(header.len()).unwrap().to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[7]).unwrap();
    }

    fn legacy_metadata_hash(store: &CpuTensorStore) -> Result<String> {
        let mut names = store.names().collect::<Vec<_>>();
        names.sort_unstable();
        let mut digest = Sha256::new();
        digest.update(b"gpt-oss-rs-native-metadata-v1");
        for name in names {
            let tensor = store.tensor(name)?;
            digest.update(name.as_bytes());
            digest.update(tensor.dtype().to_string().as_bytes());
            for dimension in tensor.shape() {
                digest.update(dimension.to_le_bytes());
            }
            digest.update(tensor.bytes().len().to_le_bytes());
            digest.update(
                tensor
                    .shard_path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("shard")
                    .as_bytes(),
            );
        }
        for path in store.shard_paths() {
            digest.update(
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("shard")
                    .as_bytes(),
            );
            digest.update(std::fs::metadata(path)?.len().to_le_bytes());
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    #[test]
    fn expected_name_and_mapping_cardinalities_are_exact() {
        for (layers, experts, native, runtime) in [(24, 32, 363, 459), (36, 128, 543, 687)] {
            let config = GptOssNativeConfig {
                num_hidden_layers: layers,
                num_experts: experts,
                experts_per_token: 4,
                vocab_size: 201_088,
                hidden_size: 2_880,
                intermediate_size: 2_880,
                head_dim: 64,
                num_attention_heads: 64,
                num_key_value_heads: 8,
            };
            config.validate().unwrap();
            assert_eq!(config.native_tensor_count(), native);
            assert_eq!(config.runtime_tensor_count(), runtime);
            assert_eq!(expected_native_names(&config).len(), native);
        }
    }

    #[test]
    fn refactored_cpu_store_metadata_hash_matches_verbatim_legacy_algorithm() {
        let root = tempdir().unwrap();
        std::fs::write(root.path().join("config.json"), b"{}").unwrap();
        #[cfg(unix)]
        let shard_name = {
            use std::os::unix::ffi::OsStringExt;
            OsString::from_vec(b"native-\xff.safetensors".to_vec())
        };
        #[cfg(not(unix))]
        let shard_name = OsString::from("native.safetensors");
        write_tiny_u8_shard(&root.path().join(shard_name));

        let store = CpuTensorStore::open(root.path()).unwrap();
        let refactored_name = NativeTensorMetadataSource::tensor_metadata(&store, "weight")
            .unwrap()
            .shard_name;
        #[cfg(unix)]
        assert_eq!(refactored_name, "shard");
        #[cfg(not(unix))]
        assert_eq!(refactored_name, "native.safetensors");
        assert_eq!(
            metadata_hash(&store).unwrap(),
            legacy_metadata_hash(&store).unwrap()
        );
    }
}
