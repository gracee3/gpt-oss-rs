//! CUDA-free safetensors header discovery.
//!
//! This module reads only safetensors header metadata. It does not read tensor
//! payloads into host vectors, convert tensor data, allocate GPU buffers, or
//! call the runtime loaders.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use gpt_oss_core::error::{LLMError, Result};

/// Header-only metadata for a safetensors model or sharded directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorHeaderManifest {
    pub tensors: Vec<SafetensorTensorInfo>,
    pub merge_policy: SafetensorHeaderMergePolicy,
    pub overridden_tensor_names: Vec<String>,
}

/// Header-only metadata for one tensor entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetensorTensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub source_file: Option<PathBuf>,
    pub byte_size: usize,
}

/// Duplicate handling policy for header-only safetensors discovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetensorHeaderMergePolicy {
    RejectDuplicates,
    AllowRestrictedSinksOverride,
}

impl SafetensorHeaderManifest {
    /// Discover safetensors header metadata from a single file or directory.
    pub fn discover(path: &Path) -> Result<Self> {
        Self::discover_with_merge_policy(path, SafetensorHeaderMergePolicy::RejectDuplicates)
    }

    /// Discover safetensors header metadata with an explicit merge policy.
    pub fn discover_with_merge_policy(
        path: &Path,
        merge_policy: SafetensorHeaderMergePolicy,
    ) -> Result<Self> {
        if path.is_file() {
            return Self::from_file_with_merge_policy(path, merge_policy);
        }
        if path.is_dir() {
            return Self::from_dir_with_merge_policy(path, merge_policy);
        }

        Err(LLMError::ModelError(format!(
            "safetensors path does not exist: {}",
            path.display()
        )))
    }

    /// Discover header metadata from one `.safetensors` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_file_with_merge_policy(path, SafetensorHeaderMergePolicy::RejectDuplicates)
    }

    /// Discover header metadata from one `.safetensors` file with a merge policy.
    pub fn from_file_with_merge_policy(
        path: &Path,
        merge_policy: SafetensorHeaderMergePolicy,
    ) -> Result<Self> {
        let mut tensors = parse_header_file(path)?;
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self {
            tensors,
            merge_policy,
            overridden_tensor_names: Vec::new(),
        })
    }

    /// Discover header metadata from all `.safetensors` files in a directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        Self::from_dir_with_merge_policy(dir, SafetensorHeaderMergePolicy::RejectDuplicates)
    }

    /// Discover header metadata from all `.safetensors` files in a directory
    /// with an explicit duplicate handling policy.
    pub fn from_dir_with_merge_policy(
        dir: &Path,
        merge_policy: SafetensorHeaderMergePolicy,
    ) -> Result<Self> {
        let shard_files = collect_safetensor_files(dir)?;
        let mut tensors = Vec::new();
        let mut tensor_indices = HashMap::new();
        let mut overridden_tensor_names = Vec::new();

        for shard_file in shard_files {
            for tensor in parse_header_file(&shard_file)? {
                if let Some(&existing_idx) = tensor_indices.get(&tensor.name) {
                    if merge_policy.allows_restricted_sinks_override(&shard_file, &tensor.name) {
                        overridden_tensor_names.push(tensor.name.clone());
                        tensors[existing_idx] = tensor;
                    } else {
                        return Err(duplicate_tensor_error(&tensor.name));
                    }
                } else if merge_policy.rejects_unique_override_tensor(&shard_file) {
                    return Err(LLMError::ModelError(format!(
                        "restricted sinks override tensor '{}' does not replace an existing base shard tensor",
                        tensor.name
                    )));
                } else {
                    tensor_indices.insert(tensor.name.clone(), tensors.len());
                    tensors.push(tensor);
                }
            }
        }

        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        overridden_tensor_names.sort();
        Ok(Self {
            tensors,
            merge_policy,
            overridden_tensor_names,
        })
    }

    /// Stable list of discovered tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect()
    }

    /// Returns true when a tensor name was discovered.
    pub fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.iter().any(|tensor| tensor.name == name)
    }

    /// Returns true when `lm_head.weight` was discovered.
    pub fn has_lm_head_weight(&self) -> bool {
        self.contains_tensor("lm_head.weight")
    }

    /// Number of tensor headers replaced by the configured merge policy.
    pub fn overridden_tensor_count(&self) -> usize {
        self.overridden_tensor_names.len()
    }
}

impl SafetensorHeaderMergePolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            SafetensorHeaderMergePolicy::RejectDuplicates => "reject_duplicates",
            SafetensorHeaderMergePolicy::AllowRestrictedSinksOverride => {
                "allow_restricted_sinks_override"
            }
        }
    }

    fn allows_restricted_sinks_override(self, shard_file: &Path, tensor_name: &str) -> bool {
        self == SafetensorHeaderMergePolicy::AllowRestrictedSinksOverride
            && shard_file_name(shard_file) == Some(RESTRICTED_SINKS_OVERRIDE_FILE)
            && is_restricted_sinks_tensor_name(tensor_name)
    }

    fn rejects_unique_override_tensor(self, shard_file: &Path) -> bool {
        self == SafetensorHeaderMergePolicy::AllowRestrictedSinksOverride
            && shard_file_name(shard_file) == Some(RESTRICTED_SINKS_OVERRIDE_FILE)
    }
}

const RESTRICTED_SINKS_OVERRIDE_FILE: &str = "zzzz-sinks-override.safetensors";

fn duplicate_tensor_error(tensor_name: &str) -> LLMError {
    LLMError::ModelError(format!(
        "duplicate tensor '{tensor_name}' across safetensors shards"
    ))
}

fn shard_file_name(path: &Path) -> Option<&str> {
    path.file_name().and_then(|name| name.to_str())
}

fn is_restricted_sinks_tensor_name(tensor_name: &str) -> bool {
    let Some(rest) = tensor_name.strip_prefix("model.layers.") else {
        return false;
    };
    let Some((layer_idx, suffix)) = rest.split_once('.') else {
        return false;
    };

    suffix == "self_attn.sinks" && layer_idx.parse::<usize>().is_ok()
}

fn collect_safetensor_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut shard_files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();
    shard_files.sort();

    if shard_files.is_empty() {
        return Err(LLMError::ModelError(format!(
            "no .safetensors files found in {}",
            dir.display()
        )));
    }

    Ok(shard_files)
}

fn parse_header_file(path: &Path) -> Result<Vec<SafetensorTensorInfo>> {
    let mut file = std::fs::File::open(path)?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes).map_err(|e| {
        LLMError::ModelError(format!(
            "failed to read safetensors header length from {}: {e}",
            path.display()
        ))
    })?;

    let header_size = u64::from_le_bytes(header_len_bytes) as usize;
    let file_len = file.metadata()?.len() as usize;
    if 8 + header_size > file_len {
        return Err(LLMError::ModelError(format!(
            "header size exceeds file length in {}",
            path.display()
        )));
    }

    let mut header_bytes = vec![0u8; header_size];
    file.seek(SeekFrom::Start(8))?;
    file.read_exact(&mut header_bytes)?;

    let header_str = std::str::from_utf8(&header_bytes)
        .map_err(|e| LLMError::ModelError(format!("invalid header utf8: {e}")))?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_str(header_str)
        .map_err(|e| LLMError::SerializationError(format!("header json: {e}")))?;

    let mut tensors = Vec::new();
    for (name, meta) in header {
        if name == "__metadata__" {
            continue;
        }

        let obj = meta.as_object().ok_or_else(|| {
            LLMError::ModelError(format!("tensor {name} has non-object header metadata"))
        })?;
        let dtype = obj
            .get("dtype")
            .and_then(|value| value.as_str())
            .ok_or_else(|| LLMError::ModelError(format!("tensor {name} missing dtype")))?
            .to_string();
        let shape = obj
            .get("shape")
            .and_then(|value| value.as_array())
            .ok_or_else(|| LLMError::ModelError(format!("tensor {name} missing shape")))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .map(|dim| dim as usize)
                    .ok_or_else(|| LLMError::ModelError(format!("tensor {name} invalid shape")))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = obj
            .get("data_offsets")
            .and_then(|value| value.as_array())
            .ok_or_else(|| LLMError::ModelError(format!("tensor {name} missing data_offsets")))?;
        if offsets.len() != 2 {
            return Err(LLMError::ModelError(format!(
                "tensor {name} has {} offsets, expected 2",
                offsets.len()
            )));
        }

        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| LLMError::ModelError(format!("tensor {name} invalid start offset")))?
            as usize;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| LLMError::ModelError(format!("tensor {name} invalid end offset")))?
            as usize;
        if end < start {
            return Err(LLMError::ModelError(format!(
                "tensor {name} has reversed data offsets"
            )));
        }
        if 8 + header_size + end > file_len {
            return Err(LLMError::ModelError(format!(
                "tensor {name} data range exceeds file length in {}",
                path.display()
            )));
        }

        tensors.push(SafetensorTensorInfo {
            name,
            dtype,
            shape,
            source_file: Some(path.to_path_buf()),
            byte_size: end - start,
        });
    }

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::super::dtype::DType;
    use super::super::safetensors::build_test_safetensors;
    use super::*;

    fn write_safetensors(path: &Path, tensors: &[(&str, &[usize], DType, &[u8])]) -> PathBuf {
        let bytes = build_test_safetensors(tensors);
        std::fs::write(path, bytes).unwrap();
        path.to_path_buf()
    }

    #[test]
    fn header_manifest_discovers_single_file_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_safetensors(
            &dir.path().join("model.safetensors"),
            &[
                ("model.embed_tokens.weight", &[2, 4], DType::F16, &[0u8; 16]),
                ("lm_head.weight", &[2, 4], DType::F32, &[1u8; 32]),
            ],
        );

        let manifest = SafetensorHeaderManifest::from_file(&path).unwrap();

        assert_eq!(
            manifest.tensor_names(),
            vec!["lm_head.weight", "model.embed_tokens.weight"]
        );
        let embed = manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == "model.embed_tokens.weight")
            .unwrap();
        assert_eq!(embed.dtype, "F16");
        assert_eq!(embed.shape, vec![2, 4]);
        assert_eq!(embed.source_file, Some(path));
        assert_eq!(embed.byte_size, 16);
        assert!(manifest.has_lm_head_weight());
    }

    #[test]
    fn header_manifest_discovers_sharded_directory_deterministically() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(
            &dir.path().join("model-00002-of-00002.safetensors"),
            &[(
                "model.layers.12.self_attn.q_proj.weight",
                &[2],
                DType::F32,
                &[2u8; 8],
            )],
        );
        write_safetensors(
            &dir.path().join("model-00001-of-00002.safetensors"),
            &[(
                "model.layers.0.self_attn.q_proj.weight",
                &[2],
                DType::F32,
                &[1u8; 8],
            )],
        );

        let manifest = SafetensorHeaderManifest::from_dir(dir.path()).unwrap();

        assert_eq!(
            manifest.tensor_names(),
            vec![
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.12.self_attn.q_proj.weight"
            ]
        );
    }

    #[test]
    fn header_manifest_ignores_non_safetensors_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("README.md"), "ignore me").unwrap();
        write_safetensors(
            &dir.path().join("model.safetensors"),
            &[("model.norm.weight", &[2], DType::F32, &[0u8; 8])],
        );

        let manifest = SafetensorHeaderManifest::discover(dir.path()).unwrap();

        assert_eq!(manifest.tensor_names(), vec!["model.norm.weight"]);
    }

    #[test]
    fn header_manifest_rejects_duplicate_tensor_names_across_shards() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(
            &dir.path().join("a.safetensors"),
            &[("model.norm.weight", &[2], DType::F32, &[0u8; 8])],
        );
        write_safetensors(
            &dir.path().join("b.safetensors"),
            &[("model.norm.weight", &[2], DType::F32, &[1u8; 8])],
        );

        let err = SafetensorHeaderManifest::from_dir(dir.path()).unwrap_err();

        assert!(
            err.to_string()
                .contains("duplicate tensor 'model.norm.weight'"),
            "got: {err}"
        );
    }

    #[test]
    fn restricted_sinks_override_allows_sink_replacement_from_override_file() {
        let dir = tempfile::tempdir().unwrap();
        let base = write_safetensors(
            &dir.path().join("model-00000-of-00002.safetensors"),
            &[
                (
                    "model.layers.0.self_attn.sinks",
                    &[2],
                    DType::F32,
                    &[0u8; 8],
                ),
                (
                    "model.layers.1.self_attn.sinks",
                    &[2],
                    DType::F32,
                    &[1u8; 8],
                ),
            ],
        );
        let override_path = write_safetensors(
            &dir.path().join("zzzz-sinks-override.safetensors"),
            &[(
                "model.layers.0.self_attn.sinks",
                &[2],
                DType::F32,
                &[2u8; 8],
            )],
        );

        let manifest = SafetensorHeaderManifest::from_dir_with_merge_policy(
            dir.path(),
            SafetensorHeaderMergePolicy::AllowRestrictedSinksOverride,
        )
        .unwrap();

        assert_eq!(
            manifest.tensor_names(),
            vec![
                "model.layers.0.self_attn.sinks",
                "model.layers.1.self_attn.sinks"
            ]
        );
        assert_eq!(
            manifest.overridden_tensor_names,
            vec!["model.layers.0.self_attn.sinks".to_string()]
        );
        assert_eq!(manifest.overridden_tensor_count(), 1);
        assert_eq!(
            manifest.merge_policy,
            SafetensorHeaderMergePolicy::AllowRestrictedSinksOverride
        );

        let overridden = manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == "model.layers.0.self_attn.sinks")
            .unwrap();
        assert_eq!(overridden.source_file, Some(override_path));

        let base_only = manifest
            .tensors
            .iter()
            .find(|tensor| tensor.name == "model.layers.1.self_attn.sinks")
            .unwrap();
        assert_eq!(base_only.source_file, Some(base));
    }

    #[test]
    fn restricted_sinks_override_rejects_duplicate_non_sink_from_override_file() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(
            &dir.path().join("model-00000-of-00002.safetensors"),
            &[("model.norm.weight", &[2], DType::F32, &[0u8; 8])],
        );
        write_safetensors(
            &dir.path().join("zzzz-sinks-override.safetensors"),
            &[("model.norm.weight", &[2], DType::F32, &[1u8; 8])],
        );

        let err = SafetensorHeaderManifest::from_dir_with_merge_policy(
            dir.path(),
            SafetensorHeaderMergePolicy::AllowRestrictedSinksOverride,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("duplicate tensor 'model.norm.weight'"),
            "got: {err}"
        );
    }

    #[test]
    fn restricted_sinks_override_rejects_sink_duplicate_from_non_override_file() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(
            &dir.path().join("model-00000-of-00002.safetensors"),
            &[(
                "model.layers.0.self_attn.sinks",
                &[2],
                DType::F32,
                &[0u8; 8],
            )],
        );
        write_safetensors(
            &dir.path().join("model-00001-of-00002.safetensors"),
            &[(
                "model.layers.0.self_attn.sinks",
                &[2],
                DType::F32,
                &[1u8; 8],
            )],
        );

        let err = SafetensorHeaderManifest::from_dir_with_merge_policy(
            dir.path(),
            SafetensorHeaderMergePolicy::AllowRestrictedSinksOverride,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("duplicate tensor 'model.layers.0.self_attn.sinks'"),
            "got: {err}"
        );
    }

    #[test]
    fn restricted_sinks_override_rejects_unique_tensor_from_override_file() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(
            &dir.path().join("model-00000-of-00002.safetensors"),
            &[("model.norm.weight", &[2], DType::F32, &[0u8; 8])],
        );
        write_safetensors(
            &dir.path().join("zzzz-sinks-override.safetensors"),
            &[(
                "model.layers.0.self_attn.sinks",
                &[2],
                DType::F32,
                &[1u8; 8],
            )],
        );

        let err = SafetensorHeaderManifest::from_dir_with_merge_policy(
            dir.path(),
            SafetensorHeaderMergePolicy::AllowRestrictedSinksOverride,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("does not replace an existing base shard tensor"),
            "got: {err}"
        );
    }

    #[test]
    fn restricted_sinks_override_preserves_deterministic_tensor_ordering() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(
            &dir.path().join("model-00000-of-00002.safetensors"),
            &[
                (
                    "model.layers.1.self_attn.sinks",
                    &[2],
                    DType::F32,
                    &[0u8; 8],
                ),
                ("model.embed_tokens.weight", &[2], DType::F16, &[1u8; 4]),
                (
                    "model.layers.0.self_attn.sinks",
                    &[2],
                    DType::F32,
                    &[2u8; 8],
                ),
            ],
        );
        write_safetensors(
            &dir.path().join("zzzz-sinks-override.safetensors"),
            &[
                (
                    "model.layers.1.self_attn.sinks",
                    &[2],
                    DType::F32,
                    &[3u8; 8],
                ),
                (
                    "model.layers.0.self_attn.sinks",
                    &[2],
                    DType::F32,
                    &[4u8; 8],
                ),
            ],
        );

        let manifest = SafetensorHeaderManifest::from_dir_with_merge_policy(
            dir.path(),
            SafetensorHeaderMergePolicy::AllowRestrictedSinksOverride,
        )
        .unwrap();

        assert_eq!(
            manifest.tensor_names(),
            vec![
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.sinks",
                "model.layers.1.self_attn.sinks",
            ]
        );
        assert_eq!(
            manifest.overridden_tensor_names,
            vec![
                "model.layers.0.self_attn.sinks".to_string(),
                "model.layers.1.self_attn.sinks".to_string()
            ]
        );
    }

    #[test]
    fn header_manifest_tensor_names_feed_upload_manifest_and_report() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(
            &dir.path().join("model.safetensors"),
            &[
                ("model.embed_tokens.weight", &[2, 4], DType::F16, &[0u8; 16]),
                (
                    "model.layers.0.self_attn.q_proj.weight",
                    &[2],
                    DType::F32,
                    &[1u8; 8],
                ),
                (
                    "model.layers.23.mlp.experts.down_proj_scales",
                    &[4],
                    DType::U8,
                    &[2u8; 4],
                ),
                ("model.norm.weight", &[4], DType::F32, &[3u8; 16]),
                ("lm_head.weight", &[2, 4], DType::F16, &[4u8; 16]),
            ],
        );
        let manifest = SafetensorHeaderManifest::discover(dir.path()).unwrap();
        let device_map =
            crate::DeviceMap::parse("split:0-11@0,12-23@1", 24, crate::DeviceId(0)).unwrap();
        let plan = crate::ShardedModelPlan::from_device_map(device_map, 24).unwrap();
        let upload_manifest = plan
            .upload_manifest_for_tensor_names(
                manifest.tensor_names(),
                crate::UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap();
        let report = plan
            .split_allocation_report(&upload_manifest, &plan.kv_cache_plan())
            .unwrap();

        assert!(manifest.has_lm_head_weight());
        assert!(report
            .shard_for_device(crate::DeviceId(0))
            .unwrap()
            .required_tensor_names
            .contains(&"model.embed_tokens.weight".to_string()));
        assert!(report
            .shard_for_device(crate::DeviceId(1))
            .unwrap()
            .host_u8_tensor_names
            .contains(&"model.layers.23.mlp.experts.down_proj_scales".to_string()));
    }

    #[test]
    fn header_manifest_absent_lm_head_allows_tied_fallback_marker() {
        let dir = tempfile::tempdir().unwrap();
        write_safetensors(
            &dir.path().join("model.safetensors"),
            &[
                ("model.embed_tokens.weight", &[2, 4], DType::F16, &[0u8; 16]),
                ("model.norm.weight", &[4], DType::F32, &[1u8; 16]),
            ],
        );
        let manifest = SafetensorHeaderManifest::discover(dir.path()).unwrap();
        let device_map =
            crate::DeviceMap::parse("split:0-11@0,12-23@1", 24, crate::DeviceId(0)).unwrap();
        let plan = crate::ShardedModelPlan::from_device_map(device_map, 24).unwrap();
        let upload_manifest = plan
            .upload_manifest_for_tensor_names(
                manifest.tensor_names(),
                crate::UploadManifestOptions {
                    tie_word_embeddings: true,
                },
            )
            .unwrap();

        let final_shard = upload_manifest
            .shard_for_device(crate::DeviceId(1))
            .unwrap();

        assert!(!manifest.has_lm_head_weight());
        assert!(final_shard
            .deferred_or_late_gpu_allocations
            .contains(&crate::LateAllocationKind::TiedLmHeadFallback));
    }
}
