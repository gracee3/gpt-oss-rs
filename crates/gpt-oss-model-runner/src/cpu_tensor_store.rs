#![allow(unsafe_code)]
//! Memory-mapped SafeTensors storage for the native CPU runner.
//!
//! Unlike `ModelWeights`, this store never uploads or copies complete tensors.
//! It owns read-only mappings of immutable Hugging Face snapshot shards and
//! returns borrowed, dtype-checked views into those mappings.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use bytemuck::try_cast_slice;
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};
use serde::Deserialize;

use gpt_oss_core::error::{LLMError, Result};

use crate::model_loader::dtype::DType;

#[derive(Debug, Clone)]
struct TensorEntry {
    shard: usize,
    dtype: DType,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

#[derive(Debug, Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Read-only, memory-mapped collection of SafeTensors shards.
pub struct CpuTensorStore {
    snapshot_dir: PathBuf,
    shard_paths: Vec<PathBuf>,
    shards: Vec<Mmap>,
    tensors: HashMap<String, TensorEntry>,
}

impl CpuTensorStore {
    pub fn open(snapshot_dir: impl AsRef<Path>) -> Result<Self> {
        let snapshot_dir = snapshot_dir.as_ref();
        if !snapshot_dir.join("config.json").is_file() {
            return Err(LLMError::ModelError(format!(
                "CPU snapshot {} has no config.json",
                snapshot_dir.display()
            )));
        }

        let mut shard_paths = std::fs::read_dir(snapshot_dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "safetensors")
            })
            .collect::<Vec<_>>();
        shard_paths.sort();
        if shard_paths.is_empty() {
            return Err(LLMError::ModelError(format!(
                "CPU snapshot {} has no SafeTensors shards",
                snapshot_dir.display()
            )));
        }

        let mut shards = Vec::with_capacity(shard_paths.len());
        let mut tensors = HashMap::new();
        for (shard_index, path) in shard_paths.iter().enumerate() {
            let file = File::open(path)?;
            // SAFETY: mappings are read-only and the store owns each mapping
            // for every returned view's lifetime. Hugging Face snapshot blobs
            // are content-addressed and treated as immutable while loaded.
            let mapping = unsafe { MmapOptions::new().map(&file) }.map_err(|error| {
                LLMError::ModelError(format!("failed to mmap {}: {error}", path.display()))
            })?;
            let entries = parse_shard_header(&mapping, shard_index, path)?;
            for (name, entry) in entries {
                if tensors.insert(name.clone(), entry).is_some() {
                    return Err(LLMError::ModelError(format!(
                        "tensor {name} occurs in more than one SafeTensors shard"
                    )));
                }
            }
            shards.push(mapping);
        }

        Ok(Self {
            snapshot_dir: snapshot_dir.to_path_buf(),
            shard_paths,
            shards,
            tensors,
        })
    }

    pub fn snapshot_dir(&self) -> &Path {
        &self.snapshot_dir
    }

    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.shard_paths
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn tensor(&self, name: &str) -> Result<CpuTensor<'_>> {
        let (stored_name, entry) = self
            .tensors
            .get_key_value(name)
            .ok_or_else(|| LLMError::ModelError(format!("missing CPU checkpoint tensor {name}")))?;
        Ok(CpuTensor {
            name: stored_name,
            dtype: entry.dtype,
            shape: &entry.shape,
            bytes: &self.shards[entry.shard][entry.start..entry.end],
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

pub struct CpuTensor<'a> {
    name: &'a str,
    dtype: DType,
    shape: &'a [usize],
    bytes: &'a [u8],
}

impl<'a> CpuTensor<'a> {
    pub fn name(&self) -> &str {
        self.name
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    pub fn bytes(&self) -> &[u8] {
        self.bytes
    }

    pub fn bf16(&self) -> Result<&[bf16]> {
        self.cast(DType::BF16, "BF16")
    }

    pub fn f16(&self) -> Result<&[f16]> {
        self.cast(DType::F16, "F16")
    }

    pub fn f32(&self) -> Result<&[f32]> {
        self.cast(DType::F32, "F32")
    }

    pub fn u8(&self) -> Result<&[u8]> {
        if self.dtype != DType::U8 {
            return Err(dtype_error(self.name, self.dtype, "U8"));
        }
        Ok(self.bytes)
    }

    fn cast<T: bytemuck::Pod>(&self, expected: DType, label: &str) -> Result<&[T]> {
        if self.dtype != expected {
            return Err(dtype_error(self.name, self.dtype, label));
        }
        try_cast_slice(self.bytes).map_err(|error| {
            LLMError::ModelError(format!(
                "tensor {} cannot be viewed as {label}: {error}",
                self.name
            ))
        })
    }
}

fn dtype_error(name: &str, actual: DType, expected: &str) -> LLMError {
    LLMError::ModelError(format!(
        "tensor {name} has dtype {actual}, expected {expected}"
    ))
}

fn parse_shard_header(
    mapping: &[u8],
    shard: usize,
    path: &Path,
) -> Result<HashMap<String, TensorEntry>> {
    if mapping.len() < 8 {
        return Err(LLMError::ModelError(format!(
            "SafeTensors shard {} is too small",
            path.display()
        )));
    }
    let header_len = u64::from_le_bytes(mapping[..8].try_into().map_err(|_| {
        LLMError::ModelError(format!("invalid SafeTensors header in {}", path.display()))
    })?) as usize;
    let data_start = 8_usize.checked_add(header_len).ok_or_else(|| {
        LLMError::ModelError(format!(
            "SafeTensors header overflows in {}",
            path.display()
        ))
    })?;
    if data_start > mapping.len() {
        return Err(LLMError::ModelError(format!(
            "SafeTensors header exceeds {}",
            path.display()
        )));
    }
    let header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&mapping[8..data_start]).map_err(|error| {
            LLMError::ModelError(format!(
                "invalid SafeTensors header in {}: {error}",
                path.display()
            ))
        })?;

    let mut entries = HashMap::new();
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let tensor: HeaderTensor = serde_json::from_value(value).map_err(|error| {
            LLMError::ModelError(format!(
                "invalid metadata for tensor {name} in {}: {error}",
                path.display()
            ))
        })?;
        let dtype = DType::from_safetensors_str(&tensor.dtype).ok_or_else(|| {
            LLMError::ModelError(format!(
                "unsupported dtype {} for tensor {name}",
                tensor.dtype
            ))
        })?;
        let start = data_start
            .checked_add(tensor.data_offsets[0])
            .ok_or_else(|| LLMError::ModelError(format!("tensor {name} offset overflows")))?;
        let end = data_start
            .checked_add(tensor.data_offsets[1])
            .ok_or_else(|| LLMError::ModelError(format!("tensor {name} offset overflows")))?;
        if start > end || end > mapping.len() {
            return Err(LLMError::ModelError(format!(
                "tensor {name} has invalid byte range [{start}, {end})"
            )));
        }
        let elements = tensor
            .shape
            .iter()
            .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| LLMError::ModelError(format!("tensor {name} shape overflows")))?;
        let expected_bytes = elements
            .checked_mul(dtype.size_of())
            .ok_or_else(|| LLMError::ModelError(format!("tensor {name} byte size overflows")))?;
        if expected_bytes != end - start {
            return Err(LLMError::ModelError(format!(
                "tensor {name} declares {expected_bytes} bytes but stores {}",
                end - start
            )));
        }
        entries.insert(
            name,
            TensorEntry {
                shard,
                dtype,
                shape: tensor.shape,
                start,
                end,
            },
        );
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;

    fn write_tensor(path: &Path, name: &str, dtype: &str, shape: &[usize], data: &[u8]) {
        let header = serde_json::json!({
            (name): {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [0, data.len()]
            }
        });
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(data).unwrap();
    }

    #[test]
    fn maps_tensors_without_copying_payloads() {
        let temp = tempdir().unwrap();
        std::fs::write(temp.path().join("config.json"), b"{}").unwrap();
        let values = [bf16::from_f32(1.0), bf16::from_f32(-2.0)];
        write_tensor(
            &temp.path().join("model.safetensors"),
            "weight",
            "BF16",
            &[2],
            bytemuck::cast_slice(&values),
        );

        let store = CpuTensorStore::open(temp.path()).unwrap();
        let tensor = store.tensor("weight").unwrap();
        assert_eq!(tensor.shape(), &[2]);
        assert_eq!(tensor.bf16().unwrap(), values);
    }

    #[test]
    fn rejects_out_of_bounds_metadata() {
        let temp = tempdir().unwrap();
        std::fs::write(temp.path().join("config.json"), b"{}").unwrap();
        write_tensor(
            &temp.path().join("model.safetensors"),
            "weight",
            "F32",
            &[2],
            &[0; 4],
        );
        assert!(CpuTensorStore::open(temp.path()).is_err());
    }
}
