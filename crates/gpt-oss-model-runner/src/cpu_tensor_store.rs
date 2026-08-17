#![allow(unsafe_code)]
//! Memory-mapped SafeTensors storage for the native CPU runner.
//!
//! Unlike `ModelWeights`, this store never uploads or copies complete tensors.
//! It owns read-only mappings of immutable Hugging Face snapshot shards and
//! returns borrowed, dtype-checked views into those mappings.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bytemuck::try_cast_slice;
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};

use gpt_oss_core::error::{LLMError, Result};

use crate::model_loader::dtype::DType;
use crate::model_loader::shard_catalog::{
    advice_telemetry, madvise_dontneed, process_source_memory_sample, AdviceTelemetry,
    ProcessSourceMemorySample,
};

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

/// Read-only, memory-mapped collection of immutable SafeTensors shards.
///
/// The snapshot directory is an operational immutability boundary: no actor
/// may replace or mutate a shard for the lifetime of this store. A descriptor
/// opened read-only does not by itself make an externally mutable mmap safe.
pub struct CpuTensorStore {
    snapshot_dir: PathBuf,
    shard_paths: Vec<PathBuf>,
    shard_files: Vec<File>,
    shard_identities: Vec<CpuTensorFileIdentity>,
    shards: Vec<Mmap>,
    tensors: HashMap<String, TensorEntry>,
}

#[derive(Debug, Clone)]
struct CpuTensorFileIdentity {
    file_name_bytes: Vec<u8>,
    device: u64,
    inode: u64,
    file_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CpuTensorMappingReleaseTelemetry {
    pub shard_index: usize,
    pub source_file_name_bytes: Vec<u8>,
    pub source_device: u64,
    pub source_inode: u64,
    pub source_file_bytes: u64,
    pub mapping_address: usize,
    pub mapping_bytes: u64,
    pub pre_release: ProcessSourceMemorySample,
    pub post_release: ProcessSourceMemorySample,
    pub mmap_advice: AdviceTelemetry,
    pub file_advice: AdviceTelemetry,
    pub unmap_close_duration_micros: u128,
    pub mapping_removed: bool,
    pub fd_closed: bool,
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
        let mut shard_files = Vec::with_capacity(shard_paths.len());
        let mut shard_identities = Vec::with_capacity(shard_paths.len());
        let mut tensors = HashMap::new();
        for (shard_index, path) in shard_paths.iter().enumerate() {
            let file = File::open(path)?;
            let metadata = file.metadata()?;
            let file_name_bytes = path
                .file_name()
                .map(OsStrExt::as_bytes)
                .unwrap_or_default()
                .to_vec();
            // SAFETY: CpuTensorStore's operational contract requires the
            // checkpoint snapshot to remain immutable for the store's entire
            // lifetime. Opening the descriptor read-only is not sufficient by
            // itself: callers must not replace or mutate these snapshot files
            // through another descriptor while any returned mapping exists.
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
            shard_identities.push(CpuTensorFileIdentity {
                file_name_bytes,
                device: metadata.dev(),
                inode: metadata.ino(),
                file_bytes: metadata.len(),
            });
            shard_files.push(file);
            shards.push(mapping);
        }

        Ok(Self {
            snapshot_dir: snapshot_dir.to_path_buf(),
            shard_paths,
            shard_files,
            shard_identities,
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
            shard_path: &self.shard_paths[entry.shard],
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Release every immutable checkpoint mapping in the frozen R2 order.
    ///
    /// The original read-only descriptor remains open until after its mapping
    /// is advised and unmapped. Advice failure is retained as telemetry; the
    /// mapping and descriptor are still released deterministically.
    pub fn release_with_advice(self) -> Vec<CpuTensorMappingReleaseTelemetry> {
        let Self {
            shard_paths: _,
            shard_files,
            shard_identities,
            shards,
            ..
        } = self;
        shard_identities
            .into_iter()
            .zip(shards)
            .zip(shard_files)
            .enumerate()
            .map(|(shard_index, ((identity, mapping), file))| {
                let mapping_address = mapping.as_ptr() as usize;
                let mapping_bytes = mapping.len() as u64;
                let pre_release = process_source_memory_sample(identity.inode);
                let started = Instant::now();
                let mmap_advice_result = madvise_dontneed(&mapping);
                let mmap_advice = advice_telemetry(
                    "madv_dontneed",
                    0,
                    mapping_bytes,
                    mmap_advice_result.as_ref().err(),
                );
                drop(mapping);
                let file_advice_result = gpt_oss_cpu_kernels::posix_fadvise_dontneed(&file, 0, 0);
                let file_advice = advice_telemetry(
                    "posix_fadv_dontneed",
                    0,
                    identity.file_bytes,
                    file_advice_result.as_ref().err(),
                );
                drop(file);
                let post_release = process_source_memory_sample(identity.inode);
                CpuTensorMappingReleaseTelemetry {
                    shard_index,
                    source_file_name_bytes: identity.file_name_bytes,
                    source_device: identity.device,
                    source_inode: identity.inode,
                    source_file_bytes: identity.file_bytes,
                    mapping_address,
                    mapping_bytes,
                    pre_release,
                    post_release,
                    mmap_advice,
                    file_advice,
                    unmap_close_duration_micros: started.elapsed().as_micros(),
                    mapping_removed: true,
                    fd_closed: true,
                }
            })
            .collect()
    }
}

/// Map a cataloged immutable checkpoint shard after revalidating its identity.
///
/// This is deliberately not a generic read-only-file mapping wrapper. The
/// catalog caller must enforce the stronger operational invariant that the
/// checkpoint file cannot be mutated or replaced by any actor for the entire
/// callback-scoped mapping lifetime. Read-only access through this descriptor
/// alone does not satisfy `memmap2`'s external-mutation safety precondition.
pub(crate) fn map_cataloged_immutable_shard(
    file: &File,
    path: &Path,
    expected_length: u64,
    expected_device: u64,
    expected_inode: u64,
) -> Result<Mmap> {
    let metadata = file.metadata()?;
    if metadata.len() != expected_length
        || metadata_device(&metadata) != expected_device
        || metadata_inode(&metadata) != expected_inode
    {
        return Err(LLMError::ModelError(format!(
            "cataloged shard {} changed before mmap",
            path.display()
        )));
    }
    // SAFETY: In addition to the identity check above, this narrowly scoped
    // caller requires the checkpoint shard to remain externally immutable for
    // the returned mapping lifetime. The catalog prevents the mapping borrow
    // from escaping its callback and revalidates path/header identity before
    // reaching this function.
    unsafe { MmapOptions::new().map(file) }.map_err(|error| {
        LLMError::ModelError(format!("failed to mmap {}: {error}", path.display()))
    })
}

#[cfg(unix)]
fn metadata_device(metadata: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    metadata.dev()
}

#[cfg(not(unix))]
fn metadata_device(_metadata: &std::fs::Metadata) -> u64 {
    0
}

#[cfg(unix)]
fn metadata_inode(metadata: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    metadata.ino()
}

#[cfg(not(unix))]
fn metadata_inode(_metadata: &std::fs::Metadata) -> u64 {
    0
}

pub struct CpuTensor<'a> {
    name: &'a str,
    dtype: DType,
    shape: &'a [usize],
    bytes: &'a [u8],
    shard_path: &'a Path,
}

impl<'a> CpuTensor<'a> {
    pub fn name(&self) -> &str {
        self.name
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub fn shard_path(&self) -> &'a Path {
        self.shard_path
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
        let releases = store.release_with_advice();
        assert_eq!(releases.len(), 1);
        assert!(releases[0].mapping_removed);
        assert!(releases[0].fd_closed);
        assert_eq!(releases[0].post_release.source_inode_mapping_count, 0);
        assert_eq!(releases[0].post_release.source_inode_pss_bytes, 0);
        assert_eq!(releases[0].mmap_advice.kind, "madv_dontneed");
        assert_eq!(releases[0].file_advice.kind, "posix_fadv_dontneed");
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
