#![allow(unsafe_code)]
//! Versioned, atomic MXFP4 repack cache for CPU expert projections.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use gpt_oss_core::error::{LLMError, Result};

use crate::cpu_tensor_store::{CpuTensor, CpuTensorStore};

const MAGIC: &[u8; 8] = b"GOSSMX4\0";
pub const REPACK_FORMAT_VERSION: u32 = 1;
pub const REPACK_LAYOUT_VERSION: u32 = 1;
const RECORD_BYTES: usize = 17;
const REPACK_BATCH_RECORDS: usize = 64 * 1024;
const LOCK_WAIT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub model_revision: String,
    pub source_hashes: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct FetchManifest {
    resolved_revision: String,
    files: Vec<FetchManifestFile>,
}

#[derive(Debug, Deserialize)]
struct FetchManifestFile {
    path: String,
    sha256: String,
}

impl SourceIdentity {
    pub fn from_store(store: &CpuTensorStore) -> Result<Self> {
        let manifest_path = store.snapshot_dir().join("gpt-oss-rs-fetch-manifest.json");
        if manifest_path.is_file() {
            let manifest: FetchManifest = serde_json::from_slice(&std::fs::read(&manifest_path)?)
                .map_err(|error| {
                LLMError::ModelError(format!(
                    "invalid fetch manifest {}: {error}",
                    manifest_path.display()
                ))
            })?;
            let mut hashes = manifest
                .files
                .into_iter()
                .filter(|file| file.path.ends_with(".safetensors"))
                .map(|file| format!("{}:{}", file.path, file.sha256))
                .collect::<Vec<_>>();
            hashes.sort();
            if hashes.len() != store.shard_paths().len() {
                return Err(LLMError::ModelError(format!(
                    "fetch manifest contains {} SafeTensors hashes for {} mapped shards",
                    hashes.len(),
                    store.shard_paths().len()
                )));
            }
            return Ok(Self {
                model_revision: manifest.resolved_revision,
                source_hashes: hashes,
            });
        }

        let mut hashes = Vec::with_capacity(store.shard_paths().len());
        for path in store.shard_paths() {
            hashes.push(format!(
                "{}:{}",
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("shard"),
                sha256_file(path)?
            ));
        }
        Ok(Self {
            model_revision: "local-unmanifested".into(),
            source_hashes: hashes,
        })
    }

    fn cache_key(&self, tensor_name: &str, shape: &[usize]) -> String {
        let mut digest = Sha256::new();
        digest.update(b"gpt-oss-rs-mxfp4-repack");
        digest.update(REPACK_FORMAT_VERSION.to_le_bytes());
        digest.update(REPACK_LAYOUT_VERSION.to_le_bytes());
        digest.update(self.model_revision.as_bytes());
        for hash in &self.source_hashes {
            digest.update(hash.as_bytes());
        }
        digest.update(tensor_name.as_bytes());
        for dimension in shape {
            digest.update(dimension.to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct RepackHeader {
    format_version: u32,
    layout_version: u32,
    tensor_name: String,
    model_revision: String,
    source_hashes: Vec<String>,
    shape: Vec<usize>,
    records: usize,
}

pub struct CpuRepackCache {
    root: PathBuf,
    identity: SourceIdentity,
}

impl CpuRepackCache {
    pub fn new(root: impl Into<PathBuf>, identity: SourceIdentity) -> Self {
        Self {
            root: root.into(),
            identity,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn open_or_create(
        &self,
        tensor_name: &str,
        blocks: &CpuTensor<'_>,
        scales: &CpuTensor<'_>,
    ) -> Result<RepackedMxfp4> {
        let block_shape = blocks.shape();
        let scale_shape = scales.shape();
        if block_shape.len() != 4
            || scale_shape.len() != 3
            || block_shape[..3] != scale_shape[..]
            || block_shape[3] != 16
        {
            return Err(LLMError::ModelError(format!(
                "invalid MXFP4 source shapes for {tensor_name}: blocks={block_shape:?}, scales={scale_shape:?}"
            )));
        }
        let block_bytes = blocks.u8()?;
        let scale_bytes = scales.u8()?;
        let records = scale_shape.iter().product::<usize>();
        if block_bytes.len() != records * 16 || scale_bytes.len() != records {
            return Err(LLMError::ModelError(format!(
                "invalid MXFP4 source byte counts for {tensor_name}"
            )));
        }

        let header = RepackHeader {
            format_version: REPACK_FORMAT_VERSION,
            layout_version: REPACK_LAYOUT_VERSION,
            tensor_name: tensor_name.to_string(),
            model_revision: self.identity.model_revision.clone(),
            source_hashes: self.identity.source_hashes.clone(),
            shape: scale_shape.to_vec(),
            records,
        };
        let key = self.identity.cache_key(tensor_name, scale_shape);
        let directory = self.root.join("mxfp4").join(key);
        let target = directory.join("weights.repack");

        if let Ok(repacked) = RepackedMxfp4::open(&target, &header) {
            return Ok(repacked);
        }
        std::fs::create_dir_all(&directory)?;
        let lock_path = directory.join("repack.lock");
        let _lock = acquire_lock(&lock_path, &target, &header)?;
        if let Ok(repacked) = RepackedMxfp4::open(&target, &header) {
            return Ok(repacked);
        }

        let temporary = directory.join(format!(".weights.repack.{}.tmp", std::process::id()));
        let write_result = write_repack(&temporary, &header, block_bytes, scale_bytes);
        if let Err(error) = write_result {
            let _ = std::fs::remove_file(&temporary);
            return Err(error);
        }
        std::fs::rename(&temporary, &target)?;
        sync_directory(&directory)?;
        RepackedMxfp4::open(&target, &header)
    }
}

pub struct RepackedMxfp4 {
    path: PathBuf,
    mapping: Mmap,
    data_start: usize,
    shape: [usize; 3],
}

impl RepackedMxfp4 {
    fn open(path: &Path, expected: &RepackHeader) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the cache mapping is read-only, retained by this object, and
        // published only after a synced atomic rename. Writers never mutate a
        // published cache file in place.
        let mapping = unsafe { MmapOptions::new().map(&file) }.map_err(|error| {
            LLMError::ModelError(format!("failed to mmap repack {}: {error}", path.display()))
        })?;
        if mapping.len() < MAGIC.len() + 8 || &mapping[..MAGIC.len()] != MAGIC {
            return Err(LLMError::ModelError(format!(
                "invalid MXFP4 repack magic in {}",
                path.display()
            )));
        }
        let header_start = MAGIC.len() + 8;
        let header_len = u64::from_le_bytes(
            mapping[MAGIC.len()..header_start]
                .try_into()
                .map_err(|_| LLMError::ModelError("invalid repack header length".into()))?,
        ) as usize;
        let data_start = header_start
            .checked_add(header_len)
            .ok_or_else(|| LLMError::ModelError("MXFP4 repack header length overflows".into()))?;
        if data_start > mapping.len() {
            return Err(LLMError::ModelError("truncated MXFP4 repack header".into()));
        }
        let actual: RepackHeader = serde_json::from_slice(&mapping[header_start..data_start])
            .map_err(|error| LLMError::ModelError(format!("invalid repack header: {error}")))?;
        if &actual != expected {
            return Err(LLMError::ModelError(format!(
                "stale MXFP4 repack metadata in {}",
                path.display()
            )));
        }
        let expected_len = data_start
            .checked_add(actual.records * RECORD_BYTES)
            .ok_or_else(|| LLMError::ModelError("MXFP4 repack size overflows".into()))?;
        if mapping.len() != expected_len {
            return Err(LLMError::ModelError(format!(
                "MXFP4 repack {} has {} bytes, expected {expected_len}",
                path.display(),
                mapping.len()
            )));
        }
        Ok(Self {
            path: path.to_path_buf(),
            mapping,
            data_start,
            shape: [actual.shape[0], actual.shape[1], actual.shape[2]],
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn shape(&self) -> [usize; 3] {
        self.shape
    }

    pub fn record(&self, expert: usize, output: usize, block: usize) -> Result<(u8, &[u8; 16])> {
        if expert >= self.shape[0] || output >= self.shape[1] || block >= self.shape[2] {
            return Err(LLMError::ModelError(format!(
                "MXFP4 repack index [{expert}, {output}, {block}] exceeds {:?}",
                self.shape
            )));
        }
        let record = ((expert * self.shape[1] + output) * self.shape[2] + block) * RECORD_BYTES;
        let start = self.data_start + record;
        let packed = self.mapping[start + 1..start + RECORD_BYTES]
            .try_into()
            .map_err(|_| LLMError::ModelError("invalid MXFP4 record".into()))?;
        Ok((self.mapping[start], packed))
    }
}

struct LockGuard {
    path: PathBuf,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn acquire_lock(path: &Path, target: &Path, expected: &RepackHeader) -> Result<LockGuard> {
    let started = Instant::now();
    loop {
        match OpenOptions::new().write(true).create_new(true).open(path) {
            Ok(mut file) => {
                writeln!(file, "pid={}", std::process::id())?;
                file.sync_all()?;
                return Ok(LockGuard {
                    path: path.to_path_buf(),
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if lock_owner_is_gone(path) {
                    std::fs::remove_file(path)?;
                    continue;
                }
                if RepackedMxfp4::open(target, expected).is_ok() {
                    // Another writer completed. Wait for it to release the
                    // lock so this guard cannot delete a lock it does not own.
                    while path.exists() && started.elapsed() < LOCK_WAIT {
                        thread::sleep(Duration::from_millis(25));
                    }
                    continue;
                }
                if started.elapsed() >= LOCK_WAIT {
                    return Err(LLMError::ModelError(format!(
                        "timed out waiting for MXFP4 repack lock {}",
                        path.display()
                    )));
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(error) => return Err(error.into()),
        }
    }
}

fn lock_owner_is_gone(path: &Path) -> bool {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return false;
    };
    let Some(pid) = contents
        .lines()
        .find_map(|line| line.strip_prefix("pid="))
        .and_then(|pid| pid.parse::<u32>().ok())
    else {
        return false;
    };
    !PathBuf::from(format!("/proc/{pid}")).exists()
}

fn write_repack(path: &Path, header: &RepackHeader, blocks: &[u8], scales: &[u8]) -> Result<()> {
    let header_bytes = serde_json::to_vec(header).map_err(|error| {
        LLMError::ModelError(format!("failed to serialize MXFP4 repack header: {error}"))
    })?;
    let mut file = File::create(path)?;
    file.write_all(MAGIC)?;
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    file.write_all(&header_bytes)?;
    for (batch_index, scale_batch) in scales.chunks(REPACK_BATCH_RECORDS).enumerate() {
        let first_record = batch_index * REPACK_BATCH_RECORDS;
        let first_byte = first_record * 16;
        let block_batch = &blocks[first_byte..first_byte + scale_batch.len() * 16];
        let mut interleaved = Vec::with_capacity(scale_batch.len() * RECORD_BYTES);
        for (record, scale) in block_batch.chunks_exact(16).zip(scale_batch) {
            interleaved.push(*scale);
            interleaved.extend_from_slice(record);
        }
        file.write_all(&interleaved)?;
    }
    file.sync_all()?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;
    use crate::cpu_tensor_store::CpuTensorStore;

    fn write_shard(path: &Path) {
        let blocks = [0x21_u8; 32];
        let scales = [126_u8, 127_u8];
        let header = serde_json::json!({
            "blocks": {"dtype":"U8", "shape":[1,2,1,16], "data_offsets":[0,32]},
            "scales": {"dtype":"U8", "shape":[1,2,1], "data_offsets":[32,34]}
        });
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&blocks).unwrap();
        file.write_all(&scales).unwrap();
    }

    fn fixture() -> (tempfile::TempDir, CpuTensorStore, CpuRepackCache) {
        let temp = tempdir().unwrap();
        std::fs::write(temp.path().join("config.json"), b"{}").unwrap();
        write_shard(&temp.path().join("model.safetensors"));
        let store = CpuTensorStore::open(temp.path()).unwrap();
        let identity = SourceIdentity::from_store(&store).unwrap();
        let cache = CpuRepackCache::new(temp.path().join("cache"), identity);
        (temp, store, cache)
    }

    #[test]
    fn repacks_and_reopens_records() {
        let (_temp, store, cache) = fixture();
        let repacked = cache
            .open_or_create(
                "fixture",
                &store.tensor("blocks").unwrap(),
                &store.tensor("scales").unwrap(),
            )
            .unwrap();
        assert_eq!(repacked.shape(), [1, 2, 1]);
        assert_eq!(repacked.record(0, 0, 0).unwrap(), (126, &[0x21; 16]));
    }

    #[test]
    fn corrupt_and_interrupted_cache_files_are_rebuilt() {
        let (_temp, store, cache) = fixture();
        let first = cache
            .open_or_create(
                "fixture",
                &store.tensor("blocks").unwrap(),
                &store.tensor("scales").unwrap(),
            )
            .unwrap();
        let target = first.path().to_path_buf();
        drop(first);
        std::fs::write(&target, b"corrupt").unwrap();
        std::fs::write(target.parent().unwrap().join(".abandoned.tmp"), b"partial").unwrap();
        std::fs::write(
            target.parent().unwrap().join("repack.lock"),
            b"pid=4294967295\n",
        )
        .unwrap();

        let rebuilt = cache
            .open_or_create(
                "fixture",
                &store.tensor("blocks").unwrap(),
                &store.tensor("scales").unwrap(),
            )
            .unwrap();
        assert_eq!(rebuilt.record(0, 1, 0).unwrap().0, 127);
    }

    #[test]
    fn format_version_is_part_of_validated_header() {
        let (_temp, store, cache) = fixture();
        let repacked = cache
            .open_or_create(
                "fixture",
                &store.tensor("blocks").unwrap(),
                &store.tensor("scales").unwrap(),
            )
            .unwrap();
        let stale = RepackHeader {
            format_version: REPACK_FORMAT_VERSION + 1,
            layout_version: REPACK_LAYOUT_VERSION,
            tensor_name: "fixture".into(),
            model_revision: cache.identity.model_revision.clone(),
            source_hashes: cache.identity.source_hashes.clone(),
            shape: vec![1, 2, 1],
            records: 2,
        };
        assert!(RepackedMxfp4::open(repacked.path(), &stale).is_err());
    }
}
