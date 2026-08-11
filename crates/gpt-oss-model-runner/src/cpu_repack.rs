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
use gpt_oss_cpu_kernels::{
    mxfp4_adjacent_to_split, Mxfp4Block, Mxfp4MatrixView, Mxfp4WeightLayout,
};

use crate::cpu_tensor_store::{CpuTensor, CpuTensorStore};

const MAGIC: &[u8; 8] = b"GOSSMX4\0";
pub const REPACK_FORMAT_VERSION: u32 = 1;
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

    fn cache_key(&self, tensor_name: &str, shape: &[usize], layout: Mxfp4WeightLayout) -> String {
        let mut digest = Sha256::new();
        digest.update(b"gpt-oss-rs-mxfp4-repack");
        digest.update(REPACK_FORMAT_VERSION.to_le_bytes());
        digest.update(layout.as_str().as_bytes());
        digest.update(layout.identifier().to_le_bytes());
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
    layout_identifier: String,
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
        layout: Mxfp4WeightLayout,
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
            layout_version: layout.identifier(),
            layout_identifier: layout.as_str().to_string(),
            tensor_name: tensor_name.to_string(),
            model_revision: self.identity.model_revision.clone(),
            source_hashes: self.identity.source_hashes.clone(),
            shape: scale_shape.to_vec(),
            records,
        };
        let key = self.identity.cache_key(tensor_name, scale_shape, layout);
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
        let write_result = write_repack(
            &temporary,
            &header,
            block_bytes,
            scale_bytes,
            [scale_shape[0], scale_shape[1], scale_shape[2]],
            layout,
        );
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
    layout: Mxfp4WeightLayout,
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
            layout: expected_layout(expected)?,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn shape(&self) -> [usize; 3] {
        self.shape
    }

    pub const fn layout(&self) -> Mxfp4WeightLayout {
        self.layout
    }

    pub fn expert_view(&self, expert: usize) -> Result<Mxfp4MatrixView<'_>> {
        if expert >= self.shape[0] {
            return Err(LLMError::ModelError(format!(
                "MXFP4 expert index {expert} exceeds {:?}",
                self.shape
            )));
        }
        let expert_bytes = self.shape[1] * self.shape[2] * RECORD_BYTES;
        let start = self.data_start + expert * expert_bytes;
        Mxfp4MatrixView::new(
            &self.mapping[start..start + expert_bytes],
            self.shape[1],
            self.shape[2],
            self.layout,
        )
        .map_err(|error| LLMError::ModelError(error.to_string()))
    }

    /// Return one canonical owned block from either cache layout.
    pub fn owned_block(&self, expert: usize, output: usize, block: usize) -> Result<Mxfp4Block> {
        self.expert_view(expert)?
            .block(output, block)
            .map_err(|error| LLMError::ModelError(error.to_string()))
    }
}

fn expected_layout(header: &RepackHeader) -> Result<Mxfp4WeightLayout> {
    match (header.layout_identifier.as_str(), header.layout_version) {
        ("CanonicalAdjacentV1", 1) => Ok(Mxfp4WeightLayout::CanonicalAdjacentV1),
        ("InterleavedSplitX8V2", 2) => Ok(Mxfp4WeightLayout::InterleavedSplitX8V2),
        _ => Err(LLMError::ModelError(format!(
            "unsupported MXFP4 repack layout {} version {}",
            header.layout_identifier, header.layout_version
        ))),
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

fn write_repack(
    path: &Path,
    header: &RepackHeader,
    blocks: &[u8],
    scales: &[u8],
    shape: [usize; 3],
    layout: Mxfp4WeightLayout,
) -> Result<()> {
    let header_bytes = serde_json::to_vec(header).map_err(|error| {
        LLMError::ModelError(format!("failed to serialize MXFP4 repack header: {error}"))
    })?;
    let mut file = File::create(path)?;
    file.write_all(MAGIC)?;
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    file.write_all(&header_bytes)?;
    match layout {
        Mxfp4WeightLayout::CanonicalAdjacentV1 => {
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
        }
        Mxfp4WeightLayout::InterleavedSplitX8V2 => {
            write_x8_payload(&mut file, blocks, scales, shape)?;
        }
    }
    file.sync_all()?;
    Ok(())
}

fn write_x8_payload(
    file: &mut File,
    blocks: &[u8],
    scales: &[u8],
    [experts, rows, blocks_per_row]: [usize; 3],
) -> Result<()> {
    let mut output = Vec::with_capacity(REPACK_BATCH_RECORDS * RECORD_BYTES);
    let source_record =
        |expert: usize, row: usize, block: usize| (expert * rows + row) * blocks_per_row + block;
    for expert in 0..experts {
        for group in 0..rows / 8 {
            for block in 0..blocks_per_row {
                for lane in 0..8 {
                    output.push(scales[source_record(expert, group * 8 + lane, block)]);
                }
                let split = std::array::from_fn::<_, 8, _>(|lane| {
                    let record = source_record(expert, group * 8 + lane, block);
                    let adjacent: [u8; 16] = blocks[record * 16..record * 16 + 16]
                        .try_into()
                        .expect("validated source record");
                    mxfp4_adjacent_to_split(adjacent)
                });
                for chunk in 0..2 {
                    for row in &split {
                        output.extend_from_slice(&row[chunk * 8..chunk * 8 + 8]);
                    }
                }
                flush_repack_batch(file, &mut output)?;
            }
        }
        for row in rows / 8 * 8..rows {
            for block in 0..blocks_per_row {
                let record = source_record(expert, row, block);
                output.push(scales[record]);
                output.extend_from_slice(&blocks[record * 16..record * 16 + 16]);
                flush_repack_batch(file, &mut output)?;
            }
        }
    }
    if !output.is_empty() {
        file.write_all(&output)?;
    }
    Ok(())
}

fn flush_repack_batch(file: &mut File, output: &mut Vec<u8>) -> Result<()> {
    if output.len() >= REPACK_BATCH_RECORDS * RECORD_BYTES {
        file.write_all(output)?;
        output.clear();
    }
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

    fn generated_block(record: usize) -> Mxfp4Block {
        Mxfp4Block {
            scale: (record as u8).wrapping_mul(17).wrapping_add(1),
            packed: std::array::from_fn(|byte| {
                let low = (record + byte * 2) as u8 & 0x0f;
                let high = (record + byte * 2 + 1) as u8 & 0x0f;
                low | (high << 4)
            }),
        }
    }

    fn write_shaped_shard(path: &Path, experts: usize, rows: usize, blocks_per_row: usize) {
        let records = experts * rows * blocks_per_row;
        let mut blocks = Vec::with_capacity(records * 16);
        let mut scales = Vec::with_capacity(records);
        for record in 0..records {
            let generated = generated_block(record);
            blocks.extend_from_slice(&generated.packed);
            scales.push(generated.scale);
        }
        let block_bytes = blocks.len();
        let header = serde_json::json!({
            "blocks": {
                "dtype":"U8",
                "shape":[experts, rows, blocks_per_row, 16],
                "data_offsets":[0, block_bytes]
            },
            "scales": {
                "dtype":"U8",
                "shape":[experts, rows, blocks_per_row],
                "data_offsets":[block_bytes, block_bytes + scales.len()]
            }
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

    fn shaped_fixture(
        experts: usize,
        rows: usize,
        blocks: usize,
    ) -> (tempfile::TempDir, CpuTensorStore, CpuRepackCache) {
        let temp = tempdir().unwrap();
        std::fs::write(temp.path().join("config.json"), b"{}").unwrap();
        write_shaped_shard(
            &temp.path().join("model.safetensors"),
            experts,
            rows,
            blocks,
        );
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
                Mxfp4WeightLayout::CanonicalAdjacentV1,
            )
            .unwrap();
        assert_eq!(repacked.shape(), [1, 2, 1]);
        assert_eq!(
            repacked.owned_block(0, 0, 0).unwrap(),
            Mxfp4Block {
                scale: 126,
                packed: [0x21; 16]
            }
        );
    }

    #[test]
    fn corrupt_and_interrupted_cache_files_are_rebuilt() {
        let (_temp, store, cache) = fixture();
        let first = cache
            .open_or_create(
                "fixture",
                &store.tensor("blocks").unwrap(),
                &store.tensor("scales").unwrap(),
                Mxfp4WeightLayout::CanonicalAdjacentV1,
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
                Mxfp4WeightLayout::CanonicalAdjacentV1,
            )
            .unwrap();
        assert_eq!(rebuilt.owned_block(0, 1, 0).unwrap().scale, 127);
    }

    #[test]
    fn format_version_is_part_of_validated_header() {
        let (_temp, store, cache) = fixture();
        let repacked = cache
            .open_or_create(
                "fixture",
                &store.tensor("blocks").unwrap(),
                &store.tensor("scales").unwrap(),
                Mxfp4WeightLayout::CanonicalAdjacentV1,
            )
            .unwrap();
        let stale = RepackHeader {
            format_version: REPACK_FORMAT_VERSION + 1,
            layout_version: Mxfp4WeightLayout::CanonicalAdjacentV1.identifier(),
            layout_identifier: Mxfp4WeightLayout::CanonicalAdjacentV1.as_str().into(),
            tensor_name: "fixture".into(),
            model_revision: cache.identity.model_revision.clone(),
            source_hashes: cache.identity.source_hashes.clone(),
            shape: vec![1, 2, 1],
            records: 2,
        };
        assert!(RepackedMxfp4::open(repacked.path(), &stale).is_err());
    }

    #[test]
    fn x8_layout_round_trips_multiple_experts_blocks_and_tails() {
        for tail in 1..=7 {
            let rows = 16 + tail;
            let (_temp, store, cache) = shaped_fixture(2, rows, 3);
            let repacked = cache
                .open_or_create(
                    "fixture",
                    &store.tensor("blocks").unwrap(),
                    &store.tensor("scales").unwrap(),
                    Mxfp4WeightLayout::InterleavedSplitX8V2,
                )
                .unwrap();
            assert_eq!(repacked.shape(), [2, rows, 3]);
            assert_eq!(
                repacked.mapping.len() - repacked.data_start,
                2 * rows * 3 * RECORD_BYTES
            );
            for expert in 0..2 {
                for row in 0..rows {
                    for block in 0..3 {
                        let record = (expert * rows + row) * 3 + block;
                        assert_eq!(
                            repacked.owned_block(expert, row, block).unwrap(),
                            generated_block(record),
                            "tail={tail}, expert={expert}, row={row}, block={block}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn layout_identifiers_have_separate_cache_keys_and_equal_payload_sizes() {
        let (_temp, store, cache) = shaped_fixture(2, 11, 4);
        let canonical = cache
            .open_or_create(
                "fixture",
                &store.tensor("blocks").unwrap(),
                &store.tensor("scales").unwrap(),
                Mxfp4WeightLayout::CanonicalAdjacentV1,
            )
            .unwrap();
        let canonical_path = canonical.path().to_path_buf();
        let canonical_len = canonical.mapping.len() - canonical.data_start;
        drop(canonical);
        let x8 = cache
            .open_or_create(
                "fixture",
                &store.tensor("blocks").unwrap(),
                &store.tensor("scales").unwrap(),
                Mxfp4WeightLayout::InterleavedSplitX8V2,
            )
            .unwrap();
        assert_ne!(canonical_path, x8.path());
        assert!(canonical_path.is_file());
        assert_eq!(canonical_len, x8.mapping.len() - x8.data_start);
        assert_eq!(canonical_len, 2 * 11 * 4 * RECORD_BYTES);
    }

    #[test]
    fn published_x8_cache_reopens_atomically() {
        let (_temp, store, cache) = shaped_fixture(2, 15, 2);
        let first = cache
            .open_or_create(
                "fixture",
                &store.tensor("blocks").unwrap(),
                &store.tensor("scales").unwrap(),
                Mxfp4WeightLayout::InterleavedSplitX8V2,
            )
            .unwrap();
        let path = first.path().to_path_buf();
        drop(first);
        let reopened = cache
            .open_or_create(
                "fixture",
                &store.tensor("blocks").unwrap(),
                &store.tensor("scales").unwrap(),
                Mxfp4WeightLayout::InterleavedSplitX8V2,
            )
            .unwrap();
        assert_eq!(reopened.path(), path);
        assert_eq!(reopened.owned_block(1, 14, 1).unwrap(), generated_block(59));
    }

    #[test]
    fn legacy_v1_header_without_layout_identifier_is_rejected() {
        let (temp, store, cache) = shaped_fixture(1, 9, 1);
        let repacked = cache
            .open_or_create(
                "fixture",
                &store.tensor("blocks").unwrap(),
                &store.tensor("scales").unwrap(),
                Mxfp4WeightLayout::InterleavedSplitX8V2,
            )
            .unwrap();
        let expected = RepackHeader {
            format_version: REPACK_FORMAT_VERSION,
            layout_version: Mxfp4WeightLayout::InterleavedSplitX8V2.identifier(),
            layout_identifier: Mxfp4WeightLayout::InterleavedSplitX8V2.as_str().into(),
            tensor_name: "fixture".into(),
            model_revision: cache.identity.model_revision.clone(),
            source_hashes: cache.identity.source_hashes.clone(),
            shape: vec![1, 9, 1],
            records: 9,
        };
        let old_header = serde_json::to_vec(&serde_json::json!({
            "format_version": 1,
            "layout_version": 1,
            "tensor_name": "fixture",
            "model_revision": cache.identity.model_revision,
            "source_hashes": cache.identity.source_hashes,
            "shape": [1, 9, 1],
            "records": 9
        }))
        .unwrap();
        let stale = temp.path().join("stale-v1.repack");
        let mut file = File::create(&stale).unwrap();
        file.write_all(MAGIC).unwrap();
        file.write_all(&(old_header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&old_header).unwrap();
        file.write_all(&vec![0; 9 * RECORD_BYTES]).unwrap();
        file.sync_all().unwrap();
        drop(repacked);
        assert!(RepackedMxfp4::open(&stale, &expected).is_err());
    }
}
