//! File-backed R2 source-release handshake shared by comparison children and
//! their supervisor. Every marker is create-new and nonce-bound inside the
//! task-unique run root.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const RELEASE_READY_SCHEMA: &str = "gpt-oss-rs.r2-release-ready/v1";
pub const RELEASE_CONTINUE_SCHEMA: &str = "gpt-oss-rs.r2-release-continue/v1";
pub const RELEASE_WAIT_TIMEOUT: Duration = Duration::from_secs(2 * 60 * 60);

#[derive(Debug, Clone)]
pub struct ChildReleaseHandshake {
    pub root: PathBuf,
    pub nonce: String,
    pub cell: String,
    pub constructor: String,
    pub expected_releases: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReleaseProof {
    pub release_report_count: usize,
    pub source_mapping_count_after_release: usize,
    pub source_mapping_pss_bytes_after_release: u64,
    pub source_payload_fds_after_release: usize,
    pub mappings_removed: bool,
    pub descriptors_closed: bool,
    pub capacity_one_mapping_high_water: Option<usize>,
}

impl ReleaseProof {
    pub fn validate(&self, constructor: &str) -> Result<()> {
        if self.release_report_count == 0
            || self.source_mapping_count_after_release != 0
            || self.source_mapping_pss_bytes_after_release != 0
            || self.source_payload_fds_after_release != 0
            || !self.mappings_removed
            || !self.descriptors_closed
        {
            bail!("source-release proof is incomplete");
        }
        match constructor {
            "monolithic-control" if self.capacity_one_mapping_high_water.is_some() => {
                bail!("monolithic release claimed a capacity-one mapping high-water")
            }
            "capacity-one" if self.capacity_one_mapping_high_water != Some(1) => {
                bail!("capacity-one mapping high-water is not exactly one")
            }
            "monolithic-control" | "capacity-one" => Ok(()),
            _ => bail!("release marker has an unsupported constructor"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReleaseReadyMarker {
    pub schema: String,
    pub captured_unix_ms: u128,
    pub nonce: String,
    pub cell: String,
    pub constructor: String,
    pub ordinal: usize,
    pub expected_releases: usize,
    pub r2_policy_sha256: String,
    pub proof: ReleaseProof,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReleaseContinueMarker {
    pub schema: String,
    pub captured_unix_ms: u128,
    pub nonce: String,
    pub cell: String,
    pub constructor: String,
    pub ordinal: usize,
    pub ready_sha256: String,
}

pub fn child_release_handshake(
    config: &ChildReleaseHandshake,
    ordinal: usize,
    r2_policy_sha256: &str,
    proof: ReleaseProof,
) -> Result<ReleaseReadyMarker> {
    validate_config(config)?;
    if ordinal >= config.expected_releases {
        bail!("release ordinal exceeds the declared handshake count");
    }
    proof.validate(&config.constructor)?;
    let marker = ReleaseReadyMarker {
        schema: RELEASE_READY_SCHEMA.into(),
        captured_unix_ms: now_unix_ms()?,
        nonce: config.nonce.clone(),
        cell: config.cell.clone(),
        constructor: config.constructor.clone(),
        ordinal,
        expected_releases: config.expected_releases,
        r2_policy_sha256: r2_policy_sha256.into(),
        proof,
    };
    let ready_path = ready_path(&config.root, ordinal);
    let ready_bytes = write_json_new(&ready_path, &marker)?;
    let ready_sha256 = sha256_bytes(&ready_bytes);
    let continue_path = continue_path(&config.root, ordinal);
    let started = Instant::now();
    loop {
        if started.elapsed() > RELEASE_WAIT_TIMEOUT {
            bail!("release handshake timed out waiting for supervisor continuation");
        }
        match fs::symlink_metadata(&continue_path) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_file() {
                    bail!("release continuation is not a regular non-symlink file");
                }
                let continuation: ReleaseContinueMarker =
                    serde_json::from_slice(&fs::read(&continue_path)?)?;
                validate_continuation(&continuation, &marker, &ready_sha256)?;
                return Ok(marker);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                thread::sleep(Duration::from_millis(100));
            }
            Err(error) => return Err(error.into()),
        }
    }
}

pub fn read_ready_marker(path: &Path) -> Result<(ReleaseReadyMarker, Vec<u8>)> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        bail!("release-ready marker is not a regular non-symlink file");
    }
    let bytes = fs::read(path)?;
    let marker = serde_json::from_slice(&bytes)?;
    Ok((marker, bytes))
}

pub fn validate_ready_marker(
    marker: &ReleaseReadyMarker,
    nonce: &str,
    cell: &str,
    constructor: &str,
    ordinal: usize,
    expected_releases: usize,
    r2_policy_sha256: &str,
) -> Result<()> {
    if marker.schema != RELEASE_READY_SCHEMA
        || marker.nonce != nonce
        || marker.cell != cell
        || marker.constructor != constructor
        || marker.ordinal != ordinal
        || marker.expected_releases != expected_releases
        || marker.r2_policy_sha256 != r2_policy_sha256
    {
        bail!("release-ready marker identity is invalid");
    }
    marker.proof.validate(constructor)
}

pub fn write_continue_marker(
    root: &Path,
    marker: &ReleaseReadyMarker,
    ready_bytes: &[u8],
) -> Result<ReleaseContinueMarker> {
    let continuation = ReleaseContinueMarker {
        schema: RELEASE_CONTINUE_SCHEMA.into(),
        captured_unix_ms: now_unix_ms()?,
        nonce: marker.nonce.clone(),
        cell: marker.cell.clone(),
        constructor: marker.constructor.clone(),
        ordinal: marker.ordinal,
        ready_sha256: sha256_bytes(ready_bytes),
    };
    write_json_new(&continue_path(root, marker.ordinal), &continuation)?;
    Ok(continuation)
}

pub fn ready_path(root: &Path, ordinal: usize) -> PathBuf {
    root.join(format!("release-{ordinal}.ready.json"))
}

pub fn continue_path(root: &Path, ordinal: usize) -> PathBuf {
    root.join(format!("release-{ordinal}.continue.json"))
}

fn validate_config(config: &ChildReleaseHandshake) -> Result<()> {
    if config.root.is_symlink() || !config.root.is_dir() {
        bail!("release handshake root must be an existing non-symlink directory");
    }
    if config.nonce.len() != 64
        || !config.nonce.bytes().all(|byte| byte.is_ascii_hexdigit())
        || config.cell.is_empty()
        || config.expected_releases == 0
    {
        bail!("release handshake identity is invalid");
    }
    Ok(())
}

fn validate_continuation(
    continuation: &ReleaseContinueMarker,
    ready: &ReleaseReadyMarker,
    ready_sha256: &str,
) -> Result<()> {
    if continuation.schema != RELEASE_CONTINUE_SCHEMA
        || continuation.nonce != ready.nonce
        || continuation.cell != ready.cell
        || continuation.constructor != ready.constructor
        || continuation.ordinal != ready.ordinal
        || continuation.ready_sha256 != ready_sha256
    {
        bail!("release continuation identity is invalid");
    }
    Ok(())
}

fn write_json_new(path: &Path, value: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    let parent = path.parent().context("release marker has no parent")?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("release marker has a non-UTF-8 file name")?;
    let temporary = parent.join(format!(
        ".{file_name}.tmp-{}-{}",
        std::process::id(),
        now_unix_ms()?
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .with_context(|| format!("create release marker temporary {}", temporary.display()))?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    drop(file);
    fs::hard_link(&temporary, path)
        .with_context(|| format!("publish release marker {}", path.display()))?;
    File::open(parent)?.sync_all()?;
    fs::remove_file(&temporary)?;
    File::open(parent)?.sync_all()?;
    Ok(bytes)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn now_unix_ms() -> Result<u128> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock precedes UNIX epoch")?
        .as_millis())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proof(constructor: &str) -> ReleaseProof {
        ReleaseProof {
            release_report_count: 2,
            source_mapping_count_after_release: 0,
            source_mapping_pss_bytes_after_release: 0,
            source_payload_fds_after_release: 0,
            mappings_removed: true,
            descriptors_closed: true,
            capacity_one_mapping_high_water: (constructor == "capacity-one").then_some(1),
        }
    }

    fn marker() -> ReleaseReadyMarker {
        ReleaseReadyMarker {
            schema: RELEASE_READY_SCHEMA.into(),
            captured_unix_ms: 1,
            nonce: "a".repeat(64),
            cell: "cold-capacity-one".into(),
            constructor: "capacity-one".into(),
            ordinal: 0,
            expected_releases: 1,
            r2_policy_sha256: "b".repeat(64),
            proof: proof("capacity-one"),
        }
    }

    #[test]
    fn proof_is_fail_closed_for_every_release_boundary() {
        assert!(proof("monolithic-control")
            .validate("monolithic-control")
            .is_ok());
        assert!(proof("capacity-one").validate("capacity-one").is_ok());
        let mut invalid = proof("capacity-one");
        invalid.source_mapping_count_after_release = 1;
        assert!(invalid.validate("capacity-one").is_err());
        let mut invalid = proof("capacity-one");
        invalid.source_mapping_pss_bytes_after_release = 1;
        assert!(invalid.validate("capacity-one").is_err());
        let mut invalid = proof("capacity-one");
        invalid.source_payload_fds_after_release = 1;
        assert!(invalid.validate("capacity-one").is_err());
        let mut invalid = proof("capacity-one");
        invalid.mappings_removed = false;
        assert!(invalid.validate("capacity-one").is_err());
        let mut invalid = proof("capacity-one");
        invalid.descriptors_closed = false;
        assert!(invalid.validate("capacity-one").is_err());
        let mut invalid = proof("capacity-one");
        invalid.capacity_one_mapping_high_water = Some(2);
        assert!(invalid.validate("capacity-one").is_err());
    }

    #[test]
    fn ready_identity_rejects_stale_nonce_cell_and_ordinal() {
        let marker = marker();
        assert!(validate_ready_marker(
            &marker,
            &"a".repeat(64),
            "cold-capacity-one",
            "capacity-one",
            0,
            1,
            &"b".repeat(64),
        )
        .is_ok());
        assert!(validate_ready_marker(
            &marker,
            &"c".repeat(64),
            "cold-capacity-one",
            "capacity-one",
            0,
            1,
            &"b".repeat(64),
        )
        .is_err());
        assert!(validate_ready_marker(
            &marker,
            &"a".repeat(64),
            "warm-capacity-one",
            "capacity-one",
            1,
            2,
            &"b".repeat(64),
        )
        .is_err());
    }

    #[test]
    fn continuation_is_hash_bound_and_create_new() {
        let root = std::env::temp_dir().join(format!(
            "gpt-oss-r2-release-{}-{}",
            std::process::id(),
            now_unix_ms().unwrap()
        ));
        fs::create_dir(&root).unwrap();
        let marker = marker();
        let ready_bytes = serde_json::to_vec(&marker).unwrap();
        let continuation = write_continue_marker(&root, &marker, &ready_bytes).unwrap();
        assert!(validate_continuation(&continuation, &marker, &sha256_bytes(&ready_bytes)).is_ok());
        assert!(validate_continuation(&continuation, &marker, &"f".repeat(64)).is_err());
        assert!(write_continue_marker(&root, &marker, &ready_bytes).is_err());
        fs::remove_dir_all(&root).unwrap();
    }
}
