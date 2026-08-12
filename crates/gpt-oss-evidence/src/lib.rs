#![forbid(unsafe_code)]
//! Stable, privacy-aware evidence records shared by CPU tools and services.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const EVIDENCE_SCHEMA_V1: &str = "gpt-oss-rs.cpu-evidence/v1";
pub const RUNTIME_SNAPSHOT_SCHEMA_V1: &str = "gpt-oss-rs.cpu-runtime/v1";
pub const DIAGNOSTIC_SCHEMA_V1: &str = "gpt-oss-rs.cpu-diagnostic/v1";

static TEMP_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, thiserror::Error)]
pub enum EvidenceError {
    #[error("invalid evidence: {0}")]
    Invalid(String),
    #[error("evidence I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("evidence serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("artifact hash mismatch for {path}: expected {expected}, observed {observed}")]
    HashMismatch {
        path: PathBuf,
        expected: String,
        observed: String,
    },
}

pub type Result<T> = std::result::Result<T, EvidenceError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStatus {
    Pass,
    Fail,
    Unsupported,
    Unavailable,
    Invalid,
    Incomplete,
    InsufficientEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub role: String,
    pub absolute_path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

impl ArtifactRef {
    pub fn from_path(role: impl Into<String>, path: impl AsRef<Path>) -> Result<Self> {
        let absolute_path = fs::canonicalize(path.as_ref())?;
        let metadata = fs::metadata(&absolute_path)?;
        if !metadata.is_file() {
            return Err(EvidenceError::Invalid(format!(
                "artifact is not a regular file: {}",
                absolute_path.display()
            )));
        }
        Ok(Self {
            role: role.into(),
            sha256: sha256_file(&absolute_path)?,
            bytes: metadata.len(),
            absolute_path,
        })
    }

    pub fn verify(&self) -> Result<()> {
        if !self.absolute_path.is_absolute() {
            return Err(EvidenceError::Invalid(format!(
                "artifact path is not absolute: {}",
                self.absolute_path.display()
            )));
        }
        validate_sha256(&self.sha256, "artifact sha256")?;
        let metadata = fs::metadata(&self.absolute_path)?;
        if metadata.len() != self.bytes {
            return Err(EvidenceError::Invalid(format!(
                "artifact byte length changed for {}: expected {}, observed {}",
                self.absolute_path.display(),
                self.bytes,
                metadata.len()
            )));
        }
        let observed = sha256_file(&self.absolute_path)?;
        if observed != self.sha256 {
            return Err(EvidenceError::HashMismatch {
                path: self.absolute_path.clone(),
                expected: self.sha256.clone(),
                observed,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceProvenance {
    pub repository_commit: String,
    pub dirty: bool,
    pub branch_role: String,
    pub cargo_lock_sha256: String,
    pub toolchain: String,
    pub profile: String,
    #[serde(default)]
    pub features: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFileEvidence {
    pub role: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepackEvidence {
    pub format: u32,
    pub layout: String,
    #[serde(default)]
    pub source_hashes: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelEvidence {
    pub id: String,
    pub revision: String,
    #[serde(default)]
    pub files: Vec<ModelFileEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repack: Option<RepackEvidence>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandEvidence {
    #[serde(default)]
    pub argv_redacted: Vec<String>,
    #[serde(default)]
    pub environment_allowlist: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadEvidence {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_sha256: Option<String>,
    pub seed: u64,
    pub repetitions: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimerEvidence {
    pub name: String,
    pub clock: String,
    #[serde(default)]
    pub includes: Vec<String>,
    #[serde(default)]
    pub excludes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunManifestV1 {
    pub schema: String,
    pub run_id: String,
    pub purpose: String,
    pub status: EvidenceStatus,
    pub source: SourceProvenance,
    pub model: ModelEvidence,
    pub host_snapshot_sha256: String,
    pub runtime_snapshot_sha256: String,
    pub command: CommandEvidence,
    pub workload: WorkloadEvidence,
    #[serde(default)]
    pub timers: Vec<TimerEvidence>,
    #[serde(default)]
    pub artifacts: Vec<ArtifactRef>,
    #[serde(default)]
    pub limitations: Vec<String>,
}

impl RunManifestV1 {
    pub fn new(
        run_id: impl Into<String>,
        purpose: impl Into<String>,
        status: EvidenceStatus,
    ) -> Self {
        Self {
            schema: EVIDENCE_SCHEMA_V1.into(),
            run_id: run_id.into(),
            purpose: purpose.into(),
            status,
            source: SourceProvenance::default(),
            model: ModelEvidence::default(),
            host_snapshot_sha256: String::new(),
            runtime_snapshot_sha256: String::new(),
            command: CommandEvidence::default(),
            workload: WorkloadEvidence::default(),
            timers: Vec::new(),
            artifacts: Vec::new(),
            limitations: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != EVIDENCE_SCHEMA_V1 {
            return Err(EvidenceError::Invalid(format!(
                "unsupported schema '{}'",
                self.schema
            )));
        }
        require_nonempty(&self.run_id, "run_id")?;
        require_nonempty(&self.purpose, "purpose")?;
        if !self.source.repository_commit.is_empty()
            && (self.source.repository_commit.len() != 40
                || !self
                    .source
                    .repository_commit
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()))
        {
            return Err(EvidenceError::Invalid(
                "repository_commit must be an empty or 40-hex value".into(),
            ));
        }
        for (name, hash) in [
            ("cargo_lock_sha256", &self.source.cargo_lock_sha256),
            ("host_snapshot_sha256", &self.host_snapshot_sha256),
            ("runtime_snapshot_sha256", &self.runtime_snapshot_sha256),
        ] {
            if !hash.is_empty() {
                validate_sha256(hash, name)?;
            }
        }
        if let Some(hash) = &self.workload.prompt_sha256 {
            validate_sha256(hash, "prompt_sha256")?;
        }
        for file in &self.model.files {
            validate_sha256(&file.sha256, "model file sha256")?;
        }
        for artifact in &self.artifacts {
            if !artifact.absolute_path.is_absolute() {
                return Err(EvidenceError::Invalid(format!(
                    "artifact path is not absolute: {}",
                    artifact.absolute_path.display()
                )));
            }
            validate_sha256(&artifact.sha256, "artifact sha256")?;
        }
        if self.workload.repetitions == 0 {
            return Err(EvidenceError::Invalid(
                "workload repetitions must be positive".into(),
            ));
        }
        Ok(())
    }

    pub fn verify_artifacts(&self) -> Result<()> {
        self.validate()?;
        self.artifacts.iter().try_for_each(ArtifactRef::verify)
    }

    /// Return a publishable copy with secret-looking values and host paths removed.
    pub fn redacted(&self) -> Self {
        let mut copy = self.clone();
        copy.command.argv_redacted = copy
            .command
            .argv_redacted
            .iter()
            .map(|value| redact_argument(value))
            .collect();
        copy.command.environment_allowlist.retain(|key, _| {
            let upper = key.to_ascii_uppercase();
            ![
                "TOKEN", "SECRET", "PASSWORD", "KEY", "PROXY", "HOME", "HOST",
            ]
            .iter()
            .any(|needle| upper.contains(needle))
        });
        copy.artifacts
            .iter_mut()
            .for_each(|artifact| artifact.absolute_path = PathBuf::from("/redacted/artifact"));
        copy
    }

    pub fn stable_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut canonical = self.clone();
        canonical.source.features.sort();
        canonical.source.features.dedup();
        canonical
            .model
            .files
            .sort_by(|left, right| (&left.role, &left.sha256).cmp(&(&right.role, &right.sha256)));
        if let Some(repack) = &mut canonical.model.repack {
            repack.source_hashes.sort();
            repack.source_hashes.dedup();
        }
        canonical
            .timers
            .sort_by(|left, right| left.name.cmp(&right.name));
        for timer in &mut canonical.timers {
            timer.includes.sort();
            timer.includes.dedup();
            timer.excludes.sort();
            timer.excludes.dedup();
        }
        canonical.artifacts.sort_by(|left, right| {
            (&left.role, &left.absolute_path).cmp(&(&right.role, &right.absolute_path))
        });
        canonical.limitations.sort();
        canonical.limitations.dedup();
        stable_json(&canonical)
    }

    pub fn write_atomic(&self, path: impl AsRef<Path>) -> Result<()> {
        atomic_write(path.as_ref(), &self.stable_json()?)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RuntimeDecision {
    pub operation_class: String,
    pub eligibility: String,
    pub selected: String,
    pub reason_code: String,
    #[serde(default)]
    pub possible_fallbacks: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectiveRuntimeSnapshot {
    pub schema: String,
    #[serde(default)]
    pub requested: BTreeMap<String, Value>,
    #[serde(default)]
    pub effective: BTreeMap<String, Value>,
    #[serde(default)]
    pub capability: BTreeMap<String, Value>,
    #[serde(default)]
    pub identity: BTreeMap<String, Value>,
    #[serde(default)]
    pub decisions: Vec<RuntimeDecision>,
    #[serde(default)]
    pub omissions: Vec<String>,
}

impl Default for EffectiveRuntimeSnapshot {
    fn default() -> Self {
        Self {
            schema: RUNTIME_SNAPSHOT_SCHEMA_V1.into(),
            requested: BTreeMap::new(),
            effective: BTreeMap::new(),
            capability: BTreeMap::new(),
            identity: BTreeMap::new(),
            decisions: Vec::new(),
            omissions: Vec::new(),
        }
    }
}

impl EffectiveRuntimeSnapshot {
    pub fn stable_json(&self) -> Result<Vec<u8>> {
        if self.schema != RUNTIME_SNAPSHOT_SCHEMA_V1 {
            return Err(EvidenceError::Invalid(format!(
                "unsupported runtime snapshot schema '{}'",
                self.schema
            )));
        }
        let mut canonical = self.clone();
        canonical.decisions.sort_by(|left, right| {
            (&left.operation_class, &left.selected, &left.reason_code).cmp(&(
                &right.operation_class,
                &right.selected,
                &right.reason_code,
            ))
        });
        for decision in &mut canonical.decisions {
            decision.possible_fallbacks.sort();
            decision.possible_fallbacks.dedup();
        }
        canonical.omissions.sort();
        canonical.omissions.dedup();
        stable_json(&canonical)
    }

    pub fn sha256(&self) -> Result<String> {
        Ok(sha256_bytes(&self.stable_json()?))
    }

    pub fn write_atomic(&self, path: impl AsRef<Path>) -> Result<String> {
        let bytes = self.stable_json()?;
        let hash = sha256_bytes(&bytes);
        atomic_write(path.as_ref(), &bytes)?;
        Ok(hash)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticMode {
    #[default]
    Off,
    Metadata,
    Summary,
    Tensor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticConfig {
    pub mode: DiagnosticMode,
    pub directory: Option<PathBuf>,
    pub byte_cap: u64,
    pub boundary: Option<String>,
    pub acknowledge_sensitive_payload: bool,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        Self {
            mode: DiagnosticMode::Off,
            directory: None,
            byte_cap: 0,
            boundary: None,
            acknowledge_sensitive_payload: false,
        }
    }
}

impl DiagnosticConfig {
    pub fn validate(&self, serving_http: bool) -> Result<()> {
        if self.mode == DiagnosticMode::Off {
            return Ok(());
        }
        if self.byte_cap == 0 {
            return Err(EvidenceError::Invalid(
                "enabled diagnostics require a positive byte cap".into(),
            ));
        }
        if self.directory.is_none() {
            return Err(EvidenceError::Invalid(
                "enabled diagnostics require an output directory".into(),
            ));
        }
        if self.mode == DiagnosticMode::Tensor {
            if serving_http {
                return Err(EvidenceError::Invalid(
                    "tensor diagnostics are unavailable while serving HTTP".into(),
                ));
            }
            if self.boundary.as_deref().is_none_or(str::is_empty)
                || !self.acknowledge_sensitive_payload
            {
                return Err(EvidenceError::Invalid(
                    "tensor diagnostics require a boundary and acknowledgement".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticRecord {
    pub schema: String,
    pub kind: String,
    pub monotonic_offset_ns: u128,
    #[serde(default)]
    pub fields: BTreeMap<String, Value>,
    #[serde(default)]
    pub truncated: bool,
}

impl DiagnosticRecord {
    pub fn new(kind: impl Into<String>, monotonic_offset_ns: u128) -> Self {
        Self {
            schema: DIAGNOSTIC_SCHEMA_V1.into(),
            kind: kind.into(),
            monotonic_offset_ns,
            fields: BTreeMap::new(),
            truncated: false,
        }
    }
}

/// Byte-capped JSONL writer. Off mode creates no directory, file, or payload.
pub struct DiagnosticSink {
    mode: DiagnosticMode,
    writer: Option<BufWriter<File>>,
    byte_cap: u64,
    bytes_written: u64,
    truncated: bool,
}

impl DiagnosticSink {
    pub fn open(config: &DiagnosticConfig, file_name: &str, serving_http: bool) -> Result<Self> {
        config.validate(serving_http)?;
        if config.mode == DiagnosticMode::Off {
            return Ok(Self {
                mode: DiagnosticMode::Off,
                writer: None,
                byte_cap: 0,
                bytes_written: 0,
                truncated: false,
            });
        }
        let directory = config.directory.as_ref().expect("validated directory");
        fs::create_dir_all(directory)?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join(file_name))?;
        Ok(Self {
            mode: config.mode,
            writer: Some(BufWriter::new(file)),
            byte_cap: config.byte_cap,
            bytes_written: 0,
            truncated: false,
        })
    }

    pub const fn mode(&self) -> DiagnosticMode {
        self.mode
    }

    pub const fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    pub const fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// Returns false when the record is omitted because the cap was reached.
    pub fn write(&mut self, record: &DiagnosticRecord) -> Result<bool> {
        if self.mode == DiagnosticMode::Off || self.truncated {
            return Ok(false);
        }
        let mut encoded = serde_json::to_vec(record)?;
        encoded.push(b'\n');
        let encoded_len = u64::try_from(encoded.len())
            .map_err(|_| EvidenceError::Invalid("diagnostic record length overflow".into()))?;
        if self
            .bytes_written
            .checked_add(encoded_len)
            .is_none_or(|total| total > self.byte_cap)
        {
            self.truncated = true;
            return Ok(false);
        }
        self.writer
            .as_mut()
            .expect("enabled diagnostic writer")
            .write_all(&encoded)?;
        self.bytes_written += encoded_len;
        Ok(true)
    }

    pub fn flush(&mut self) -> Result<()> {
        if let Some(writer) = &mut self.writer {
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
        Ok(())
    }
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    std::io::copy(&mut file, &mut digest)?;
    Ok(format!("{:x}", digest.finalize()))
}

pub fn stable_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut value = serde_json::to_value(value)?;
    canonicalize_json(&mut value);
    let mut bytes = serde_json::to_vec_pretty(&value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn canonicalize_json(value: &mut Value) {
    match value {
        Value::Object(object) => {
            let old = std::mem::take(object);
            let mut entries = old.into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            for (key, mut nested) in entries {
                canonicalize_json(&mut nested);
                object.insert(key, nested);
            }
        }
        Value::Array(values) => values.iter_mut().for_each(canonicalize_json),
        _ => {}
    }
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| EvidenceError::Invalid("output path has no UTF-8 file name".into()))?;
    let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(".{file_name}.{}.{}.tmp", std::process::id(), id));
    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        if let Ok(directory) = File::open(parent) {
            let _ = directory.sync_all();
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn require_nonempty(value: &str, name: &str) -> Result<()> {
    if value.trim().is_empty() {
        Err(EvidenceError::Invalid(format!("{name} must not be empty")))
    } else {
        Ok(())
    }
}

fn validate_sha256(hash: &str, name: &str) -> Result<()> {
    if hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(EvidenceError::Invalid(format!(
            "{name} must contain exactly 64 hexadecimal characters"
        )))
    }
}

fn redact_argument(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    let assigned_path = value
        .split_once('=')
        .is_some_and(|(_, assigned)| Path::new(assigned).is_absolute());
    if lower.contains("token=")
        || lower.contains("password=")
        || lower.contains("secret=")
        || Path::new(value).is_absolute()
        || assigned_path
    {
        "<redacted>".into()
    } else {
        value.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(artifact: ArtifactRef) -> RunManifestV1 {
        let mut manifest = RunManifestV1::new("run-1", "probe", EvidenceStatus::Pass);
        manifest.workload.repetitions = 1;
        manifest.artifacts.push(artifact);
        manifest
    }

    #[test]
    fn all_negative_statuses_have_stable_json_values() {
        for (status, expected) in [
            (EvidenceStatus::Fail, "fail"),
            (EvidenceStatus::Unsupported, "unsupported"),
            (EvidenceStatus::Unavailable, "unavailable"),
            (EvidenceStatus::Invalid, "invalid"),
            (EvidenceStatus::Incomplete, "incomplete"),
            (
                EvidenceStatus::InsufficientEvidence,
                "insufficient_evidence",
            ),
        ] {
            assert_eq!(
                serde_json::to_string(&status).unwrap(),
                format!("\"{expected}\"")
            );
        }
    }

    #[test]
    fn artifact_paths_are_absolute_and_hashes_are_verified() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("raw.json");
        fs::write(&path, b"raw").unwrap();
        let artifact = ArtifactRef::from_path("raw-output", &path).unwrap();
        assert!(artifact.absolute_path.is_absolute());
        artifact.verify().unwrap();
        fs::write(&path, b"bad").unwrap();
        assert!(artifact.verify().is_err());
    }

    #[test]
    fn stable_serialization_and_atomic_write() {
        let temp = tempfile::tempdir().unwrap();
        let raw = temp.path().join("raw.json");
        fs::write(&raw, b"raw").unwrap();
        let manifest = manifest(ArtifactRef::from_path("raw", raw).unwrap());
        assert_eq!(
            manifest.stable_json().unwrap(),
            manifest.stable_json().unwrap()
        );
        let output = temp.path().join("nested/manifest.json");
        manifest.write_atomic(&output).unwrap();
        assert_eq!(fs::read(output).unwrap(), manifest.stable_json().unwrap());
    }

    #[test]
    fn redaction_removes_secret_arguments_paths_and_environment() {
        let temp = tempfile::tempdir().unwrap();
        let raw = temp.path().join("raw");
        fs::write(&raw, b"raw").unwrap();
        let mut manifest = manifest(ArtifactRef::from_path("raw", raw).unwrap());
        manifest.command.argv_redacted = vec![
            "--ok".into(),
            "token=secret".into(),
            "/home/a/model".into(),
            "--model=/srv/private/model".into(),
        ];
        manifest
            .command
            .environment_allowlist
            .insert("API_TOKEN".into(), "secret".into());
        let redacted = manifest.redacted();
        assert_eq!(redacted.command.argv_redacted[0], "--ok");
        assert!(redacted.command.argv_redacted[1..]
            .iter()
            .all(|value| value == "<redacted>"));
        assert!(redacted.command.environment_allowlist.is_empty());
        assert!(!redacted.artifacts[0]
            .absolute_path
            .to_string_lossy()
            .contains(temp.path().to_string_lossy().as_ref()));
    }

    #[test]
    fn dirty_incomplete_provenance_is_preserved_not_promoted() {
        let temp = tempfile::tempdir().unwrap();
        let raw = temp.path().join("raw");
        fs::write(&raw, b"raw").unwrap();
        let mut manifest = manifest(ArtifactRef::from_path("raw", raw).unwrap());
        manifest.status = EvidenceStatus::Incomplete;
        manifest.source.dirty = true;
        manifest.source.repository_commit = "a".repeat(40);
        let value: Value = serde_json::from_slice(&manifest.stable_json().unwrap()).unwrap();
        assert_eq!(value["status"], "incomplete");
        assert_eq!(value["source"]["dirty"], true);
    }

    #[test]
    fn disabled_diagnostics_create_no_payload() {
        let temp = tempfile::tempdir().unwrap();
        let mut sink =
            DiagnosticSink::open(&DiagnosticConfig::default(), "trace.jsonl", true).unwrap();
        assert!(!sink.write(&DiagnosticRecord::new("event", 0)).unwrap());
        assert_eq!(sink.bytes_written(), 0);
        assert!(!temp.path().join("trace.jsonl").exists());
    }

    #[test]
    fn diagnostic_cap_is_hard_and_records_truncation() {
        let temp = tempfile::tempdir().unwrap();
        let config = DiagnosticConfig {
            mode: DiagnosticMode::Metadata,
            directory: Some(temp.path().to_path_buf()),
            byte_cap: 8,
            boundary: None,
            acknowledge_sensitive_payload: false,
        };
        let mut sink = DiagnosticSink::open(&config, "trace.jsonl", true).unwrap();
        assert!(!sink.write(&DiagnosticRecord::new("too-large", 0)).unwrap());
        assert!(sink.is_truncated());
        assert_eq!(sink.bytes_written(), 0);
    }

    #[test]
    fn tensor_diagnostics_require_offline_boundary_and_acknowledgement() {
        let config = DiagnosticConfig {
            mode: DiagnosticMode::Tensor,
            directory: Some(PathBuf::from("out")),
            byte_cap: 1,
            boundary: Some("prefill.layer".into()),
            acknowledge_sensitive_payload: true,
        };
        assert!(config.validate(false).is_ok());
        assert!(config.validate(true).is_err());
    }
}
