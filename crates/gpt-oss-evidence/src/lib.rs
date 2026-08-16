#![forbid(unsafe_code)]
//! Stable, privacy-aware evidence records shared by CPU tools and services.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const EVIDENCE_SCHEMA_V1: &str = "gpt-oss-rs.cpu-evidence/v1";
pub const RUNTIME_SNAPSHOT_SCHEMA_V1: &str = "gpt-oss-rs.cpu-runtime/v1";
pub const DIAGNOSTIC_SCHEMA_V1: &str = "gpt-oss-rs.cpu-diagnostic/v1";
pub const CAMPAIGN_INDEX_SCHEMA_V1: &str = "gpt-oss-rs.cpu-campaign-index/v1";
pub const HETEROGENEOUS_STEP_TRACE_SCHEMA_V1: &str = "gpt-oss-rs.heterogeneous-step-trace/v1";

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

/// Campaign coordinates are optional on legacy v1 records, but mandatory for
/// campaign-complete records. Keeping them inside v1 preserves the stable
/// schema identity while allowing older captures to deserialize unchanged.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignIdentity {
    pub campaign_id: String,
    pub candidate_sha: String,
    pub phase: String,
    pub scenario: String,
    pub requested_kernel: String,
    pub attempt_number: u32,
    pub attempt_id: String,
    pub cell_key: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryEvidence {
    pub role: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispatchEvidence {
    pub requested_kernel: String,
    pub effective_kernel: String,
    pub requested_matrix_backend: String,
    pub effective_matrix_backend: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasuredResources {
    pub wall_time_ns: u64,
    pub startup_time_ns: u64,
    pub prompt_time_ns: u64,
    pub generation_time_ns: u64,
    pub peak_rss_bytes: u64,
    pub process_swap_bytes: u64,
    pub system_swap_used_bytes: u64,
    pub available_memory_bytes: u64,
}

/// Sanitized durable GPU identity. CUDA ordinals are retained only as the
/// process-local resolution of the PCI identity, never as placement identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeterogeneousDeviceEvidenceV1 {
    pub role: String,
    pub pci_bus_id: String,
    pub transient_ordinal: u32,
    pub expected_name: String,
    pub compute_capability: (u32, u32),
    pub minimum_memory_bytes: u64,
}

/// One canonical row/rank route. BF16 selected weights remain serialized as
/// bits so evidence cannot silently widen and reround them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeterogeneousRouteEvidenceV1 {
    pub source_row: u32,
    pub route_rank: u8,
    pub expert_id: u16,
    pub selected_weight_bf16_bits: u16,
    pub activation_slot: u32,
    pub owner: String,
    pub result_slot: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeterogeneousIntervalEvidenceV1 {
    pub name: String,
    pub owner: String,
    pub clock: String,
    pub start_ns: u64,
    pub end_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeterogeneousErrorEvidenceV1 {
    pub precedence: u32,
    pub kind: String,
    pub owner: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route_slot: Option<u32>,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeterogeneousStepOutcomeV1 {
    Committed,
    Discarded,
}

/// Bounded terminal trace for one heterogeneous prepared step.
///
/// Active traces are deliberately not representable as terminal evidence.
/// A caller may publish this record only after commit, or after mandatory
/// drain and discard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeterogeneousStepTraceV1 {
    pub schema: String,
    pub trace_id: String,
    pub model_id: String,
    pub model_revision: String,
    pub config_sha256: String,
    pub index_sha256: String,
    pub mapping_sha256: String,
    pub placement_sha256: String,
    pub build_sha256: String,
    pub devices: Vec<HeterogeneousDeviceEvidenceV1>,
    pub sequence_id: u64,
    pub expected_revision: u64,
    pub expected_visibility_epoch: u64,
    pub terminal_visibility_epoch: u64,
    pub placement_epoch: u64,
    pub generation: u64,
    pub layer: u16,
    pub phase: String,
    pub chunk: u32,
    pub rows: u32,
    pub routes: Vec<HeterogeneousRouteEvidenceV1>,
    #[serde(default)]
    pub reserved_bytes: BTreeMap<String, u64>,
    #[serde(default)]
    pub high_water_bytes: BTreeMap<String, u64>,
    #[serde(default)]
    pub intervals: Vec<HeterogeneousIntervalEvidenceV1>,
    #[serde(default)]
    pub errors: Vec<HeterogeneousErrorEvidenceV1>,
    pub outcome: HeterogeneousStepOutcomeV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_divergence: Option<ArtifactRef>,
}

impl HeterogeneousStepTraceV1 {
    pub fn validate(&self) -> Result<()> {
        if self.schema != HETEROGENEOUS_STEP_TRACE_SCHEMA_V1 {
            return Err(EvidenceError::Invalid(format!(
                "unsupported heterogeneous trace schema '{}'",
                self.schema
            )));
        }
        for (name, value) in [
            ("trace_id", self.trace_id.as_str()),
            ("model_id", self.model_id.as_str()),
            ("model_revision", self.model_revision.as_str()),
            ("phase", self.phase.as_str()),
        ] {
            require_nonempty(value, name)?;
        }
        for (name, hash) in [
            ("config_sha256", self.config_sha256.as_str()),
            ("index_sha256", self.index_sha256.as_str()),
            ("mapping_sha256", self.mapping_sha256.as_str()),
            ("placement_sha256", self.placement_sha256.as_str()),
            ("build_sha256", self.build_sha256.as_str()),
        ] {
            validate_sha256(hash, name)?;
        }
        if !matches!(self.phase.as_str(), "decode" | "prefill") || self.rows == 0 {
            return Err(EvidenceError::Invalid(
                "phase must be decode or prefill and rows must be positive".into(),
            ));
        }
        if self.routes.len() != self.rows as usize * 4 {
            return Err(EvidenceError::Invalid(format!(
                "heterogeneous trace has {} routes, expected {}",
                self.routes.len(),
                self.rows as usize * 4
            )));
        }
        let mut device_roles = BTreeSet::new();
        let mut pci_identities = BTreeSet::new();
        for device in &self.devices {
            require_nonempty(&device.role, "device role")?;
            require_nonempty(&device.pci_bus_id, "device PCI identity")?;
            require_nonempty(&device.expected_name, "device name")?;
            if !device_roles.insert(device.role.as_str()) {
                return Err(EvidenceError::Invalid(format!(
                    "duplicate device role '{}'",
                    device.role
                )));
            }
            if !pci_identities.insert(device.pci_bus_id.as_str()) {
                return Err(EvidenceError::Invalid(format!(
                    "duplicate device PCI identity '{}'",
                    device.pci_bus_id
                )));
            }
        }
        if device_roles != BTreeSet::from(["layer_owner_gpu", "remote_gpu"]) {
            return Err(EvidenceError::Invalid(
                "heterogeneous trace requires exactly layer_owner_gpu and remote_gpu identities"
                    .into(),
            ));
        }
        for (slot, route) in self.routes.iter().enumerate() {
            let expected_row = (slot / 4) as u32;
            let expected_rank = (slot % 4) as u8;
            if route.source_row != expected_row
                || route.route_rank != expected_rank
                || route.result_slot != slot as u32
                || route.activation_slot >= self.rows
            {
                return Err(EvidenceError::Invalid(format!(
                    "route slot {slot} does not preserve canonical row/rank/result identity"
                )));
            }
            require_nonempty(&route.owner, "route owner")?;
        }
        for interval in &self.intervals {
            require_nonempty(&interval.name, "interval name")?;
            require_nonempty(&interval.owner, "interval owner")?;
            require_nonempty(&interval.clock, "interval clock")?;
            if interval.end_ns < interval.start_ns {
                return Err(EvidenceError::Invalid(format!(
                    "interval '{}' ends before it starts",
                    interval.name
                )));
            }
        }
        if self
            .errors
            .windows(2)
            .any(|pair| pair[0].precedence > pair[1].precedence)
        {
            return Err(EvidenceError::Invalid(
                "heterogeneous errors are not in deterministic precedence order".into(),
            ));
        }
        match self.outcome {
            HeterogeneousStepOutcomeV1::Committed
                if self.terminal_visibility_epoch
                    != self.expected_visibility_epoch.saturating_add(1) =>
            {
                return Err(EvidenceError::Invalid(
                    "committed trace must advance visibility epoch exactly once".into(),
                ));
            }
            HeterogeneousStepOutcomeV1::Discarded
                if self.terminal_visibility_epoch != self.expected_visibility_epoch =>
            {
                return Err(EvidenceError::Invalid(
                    "discarded trace must not advance visibility epoch".into(),
                ));
            }
            _ => {}
        }
        if self.outcome == HeterogeneousStepOutcomeV1::Committed && !self.errors.is_empty() {
            return Err(EvidenceError::Invalid(
                "committed heterogeneous trace must not contain terminal errors".into(),
            ));
        }
        if let Some(divergence) = &self.first_divergence {
            divergence.verify()?;
        }
        Ok(())
    }

    pub fn stable_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        stable_json(self)
    }

    pub fn write_atomic_new(&self, path: impl AsRef<Path>) -> Result<()> {
        atomic_write_new(path.as_ref(), &self.stable_json()?)
    }
}

/// Immutable CPU-oracle coordinates. All fields remain optional so evidence
/// v1 records created before the container oracle continue to deserialize.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleIdentityEvidence {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_manifest_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_config_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub software_lock_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub official_source_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container_policy_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe_artifact_sha256: Option<String>,
}

impl OracleIdentityEvidence {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
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
    #[serde(default)]
    pub campaign: CampaignIdentity,
    #[serde(default)]
    pub build_binaries: Vec<BinaryEvidence>,
    #[serde(default)]
    pub dispatch: DispatchEvidence,
    #[serde(default)]
    pub measured: MeasuredResources,
    #[serde(default)]
    pub related_runs: Vec<String>,
    #[serde(default, skip_serializing_if = "OracleIdentityEvidence::is_empty")]
    pub oracle_identity: OracleIdentityEvidence,
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
            campaign: CampaignIdentity::default(),
            build_binaries: Vec::new(),
            dispatch: DispatchEvidence::default(),
            measured: MeasuredResources::default(),
            related_runs: Vec::new(),
            oracle_identity: OracleIdentityEvidence::default(),
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
        for (name, hash) in [
            (
                "oracle image manifest digest",
                &self.oracle_identity.image_manifest_digest,
            ),
            (
                "oracle image config digest",
                &self.oracle_identity.image_config_digest,
            ),
            (
                "oracle software lock sha256",
                &self.oracle_identity.software_lock_sha256,
            ),
            (
                "oracle host fingerprint",
                &self.oracle_identity.host_fingerprint,
            ),
            (
                "oracle container policy sha256",
                &self.oracle_identity.container_policy_sha256,
            ),
            (
                "oracle probe artifact sha256",
                &self.oracle_identity.probe_artifact_sha256,
            ),
        ] {
            if let Some(hash) = hash {
                validate_sha256(hash, name)?;
            }
        }
        if let Some(revision) = &self.oracle_identity.official_source_revision {
            if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(EvidenceError::Invalid(
                    "oracle official source revision must be 40 hexadecimal characters".into(),
                ));
            }
        }
        if let Some(mode) = &self.oracle_identity.execution_mode {
            if !matches!(mode.as_str(), "native" | "generic") {
                return Err(EvidenceError::Invalid(
                    "oracle execution mode must be native or generic".into(),
                ));
            }
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

    /// Validate fields required for a terminal campaign attempt. Legacy v1
    /// captures should continue to call `validate`; campaign finalization must
    /// call this stricter surface.
    pub fn validate_campaign_complete(&self) -> Result<()> {
        self.validate()?;
        for (name, value) in [
            ("campaign_id", self.campaign.campaign_id.as_str()),
            ("phase", self.campaign.phase.as_str()),
            ("scenario", self.campaign.scenario.as_str()),
            ("requested_kernel", self.campaign.requested_kernel.as_str()),
            ("attempt_id", self.campaign.attempt_id.as_str()),
            ("cell_key", self.campaign.cell_key.as_str()),
        ] {
            require_nonempty(value, name)?;
        }
        if self.campaign.candidate_sha.len() != 40
            || !self
                .campaign
                .candidate_sha
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(EvidenceError::Invalid(
                "candidate_sha must contain exactly 40 hexadecimal characters".into(),
            ));
        }
        if self.campaign.attempt_number == 0 {
            return Err(EvidenceError::Invalid(
                "attempt_number must be positive".into(),
            ));
        }
        if self.run_id != self.campaign.attempt_id {
            return Err(EvidenceError::Invalid(
                "run_id must equal campaign attempt_id".into(),
            ));
        }
        if self.build_binaries.is_empty() {
            return Err(EvidenceError::Invalid(
                "campaign attempt requires at least one build binary hash".into(),
            ));
        }
        for binary in &self.build_binaries {
            require_nonempty(&binary.role, "binary role")?;
            validate_sha256(&binary.sha256, "binary sha256")?;
        }
        for (name, value) in [
            (
                "requested dispatch",
                self.dispatch.requested_kernel.as_str(),
            ),
            (
                "effective dispatch",
                self.dispatch.effective_kernel.as_str(),
            ),
            (
                "requested matrix backend",
                self.dispatch.requested_matrix_backend.as_str(),
            ),
            (
                "effective matrix backend",
                self.dispatch.effective_matrix_backend.as_str(),
            ),
        ] {
            require_nonempty(value, name)?;
        }
        let mut related = BTreeSet::new();
        for run in &self.related_runs {
            require_nonempty(run, "related run")?;
            if !related.insert(run) {
                return Err(EvidenceError::Invalid(format!(
                    "duplicate related run '{run}'"
                )));
            }
        }
        self.verify_artifacts()
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
        canonical
            .build_binaries
            .sort_by(|left, right| (&left.role, &left.sha256).cmp(&(&right.role, &right.sha256)));
        canonical.related_runs.sort();
        canonical.related_runs.dedup();
        canonical.limitations.sort();
        canonical.limitations.dedup();
        stable_json(&canonical)
    }

    pub fn write_atomic(&self, path: impl AsRef<Path>) -> Result<()> {
        atomic_write(path.as_ref(), &self.stable_json()?)
    }

    /// Publish a terminal artifact exactly once. Existing output is never
    /// replaced, even if a prior process raced this writer.
    pub fn write_atomic_new(&self, path: impl AsRef<Path>) -> Result<()> {
        atomic_write_new(path.as_ref(), &self.stable_json()?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignAttemptV1 {
    pub cell_key: String,
    pub attempt_id: String,
    pub attempt_number: u32,
    pub status: EvidenceStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terminal_manifest: Option<ArtifactRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignIndexV1 {
    pub schema: String,
    pub campaign_id: String,
    pub candidate_sha: String,
    #[serde(default)]
    pub parent_candidate_sha: Option<String>,
    #[serde(default)]
    pub attempts: Vec<CampaignAttemptV1>,
}

impl CampaignIndexV1 {
    pub fn new(campaign_id: impl Into<String>, candidate_sha: impl Into<String>) -> Self {
        Self {
            schema: CAMPAIGN_INDEX_SCHEMA_V1.into(),
            campaign_id: campaign_id.into(),
            candidate_sha: candidate_sha.into(),
            parent_candidate_sha: None,
            attempts: Vec::new(),
        }
    }

    pub fn stable_cell_key(
        phase: &str,
        scenario: &str,
        requested_kernel: &str,
        backend: &str,
    ) -> Result<String> {
        for (name, value) in [
            ("phase", phase),
            ("scenario", scenario),
            ("requested_kernel", requested_kernel),
            ("backend", backend),
        ] {
            require_nonempty(value, name)?;
            if value.contains('/') || value.contains(char::is_whitespace) {
                return Err(EvidenceError::Invalid(format!(
                    "{name} contains a path separator or whitespace"
                )));
            }
        }
        Ok(format!(
            "{phase}--{scenario}--{requested_kernel}--{backend}"
        ))
    }

    pub fn next_attempt(&self, cell_key: &str) -> u32 {
        self.attempts
            .iter()
            .filter(|attempt| attempt.cell_key == cell_key)
            .map(|attempt| attempt.attempt_number)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    pub fn push(&mut self, attempt: CampaignAttemptV1) -> Result<()> {
        if self
            .attempts
            .iter()
            .any(|existing| existing.attempt_id == attempt.attempt_id)
        {
            return Err(EvidenceError::Invalid(format!(
                "duplicate attempt ID '{}'",
                attempt.attempt_id
            )));
        }
        let expected = self.next_attempt(&attempt.cell_key);
        if attempt.attempt_number != expected {
            return Err(EvidenceError::Invalid(format!(
                "attempt number {} for '{}' is not next expected number {expected}",
                attempt.attempt_number, attempt.cell_key
            )));
        }
        self.attempts.push(attempt);
        self.validate()
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema != CAMPAIGN_INDEX_SCHEMA_V1 {
            return Err(EvidenceError::Invalid(format!(
                "unsupported campaign index schema '{}'",
                self.schema
            )));
        }
        require_nonempty(&self.campaign_id, "campaign_id")?;
        if self.candidate_sha.len() != 40
            || !self
                .candidate_sha
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(EvidenceError::Invalid(
                "candidate_sha must contain exactly 40 hexadecimal characters".into(),
            ));
        }
        if let Some(parent) = &self.parent_candidate_sha {
            if parent.len() != 40 || !parent.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(EvidenceError::Invalid(
                    "parent_candidate_sha must contain exactly 40 hexadecimal characters".into(),
                ));
            }
            if parent == &self.candidate_sha {
                return Err(EvidenceError::Invalid(
                    "parent_candidate_sha must differ from candidate_sha".into(),
                ));
            }
        }
        let mut ids = BTreeSet::new();
        let mut numbers: BTreeMap<&str, u32> = BTreeMap::new();
        for attempt in &self.attempts {
            require_nonempty(&attempt.cell_key, "cell_key")?;
            require_nonempty(&attempt.attempt_id, "attempt_id")?;
            if !ids.insert(attempt.attempt_id.as_str()) {
                return Err(EvidenceError::Invalid(format!(
                    "duplicate attempt ID '{}'",
                    attempt.attempt_id
                )));
            }
            let next = numbers.entry(&attempt.cell_key).or_default();
            *next += 1;
            if attempt.attempt_number != *next {
                return Err(EvidenceError::Invalid(format!(
                    "non-contiguous attempt number for '{}'",
                    attempt.cell_key
                )));
            }
            if let Some(manifest) = &attempt.terminal_manifest {
                manifest.verify()?;
            }
        }
        Ok(())
    }

    pub fn stable_json(&self) -> Result<Vec<u8>> {
        self.validate()?;
        stable_json(self)
    }

    pub fn write_atomic(&self, path: impl AsRef<Path>) -> Result<()> {
        atomic_write(path.as_ref(), &self.stable_json()?)
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let index: Self = serde_json::from_slice(&fs::read(path)?)?;
        index.validate()?;
        Ok(index)
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

/// Atomically publish bytes without replacing an existing destination.
pub fn atomic_write_new(path: &Path, bytes: &[u8]) -> Result<()> {
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
        fs::hard_link(&temporary, path)?;
        fs::remove_file(&temporary)?;
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

    fn heterogeneous_trace() -> HeterogeneousStepTraceV1 {
        HeterogeneousStepTraceV1 {
            schema: HETEROGENEOUS_STEP_TRACE_SCHEMA_V1.into(),
            trace_id: "step-7".into(),
            model_id: "gpt-oss-20b".into(),
            model_revision: "fixture-revision".into(),
            config_sha256: "1".repeat(64),
            index_sha256: "2".repeat(64),
            mapping_sha256: "3".repeat(64),
            placement_sha256: "4".repeat(64),
            build_sha256: "5".repeat(64),
            devices: vec![
                HeterogeneousDeviceEvidenceV1 {
                    role: "layer_owner_gpu".into(),
                    pci_bus_id: "0000:19:00.0".into(),
                    transient_ordinal: 0,
                    expected_name: "NVIDIA GeForce RTX 3090".into(),
                    compute_capability: (8, 6),
                    minimum_memory_bytes: 25_769_803_776,
                },
                HeterogeneousDeviceEvidenceV1 {
                    role: "remote_gpu".into(),
                    pci_bus_id: "0000:65:00.0".into(),
                    transient_ordinal: 1,
                    expected_name: "NVIDIA GeForce RTX 3090".into(),
                    compute_capability: (8, 6),
                    minimum_memory_bytes: 25_769_803_776,
                },
            ],
            sequence_id: 7,
            expected_revision: 11,
            expected_visibility_epoch: 13,
            terminal_visibility_epoch: 14,
            placement_epoch: 17,
            generation: 19,
            layer: 0,
            phase: "decode".into(),
            chunk: 0,
            rows: 1,
            routes: [31_u16, 21, 22, 6]
                .into_iter()
                .enumerate()
                .map(|(rank, expert_id)| HeterogeneousRouteEvidenceV1 {
                    source_row: 0,
                    route_rank: rank as u8,
                    expert_id,
                    selected_weight_bf16_bits: 0x3e80,
                    activation_slot: 0,
                    owner: ["layer_owner_gpu", "cpu", "remote_gpu", "layer_owner_gpu"][rank].into(),
                    result_slot: rank as u32,
                })
                .collect(),
            reserved_bytes: BTreeMap::from([("pinned_relay".into(), 23_040)]),
            high_water_bytes: BTreeMap::from([("pinned_relay".into(), 23_040)]),
            intervals: vec![HeterogeneousIntervalEvidenceV1 {
                name: "commit".into(),
                owner: "coordinator".into(),
                clock: "cpu_monotonic".into(),
                start_ns: 100,
                end_ns: 101,
            }],
            errors: Vec::new(),
            outcome: HeterogeneousStepOutcomeV1::Committed,
            first_divergence: None,
        }
    }

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
    fn heterogeneous_trace_schema_matches_golden_fixture() {
        let trace = heterogeneous_trace();
        trace.validate().unwrap();
        assert_eq!(
            String::from_utf8(trace.stable_json().unwrap()).unwrap(),
            include_str!("../fixtures/heterogeneous-step-trace-v1.json")
        );
    }

    #[test]
    fn heterogeneous_trace_rejects_rank_loss_and_visibility_errors() {
        let mut trace = heterogeneous_trace();
        trace.routes.swap(0, 1);
        assert!(trace.validate().is_err());

        let mut trace = heterogeneous_trace();
        trace.terminal_visibility_epoch = trace.expected_visibility_epoch;
        assert!(trace.validate().is_err());

        let mut trace = heterogeneous_trace();
        trace.outcome = HeterogeneousStepOutcomeV1::Discarded;
        trace.terminal_visibility_epoch = trace.expected_visibility_epoch;
        trace.errors = vec![
            HeterogeneousErrorEvidenceV1 {
                precedence: 2,
                kind: "cancelled".into(),
                owner: "coordinator".into(),
                route_slot: None,
                message: "cancel".into(),
            },
            HeterogeneousErrorEvidenceV1 {
                precedence: 1,
                kind: "cuda_async".into(),
                owner: "remote_gpu".into(),
                route_slot: Some(2),
                message: "worker".into(),
            },
        ];
        assert!(trace.validate().is_err());
    }

    #[test]
    fn create_new_atomic_output_never_replaces_terminal_artifact() {
        let temp = tempfile::tempdir().unwrap();
        let output = temp.path().join("terminal.json");
        atomic_write_new(&output, b"first").unwrap();
        assert!(atomic_write_new(&output, b"second").is_err());
        assert_eq!(fs::read(output).unwrap(), b"first");
    }

    #[test]
    fn campaign_index_has_stable_cells_unique_attempts_and_verified_terminals() {
        let temp = tempfile::tempdir().unwrap();
        let terminal = temp.path().join("terminal.json");
        atomic_write_new(&terminal, b"terminal").unwrap();
        let artifact = ArtifactRef::from_path("terminal-manifest", terminal).unwrap();
        let cell =
            CampaignIndexV1::stable_cell_key("official", "harmony_262", "scalar", "auto").unwrap();
        let mut index = CampaignIndexV1::new("campaign", "a".repeat(40));
        index
            .push(CampaignAttemptV1 {
                cell_key: cell.clone(),
                attempt_id: "campaign--candidate--official--harmony_262--scalar--1".into(),
                attempt_number: 1,
                status: EvidenceStatus::InsufficientEvidence,
                terminal_manifest: Some(artifact),
            })
            .unwrap();
        assert_eq!(index.next_attempt(&cell), 2);
        assert!(index
            .push(CampaignAttemptV1 {
                cell_key: cell,
                attempt_id: "campaign--candidate--official--harmony_262--scalar--1".into(),
                attempt_number: 2,
                status: EvidenceStatus::Incomplete,
                terminal_manifest: None,
            })
            .is_err());
    }

    #[test]
    fn strict_campaign_validation_requires_identity_dispatch_and_binary_hashes() {
        let temp = tempfile::tempdir().unwrap();
        let raw = temp.path().join("raw");
        fs::write(&raw, b"raw").unwrap();
        let mut record = RunManifestV1::new(
            "campaign--candidate--phase--scenario--scalar--1",
            "correctness",
            EvidenceStatus::InsufficientEvidence,
        );
        record.workload.repetitions = 1;
        record.campaign = CampaignIdentity {
            campaign_id: "campaign".into(),
            candidate_sha: "a".repeat(40),
            phase: "phase".into(),
            scenario: "scenario".into(),
            requested_kernel: "scalar".into(),
            attempt_number: 1,
            attempt_id: record.run_id.clone(),
            cell_key: "phase--scenario--scalar--auto".into(),
        };
        record.build_binaries.push(BinaryEvidence {
            role: "worker".into(),
            sha256: "b".repeat(64),
        });
        record.dispatch = DispatchEvidence {
            requested_kernel: "scalar".into(),
            effective_kernel: "scalar".into(),
            requested_matrix_backend: "auto".into(),
            effective_matrix_backend: "rayon".into(),
        };
        record
            .artifacts
            .push(ArtifactRef::from_path("raw", raw).unwrap());
        record.validate_campaign_complete().unwrap();
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
    fn oracle_identity_is_optional_but_present_coordinates_are_validated() {
        let temp = tempfile::tempdir().unwrap();
        let raw = temp.path().join("raw");
        fs::write(&raw, b"raw").unwrap();
        let mut manifest = manifest(ArtifactRef::from_path("raw", raw).unwrap());
        manifest.validate().unwrap();
        manifest.oracle_identity = OracleIdentityEvidence {
            image_manifest_digest: Some("a".repeat(64)),
            image_config_digest: Some("b".repeat(64)),
            software_lock_sha256: Some("c".repeat(64)),
            official_source_revision: Some("d".repeat(40)),
            execution_mode: Some("native".into()),
            host_fingerprint: Some("e".repeat(64)),
            container_policy_sha256: Some("f".repeat(64)),
            probe_artifact_sha256: Some("0".repeat(64)),
        };
        manifest.validate().unwrap();
        manifest.oracle_identity.execution_mode = Some("mixed".into());
        assert!(manifest.validate().is_err());
        manifest.oracle_identity.execution_mode = Some("native".into());
        manifest.oracle_identity.host_fingerprint = Some("short".into());
        assert!(manifest.validate().is_err());
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
