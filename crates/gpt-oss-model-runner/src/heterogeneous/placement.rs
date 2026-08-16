//! Static single-owner placement for GPT-OSS routed experts.

use std::collections::{BTreeMap, BTreeSet};

use gpt_oss_gpu::device::{
    resolve_stable_device, GpuDevice, ResolvedCudaDevice, StableCudaDeviceId,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const HETEROGENEOUS_PLACEMENT_SCHEMA_V1: &str = "gpt-oss-rs.heterogeneous-placement/v1";
pub const CONSERVATIVE_OWNER_EXPERT_BYTES: u64 = 13_253_760;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GptOssExpertKey {
    pub layer: u16,
    pub expert: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CpuPoolId(pub u16);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExpertOwner {
    Cpu { pool: CpuPoolId },
    LayerOwnerGpu { device: StableCudaDeviceId },
    RemoteGpu { device: StableCudaDeviceId },
}

impl ExpertOwner {
    pub const fn role_name(&self) -> &'static str {
        match self {
            Self::Cpu { .. } => "cpu",
            Self::LayerOwnerGpu { .. } => "layer_owner_gpu",
            Self::RemoteGpu { .. } => "remote_gpu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementPolicyClass {
    Proof,
    Performance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GptOssPlacementModel {
    pub revision: String,
    pub config_sha256: String,
    pub index_sha256: String,
    pub mapping_sha256: String,
    pub num_layers: u16,
    pub experts_per_layer: u16,
    pub hidden_size: u16,
    pub intermediate_size: u16,
    pub top_k: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementBudgets {
    pub max_cpu_experts: u32,
    pub max_layer_owner_experts: u32,
    pub max_remote_gpu_experts: u32,
    pub max_host_owner_bytes: u64,
    pub max_layer_owner_bytes: u64,
    pub max_remote_gpu_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertAssignment {
    pub key: GptOssExpertKey,
    pub owner: ExpertOwner,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GptOssExpertPlacementManifestV1 {
    pub schema: String,
    pub model: GptOssPlacementModel,
    pub layer_owner: StableCudaDeviceId,
    pub remote_worker: StableCudaDeviceId,
    pub policy: PlacementPolicyClass,
    pub policy_seed: u64,
    pub placement_epoch: u64,
    pub budgets: PlacementBudgets,
    pub assignments: Vec<ExpertAssignment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementOwnerCounts {
    pub cpu: u32,
    pub layer_owner_gpu: u32,
    pub remote_gpu: u32,
}

pub struct ResolvedExpertPlacement {
    manifest_hash: String,
    placement_epoch: u64,
    layer_owner: ResolvedCudaDevice,
    remote_worker: ResolvedCudaDevice,
    assignments: BTreeMap<GptOssExpertKey, ExpertOwner>,
    counts: PlacementOwnerCounts,
}

/// Device-independent, fully validated placement used before CUDA discovery.
pub struct StaticExpertPlacement {
    manifest_hash: String,
    placement_epoch: u64,
    layer_owner: StableCudaDeviceId,
    remote_worker: StableCudaDeviceId,
    assignments: BTreeMap<GptOssExpertKey, ExpertOwner>,
    counts: PlacementOwnerCounts,
}

impl StaticExpertPlacement {
    pub fn manifest_hash(&self) -> &str {
        &self.manifest_hash
    }

    pub const fn placement_epoch(&self) -> u64 {
        self.placement_epoch
    }

    pub const fn layer_owner(&self) -> &StableCudaDeviceId {
        &self.layer_owner
    }

    pub const fn remote_worker(&self) -> &StableCudaDeviceId {
        &self.remote_worker
    }

    pub fn owner(&self, key: GptOssExpertKey) -> Option<&ExpertOwner> {
        self.assignments.get(&key)
    }

    pub fn assignments(&self) -> impl Iterator<Item = (&GptOssExpertKey, &ExpertOwner)> {
        self.assignments.iter()
    }

    pub const fn counts(&self) -> &PlacementOwnerCounts {
        &self.counts
    }
}

impl ResolvedExpertPlacement {
    pub fn manifest_hash(&self) -> &str {
        &self.manifest_hash
    }

    pub const fn placement_epoch(&self) -> u64 {
        self.placement_epoch
    }

    pub const fn layer_owner(&self) -> &ResolvedCudaDevice {
        &self.layer_owner
    }

    pub const fn remote_worker(&self) -> &ResolvedCudaDevice {
        &self.remote_worker
    }

    pub fn owner(&self, key: GptOssExpertKey) -> Option<&ExpertOwner> {
        self.assignments.get(&key)
    }

    pub fn assignments(&self) -> impl Iterator<Item = (&GptOssExpertKey, &ExpertOwner)> {
        self.assignments.iter()
    }

    pub const fn counts(&self) -> &PlacementOwnerCounts {
        &self.counts
    }

    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlacementError {
    #[error("unsupported placement schema '{0}'")]
    Schema(String),
    #[error("unsupported GPT-OSS dimensions: layers={layers}, experts={experts}, hidden={hidden}, intermediate={intermediate}, top_k={top_k}")]
    ModelDimensions {
        layers: u16,
        experts: u16,
        hidden: u16,
        intermediate: u16,
        top_k: u8,
    },
    #[error("placement model revision must not be empty")]
    MissingModelRevision,
    #[error("{field} must be a 64-character hexadecimal SHA-256")]
    InvalidSha256 { field: &'static str },
    #[error("layer-owner and remote-worker PCI identities must differ")]
    DuplicateDevice,
    #[error("stable device resolution failed: {0}")]
    StableDevice(String),
    #[error("expert key ({layer}, {expert}) is outside the model rectangle")]
    OutOfRange { layer: u16, expert: u16 },
    #[error("expert key ({layer}, {expert}) is assigned more than once")]
    DuplicateAssignment { layer: u16, expert: u16 },
    #[error("expert key ({layer}, {expert}) has no owner")]
    MissingAssignment { layer: u16, expert: u16 },
    #[error("assignment for ({layer}, {expert}) names a device inconsistent with owner role")]
    OwnerDeviceMismatch { layer: u16, expert: u16 },
    #[error("owner counts exceed placement budgets: cpu={cpu}, layer_owner={layer_owner}, remote={remote}")]
    CountBudgetExceeded {
        cpu: u32,
        layer_owner: u32,
        remote: u32,
    },
    #[error("owner bytes exceed placement budgets: cpu={cpu}, layer_owner={layer_owner}, remote={remote}")]
    ByteBudgetExceeded {
        cpu: u64,
        layer_owner: u64,
        remote: u64,
    },
    #[error("placement serialization failed: {0}")]
    Serialization(String),
}

impl GptOssExpertPlacementManifestV1 {
    pub fn stable_json(&self) -> Result<Vec<u8>, PlacementError> {
        let mut canonical = self.clone();
        canonical
            .assignments
            .sort_by_key(|assignment| assignment.key);
        let mut bytes = serde_json::to_vec_pretty(&canonical)
            .map_err(|error| PlacementError::Serialization(error.to_string()))?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    pub fn sha256(&self) -> Result<String, PlacementError> {
        let mut digest = Sha256::new();
        digest.update(self.stable_json()?);
        Ok(format!("{:x}", digest.finalize()))
    }

    pub fn validate(
        &self,
        devices: &[GpuDevice],
    ) -> Result<ResolvedExpertPlacement, PlacementError> {
        self.validate_header()?;
        let layer_owner = resolve_stable_device(&self.layer_owner, devices)
            .map_err(|error| PlacementError::StableDevice(error.to_string()))?;
        let remote_worker = resolve_stable_device(&self.remote_worker, devices)
            .map_err(|error| PlacementError::StableDevice(error.to_string()))?;
        let (assignments, counts) = self.validate_assignments()?;
        Ok(ResolvedExpertPlacement {
            manifest_hash: self.sha256()?,
            placement_epoch: self.placement_epoch,
            layer_owner,
            remote_worker,
            assignments,
            counts,
        })
    }

    /// Validate the complete manifest without resolving CUDA ordinals.
    pub fn validate_static(&self) -> Result<StaticExpertPlacement, PlacementError> {
        self.validate_header()?;
        let (assignments, counts) = self.validate_assignments()?;
        Ok(StaticExpertPlacement {
            manifest_hash: self.sha256()?,
            placement_epoch: self.placement_epoch,
            layer_owner: self.layer_owner.clone(),
            remote_worker: self.remote_worker.clone(),
            assignments,
            counts,
        })
    }

    fn validate_header(&self) -> Result<(), PlacementError> {
        if self.schema != HETEROGENEOUS_PLACEMENT_SCHEMA_V1 {
            return Err(PlacementError::Schema(self.schema.clone()));
        }
        if !matches!(
            (self.model.num_layers, self.model.experts_per_layer),
            (24, 32) | (36, 128)
        ) || self.model.hidden_size != 2_880
            || self.model.intermediate_size != 2_880
            || self.model.top_k != 4
        {
            return Err(PlacementError::ModelDimensions {
                layers: self.model.num_layers,
                experts: self.model.experts_per_layer,
                hidden: self.model.hidden_size,
                intermediate: self.model.intermediate_size,
                top_k: self.model.top_k,
            });
        }
        if self.model.revision.trim().is_empty() {
            return Err(PlacementError::MissingModelRevision);
        }
        for (field, value) in [
            ("config_sha256", self.model.config_sha256.as_str()),
            ("index_sha256", self.model.index_sha256.as_str()),
            ("mapping_sha256", self.model.mapping_sha256.as_str()),
        ] {
            if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(PlacementError::InvalidSha256 { field });
            }
        }
        if self.layer_owner.pci_bus_id == self.remote_worker.pci_bus_id {
            return Err(PlacementError::DuplicateDevice);
        }
        Ok(())
    }

    fn validate_assignments(
        &self,
    ) -> Result<(BTreeMap<GptOssExpertKey, ExpertOwner>, PlacementOwnerCounts), PlacementError>
    {
        let mut assignments = BTreeMap::new();
        let mut seen = BTreeSet::new();
        let mut counts = PlacementOwnerCounts {
            cpu: 0,
            layer_owner_gpu: 0,
            remote_gpu: 0,
        };
        for assignment in &self.assignments {
            let key = assignment.key;
            if key.layer >= self.model.num_layers || key.expert >= self.model.experts_per_layer {
                return Err(PlacementError::OutOfRange {
                    layer: key.layer,
                    expert: key.expert,
                });
            }
            if !seen.insert(key) {
                return Err(PlacementError::DuplicateAssignment {
                    layer: key.layer,
                    expert: key.expert,
                });
            }
            match &assignment.owner {
                ExpertOwner::Cpu { .. } => counts.cpu += 1,
                ExpertOwner::LayerOwnerGpu { device } if device == &self.layer_owner => {
                    counts.layer_owner_gpu += 1;
                }
                ExpertOwner::RemoteGpu { device } if device == &self.remote_worker => {
                    counts.remote_gpu += 1;
                }
                ExpertOwner::LayerOwnerGpu { .. } | ExpertOwner::RemoteGpu { .. } => {
                    return Err(PlacementError::OwnerDeviceMismatch {
                        layer: key.layer,
                        expert: key.expert,
                    });
                }
            }
            assignments.insert(key, assignment.owner.clone());
        }
        for layer in 0..self.model.num_layers {
            for expert in 0..self.model.experts_per_layer {
                let key = GptOssExpertKey { layer, expert };
                if !seen.contains(&key) {
                    return Err(PlacementError::MissingAssignment { layer, expert });
                }
            }
        }
        if counts.cpu > self.budgets.max_cpu_experts
            || counts.layer_owner_gpu > self.budgets.max_layer_owner_experts
            || counts.remote_gpu > self.budgets.max_remote_gpu_experts
        {
            return Err(PlacementError::CountBudgetExceeded {
                cpu: counts.cpu,
                layer_owner: counts.layer_owner_gpu,
                remote: counts.remote_gpu,
            });
        }
        let cpu_bytes = u64::from(counts.cpu) * CONSERVATIVE_OWNER_EXPERT_BYTES;
        let layer_owner_bytes = u64::from(counts.layer_owner_gpu) * CONSERVATIVE_OWNER_EXPERT_BYTES;
        let remote_bytes = u64::from(counts.remote_gpu) * CONSERVATIVE_OWNER_EXPERT_BYTES;
        if cpu_bytes > self.budgets.max_host_owner_bytes
            || layer_owner_bytes > self.budgets.max_layer_owner_bytes
            || remote_bytes > self.budgets.max_remote_gpu_bytes
        {
            return Err(PlacementError::ByteBudgetExceeded {
                cpu: cpu_bytes,
                layer_owner: layer_owner_bytes,
                remote: remote_bytes,
            });
        }
        Ok((assignments, counts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpt_oss_gpu::device::{GpuDevice, PciBusId};

    fn stable(pci: &str) -> StableCudaDeviceId {
        StableCudaDeviceId {
            pci_bus_id: pci.parse::<PciBusId>().unwrap(),
            expected_name: "NVIDIA GeForce RTX 3090".into(),
            compute_capability: (8, 6),
            minimum_memory: 24 * 1024 * 1024 * 1024,
        }
    }

    fn devices(permuted: bool) -> Vec<GpuDevice> {
        let mut devices = vec![
            GpuDevice {
                id: 0,
                name: "NVIDIA GeForce RTX 3090".into(),
                compute_capability: (8, 6),
                total_memory: 24 * 1024 * 1024 * 1024,
                pci_bus_id: Some("0000:19:00.0".parse().unwrap()),
            },
            GpuDevice {
                id: 1,
                name: "NVIDIA GeForce RTX 3090".into(),
                compute_capability: (8, 6),
                total_memory: 24 * 1024 * 1024 * 1024,
                pci_bus_id: Some("0000:65:00.0".parse().unwrap()),
            },
        ];
        if permuted {
            devices.swap(0, 1);
            devices[0].id = 0;
            devices[1].id = 1;
        }
        devices
    }

    fn manifest(layers: u16, experts: u16) -> GptOssExpertPlacementManifestV1 {
        let layer_owner = stable("0000:19:00.0");
        let remote_worker = stable("0000:65:00.0");
        let assignments = (0..layers)
            .flat_map(|layer| {
                let layer_owner = layer_owner.clone();
                let remote_worker = remote_worker.clone();
                (0..experts).map(move |expert| ExpertAssignment {
                    key: GptOssExpertKey { layer, expert },
                    owner: match expert % 3 {
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
        GptOssExpertPlacementManifestV1 {
            schema: HETEROGENEOUS_PLACEMENT_SCHEMA_V1.into(),
            model: GptOssPlacementModel {
                revision: "revision".into(),
                config_sha256: "1".repeat(64),
                index_sha256: "2".repeat(64),
                mapping_sha256: "3".repeat(64),
                num_layers: layers,
                experts_per_layer: experts,
                hidden_size: 2_880,
                intermediate_size: 2_880,
                top_k: 4,
            },
            layer_owner,
            remote_worker,
            policy: PlacementPolicyClass::Proof,
            policy_seed: 7,
            placement_epoch: 1,
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

    #[test]
    fn validates_complete_20b_and_120b_rectangles() {
        let twenty_manifest = manifest(24, 32);
        let static_twenty = twenty_manifest.validate_static().unwrap();
        let twenty = twenty_manifest.validate(&devices(false)).unwrap();
        assert_eq!(twenty.len(), 24 * 32);
        assert_eq!(static_twenty.assignments().count(), twenty.len());
        assert_eq!(static_twenty.counts(), twenty.counts());
        assert_eq!(static_twenty.manifest_hash(), twenty.manifest_hash());
        assert_eq!(static_twenty.layer_owner(), &twenty_manifest.layer_owner);
        assert_eq!(
            static_twenty.remote_worker(),
            &twenty_manifest.remote_worker
        );
        let one_twenty = manifest(36, 128).validate(&devices(false)).unwrap();
        assert_eq!(one_twenty.len(), 36 * 128);
    }

    #[test]
    fn stable_devices_resolve_after_ordinal_permutation() {
        let resolved = manifest(24, 32).validate(&devices(true)).unwrap();
        assert_eq!(resolved.layer_owner().transient_ordinal, 1);
        assert_eq!(resolved.remote_worker().transient_ordinal, 0);
    }

    #[test]
    fn rejects_pci_identity_or_model_revision_mismatch() {
        let mut wrong_pci = manifest(24, 32);
        wrong_pci.layer_owner.pci_bus_id = "0000:20:00.0".parse().unwrap();
        assert!(matches!(
            wrong_pci.validate(&devices(false)),
            Err(PlacementError::StableDevice(_))
        ));

        let mut missing_revision = manifest(24, 32);
        missing_revision.model.revision.clear();
        assert!(matches!(
            missing_revision.validate(&devices(false)),
            Err(PlacementError::MissingModelRevision)
        ));
    }

    #[test]
    fn rejects_duplicate_missing_and_wrong_device_assignments() {
        let mut duplicate = manifest(24, 32);
        duplicate.assignments.push(duplicate.assignments[0].clone());
        assert!(matches!(
            duplicate.validate(&devices(false)),
            Err(PlacementError::DuplicateAssignment { .. })
        ));

        let mut missing = manifest(24, 32);
        missing.assignments.pop();
        assert!(matches!(
            missing.validate(&devices(false)),
            Err(PlacementError::MissingAssignment { .. })
        ));

        let mut wrong = manifest(24, 32);
        wrong.assignments[0].owner = ExpertOwner::RemoteGpu {
            device: wrong.layer_owner.clone(),
        };
        assert!(matches!(
            wrong.validate(&devices(false)),
            Err(PlacementError::OwnerDeviceMismatch { .. })
        ));
    }

    #[test]
    fn manifest_hash_is_assignment_order_independent() {
        let first = manifest(24, 32);
        let mut reversed = first.clone();
        reversed.assignments.reverse();
        assert_eq!(first.sha256().unwrap(), reversed.sha256().unwrap());
        assert_eq!(
            first.stable_json().unwrap(),
            reversed.stable_json().unwrap()
        );
        let decoded: GptOssExpertPlacementManifestV1 =
            serde_json::from_slice(&first.stable_json().unwrap()).unwrap();
        assert_eq!(decoded, first);
    }

    #[test]
    fn rejects_a_byte_budget_that_count_budgets_would_accept() {
        let mut manifest = manifest(24, 32);
        manifest.budgets.max_host_owner_bytes = 0;
        assert!(matches!(
            manifest.validate(&devices(false)),
            Err(PlacementError::ByteBudgetExceeded { .. })
        ));
    }
}
