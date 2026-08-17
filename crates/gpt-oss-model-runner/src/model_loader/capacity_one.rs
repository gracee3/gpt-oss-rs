//! Payload-free planning, bounded split state, and publication proofs for the
//! capacity-one heterogeneous constructor.

use std::collections::{BTreeMap, BTreeSet};

use gpt_oss_core::error::{LLMError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::heterogeneous::placement::{ExpertOwner, GptOssExpertKey};

use super::shard_consumer_plan::{
    GptOssExpertSurface, GptOssShardConsumer, GptOssShardConsumerAction, GptOssShardConsumerPlan,
};

pub const CAPACITY_ONE_POLICY_SHA256: &str =
    "f269a4c984bbfa0d2a18c037b42ded2c81330094b18c6fc8dc668b7ad81bb90f";
pub const SPLIT_BIAS_BYTES: u64 = 11_520;
pub const RETAINED_120B_SPLIT_BOUND_BYTES: u64 = 1_474_560;
pub const RETAINED_MAX_SOURCE_MAPPING_BYTES: u64 = 10_544_040_680;
pub const RETAINED_MAX_PINNED_CONSTRUCTION_BYTES: u64 = 33_554_432;
pub const RETAINED_MAX_DIRTY_CPU_OUTPUT_BYTES: u64 = 675_941_760;
pub const MAX_BOUNDED_JOURNAL_ACTION_IDS: usize = 100_000;
pub const R2_PREFLIGHT_SAMPLE_COUNT: u32 = 5;
pub const R2_PREFLIGHT_DURATION_MILLIS: u64 = 120_000;
pub const R2_PREFLIGHT_SAMPLE_INTERVAL_MILLIS: u64 = 1_000;
pub const R2_SETTLE_DURATION_SECONDS: u64 = 30;
pub const R2_MEM_AVAILABLE_FLOOR_BYTES: u64 = 12_884_901_888;
pub const R2_CGROUP_CLEAN_FILE_ALLOWANCE_BYTES: u64 = 11_488_417_896;
pub const R2_DIRTY_WRITEBACK_ALLOWANCE_BYTES: u64 = 944_377_216;
pub const R2_CLEANUP_DRIFT_ALLOWANCE_BYTES: u64 = 67_108_864;
pub const R2_DISK_RESERVE_BYTES: u64 = 68_719_476_736;
pub const R2_MEMORY_PSI_SOME_AVG10_MAX_MICROS: u64 = 0;
pub const R2_MEMORY_PSI_FULL_AVG10_MAX_MICROS: u64 = 0;

/// Frozen numeric admission gates. These values are policy identity inputs,
/// not adaptive tuning knobs, and must not be changed after a model failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacityOneAdmissionPolicy {
    pub preflight_sample_count: u32,
    pub preflight_duration_millis: u64,
    pub preflight_sample_interval_millis: u64,
    pub settle_duration_seconds: u64,
    pub mem_available_floor_bytes: u64,
    pub cgroup_clean_file_allowance_bytes: u64,
    pub dirty_writeback_allowance_bytes: u64,
    pub cleanup_drift_allowance_bytes: u64,
    pub disk_reserve_bytes: u64,
    pub memory_psi_some_avg10_max_micros: u64,
    pub memory_psi_full_avg10_max_micros: u64,
}

impl CapacityOneAdmissionPolicy {
    pub const fn frozen_r2() -> Self {
        Self {
            preflight_sample_count: R2_PREFLIGHT_SAMPLE_COUNT,
            preflight_duration_millis: R2_PREFLIGHT_DURATION_MILLIS,
            preflight_sample_interval_millis: R2_PREFLIGHT_SAMPLE_INTERVAL_MILLIS,
            settle_duration_seconds: R2_SETTLE_DURATION_SECONDS,
            mem_available_floor_bytes: R2_MEM_AVAILABLE_FLOOR_BYTES,
            cgroup_clean_file_allowance_bytes: R2_CGROUP_CLEAN_FILE_ALLOWANCE_BYTES,
            dirty_writeback_allowance_bytes: R2_DIRTY_WRITEBACK_ALLOWANCE_BYTES,
            cleanup_drift_allowance_bytes: R2_CLEANUP_DRIFT_ALLOWANCE_BYTES,
            disk_reserve_bytes: R2_DISK_RESERVE_BYTES,
            memory_psi_some_avg10_max_micros: R2_MEMORY_PSI_SOME_AVG10_MAX_MICROS,
            memory_psi_full_avg10_max_micros: R2_MEMORY_PSI_FULL_AVG10_MAX_MICROS,
        }
    }

    pub fn validate_identity(&self) -> Result<()> {
        if *self != Self::frozen_r2() {
            return Err(model_error(
                "capacity-one admission policy differs from frozen R2",
            ));
        }
        Ok(())
    }

    pub fn admit_sample(&self, sample: CapacityOneAdmissionSample) -> Result<()> {
        self.validate_identity()?;
        if sample.process_vm_swap_bytes != 0
            || sample.cgroup_swap_current_bytes != 0
            || sample.global_swap_used_bytes > sample.admitted_global_swap_used_bytes
            || sample.mem_available_bytes < self.mem_available_floor_bytes
            || sample.memory_psi_some_avg10_micros > self.memory_psi_some_avg10_max_micros
            || sample.memory_psi_full_avg10_micros > self.memory_psi_full_avg10_max_micros
        {
            return Err(model_error(
                "capacity-one memory/swap/PSI admission gate failed",
            ));
        }
        Ok(())
    }

    pub fn admit_settled_cleanup(&self, sample: CapacityOneSettleSample) -> Result<()> {
        self.validate_identity()?;
        if sample.cgroup_clean_file_bytes > self.cgroup_clean_file_allowance_bytes
            || sample.dirty_writeback_bytes > self.dirty_writeback_allowance_bytes
            || sample.cleanup_drift_bytes > self.cleanup_drift_allowance_bytes
        {
            return Err(model_error("capacity-one settle/cleanup allowance failed"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacityOneAdmissionSample {
    pub process_vm_swap_bytes: u64,
    pub cgroup_swap_current_bytes: u64,
    pub admitted_global_swap_used_bytes: u64,
    pub global_swap_used_bytes: u64,
    pub mem_available_bytes: u64,
    pub memory_psi_some_avg10_micros: u64,
    pub memory_psi_full_avg10_micros: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapacityOneSettleSample {
    pub cgroup_clean_file_bytes: u64,
    pub dirty_writeback_bytes: u64,
    pub cleanup_drift_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct ExpertPartialKey {
    pub key: GptOssExpertKey,
    pub owner: ExpertOwner,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SplitExpertPlanEntry {
    pub identity: ExpertPartialKey,
    pub earlier_action_id_sha256: String,
    pub earlier_shard_index: usize,
    pub later_shard_index: usize,
    pub later_action_ids_sha256: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExpertPartialPlan {
    entries: BTreeMap<ExpertPartialKey, SplitExpertPlanEntry>,
    derived_high_water_count: usize,
    derived_high_water_bytes: u64,
    derived_owner_high_waters: Vec<OwnerPartialHighWater>,
    approved_bound_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OwnerPartialHighWater {
    pub owner: ExpertOwner,
    pub count: usize,
    pub bytes: u64,
}

impl ExpertPartialPlan {
    pub fn derive(plan: &GptOssShardConsumerPlan, approved_bound_bytes: u64) -> Result<Self> {
        plan.validate_identity()?;
        if approved_bound_bytes == 0 {
            return Err(model_error("partial-store approved bound is zero"));
        }
        let mut expert_actions =
            BTreeMap::<ExpertPartialKey, BTreeMap<GptOssExpertSurface, (usize, String, u64)>>::new(
            );
        for (shard_index, shard) in plan.shards().iter().enumerate() {
            for action in &shard.actions {
                let GptOssShardConsumer::OwnedExpert {
                    key,
                    owner,
                    surface,
                } = &action.consumer
                else {
                    continue;
                };
                let identity = ExpertPartialKey {
                    key: *key,
                    owner: owner.clone(),
                };
                if expert_actions
                    .entry(identity)
                    .or_default()
                    .insert(
                        *surface,
                        (
                            shard_index,
                            action.action_id_sha256.clone(),
                            action.byte_len()?,
                        ),
                    )
                    .is_some()
                {
                    return Err(model_error(format!(
                        "duplicate expert surface in plan for ({},{})",
                        key.layer, key.expert
                    )));
                }
            }
        }

        let expected_surfaces = all_surfaces().into_iter().collect::<BTreeSet<_>>();
        let mut entries = BTreeMap::new();
        let mut boundary_counts = BTreeMap::<usize, usize>::new();
        let mut boundary_owner_counts = BTreeMap::<usize, BTreeMap<ExpertOwner, usize>>::new();
        for (identity, surfaces) in expert_actions {
            if surfaces.keys().copied().collect::<BTreeSet<_>>() != expected_surfaces {
                return Err(model_error(format!(
                    "expert ({},{}) does not have exactly six planned surfaces",
                    identity.key.layer, identity.key.expert
                )));
            }
            for (surface, (_, _, bytes)) in &surfaces {
                let expected = surface_bytes(*surface);
                if *bytes != expected {
                    return Err(model_error(format!(
                        "expert ({},{}) surface {surface:?} has {bytes} bytes, expected {expected}",
                        identity.key.layer, identity.key.expert
                    )));
                }
            }
            let (bias_shard, bias_action, _) = surfaces
                .get(&GptOssExpertSurface::GateUpBias)
                .expect("six-surface set contains bias");
            let later_shards = all_surfaces()
                .into_iter()
                .filter(|surface| *surface != GptOssExpertSurface::GateUpBias)
                .map(|surface| surfaces[&surface].0)
                .collect::<BTreeSet<_>>();
            if later_shards.len() != 1 {
                return Err(model_error(format!(
                    "expert ({},{}) later surfaces span more than one shard",
                    identity.key.layer, identity.key.expert
                )));
            }
            let later_shard = *later_shards.iter().next().expect("one later shard");
            if *bias_shard == later_shard {
                continue;
            }
            if bias_shard.checked_add(1) != Some(later_shard) {
                return Err(model_error(format!(
                    "expert ({},{}) split is not across adjacent shards",
                    identity.key.layer, identity.key.expert
                )));
            }
            let later_action_ids_sha256 = all_surfaces()
                .into_iter()
                .filter(|surface| *surface != GptOssExpertSurface::GateUpBias)
                .map(|surface| surfaces[&surface].1.clone())
                .collect::<Vec<_>>();
            *boundary_counts.entry(*bias_shard).or_default() += 1;
            *boundary_owner_counts
                .entry(*bias_shard)
                .or_default()
                .entry(identity.owner.clone())
                .or_default() += 1;
            entries.insert(
                identity.clone(),
                SplitExpertPlanEntry {
                    identity,
                    earlier_action_id_sha256: bias_action.clone(),
                    earlier_shard_index: *bias_shard,
                    later_shard_index: later_shard,
                    later_action_ids_sha256,
                },
            );
        }
        let derived_high_water_count = boundary_counts.values().copied().max().unwrap_or(0);
        let derived_high_water_bytes = u64::try_from(derived_high_water_count)
            .ok()
            .and_then(|count| count.checked_mul(SPLIT_BIAS_BYTES))
            .ok_or_else(|| model_error("partial-store derived byte bound overflows"))?;
        if derived_high_water_bytes > approved_bound_bytes {
            return Err(model_error(format!(
                "partial-store plan requires {derived_high_water_bytes} bytes, above approved {approved_bound_bytes}"
            )));
        }
        let mut owner_counts = BTreeMap::<ExpertOwner, usize>::new();
        for boundary in boundary_owner_counts.values() {
            for (owner, count) in boundary {
                owner_counts
                    .entry(owner.clone())
                    .and_modify(|maximum| *maximum = (*maximum).max(*count))
                    .or_insert(*count);
            }
        }
        let derived_owner_high_waters = owner_counts
            .into_iter()
            .map(|(owner, count)| {
                let bytes = u64::try_from(count)
                    .ok()
                    .and_then(|count| count.checked_mul(SPLIT_BIAS_BYTES))
                    .ok_or_else(|| model_error("owner partial-store bound overflows"))?;
                Ok(OwnerPartialHighWater {
                    owner,
                    count,
                    bytes,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            entries,
            derived_high_water_count,
            derived_high_water_bytes,
            derived_owner_high_waters,
            approved_bound_bytes,
        })
    }

    pub fn entry(&self, key: &ExpertPartialKey) -> Option<&SplitExpertPlanEntry> {
        self.entries.get(key)
    }

    pub fn entries(&self) -> impl Iterator<Item = &SplitExpertPlanEntry> {
        self.entries.values()
    }

    pub const fn derived_high_water_count(&self) -> usize {
        self.derived_high_water_count
    }

    pub const fn derived_high_water_bytes(&self) -> u64 {
        self.derived_high_water_bytes
    }

    pub const fn approved_bound_bytes(&self) -> u64 {
        self.approved_bound_bytes
    }

    pub fn derived_owner_high_waters(&self) -> &[OwnerPartialHighWater] {
        &self.derived_owner_high_waters
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ExpertPartialStoreStats {
    pub current_count: usize,
    pub current_bytes: u64,
    pub high_water_count: usize,
    pub high_water_bytes: u64,
}

struct OwnedSplitBias {
    bytes: Vec<u8>,
    sha256: String,
    action_id_sha256: String,
    earlier_shard_index: usize,
}

pub struct CompletedSplitBias {
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub earlier_action_id_sha256: String,
}

pub struct ExpertPartialStore {
    plan: ExpertPartialPlan,
    entries: BTreeMap<ExpertPartialKey, OwnedSplitBias>,
    stats: ExpertPartialStoreStats,
}

impl ExpertPartialStore {
    pub fn new(plan: ExpertPartialPlan) -> Self {
        Self {
            plan,
            entries: BTreeMap::new(),
            stats: ExpertPartialStoreStats::default(),
        }
    }

    pub fn insert_bias(
        &mut self,
        shard_index: usize,
        action: &GptOssShardConsumerAction,
        bytes: &[u8],
    ) -> Result<()> {
        let GptOssShardConsumer::OwnedExpert {
            key,
            owner,
            surface: GptOssExpertSurface::GateUpBias,
        } = &action.consumer
        else {
            return Err(model_error(
                "partial store accepts only gate/up bias actions",
            ));
        };
        let identity = ExpertPartialKey {
            key: *key,
            owner: owner.clone(),
        };
        let expected = self.plan.entry(&identity).ok_or_else(|| {
            model_error(format!(
                "unexpected split bias for ({},{})",
                key.layer, key.expert
            ))
        })?;
        if shard_index != expected.earlier_shard_index
            || action.action_id_sha256 != expected.earlier_action_id_sha256
            || action.byte_len()? != SPLIT_BIAS_BYTES
            || bytes.len() as u64 != SPLIT_BIAS_BYTES
        {
            return Err(model_error(
                "split bias identity, shard, or length mismatch",
            ));
        }
        if self.entries.contains_key(&identity) {
            return Err(model_error("duplicate split bias"));
        }
        let next_bytes = self
            .stats
            .current_bytes
            .checked_add(SPLIT_BIAS_BYTES)
            .ok_or_else(|| model_error("partial-store byte count overflows"))?;
        if next_bytes > self.plan.approved_bound_bytes() {
            return Err(model_error("partial-store runtime bound exceeded"));
        }
        self.entries.insert(
            identity,
            OwnedSplitBias {
                bytes: bytes.to_vec(),
                sha256: hash_bytes(bytes),
                action_id_sha256: action.action_id_sha256.clone(),
                earlier_shard_index: shard_index,
            },
        );
        self.stats.current_count += 1;
        self.stats.current_bytes = next_bytes;
        self.stats.high_water_count = self.stats.high_water_count.max(self.stats.current_count);
        self.stats.high_water_bytes = self.stats.high_water_bytes.max(self.stats.current_bytes);
        Ok(())
    }

    pub fn take_for_completion(
        &mut self,
        identity: &ExpertPartialKey,
        shard_index: usize,
        later_action_ids_sha256: &[String],
    ) -> Result<CompletedSplitBias> {
        let expected = self
            .plan
            .entry(identity)
            .ok_or_else(|| model_error("expert is not a planned split"))?;
        let stored = self
            .entries
            .get(identity)
            .ok_or_else(|| model_error("split expert is missing its earlier bias"))?;
        if shard_index != expected.later_shard_index
            || later_action_ids_sha256 != expected.later_action_ids_sha256
            || stored.action_id_sha256 != expected.earlier_action_id_sha256
            || stored.earlier_shard_index != expected.earlier_shard_index
            || stored.bytes.len() as u64 != SPLIT_BIAS_BYTES
            || hash_bytes(&stored.bytes) != stored.sha256
        {
            return Err(model_error(
                "split expert later completion does not match stored bias",
            ));
        }
        let stored = self
            .entries
            .remove(identity)
            .expect("validated partial entry is present");
        self.stats.current_count -= 1;
        self.stats.current_bytes -= SPLIT_BIAS_BYTES;
        Ok(CompletedSplitBias {
            bytes: stored.bytes,
            sha256: stored.sha256,
            earlier_action_id_sha256: stored.action_id_sha256,
        })
    }

    pub const fn stats(&self) -> ExpertPartialStoreStats {
        self.stats
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn require_empty(&self) -> Result<()> {
        if self.is_empty() && self.stats.current_count == 0 && self.stats.current_bytes == 0 {
            Ok(())
        } else {
            Err(model_error("partial store is not empty"))
        }
    }

    pub fn cancel(&mut self) {
        self.entries.clear();
        self.stats.current_count = 0;
        self.stats.current_bytes = 0;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WarmRecordElisionProof {
    pub catalog_sha256: String,
    pub source_revision: String,
    pub mapping_sha256: String,
    pub placement_sha256: String,
    pub placement_epoch: u64,
    pub format_version: u32,
    pub layer: u16,
    pub ordered_expert_ids: Vec<u16>,
    pub record_identity_sha256: String,
    pub action_ids_sha256: Vec<String>,
    pub native_bytes: u64,
}

pub struct ExactActionCoverage {
    expected: BTreeMap<String, u64>,
    consumed: BTreeSet<String>,
    elided: BTreeSet<String>,
    consumed_bytes: u64,
    elided_bytes: u64,
}

impl ExactActionCoverage {
    pub fn new(plan: &GptOssShardConsumerPlan) -> Result<Self> {
        plan.validate_identity()?;
        if plan.total_actions() > MAX_BOUNDED_JOURNAL_ACTION_IDS {
            return Err(model_error(
                "action coverage exceeds bounded journal capacity",
            ));
        }
        let mut expected = BTreeMap::new();
        for shard in plan.shards() {
            for action in &shard.actions {
                if expected
                    .insert(action.action_id_sha256.clone(), action.byte_len()?)
                    .is_some()
                {
                    return Err(model_error("duplicate action identity in plan"));
                }
            }
        }
        Ok(Self {
            expected,
            consumed: BTreeSet::new(),
            elided: BTreeSet::new(),
            consumed_bytes: 0,
            elided_bytes: 0,
        })
    }

    pub fn consume(&mut self, action: &GptOssShardConsumerAction) -> Result<()> {
        self.mark(&action.action_id_sha256, action.byte_len()?, false)
    }

    pub fn elide(&mut self, action_id_sha256: &str, bytes: u64) -> Result<()> {
        self.mark(action_id_sha256, bytes, true)
    }

    fn mark(&mut self, action_id_sha256: &str, bytes: u64, elided: bool) -> Result<()> {
        let expected = self
            .expected
            .get(action_id_sha256)
            .ok_or_else(|| model_error("unexpected action identity"))?;
        if *expected != bytes
            || self.consumed.contains(action_id_sha256)
            || self.elided.contains(action_id_sha256)
        {
            return Err(model_error("duplicate action or action byte mismatch"));
        }
        if elided {
            self.elided.insert(action_id_sha256.to_owned());
            self.elided_bytes = self
                .elided_bytes
                .checked_add(bytes)
                .ok_or_else(|| model_error("elided byte count overflows"))?;
        } else {
            self.consumed.insert(action_id_sha256.to_owned());
            self.consumed_bytes = self
                .consumed_bytes
                .checked_add(bytes)
                .ok_or_else(|| model_error("consumed byte count overflows"))?;
        }
        Ok(())
    }

    pub fn validate_complete(&self) -> Result<()> {
        if self.consumed.len() + self.elided.len() != self.expected.len()
            || self
                .expected
                .keys()
                .any(|id| !self.consumed.contains(id) && !self.elided.contains(id))
        {
            return Err(model_error("action coverage is incomplete"));
        }
        Ok(())
    }

    pub fn consumed_count(&self) -> usize {
        self.consumed.len()
    }

    pub fn elided_count(&self) -> usize {
        self.elided.len()
    }

    pub const fn consumed_bytes(&self) -> u64 {
        self.consumed_bytes
    }

    pub const fn elided_bytes(&self) -> u64 {
        self.elided_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OwnerSelectivePublicationProof {
    pub catalog_identity_exact: bool,
    pub action_coverage_complete: bool,
    pub warm_elision_complete: bool,
    pub active_source_mappings: usize,
    pub active_source_payload_fds: usize,
    pub source_payload_views: usize,
    pub borrowed_source_slices: usize,
    pub source_inode_mappings: usize,
    pub source_inode_pss_bytes: u64,
    pub partial_store_entries: usize,
    pub partial_store_bytes: u64,
    pub incomplete_cpu_experts: usize,
    pub task_temporaries: usize,
    pub pending_cuda_receipts: usize,
    pub quarantined_cuda_receipts: usize,
    pub cold_records_directory_synced: bool,
    pub records_freshly_validated: bool,
    pub runtime_maps_after_source_release: bool,
    pub stable_device_ownership_complete: bool,
    pub bounded_journal_complete: bool,
    pub visibility_contract_unchanged: bool,
}

impl OwnerSelectivePublicationProof {
    pub fn validate(&self) -> Result<()> {
        if !self.catalog_identity_exact
            || !self.action_coverage_complete
            || !self.warm_elision_complete
            || self.active_source_mappings != 0
            || self.active_source_payload_fds != 0
            || self.source_payload_views != 0
            || self.borrowed_source_slices != 0
            || self.source_inode_mappings != 0
            || self.source_inode_pss_bytes != 0
            || self.partial_store_entries != 0
            || self.partial_store_bytes != 0
            || self.incomplete_cpu_experts != 0
            || self.task_temporaries != 0
            || self.pending_cuda_receipts != 0
            || self.quarantined_cuda_receipts != 0
            || !self.cold_records_directory_synced
            || !self.records_freshly_validated
            || !self.runtime_maps_after_source_release
            || !self.stable_device_ownership_complete
            || !self.bounded_journal_complete
            || !self.visibility_contract_unchanged
        {
            return Err(model_error(
                "owner-selective publication proof is incomplete",
            ));
        }
        Ok(())
    }
}

pub const fn surface_bytes(surface: GptOssExpertSurface) -> u64 {
    match surface {
        GptOssExpertSurface::GateUpBias => 11_520,
        GptOssExpertSurface::GateUpBlocks => 8_294_400,
        GptOssExpertSurface::GateUpScales => 518_400,
        GptOssExpertSurface::DownBias => 5_760,
        GptOssExpertSurface::DownBlocks => 4_147_200,
        GptOssExpertSurface::DownScales => 259_200,
    }
}

pub const fn all_surfaces() -> [GptOssExpertSurface; 6] {
    [
        GptOssExpertSurface::GateUpBias,
        GptOssExpertSurface::GateUpBlocks,
        GptOssExpertSurface::GateUpScales,
        GptOssExpertSurface::DownBias,
        GptOssExpertSurface::DownBlocks,
        GptOssExpertSurface::DownScales,
    ]
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn model_error(message: impl Into<String>) -> LLMError {
    LLMError::ModelError(message.into())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs::{File, OpenOptions};
    use std::io::{Seek, SeekFrom, Write};
    use std::os::unix::fs::FileExt;
    use std::path::Path;

    use gpt_oss_gpu::device::{PciBusId, StableCudaDeviceId};
    use serde::Serialize;
    use tempfile::{tempdir, TempDir};

    use super::*;
    use crate::heterogeneous::placement::CpuPoolId;
    use crate::model_loader::shard_catalog::SafeTensorFileIdentity;
    use crate::model_loader::shard_catalog::{SafeTensorShardCatalog, ShardReleaseLogicalLedger};
    use crate::model_loader::shard_consumer_plan::GptOssShardConsumption;

    fn owner() -> ExpertOwner {
        ExpertOwner::LayerOwnerGpu {
            device: StableCudaDeviceId {
                pci_bus_id: PciBusId::new(0, 0x19, 0, 0),
                expected_name: "fixture".into(),
                compute_capability: (8, 6),
                minimum_memory: 1,
            },
        }
    }

    fn split_plan(identity: ExpertPartialKey) -> ExpertPartialPlan {
        let entry = SplitExpertPlanEntry {
            identity: identity.clone(),
            earlier_action_id_sha256: "a".repeat(64),
            earlier_shard_index: 0,
            later_shard_index: 1,
            later_action_ids_sha256: (0..5).map(|index| format!("later-{index}")).collect(),
        };
        ExpertPartialPlan {
            entries: BTreeMap::from([(identity, entry)]),
            derived_high_water_count: 1,
            derived_high_water_bytes: SPLIT_BIAS_BYTES,
            derived_owner_high_waters: Vec::new(),
            approved_bound_bytes: SPLIT_BIAS_BYTES,
        }
    }

    fn bias_action(identity: &ExpertPartialKey) -> GptOssShardConsumerAction {
        GptOssShardConsumerAction {
            action_id_sha256: "a".repeat(64),
            native_tensor: "bias".into(),
            native_tensor_range: [0, SPLIT_BIAS_BYTES],
            shard_absolute_range: [8, 8 + SPLIT_BIAS_BYTES],
            consumer: GptOssShardConsumer::OwnedExpert {
                key: identity.key,
                owner: identity.owner.clone(),
                surface: GptOssExpertSurface::GateUpBias,
            },
        }
    }

    fn retained_bound_fixture(boundaries: &[[usize; 3]]) -> GptOssShardConsumerPlan {
        let remote = ExpertOwner::RemoteGpu {
            device: StableCudaDeviceId {
                pci_bus_id: PciBusId::new(0, 0x65, 0, 0),
                expected_name: "fixture-remote".into(),
                compute_capability: (8, 6),
                minimum_memory: 1,
            },
        };
        let gpu0 = owner();
        let mut shard_actions = (0..boundaries.len() * 2)
            .map(|_| Vec::new())
            .collect::<Vec<_>>();
        let mut offsets = vec![8_u64; shard_actions.len()];
        for (boundary, [gpu0_count, gpu1_count, cpu_count]) in
            boundaries.iter().copied().enumerate()
        {
            let split_count = gpu0_count + gpu1_count + cpu_count;
            for expert in 0..split_count {
                let expert_owner = if expert < gpu0_count {
                    gpu0.clone()
                } else if expert < gpu0_count + gpu1_count {
                    remote.clone()
                } else {
                    ExpertOwner::Cpu { pool: CpuPoolId(0) }
                };
                let key = GptOssExpertKey {
                    layer: u16::try_from(boundary).unwrap(),
                    expert: u16::try_from(expert).unwrap(),
                };
                for surface in all_surfaces() {
                    let shard_index =
                        boundary * 2 + usize::from(surface != GptOssExpertSurface::GateUpBias);
                    let bytes = surface_bytes(surface);
                    let start = offsets[shard_index];
                    let end = start + bytes;
                    offsets[shard_index] = end;
                    shard_actions[shard_index].push(GptOssShardConsumerAction {
                        action_id_sha256: String::new(),
                        native_tensor: format!("boundary-{boundary}-expert-{expert}-{surface:?}"),
                        native_tensor_range: [0, bytes],
                        shard_absolute_range: [start, end],
                        consumer: GptOssShardConsumer::OwnedExpert {
                            key,
                            owner: expert_owner.clone(),
                            surface,
                        },
                    });
                }
            }
        }
        let shards = shard_actions
            .into_iter()
            .enumerate()
            .map(|(index, actions)| GptOssShardConsumption {
                shard: SafeTensorFileIdentity {
                    file_name: format!("fixture-{index}.safetensors"),
                    file_length: offsets[index],
                    header_sha256: format!("{:064x}", index + 1),
                    data_start: 8,
                    payload_length: offsets[index] - 8,
                    device: 0,
                    inode: 0,
                },
                planned_payload_bytes: 0,
                actions,
            })
            .collect();
        GptOssShardConsumerPlan::from_capacity_one_test_shards("c".repeat(64), shards)
    }

    #[derive(Serialize)]
    struct FixtureTensorHeader {
        dtype: &'static str,
        shape: [u64; 1],
        data_offsets: [u64; 2],
    }

    struct DeterministicFixture {
        _root: TempDir,
        catalog: SafeTensorShardCatalog,
        plan: GptOssShardConsumerPlan,
    }

    fn push_expert_specs(
        target: &mut Vec<(String, u64, GptOssShardConsumer)>,
        label: &str,
        key: GptOssExpertKey,
        owner: ExpertOwner,
        surfaces: impl IntoIterator<Item = GptOssExpertSurface>,
    ) {
        for surface in surfaces {
            target.push((
                format!("{label}-{surface:?}"),
                surface_bytes(surface),
                GptOssShardConsumer::OwnedExpert {
                    key,
                    owner: owner.clone(),
                    surface,
                },
            ));
        }
    }

    fn write_fixture_shard(path: &Path, specs: &[(String, u64, GptOssShardConsumer)], seed: u8) {
        let mut next = 0_u64;
        let mut header = BTreeMap::new();
        let mut ranges = Vec::new();
        for (name, bytes, _) in specs {
            let start = next;
            next += *bytes;
            header.insert(
                name.clone(),
                FixtureTensorHeader {
                    dtype: "U8",
                    shape: [*bytes],
                    data_offsets: [start, next],
                },
            );
            ranges.push((start, *bytes));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        let data_start = 8 + header.len() as u64;
        file.set_len(data_start + next).unwrap();
        for (ordinal, (start, bytes)) in ranges.into_iter().enumerate() {
            file.seek(SeekFrom::Start(data_start + start)).unwrap();
            let marker_len = usize::try_from(bytes.min(64)).unwrap();
            let marker = (0..marker_len)
                .map(|byte| seed.wrapping_add(ordinal as u8).wrapping_add(byte as u8))
                .collect::<Vec<_>>();
            file.write_all(&marker).unwrap();
        }
        file.sync_all().unwrap();
    }

    fn deterministic_fixture() -> DeterministicFixture {
        let root = tempdir().unwrap();
        let gpu0 = owner();
        let gpu1 = ExpertOwner::RemoteGpu {
            device: StableCudaDeviceId {
                pci_bus_id: PciBusId::new(0, 0x65, 0, 0),
                expected_name: "fixture-remote".into(),
                compute_capability: (8, 6),
                minimum_memory: 1,
            },
        };
        let cpu = ExpertOwner::Cpu { pool: CpuPoolId(0) };
        let local0 = GptOssExpertKey {
            layer: 0,
            expert: 0,
        };
        let local1 = GptOssExpertKey {
            layer: 0,
            expert: 1,
        };
        let split_gpu = GptOssExpertKey {
            layer: 0,
            expert: 2,
        };
        let split_cpu = GptOssExpertKey {
            layer: 0,
            expert: 3,
        };
        let warm_cpu = GptOssExpertKey {
            layer: 1,
            expert: 0,
        };
        let invalid_warm_cpu = GptOssExpertKey {
            layer: 2,
            expert: 0,
        };
        let later = all_surfaces()
            .into_iter()
            .filter(|surface| *surface != GptOssExpertSurface::GateUpBias)
            .collect::<Vec<_>>();
        let mut specs = [Vec::new(), Vec::new(), Vec::new()];
        specs[0].push((
            "dense".into(),
            37,
            GptOssShardConsumer::LayerOwnerDense {
                runtime_tensor: "model.embed_tokens.weight".into(),
            },
        ));
        push_expert_specs(
            &mut specs[0],
            "gpu0-local",
            local0,
            gpu0.clone(),
            all_surfaces(),
        );
        push_expert_specs(
            &mut specs[0],
            "gpu1-local",
            local1,
            gpu1.clone(),
            all_surfaces(),
        );
        push_expert_specs(
            &mut specs[1],
            "gpu-split",
            split_gpu,
            gpu0.clone(),
            [GptOssExpertSurface::GateUpBias],
        );
        push_expert_specs(
            &mut specs[1],
            "cpu-split",
            split_cpu,
            cpu.clone(),
            [GptOssExpertSurface::GateUpBias],
        );
        push_expert_specs(
            &mut specs[1],
            "cpu-warm-valid",
            warm_cpu,
            cpu.clone(),
            all_surfaces(),
        );
        push_expert_specs(
            &mut specs[1],
            "cpu-warm-invalid",
            invalid_warm_cpu,
            cpu.clone(),
            all_surfaces(),
        );
        push_expert_specs(
            &mut specs[2],
            "gpu-split",
            split_gpu,
            gpu0,
            later.iter().copied(),
        );
        push_expert_specs(&mut specs[2], "cpu-split", split_cpu, cpu, later);
        for (index, shard) in specs.iter().enumerate() {
            write_fixture_shard(
                &root.path().join(format!("model-{index:02}.safetensors")),
                shard,
                17 + index as u8,
            );
        }
        let weight_map = specs
            .iter()
            .enumerate()
            .flat_map(|(index, shard)| {
                shard.iter().map(move |(name, _, _)| {
                    (name.clone(), format!("model-{index:02}.safetensors"))
                })
            })
            .collect::<BTreeMap<_, _>>();
        let total_size = specs
            .iter()
            .flatten()
            .map(|(_, bytes, _)| *bytes)
            .sum::<u64>();
        std::fs::write(
            root.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }))
            .unwrap(),
        )
        .unwrap();
        let catalog = SafeTensorShardCatalog::open(root.path()).unwrap();
        let consumers = specs
            .into_iter()
            .flatten()
            .map(|(name, _, consumer)| (name, consumer))
            .collect::<BTreeMap<_, _>>();
        let shards = catalog
            .shards()
            .iter()
            .enumerate()
            .map(|(shard_index, shard)| {
                let mut tensors = catalog
                    .tensors()
                    .filter(|tensor| tensor.shard_index == shard_index)
                    .collect::<Vec<_>>();
                tensors.sort_by_key(|tensor| tensor.absolute_range);
                let actions = tensors
                    .into_iter()
                    .map(|tensor| GptOssShardConsumerAction {
                        action_id_sha256: String::new(),
                        native_tensor: tensor.name.clone(),
                        native_tensor_range: [0, tensor.byte_len()],
                        shard_absolute_range: tensor.absolute_range,
                        consumer: consumers[&tensor.name].clone(),
                    })
                    .collect();
                GptOssShardConsumption {
                    shard: shard.identity.clone(),
                    actions,
                    planned_payload_bytes: 0,
                }
            })
            .collect();
        let plan = GptOssShardConsumerPlan::from_capacity_one_test_shards(
            catalog.metadata_sha256().into(),
            shards,
        );
        DeterministicFixture {
            _root: root,
            catalog,
            plan,
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct FixtureRun {
        destination_sha256: String,
        consumed: usize,
        elided: usize,
        partial: ExpertPartialStoreStats,
    }

    fn expected_action_sha256(
        catalog: &SafeTensorShardCatalog,
        shard_index: usize,
        action: &GptOssShardConsumerAction,
    ) -> String {
        let file = OpenOptions::new()
            .read(true)
            .open(catalog.shards()[shard_index].path())
            .unwrap();
        let mut bytes = vec![0_u8; action.byte_len().unwrap() as usize];
        file.read_exact_at(&mut bytes, action.shard_absolute_range[0])
            .unwrap();
        hash_bytes(&bytes)
    }

    fn run_deterministic_fixture(fixture: &DeterministicFixture) -> FixtureRun {
        let partial_plan =
            ExpertPartialPlan::derive(&fixture.plan, RETAINED_120B_SPLIT_BOUND_BYTES).unwrap();
        assert_eq!(partial_plan.derived_high_water_count(), 2);
        assert_eq!(
            partial_plan.derived_high_water_bytes(),
            2 * SPLIT_BIAS_BYTES
        );
        let split_entries = partial_plan.entries().cloned().collect::<Vec<_>>();
        let mut partial = ExpertPartialStore::new(partial_plan);
        let mut coverage = ExactActionCoverage::new(&fixture.plan).unwrap();
        let mut destination = Sha256::new();
        for (shard_index, shard) in fixture.plan.shards().iter().enumerate() {
            fixture
                .catalog
                .with_scoped_shard_transaction(&fixture.plan, shard_index, |transaction| {
                    let warm = shard
                        .actions
                        .iter()
                        .enumerate()
                        .filter_map(|(index, action)| match &action.consumer {
                            GptOssShardConsumer::OwnedExpert {
                                key: GptOssExpertKey { layer: 1, .. },
                                owner: ExpertOwner::Cpu { .. },
                                ..
                            } => Some(index),
                            _ => None,
                        })
                        .collect::<Vec<_>>();
                    if !warm.is_empty() {
                        transaction.record_warm_elisions(&warm)?;
                    }
                    for (action_index, action) in shard.actions.iter().enumerate() {
                        if warm.binary_search(&action_index).is_ok() {
                            coverage.elide(&action.action_id_sha256, action.byte_len()?)?;
                            continue;
                        }
                        let split = split_entries.iter().find(|entry| {
                            entry.earlier_shard_index == shard_index
                                && entry.earlier_action_id_sha256 == action.action_id_sha256
                        });
                        if split.is_some() {
                            transaction.with_synchronous_action(action_index, |scoped| {
                                partial.insert_bias(shard_index, scoped.action(), scoped.bytes())
                            })?;
                        } else {
                            transaction.with_synchronous_action(action_index, |scoped| {
                                let actual = hash_bytes(scoped.bytes());
                                let expected = expected_action_sha256(
                                    &fixture.catalog,
                                    shard_index,
                                    scoped.action(),
                                );
                                if actual != expected {
                                    return Err(model_error("old/new fixture bytes differ"));
                                }
                                destination.update(scoped.action().action_id_sha256.as_bytes());
                                destination.update(scoped.bytes().len().to_le_bytes());
                                destination.update(actual.as_bytes());
                                Ok(())
                            })?;
                        }
                        coverage.consume(action)?;
                    }
                    for split in split_entries
                        .iter()
                        .filter(|entry| entry.later_shard_index == shard_index)
                    {
                        let completed = partial.take_for_completion(
                            &split.identity,
                            shard_index,
                            &split.later_action_ids_sha256,
                        )?;
                        destination.update(completed.earlier_action_id_sha256.as_bytes());
                        destination.update(completed.sha256.as_bytes());
                    }
                    let stats = partial.stats();
                    transaction.record_terminal_audit(ShardReleaseLogicalLedger {
                        partial_store_current_count: stats.current_count,
                        partial_store_current_bytes: stats.current_bytes,
                        partial_store_high_water_count: stats.high_water_count,
                        partial_store_high_water_bytes: stats.high_water_bytes,
                        ..ShardReleaseLogicalLedger::default()
                    })?;
                    Ok(())
                })
                .unwrap();
        }
        coverage.validate_complete().unwrap();
        partial.require_empty().unwrap();
        FixtureRun {
            destination_sha256: format!("{:x}", destination.finalize()),
            consumed: coverage.consumed_count(),
            elided: coverage.elided_count(),
            partial: partial.stats(),
        }
    }

    #[test]
    fn deterministic_tiny_shards_cover_all_owners_splits_warm_elision_and_repeat() {
        let fixture = deterministic_fixture();
        let first = run_deterministic_fixture(&fixture);
        let second = run_deterministic_fixture(&fixture);
        assert_eq!(first, second);
        assert_eq!(first.elided, 6);
        assert_eq!(first.consumed + first.elided, fixture.plan.total_actions());
        assert_eq!(first.partial.high_water_count, 2);
        assert_eq!(first.partial.high_water_bytes, 23_040);
        assert_eq!(first.partial.current_count, 0);
        assert_eq!(fixture.catalog.mapping_activity().high_water, 1);
        assert_eq!(fixture.catalog.mapping_activity().current, 0);
        assert_eq!(fixture.catalog.active_source_payload_fds(), 0);
        let reports = fixture.catalog.release_reports();
        assert_eq!(reports.len(), fixture.plan.shards().len() * 2);
        assert!(reports.iter().all(|report| {
            report.terminal_audit_complete
                && report.mapping_removed
                && report.fd_closed
                && report.post_release.source_inode_mapping_count == 0
                && report.post_release.source_inode_pss_bytes == 0
        }));
        // Layer 2 carried a mismatched warm identity in the fixture contract;
        // unlike the exact layer-1 proof, all six actions were consumed.
        assert!(fixture
            .plan
            .shards()
            .iter()
            .flat_map(|shard| &shard.actions)
            .all(|action| match &action.consumer {
                GptOssShardConsumer::OwnedExpert {
                    key: GptOssExpertKey { layer: 2, .. },
                    ..
                } => reports.iter().any(|report| {
                    report
                        .terminal_action_ids_sha256
                        .contains(&action.action_id_sha256)
                }),
                _ => true,
            }));
    }

    #[test]
    fn retained_120b_partial_bound_is_derived_exactly_and_enforced() {
        let retained = retained_bound_fixture(&[
            [35, 42, 51],
            [34, 43, 51],
            [35, 43, 50],
            [35, 43, 50],
            [35, 43, 50],
        ]);
        let derived =
            ExpertPartialPlan::derive(&retained, RETAINED_120B_SPLIT_BOUND_BYTES).unwrap();
        assert_eq!(derived.derived_high_water_count(), 128);
        assert_eq!(derived.derived_high_water_bytes(), 1_474_560);
        assert_eq!(derived.entries().count(), 640);
        let owner_bytes = derived
            .derived_owner_high_waters()
            .iter()
            .map(|high_water| (high_water.owner.role_name(), high_water.bytes))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(owner_bytes["layer_owner_gpu"], 403_200);
        assert_eq!(owner_bytes["remote_gpu"], 495_360);
        assert_eq!(owner_bytes["cpu"], 587_520);
        assert!(ExpertPartialPlan::derive(
            &retained_bound_fixture(&[[43, 43, 43]]),
            RETAINED_120B_SPLIT_BOUND_BYTES
        )
        .is_err());
    }

    #[test]
    fn split_store_is_exact_bounded_and_erases_once() {
        let identity = ExpertPartialKey {
            key: GptOssExpertKey {
                layer: 2,
                expert: 7,
            },
            owner: owner(),
        };
        let mut store = ExpertPartialStore::new(split_plan(identity.clone()));
        let action = bias_action(&identity);
        let bytes = vec![7_u8; SPLIT_BIAS_BYTES as usize];
        store.insert_bias(0, &action, &bytes).unwrap();
        assert_eq!(store.stats().current_count, 1);
        assert_eq!(store.stats().high_water_bytes, SPLIT_BIAS_BYTES);
        assert!(store.insert_bias(0, &action, &bytes).is_err());
        let ids = (0..5)
            .map(|index| format!("later-{index}"))
            .collect::<Vec<_>>();
        let completed = store.take_for_completion(&identity, 1, &ids).unwrap();
        assert_eq!(completed.bytes, bytes);
        assert!(store.take_for_completion(&identity, 1, &ids).is_err());
        store.require_empty().unwrap();
    }

    #[test]
    fn split_store_rejects_wrong_length_owner_and_later_identity() {
        let identity = ExpertPartialKey {
            key: GptOssExpertKey {
                layer: 4,
                expert: 9,
            },
            owner: owner(),
        };
        let mut store = ExpertPartialStore::new(split_plan(identity.clone()));
        let action = bias_action(&identity);
        assert!(store.insert_bias(0, &action, &[0; 3]).is_err());
        let mut wrong = action.clone();
        wrong.consumer = GptOssShardConsumer::OwnedExpert {
            key: identity.key,
            owner: ExpertOwner::Cpu {
                pool: crate::heterogeneous::placement::CpuPoolId(0),
            },
            surface: GptOssExpertSurface::GateUpBias,
        };
        assert!(store
            .insert_bias(0, &wrong, &vec![0; SPLIT_BIAS_BYTES as usize])
            .is_err());
        store
            .insert_bias(0, &action, &vec![0; SPLIT_BIAS_BYTES as usize])
            .unwrap();
        assert!(store
            .take_for_completion(&identity, 1, &["wrong".into()])
            .is_err());
        assert_eq!(store.stats().current_count, 1);
        store.entries.get_mut(&identity).unwrap().bytes[0] ^= 0xff;
        let ids = (0..5)
            .map(|index| format!("later-{index}"))
            .collect::<Vec<_>>();
        assert!(store.take_for_completion(&identity, 1, &ids).is_err());
        store.cancel();
        store.require_empty().unwrap();
    }

    #[test]
    fn publication_proof_fails_each_missing_zero_or_gate() {
        let complete = OwnerSelectivePublicationProof {
            catalog_identity_exact: true,
            action_coverage_complete: true,
            warm_elision_complete: true,
            active_source_mappings: 0,
            active_source_payload_fds: 0,
            source_payload_views: 0,
            borrowed_source_slices: 0,
            source_inode_mappings: 0,
            source_inode_pss_bytes: 0,
            partial_store_entries: 0,
            partial_store_bytes: 0,
            incomplete_cpu_experts: 0,
            task_temporaries: 0,
            pending_cuda_receipts: 0,
            quarantined_cuda_receipts: 0,
            cold_records_directory_synced: true,
            records_freshly_validated: true,
            runtime_maps_after_source_release: true,
            stable_device_ownership_complete: true,
            bounded_journal_complete: true,
            visibility_contract_unchanged: true,
        };
        complete.validate().unwrap();
        macro_rules! rejects {
            ($field:ident = $value:expr) => {{
                let mut missing = complete.clone();
                missing.$field = $value;
                assert!(missing.validate().is_err(), stringify!($field));
            }};
        }
        rejects!(catalog_identity_exact = false);
        rejects!(action_coverage_complete = false);
        rejects!(warm_elision_complete = false);
        rejects!(active_source_mappings = 1);
        rejects!(active_source_payload_fds = 1);
        rejects!(source_payload_views = 1);
        rejects!(borrowed_source_slices = 1);
        rejects!(source_inode_mappings = 1);
        rejects!(source_inode_pss_bytes = 1);
        rejects!(partial_store_entries = 1);
        rejects!(partial_store_bytes = 1);
        rejects!(incomplete_cpu_experts = 1);
        rejects!(task_temporaries = 1);
        rejects!(pending_cuda_receipts = 1);
        rejects!(quarantined_cuda_receipts = 1);
        rejects!(cold_records_directory_synced = false);
        rejects!(records_freshly_validated = false);
        rejects!(runtime_maps_after_source_release = false);
        rejects!(stable_device_ownership_complete = false);
        rejects!(bounded_journal_complete = false);
        rejects!(visibility_contract_unchanged = false);
    }

    #[test]
    fn frozen_admission_policy_rejects_swap_psi_memory_cache_and_cleanup_drift() {
        let policy = CapacityOneAdmissionPolicy::frozen_r2();
        policy.validate_identity().unwrap();
        let admitted = CapacityOneAdmissionSample {
            process_vm_swap_bytes: 0,
            cgroup_swap_current_bytes: 0,
            admitted_global_swap_used_bytes: 19,
            global_swap_used_bytes: 19,
            mem_available_bytes: R2_MEM_AVAILABLE_FLOOR_BYTES,
            memory_psi_some_avg10_micros: 0,
            memory_psi_full_avg10_micros: 0,
        };
        policy.admit_sample(admitted).unwrap();
        for failed in [
            CapacityOneAdmissionSample {
                process_vm_swap_bytes: 1,
                ..admitted
            },
            CapacityOneAdmissionSample {
                cgroup_swap_current_bytes: 1,
                ..admitted
            },
            CapacityOneAdmissionSample {
                global_swap_used_bytes: 20,
                ..admitted
            },
            CapacityOneAdmissionSample {
                mem_available_bytes: R2_MEM_AVAILABLE_FLOOR_BYTES - 1,
                ..admitted
            },
            CapacityOneAdmissionSample {
                memory_psi_some_avg10_micros: 1,
                ..admitted
            },
            CapacityOneAdmissionSample {
                memory_psi_full_avg10_micros: 1,
                ..admitted
            },
        ] {
            assert!(policy.admit_sample(failed).is_err());
        }
        let settled = CapacityOneSettleSample {
            cgroup_clean_file_bytes: R2_CGROUP_CLEAN_FILE_ALLOWANCE_BYTES,
            dirty_writeback_bytes: R2_DIRTY_WRITEBACK_ALLOWANCE_BYTES,
            cleanup_drift_bytes: R2_CLEANUP_DRIFT_ALLOWANCE_BYTES,
        };
        policy.admit_settled_cleanup(settled).unwrap();
        assert!(policy
            .admit_settled_cleanup(CapacityOneSettleSample {
                cgroup_clean_file_bytes: R2_CGROUP_CLEAN_FILE_ALLOWANCE_BYTES + 1,
                ..settled
            })
            .is_err());
        let mut changed = policy;
        changed.settle_duration_seconds += 1;
        assert!(changed.validate_identity().is_err());
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum FakeUploadFault {
        None,
        BeforeEnqueue,
        AfterEnqueue,
        Synchronize,
    }

    #[derive(Default)]
    struct FakeTerminalDeviceSink {
        lease_borrowed: bool,
        pending_receipts: usize,
        quarantined_receipts: usize,
        terminal_receipts: usize,
    }

    impl FakeTerminalDeviceSink {
        fn upload(&mut self, fault: FakeUploadFault) -> Result<()> {
            if self.lease_borrowed {
                return Err(model_error("pinned lease was reused before terminal proof"));
            }
            if fault == FakeUploadFault::BeforeEnqueue {
                return Err(model_error("failure before enqueue"));
            }
            self.lease_borrowed = true;
            self.pending_receipts += 1;
            if fault == FakeUploadFault::AfterEnqueue {
                return Err(model_error("failure after enqueue"));
            }
            if fault == FakeUploadFault::Synchronize {
                self.pending_receipts -= 1;
                self.quarantined_receipts += 1;
                return Err(model_error("synchronization failed"));
            }
            self.pending_receipts -= 1;
            self.terminal_receipts += 1;
            self.lease_borrowed = false;
            Ok(())
        }
    }

    #[test]
    fn fake_terminal_sink_proves_lease_reuse_and_fatal_quarantine_boundaries() {
        let mut sink = FakeTerminalDeviceSink::default();
        assert!(sink.upload(FakeUploadFault::BeforeEnqueue).is_err());
        assert!(!sink.lease_borrowed);
        sink.upload(FakeUploadFault::None).unwrap();
        assert_eq!(sink.terminal_receipts, 1);
        assert!(!sink.lease_borrowed);

        assert!(sink.upload(FakeUploadFault::AfterEnqueue).is_err());
        assert_eq!(sink.pending_receipts, 1);
        assert!(sink.upload(FakeUploadFault::None).is_err());

        let mut sync_failure = FakeTerminalDeviceSink::default();
        assert!(sync_failure.upload(FakeUploadFault::Synchronize).is_err());
        assert_eq!(sync_failure.pending_receipts, 0);
        assert_eq!(sync_failure.quarantined_receipts, 1);
        assert!(sync_failure.lease_borrowed);
    }
}
