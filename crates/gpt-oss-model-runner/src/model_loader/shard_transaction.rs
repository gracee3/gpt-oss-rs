//! Capacity-one transaction joining a validated shard catalog to its consumer plan.
//!
//! This module deliberately stops before owner-selective construction. It
//! provides the lifetime state needed by a future integration without claiming
//! that CUDA copies or CPU records are currently published through this seam.

use gpt_oss_core::error::{LLMError, Result};

use super::shard_catalog::{CatalogMappedShard, SafeTensorShardCatalog};
use super::shard_consumer_plan::{
    GptOssShardConsumerAction, GptOssShardConsumerPlan, GptOssShardConsumption,
};

trait ShardTransactionPlanSource {
    fn validate_transaction_identity(&self) -> Result<()>;
    fn catalog_sha256(&self) -> &str;
    fn plan_sha256(&self) -> &str;
    fn shards(&self) -> &[GptOssShardConsumption];
}

impl ShardTransactionPlanSource for GptOssShardConsumerPlan {
    fn validate_transaction_identity(&self) -> Result<()> {
        self.validate_identity()
    }

    fn catalog_sha256(&self) -> &str {
        self.catalog_sha256()
    }

    fn plan_sha256(&self) -> &str {
        self.plan_sha256()
    }

    fn shards(&self) -> &[GptOssShardConsumption] {
        self.shards()
    }
}

impl SafeTensorShardCatalog {
    /// Consume one validated plan shard inside a capacity-one mapping transaction.
    ///
    /// Synchronous action callbacks must not create a lifetime that outlives
    /// their call. Any asynchronous or otherwise external use must instead go
    /// through [`ScopedShardConsumerTransaction::begin_external_handoff`].
    /// Returning or unwinding with an unproven external handoff intentionally
    /// retains the mapping for process life and permanently quarantines this
    /// catalog instance. Because that mmap remains live, the checkpoint shard
    /// must remain externally immutable until process exit. Retaining a `File`
    /// handle would not prevent another actor from mutating or truncating the
    /// inode, and this process-local quarantine is not artifact-level
    /// revocation.
    pub fn with_scoped_shard_transaction<R>(
        &self,
        plan: &GptOssShardConsumerPlan,
        shard_index: usize,
        use_transaction: impl FnOnce(&mut ScopedShardConsumerTransaction<'_, '_>) -> Result<R>,
    ) -> Result<R> {
        with_plan_source(self, plan, shard_index, use_transaction)
    }
}

fn with_plan_source<R>(
    catalog: &SafeTensorShardCatalog,
    plan: &impl ShardTransactionPlanSource,
    shard_index: usize,
    use_transaction: impl FnOnce(&mut ScopedShardConsumerTransaction<'_, '_>) -> Result<R>,
) -> Result<R> {
    plan.validate_transaction_identity()?;
    if plan.catalog_sha256() != catalog.metadata_sha256() {
        return Err(model_error(
            "shard transaction plan identifies a different catalog",
        ));
    }
    let planned_shard = plan
        .shards()
        .get(shard_index)
        .ok_or_else(|| model_error("shard transaction index is outside the plan"))?;
    validate_planned_shard(catalog, shard_index, planned_shard)?;

    let mapped = catalog.map_scoped_shard(shard_index)?;
    let mut transaction = ScopedShardConsumerTransaction {
        mapped,
        plan_sha256: plan.plan_sha256(),
        planned_shard,
        shard_index,
        lifecycle: ShardTransactionLifecycle::PreHandoff,
    };
    let callback_result = use_transaction(&mut transaction);
    transaction.finish(callback_result)
}

fn validate_planned_shard(
    catalog: &SafeTensorShardCatalog,
    shard_index: usize,
    planned: &GptOssShardConsumption,
) -> Result<()> {
    let catalog_shard = catalog
        .shards()
        .get(shard_index)
        .ok_or_else(|| model_error("shard transaction index is outside the catalog"))?;
    if planned.shard != catalog_shard.identity {
        return Err(model_error(
            "shard transaction plan and catalog shard identities differ",
        ));
    }
    let mut next = catalog_shard.identity.data_start;
    let mut planned_bytes = 0_u64;
    for action in &planned.actions {
        let [absolute_start, absolute_end] = action.shard_absolute_range;
        if absolute_start != next || absolute_start >= absolute_end {
            return Err(model_error(
                "shard transaction action ranges overlap, leave a gap, or are empty",
            ));
        }
        let action_bytes = absolute_end
            .checked_sub(absolute_start)
            .ok_or_else(|| model_error("shard transaction action range reverses"))?;
        let native_bytes = action.native_tensor_range[1]
            .checked_sub(action.native_tensor_range[0])
            .ok_or_else(|| model_error("shard transaction native range reverses"))?;
        if action_bytes != native_bytes {
            return Err(model_error(
                "shard transaction action and native ranges differ",
            ));
        }
        let tensor = catalog.tensor(&action.native_tensor)?;
        if tensor.shard_index != shard_index {
            return Err(model_error(
                "shard transaction action identifies a tensor in another shard",
            ));
        }
        let expected_start = tensor.absolute_range[0]
            .checked_add(action.native_tensor_range[0])
            .ok_or_else(|| model_error("shard transaction action start overflows"))?;
        let expected_end = tensor.absolute_range[0]
            .checked_add(action.native_tensor_range[1])
            .ok_or_else(|| model_error("shard transaction action end overflows"))?;
        if [expected_start, expected_end] != action.shard_absolute_range
            || expected_end > tensor.absolute_range[1]
            || absolute_start < catalog_shard.identity.data_start
            || absolute_end > catalog_shard.identity.file_length
        {
            return Err(model_error(
                "shard transaction action falls outside its checked tensor range",
            ));
        }
        planned_bytes = planned_bytes
            .checked_add(action_bytes)
            .ok_or_else(|| model_error("shard transaction planned bytes overflow"))?;
        next = absolute_end;
    }
    if next != catalog_shard.identity.file_length
        || planned_bytes != catalog_shard.identity.payload_length
        || planned_bytes != planned.planned_payload_bytes
    {
        return Err(model_error(
            "shard transaction actions do not exactly cover the shard payload",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardTransactionLifecycle {
    PreHandoff,
    ExternalHandoffPending,
    ExternalOwnershipTerminal,
    Quarantined,
}

/// Callback-scoped mapping whose action views cannot escape the transaction.
pub struct ScopedShardConsumerTransaction<'catalog, 'plan> {
    mapped: CatalogMappedShard<'catalog>,
    plan_sha256: &'plan str,
    planned_shard: &'plan GptOssShardConsumption,
    shard_index: usize,
    lifecycle: ShardTransactionLifecycle,
}

impl<'catalog, 'plan> ScopedShardConsumerTransaction<'catalog, 'plan> {
    pub fn plan_sha256(&self) -> &str {
        self.plan_sha256
    }

    pub const fn shard_index(&self) -> usize {
        self.shard_index
    }

    pub fn action_count(&self) -> usize {
        self.planned_shard.actions.len()
    }

    pub const fn lifecycle(&self) -> ShardTransactionLifecycle {
        self.lifecycle
    }

    /// Borrow one exact action range for wholly synchronous consumption.
    pub fn with_synchronous_action<R>(
        &mut self,
        action_index: usize,
        use_action: impl FnOnce(ScopedShardAction<'_>) -> Result<R>,
    ) -> Result<R> {
        if self.lifecycle != ShardTransactionLifecycle::PreHandoff {
            return Err(model_error(
                "synchronous shard action is unavailable after external handoff",
            ));
        }
        use_action(self.checked_action(action_index)?)
    }

    /// Enter the only state permitted to create an external lifetime.
    ///
    /// The returned guard may expose any number of checked action ranges. It
    /// must call `prove_terminal_with` after the external consumer has drained.
    /// Merely dropping it leaves the transaction unproven and therefore forces
    /// quarantine when the outer callback returns or unwinds.
    pub fn begin_external_handoff<'transaction>(
        &'transaction mut self,
    ) -> Result<ScopedShardExternalHandoff<'transaction, 'catalog, 'plan>> {
        if self.lifecycle != ShardTransactionLifecycle::PreHandoff {
            return Err(model_error(
                "shard transaction external handoff has already begun",
            ));
        }
        self.lifecycle = ShardTransactionLifecycle::ExternalHandoffPending;
        Ok(ScopedShardExternalHandoff { transaction: self })
    }

    fn checked_action(&self, action_index: usize) -> Result<ScopedShardAction<'_>> {
        let action = self
            .planned_shard
            .actions
            .get(action_index)
            .ok_or_else(|| model_error("shard transaction action index is outside the plan"))?;
        let bytes = self.mapped.checked_bytes(action.shard_absolute_range)?;
        Ok(ScopedShardAction { action, bytes })
    }

    fn quarantine(&mut self) {
        self.mapped.quarantine_for_process_lifetime();
        self.lifecycle = ShardTransactionLifecycle::Quarantined;
    }

    fn finish<R>(&mut self, callback_result: Result<R>) -> Result<R> {
        match self.lifecycle {
            ShardTransactionLifecycle::PreHandoff
            | ShardTransactionLifecycle::ExternalOwnershipTerminal => callback_result,
            ShardTransactionLifecycle::ExternalHandoffPending => {
                self.quarantine();
                match callback_result {
                    Ok(_) => Err(model_error(
                        "shard transaction returned with unproven external ownership; mapping quarantined",
                    )),
                    Err(primary) => Err(model_error(format!(
                        "shard transaction failed after external handoff ({primary}); mapping quarantined"
                    ))),
                }
            }
            ShardTransactionLifecycle::Quarantined => Err(model_error(
                "shard transaction external terminal proof failed; mapping quarantined",
            )),
        }
    }
}

impl Drop for ScopedShardConsumerTransaction<'_, '_> {
    fn drop(&mut self) {
        if self.lifecycle == ShardTransactionLifecycle::ExternalHandoffPending {
            self.quarantine();
        }
    }
}

pub struct ScopedShardAction<'a> {
    action: &'a GptOssShardConsumerAction,
    bytes: &'a [u8],
}

impl<'a> ScopedShardAction<'a> {
    pub const fn action(&self) -> &'a GptOssShardConsumerAction {
        self.action
    }

    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

/// External-lifetime guard. No CUDA claim is made: the future integration must
/// supply its real drain operation to `prove_terminal_with`.
pub struct ScopedShardExternalHandoff<'transaction, 'catalog, 'plan> {
    transaction: &'transaction mut ScopedShardConsumerTransaction<'catalog, 'plan>,
}

impl ScopedShardExternalHandoff<'_, '_, '_> {
    pub fn action(&self, action_index: usize) -> Result<ScopedShardAction<'_>> {
        self.transaction.checked_action(action_index)
    }

    pub fn prove_terminal_with(self, prove_terminal: impl FnOnce() -> Result<()>) -> Result<()> {
        match prove_terminal() {
            Ok(()) => {
                self.transaction.lifecycle = ShardTransactionLifecycle::ExternalOwnershipTerminal;
                Ok(())
            }
            Err(primary) => {
                self.transaction.quarantine();
                Err(model_error(format!(
                    "external ownership terminal proof failed ({primary}); shard mapping quarantined"
                )))
            }
        }
    }
}

fn model_error(message: impl Into<String>) -> LLMError {
    LLMError::ModelError(message.into())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs::{File, OpenOptions};
    use std::io::{Seek, SeekFrom, Write};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::path::Path;
    use std::sync::mpsc::sync_channel;

    use serde::Serialize;
    use tempfile::{tempdir, TempDir};

    use super::*;
    use crate::model_loader::shard_catalog::ShardMappingActivity;
    use crate::model_loader::shard_consumer_plan::GptOssShardConsumer;

    #[derive(Serialize)]
    struct HeaderTensor {
        dtype: &'static str,
        shape: [usize; 1],
        data_offsets: [u64; 2],
    }

    struct TinyFixture {
        root: TempDir,
        catalog: SafeTensorShardCatalog,
        plan: TestPlan,
        file_length: u64,
    }

    #[derive(Clone)]
    struct TestPlan {
        identity_valid: bool,
        catalog_sha256: String,
        plan_sha256: String,
        shards: Vec<GptOssShardConsumption>,
    }

    impl ShardTransactionPlanSource for TestPlan {
        fn validate_transaction_identity(&self) -> Result<()> {
            if self.identity_valid {
                Ok(())
            } else {
                Err(model_error("synthetic plan identity mismatch"))
            }
        }

        fn catalog_sha256(&self) -> &str {
            &self.catalog_sha256
        }

        fn plan_sha256(&self) -> &str {
            &self.plan_sha256
        }

        fn shards(&self) -> &[GptOssShardConsumption] {
            &self.shards
        }
    }

    fn write_tiny_shard(path: &Path) {
        let tensors = [("a", vec![1_u8, 2, 3]), ("b", vec![4_u8, 5])];
        let mut header = BTreeMap::new();
        let mut offset = 0_u64;
        for (name, payload) in &tensors {
            let start = offset;
            offset += payload.len() as u64;
            header.insert(
                *name,
                HeaderTensor {
                    dtype: "U8",
                    shape: [payload.len()],
                    data_offsets: [start, offset],
                },
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        for (_, payload) in &tensors {
            file.write_all(payload).unwrap();
        }
        file.sync_all().unwrap();
    }

    fn tiny_fixture() -> TinyFixture {
        let root = tempdir().unwrap();
        write_tiny_shard(&root.path().join("model.safetensors"));
        let catalog = SafeTensorShardCatalog::open(root.path()).unwrap();
        let shard = catalog.shards()[0].identity.clone();
        let mut actions = Vec::new();
        for (name, runtime) in [("a", "runtime.a"), ("b", "runtime.b")] {
            let tensor = catalog.tensor(name).unwrap();
            actions.push(GptOssShardConsumerAction {
                native_tensor: name.into(),
                native_tensor_range: [0, tensor.byte_len()],
                shard_absolute_range: tensor.absolute_range,
                consumer: GptOssShardConsumer::LayerOwnerDense {
                    runtime_tensor: runtime.into(),
                },
            });
        }
        let file_length = shard.file_length;
        let plan = TestPlan {
            identity_valid: true,
            catalog_sha256: catalog.metadata_sha256().into(),
            plan_sha256: "synthetic-plan-v1".into(),
            shards: vec![GptOssShardConsumption {
                shard,
                actions,
                planned_payload_bytes: 5,
            }],
        };
        TinyFixture {
            root,
            catalog,
            plan,
            file_length,
        }
    }

    fn assert_released(activity: ShardMappingActivity, file_length: u64) {
        assert_eq!(activity.current, 0);
        assert_eq!(activity.high_water, 1);
        assert_eq!(activity.current_mapped_bytes, 0);
        assert_eq!(activity.mapped_byte_high_water, file_length);
        assert!(!activity.quarantined);
    }

    fn assert_quarantined(activity: ShardMappingActivity, file_length: u64) {
        assert_eq!(activity.current, 1);
        assert_eq!(activity.high_water, 1);
        assert_eq!(activity.current_mapped_bytes, file_length);
        assert_eq!(activity.mapped_byte_high_water, file_length);
        assert!(activity.quarantined);
    }

    #[test]
    fn success_is_capacity_one_range_checked_and_exact_high_water() {
        let fixture = tiny_fixture();
        let copied = with_plan_source(&fixture.catalog, &fixture.plan, 0, |transaction| {
            assert_eq!(
                transaction.lifecycle(),
                ShardTransactionLifecycle::PreHandoff
            );
            assert_eq!(transaction.plan_sha256(), "synthetic-plan-v1");
            assert_eq!(transaction.shard_index(), 0);
            assert_eq!(transaction.action_count(), 2);
            let active = fixture.catalog.mapping_activity();
            assert_eq!(active.current, 1);
            assert_eq!(active.current_mapped_bytes, fixture.file_length);
            assert_eq!(active.mapped_byte_high_water, fixture.file_length);
            assert!(with_plan_source(&fixture.catalog, &fixture.plan, 0, |_| Ok(())).is_err());
            assert!(fixture.catalog.with_mapped_shard(0, |_| Ok(())).is_err());
            assert!(transaction.with_synchronous_action(2, |_| Ok(())).is_err());
            transaction.with_synchronous_action(0, |action| {
                assert_eq!(action.action().native_tensor, "a");
                Ok(action.bytes().to_vec())
            })
        })
        .unwrap();
        assert_eq!(copied, [1, 2, 3]);
        assert_released(fixture.catalog.mapping_activity(), fixture.file_length);
    }

    #[test]
    fn simultaneous_cross_thread_admission_rejects_then_cleanly_retries() {
        let fixture = tiny_fixture();
        std::thread::scope(|scope| {
            let (admitted_tx, admitted_rx) = sync_channel(0);
            let (release_tx, release_rx) = sync_channel(0);
            let catalog = &fixture.catalog;
            let plan = &fixture.plan;
            let holder = scope.spawn(move || {
                with_plan_source(catalog, plan, 0, |transaction| {
                    assert_eq!(transaction.action_count(), 2);
                    admitted_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                })
                .unwrap();
            });

            admitted_rx.recv().unwrap();
            let transaction_attempt: Result<()> =
                with_plan_source(&fixture.catalog, &fixture.plan, 0, |_| Ok(()));
            let mapping_attempt: Result<()> = fixture.catalog.with_mapped_shard(0, |_| Ok(()));
            let while_held = fixture.catalog.mapping_activity();

            // Always unblock and join the holder before making assertions, so
            // a failed expectation cannot strand this deterministic protocol.
            release_tx.send(()).unwrap();
            holder.join().unwrap();

            assert!(transaction_attempt
                .unwrap_err()
                .to_string()
                .contains("a shard mapping is already active"));
            assert!(mapping_attempt
                .unwrap_err()
                .to_string()
                .contains("a shard mapping is already active"));
            assert_eq!(while_held.current, 1);
            assert_eq!(while_held.high_water, 1);
            assert_eq!(while_held.current_mapped_bytes, fixture.file_length);
            assert_eq!(while_held.mapped_byte_high_water, fixture.file_length);
            assert!(!while_held.quarantined);
        });

        assert_released(fixture.catalog.mapping_activity(), fixture.file_length);
        let retry = with_plan_source(&fixture.catalog, &fixture.plan, 0, |transaction| {
            transaction.with_synchronous_action(0, |action| Ok(action.bytes().to_vec()))
        })
        .unwrap();
        assert_eq!(retry, [1, 2, 3]);
        assert_released(fixture.catalog.mapping_activity(), fixture.file_length);
    }

    #[test]
    fn plan_catalog_shard_and_action_mismatches_reject_before_mapping() {
        let fixture = tiny_fixture();
        let mut wrong_plan = fixture.plan.clone();
        wrong_plan.identity_valid = false;
        assert!(with_plan_source(&fixture.catalog, &wrong_plan, 0, |_| Ok(())).is_err());

        let mut wrong_catalog = fixture.plan.clone();
        wrong_catalog.catalog_sha256 = "different-catalog".into();
        assert!(with_plan_source(&fixture.catalog, &wrong_catalog, 0, |_| Ok(())).is_err());

        let mut wrong_shard = fixture.plan.clone();
        wrong_shard.shards[0].shard.header_sha256 = "different-header".into();
        assert!(with_plan_source(&fixture.catalog, &wrong_shard, 0, |_| Ok(())).is_err());

        let mut wrong_action = fixture.plan.clone();
        wrong_action.shards[0].actions[0].shard_absolute_range[1] += 1;
        assert!(with_plan_source(&fixture.catalog, &wrong_action, 0, |_| Ok(())).is_err());

        assert_eq!(
            fixture.catalog.mapping_activity(),
            ShardMappingActivity {
                current: 0,
                high_water: 0,
                current_mapped_bytes: 0,
                mapped_byte_high_water: 0,
                quarantined: false,
            }
        );
    }

    #[test]
    fn stale_file_and_header_identities_fail_closed() {
        let replaced = tiny_fixture();
        let path = replaced.root.path().join("model.safetensors");
        std::fs::rename(&path, replaced.root.path().join("old.safetensors")).unwrap();
        write_tiny_shard(&path);
        assert!(with_plan_source(&replaced.catalog, &replaced.plan, 0, |_| Ok(())).is_err());
        assert_eq!(
            replaced.catalog.mapping_activity(),
            ShardMappingActivity {
                current: 0,
                high_water: 1,
                current_mapped_bytes: 0,
                mapped_byte_high_water: 0,
                quarantined: false,
            }
        );

        let changed = tiny_fixture();
        let path = changed.root.path().join("model.safetensors");
        let mut file = OpenOptions::new().write(true).open(path).unwrap();
        file.seek(SeekFrom::Start(8)).unwrap();
        file.write_all(b"[").unwrap();
        file.sync_all().unwrap();
        assert!(with_plan_source(&changed.catalog, &changed.plan, 0, |_| Ok(())).is_err());
        assert_eq!(
            changed.catalog.mapping_activity(),
            ShardMappingActivity {
                current: 0,
                high_water: 1,
                current_mapped_bytes: 0,
                mapped_byte_high_water: 0,
                quarantined: false,
            }
        );
    }

    #[test]
    fn pre_handoff_error_and_panic_release_and_allow_retry() {
        let fixture = tiny_fixture();
        let error: Result<()> = with_plan_source(&fixture.catalog, &fixture.plan, 0, |_| {
            Err(model_error("synthetic pre-handoff error"))
        });
        assert!(error.is_err());
        assert_released(fixture.catalog.mapping_activity(), fixture.file_length);

        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = with_plan_source::<()>(&fixture.catalog, &fixture.plan, 0, |_| {
                panic!("synthetic pre-handoff panic")
            });
        }));
        assert!(panic.is_err());
        assert_released(fixture.catalog.mapping_activity(), fixture.file_length);

        let retry = with_plan_source(&fixture.catalog, &fixture.plan, 0, |transaction| {
            transaction.with_synchronous_action(1, |action| Ok(action.bytes().to_vec()))
        })
        .unwrap();
        assert_eq!(retry, [4, 5]);
    }

    #[test]
    fn unproven_post_handoff_error_quarantines_and_prohibits_reuse() {
        let fixture = tiny_fixture();
        let result: Result<()> =
            with_plan_source(&fixture.catalog, &fixture.plan, 0, |transaction| {
                let handoff = transaction.begin_external_handoff()?;
                assert_eq!(handoff.action(0)?.bytes(), [1, 2, 3]);
                drop(handoff);
                Err(model_error("synthetic post-handoff error"))
            });
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("mapping quarantined"));
        assert_quarantined(fixture.catalog.mapping_activity(), fixture.file_length);
        assert!(with_plan_source(&fixture.catalog, &fixture.plan, 0, |_| Ok(())).is_err());
        assert!(fixture.catalog.with_mapped_shard(0, |_| Ok(())).is_err());
    }

    #[test]
    fn unproven_post_handoff_ok_return_quarantines_and_prohibits_reuse() {
        let fixture = tiny_fixture();
        let result: Result<()> =
            with_plan_source(&fixture.catalog, &fixture.plan, 0, |transaction| {
                let handoff = transaction.begin_external_handoff()?;
                assert_eq!(handoff.action(0)?.bytes(), [1, 2, 3]);
                drop(handoff);
                Ok(())
            });
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("returned with unproven external ownership"));
        assert_quarantined(fixture.catalog.mapping_activity(), fixture.file_length);
        assert!(with_plan_source(&fixture.catalog, &fixture.plan, 0, |_| Ok(())).is_err());
        assert!(fixture.catalog.with_mapped_shard(0, |_| Ok(())).is_err());
    }

    #[test]
    fn unproven_post_handoff_panic_quarantines_and_prohibits_reuse() {
        let fixture = tiny_fixture();
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let _ = with_plan_source::<()>(&fixture.catalog, &fixture.plan, 0, |transaction| {
                let handoff = transaction.begin_external_handoff()?;
                assert_eq!(handoff.action(1)?.bytes(), [4, 5]);
                panic!("synthetic post-handoff panic")
            });
        }));
        assert!(panic.is_err());
        assert_quarantined(fixture.catalog.mapping_activity(), fixture.file_length);
        assert!(with_plan_source(&fixture.catalog, &fixture.plan, 0, |_| Ok(())).is_err());
    }

    #[test]
    fn terminal_proof_releases_and_failed_proof_quarantines() {
        let released = tiny_fixture();
        with_plan_source(&released.catalog, &released.plan, 0, |transaction| {
            let handoff = transaction.begin_external_handoff()?;
            assert_eq!(handoff.action(0)?.bytes(), [1, 2, 3]);
            handoff.prove_terminal_with(|| Ok(()))?;
            assert_eq!(
                transaction.lifecycle(),
                ShardTransactionLifecycle::ExternalOwnershipTerminal
            );
            Ok(())
        })
        .unwrap();
        assert_released(released.catalog.mapping_activity(), released.file_length);

        let quarantined = tiny_fixture();
        let result: Result<()> =
            with_plan_source(&quarantined.catalog, &quarantined.plan, 0, |transaction| {
                transaction
                    .begin_external_handoff()?
                    .prove_terminal_with(|| Err(model_error("synthetic drain failure")))?;
                Ok(())
            });
        assert!(result.is_err());
        assert_quarantined(
            quarantined.catalog.mapping_activity(),
            quarantined.file_length,
        );
    }
}
