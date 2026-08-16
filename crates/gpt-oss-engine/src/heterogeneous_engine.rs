//! Generation-tagged private K/V metadata and atomic heterogeneous step publication.
//!
//! This coordinator is deliberately opt-in. H5's GPU-engine and worker-input
//! adapters consume its private views; H6 supplies the complete owner shell.
//! Committed readers never receive a handle to an active step's private table
//! or output image.

use std::collections::BTreeMap;

use gpt_oss_core::error::{LLMError, Result};
use gpt_oss_model_runner::heterogeneous::{
    sort_errors_by_precedence, ErrorOwner, HeterogeneousErrorKind, HeterogeneousErrorRecord,
    PreparedStepState,
};
use serde::{Deserialize, Serialize};

pub type HeterogeneousSequenceId = u64;
pub type HeterogeneousStepId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GenerationBlockRef {
    pub block_id: u32,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuSequenceVisibility {
    pub sequence_id: HeterogeneousSequenceId,
    pub committed_length: u32,
    pub committed_block_table: Vec<GenerationBlockRef>,
    pub request_revision: u64,
    pub placement_epoch: u64,
    pub visibility_epoch: u64,
    pub token_ids: Vec<u32>,
    pub output_image: Vec<u8>,
    pub evidence_image: Vec<u8>,
    pub delivery_failure: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvisionalKvView<'a> {
    pub transaction_generation: HeterogeneousStepId,
    pub sequence_id: HeterogeneousSequenceId,
    pub private_length: u32,
    pub private_block_table: &'a [GenerationBlockRef],
    pub append_slot_mapping: &'a [u32],
    pub expected_revision: u64,
    pub expected_visibility_epoch: u64,
    pub placement_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceCommitImage {
    pub next_revision: u64,
    pub token_ids: Vec<u32>,
    pub output_image: Vec<u8>,
    pub evidence_image: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrainRole {
    LayerOwnerRouter,
    LayerOwnerExpert,
    LayerOwnerRelay,
    CpuExpert,
    RemoteGpuExpert,
    RankReduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DrainObligation {
    role: DrainRole,
    terminal: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransactionOutcome {
    Committed,
    Discarded,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransactionTerminalRecord {
    pub step_id: HeterogeneousStepId,
    pub sequence_id: HeterogeneousSequenceId,
    pub outcome: TransactionOutcome,
    pub final_state: PreparedStepState,
    pub request_revision: u64,
    pub visibility_epoch: u64,
    pub placement_epoch: u64,
    pub publication_forbidden: bool,
    pub drained_roles: Vec<DrainRole>,
    pub errors: Vec<HeterogeneousErrorRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockState {
    Free,
    Leased {
        step_id: HeterogeneousStepId,
    },
    Committed {
        sequence_id: HeterogeneousSequenceId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockEntry {
    generation: u64,
    state: BlockState,
}

#[derive(Debug)]
struct GenerationBlockAllocator {
    entries: Vec<BlockEntry>,
    free: Vec<u32>,
}

impl GenerationBlockAllocator {
    fn new(capacity: u32) -> Result<Self> {
        if capacity == 0 {
            return Err(LLMError::ConfigError(
                "heterogeneous K/V block capacity must be nonzero".into(),
            ));
        }
        let entries = vec![
            BlockEntry {
                generation: 0,
                state: BlockState::Free,
            };
            capacity as usize
        ];
        let free = (0..capacity).rev().collect();
        Ok(Self { entries, free })
    }

    fn lease(
        &mut self,
        step_id: HeterogeneousStepId,
        count: usize,
        fail_after: Option<usize>,
    ) -> Result<Vec<GenerationBlockRef>> {
        let mut leased = Vec::with_capacity(count);
        for index in 0..count {
            if fail_after == Some(index) {
                self.release_leased(step_id, &leased)?;
                return Err(LLMError::MemoryError(format!(
                    "injected K/V reservation failure after {index} blocks"
                )));
            }
            let block_id = match self.free.pop() {
                Some(block_id) => block_id,
                None => {
                    self.release_leased(step_id, &leased)?;
                    return Err(LLMError::MemoryError(
                        "heterogeneous K/V block pool exhausted".into(),
                    ));
                }
            };
            let entry = &mut self.entries[block_id as usize];
            if entry.state != BlockState::Free {
                self.release_leased(step_id, &leased)?;
                return Err(LLMError::ModelError(format!(
                    "free-list block {block_id} is not free"
                )));
            }
            let next_generation = match entry.generation.checked_add(1) {
                Some(generation) => generation,
                None => {
                    self.free.push(block_id);
                    self.release_leased(step_id, &leased)?;
                    return Err(LLMError::ModelError(format!(
                        "K/V block {block_id} generation exhausted"
                    )));
                }
            };
            entry.generation = next_generation;
            entry.state = BlockState::Leased { step_id };
            leased.push(GenerationBlockRef {
                block_id,
                generation: entry.generation,
            });
        }
        Ok(leased)
    }

    fn validate_private_table(
        &self,
        step_id: HeterogeneousStepId,
        sequence_id: HeterogeneousSequenceId,
        table: &[GenerationBlockRef],
        new_blocks: &[GenerationBlockRef],
    ) -> Result<()> {
        for block in table {
            let entry = self.entries.get(block.block_id as usize).ok_or_else(|| {
                LLMError::ModelError(format!("K/V block {} is out of range", block.block_id))
            })?;
            if entry.generation != block.generation {
                return Err(LLMError::ModelError(format!(
                    "K/V block {} generation changed: expected {}, observed {}",
                    block.block_id, block.generation, entry.generation
                )));
            }
            let expected = if new_blocks.contains(block) {
                BlockState::Leased { step_id }
            } else {
                BlockState::Committed { sequence_id }
            };
            if entry.state != expected {
                return Err(LLMError::ModelError(format!(
                    "K/V block {} ownership changed before publication",
                    block.block_id
                )));
            }
        }
        Ok(())
    }

    fn release_leased(
        &mut self,
        step_id: HeterogeneousStepId,
        blocks: &[GenerationBlockRef],
    ) -> Result<()> {
        // Validate the entire release before mutating a single entry. A bad
        // generation quarantines every block under the authoritative step.
        for block in blocks {
            let entry = self
                .entries
                .get(block.block_id as usize)
                .ok_or_else(|| LLMError::ModelError("leased K/V block is out of range".into()))?;
            if entry.generation != block.generation
                || entry.state != (BlockState::Leased { step_id })
            {
                return Err(LLMError::ModelError(format!(
                    "leased K/V block {} changed before discard",
                    block.block_id
                )));
            }
            entry.generation.checked_add(1).ok_or_else(|| {
                LLMError::ModelError(format!(
                    "leased K/V block {} generation exhausted during discard",
                    block.block_id
                ))
            })?;
        }
        for block in blocks {
            let entry = &mut self.entries[block.block_id as usize];
            entry.generation = entry
                .generation
                .checked_add(1)
                .expect("release generation prevalidated");
            entry.state = BlockState::Free;
            self.free.push(block.block_id);
        }
        Ok(())
    }

    fn commit_leased(
        &mut self,
        step_id: HeterogeneousStepId,
        sequence_id: HeterogeneousSequenceId,
        blocks: &[GenerationBlockRef],
    ) {
        for block in blocks {
            let entry = &mut self.entries[block.block_id as usize];
            debug_assert_eq!(entry.generation, block.generation);
            debug_assert_eq!(entry.state, BlockState::Leased { step_id });
            entry.state = BlockState::Committed { sequence_id };
        }
    }

    fn release_committed(
        &mut self,
        sequence_id: HeterogeneousSequenceId,
        blocks: &[GenerationBlockRef],
    ) -> Result<()> {
        for block in blocks {
            let entry = self.entries.get(block.block_id as usize).ok_or_else(|| {
                LLMError::ModelError("committed K/V block is out of range".into())
            })?;
            if entry.generation != block.generation
                || entry.state != (BlockState::Committed { sequence_id })
            {
                return Err(LLMError::ModelError(format!(
                    "committed K/V block {} changed before recycle",
                    block.block_id
                )));
            }
            entry.generation.checked_add(1).ok_or_else(|| {
                LLMError::ModelError(format!(
                    "committed K/V block {} generation exhausted during recycle",
                    block.block_id
                ))
            })?;
        }
        for block in blocks {
            let entry = &mut self.entries[block.block_id as usize];
            entry.generation = entry
                .generation
                .checked_add(1)
                .expect("recycle generation prevalidated");
            entry.state = BlockState::Free;
            self.free.push(block.block_id);
        }
        Ok(())
    }

    fn free_count(&self) -> usize {
        self.free.len()
    }
}

#[derive(Debug)]
struct ProvisionalKvLease {
    sequence_id: HeterogeneousSequenceId,
    expected_revision: u64,
    expected_visibility_epoch: u64,
    placement_epoch: u64,
    private_length: u32,
    private_block_table: Vec<GenerationBlockRef>,
    append_slot_mapping: Vec<u32>,
    new_blocks: Vec<GenerationBlockRef>,
    committable: bool,
    invalidated: bool,
    drained: bool,
}

#[derive(Debug)]
struct PreparedHeterogeneousStep {
    step_id: HeterogeneousStepId,
    state: PreparedStepState,
    lease: ProvisionalKvLease,
    obligations: Vec<DrainObligation>,
    terminal_roles: Vec<DrainRole>,
    publication_forbidden: bool,
    cancelled: bool,
    errors: Vec<HeterogeneousErrorRecord>,
    reduced_output_bf16_bits: Vec<u16>,
    commit_image: Option<SequenceCommitImage>,
}

impl PreparedHeterogeneousStep {
    fn transition(&mut self, next: PreparedStepState) -> Result<()> {
        if !self.state.allows(next) {
            return Err(LLMError::ModelError(format!(
                "invalid heterogeneous step transition {:?}->{next:?}",
                self.state
            )));
        }
        self.state = next;
        Ok(())
    }

    fn all_terminal(&self) -> bool {
        self.obligations
            .iter()
            .all(|obligation| obligation.terminal)
    }
}

/// Opt-in H5 transaction coordinator. It owns every provisional table and is
/// the only object allowed to advance a committed visibility epoch.
pub struct HeterogeneousTransactionCoordinator {
    block_size: u32,
    allocator: GenerationBlockAllocator,
    sequences: BTreeMap<HeterogeneousSequenceId, GpuSequenceVisibility>,
    steps: BTreeMap<HeterogeneousStepId, PreparedHeterogeneousStep>,
    active_by_sequence: BTreeMap<HeterogeneousSequenceId, HeterogeneousStepId>,
    next_step_id: HeterogeneousStepId,
    admission_closed: bool,
}

impl HeterogeneousTransactionCoordinator {
    pub fn new(block_size: u32, block_capacity: u32, prefix_caching: bool) -> Result<Self> {
        if block_size == 0 {
            return Err(LLMError::ConfigError(
                "heterogeneous K/V block size must be nonzero".into(),
            ));
        }
        if prefix_caching {
            return Err(LLMError::ConfigError(
                "prefix caching is disabled for the first heterogeneous transaction proof".into(),
            ));
        }
        let addressable_slots = u64::from(block_size)
            .checked_mul(u64::from(block_capacity))
            .ok_or_else(|| {
                LLMError::ConfigError("heterogeneous K/V slot geometry overflows".into())
            })?;
        if addressable_slots > u64::from(u32::MAX) + 1 {
            return Err(LLMError::ConfigError(format!(
                "heterogeneous K/V geometry {block_capacity} blocks * {block_size} slots exceeds u32 physical-slot addressing"
            )));
        }
        Ok(Self {
            block_size,
            allocator: GenerationBlockAllocator::new(block_capacity)?,
            sequences: BTreeMap::new(),
            steps: BTreeMap::new(),
            active_by_sequence: BTreeMap::new(),
            next_step_id: 1,
            admission_closed: false,
        })
    }

    pub fn register_sequence(
        &mut self,
        sequence_id: HeterogeneousSequenceId,
        committed_length: u32,
        placement_epoch: u64,
        token_ids: Vec<u32>,
    ) -> Result<()> {
        if self.admission_closed {
            return Err(LLMError::ModelError(
                "heterogeneous transaction admission is closed".into(),
            ));
        }
        if self.sequences.contains_key(&sequence_id) {
            return Err(LLMError::ModelError(format!(
                "sequence {sequence_id} is already registered"
            )));
        }
        let expected_tokens = usize::try_from(committed_length).map_err(|_| {
            LLMError::ModelError("committed sequence length cannot be represented as usize".into())
        })?;
        if token_ids.len() != expected_tokens {
            return Err(LLMError::ModelError(format!(
                "registered token count {} does not match committed length {committed_length}",
                token_ids.len()
            )));
        }
        let block_count = blocks_for(committed_length, self.block_size);
        let bootstrap_step = 0;
        let blocks = self.allocator.lease(bootstrap_step, block_count, None)?;
        self.allocator
            .commit_leased(bootstrap_step, sequence_id, &blocks);
        self.sequences.insert(
            sequence_id,
            GpuSequenceVisibility {
                sequence_id,
                committed_length,
                committed_block_table: blocks,
                request_revision: 0,
                placement_epoch,
                visibility_epoch: 0,
                token_ids,
                output_image: Vec::new(),
                evidence_image: Vec::new(),
                delivery_failure: None,
            },
        );
        Ok(())
    }

    pub fn committed_view(
        &self,
        sequence_id: HeterogeneousSequenceId,
    ) -> Option<&GpuSequenceVisibility> {
        self.sequences.get(&sequence_id)
    }

    pub fn private_kv_view(&self, step_id: HeterogeneousStepId) -> Option<ProvisionalKvView<'_>> {
        let step = self.steps.get(&step_id)?;
        Some(ProvisionalKvView {
            transaction_generation: step.step_id,
            sequence_id: step.lease.sequence_id,
            private_length: step.lease.private_length,
            private_block_table: &step.lease.private_block_table,
            append_slot_mapping: &step.lease.append_slot_mapping,
            expected_revision: step.lease.expected_revision,
            expected_visibility_epoch: step.lease.expected_visibility_epoch,
            placement_epoch: step.lease.placement_epoch,
        })
    }

    pub fn reserve_step(
        &mut self,
        sequence_id: HeterogeneousSequenceId,
        append_tokens: u32,
        placement_epoch: u64,
    ) -> Result<HeterogeneousStepId> {
        self.reserve_step_inner(sequence_id, append_tokens, placement_epoch, None)
    }

    fn reserve_step_inner(
        &mut self,
        sequence_id: HeterogeneousSequenceId,
        append_tokens: u32,
        placement_epoch: u64,
        fail_after_blocks: Option<usize>,
    ) -> Result<HeterogeneousStepId> {
        if self.admission_closed {
            return Err(LLMError::ModelError(
                "heterogeneous transaction admission is closed".into(),
            ));
        }
        if append_tokens == 0 {
            return Err(LLMError::ModelError(
                "heterogeneous step must append at least one K/V position".into(),
            ));
        }
        if self.active_by_sequence.contains_key(&sequence_id) {
            return Err(LLMError::ModelError(format!(
                "sequence {sequence_id} already has an in-flight heterogeneous step"
            )));
        }
        let committed = self.sequences.get(&sequence_id).ok_or_else(|| {
            LLMError::ModelError(format!("sequence {sequence_id} is not registered"))
        })?;
        if placement_epoch != committed.placement_epoch {
            return Err(LLMError::ModelError(format!(
                "placement epoch {placement_epoch} does not match committed {}",
                committed.placement_epoch
            )));
        }
        let committed_length = committed.committed_length;
        let committed_block_table = committed.committed_block_table.clone();
        let expected_revision = committed.request_revision;
        let expected_visibility_epoch = committed.visibility_epoch;
        let private_length = committed_length
            .checked_add(append_tokens)
            .ok_or_else(|| LLMError::ModelError("private K/V length overflows".into()))?;
        let required_blocks = blocks_for(private_length, self.block_size);
        let additional = required_blocks.saturating_sub(committed_block_table.len());
        let step_id = self.next_step_id;
        let next_step_id = step_id
            .checked_add(1)
            .ok_or_else(|| LLMError::ModelError("heterogeneous step identity exhausted".into()))?;
        if self.steps.contains_key(&step_id) {
            return Err(LLMError::ModelError(
                "heterogeneous step identity would alias live state".into(),
            ));
        }
        let new_blocks = self
            .allocator
            .lease(step_id, additional, fail_after_blocks)?;
        let mapping = (|| -> Result<_> {
            let mut private_block_table = committed_block_table;
            private_block_table.extend_from_slice(&new_blocks);
            let append_capacity = usize::try_from(append_tokens).map_err(|_| {
                LLMError::ModelError("append token count cannot be represented as usize".into())
            })?;
            let mut append_slot_mapping = Vec::with_capacity(append_capacity);
            for position in committed_length..private_length {
                let logical_block = usize::try_from(position / self.block_size).map_err(|_| {
                    LLMError::ModelError("logical K/V block cannot be represented as usize".into())
                })?;
                let block_offset = position % self.block_size;
                let block = private_block_table.get(logical_block).ok_or_else(|| {
                    LLMError::ModelError("private K/V table does not cover append slot".into())
                })?;
                let physical_slot = block
                    .block_id
                    .checked_mul(self.block_size)
                    .and_then(|base| base.checked_add(block_offset))
                    .ok_or_else(|| {
                        LLMError::ModelError("physical K/V append slot overflows u32".into())
                    })?;
                append_slot_mapping.push(physical_slot);
            }
            Ok((private_block_table, append_slot_mapping))
        })();
        let (private_block_table, append_slot_mapping) = match mapping {
            Ok(mapping) => mapping,
            Err(error) => {
                self.allocator.release_leased(step_id, &new_blocks)?;
                return Err(error);
            }
        };
        let step = PreparedHeterogeneousStep {
            step_id,
            state: PreparedStepState::Reserved,
            lease: ProvisionalKvLease {
                sequence_id,
                expected_revision,
                expected_visibility_epoch,
                placement_epoch,
                private_length,
                private_block_table,
                append_slot_mapping,
                new_blocks,
                committable: true,
                invalidated: false,
                drained: false,
            },
            obligations: Vec::with_capacity(6),
            terminal_roles: Vec::with_capacity(6),
            publication_forbidden: false,
            cancelled: false,
            errors: Vec::with_capacity(32),
            reduced_output_bf16_bits: vec![0_u16; 2_880],
            commit_image: None,
        };
        self.next_step_id = next_step_id;
        self.active_by_sequence.insert(sequence_id, step_id);
        self.steps.insert(step_id, step);
        Ok(step_id)
    }

    pub fn mark_prepared(&mut self, step_id: HeterogeneousStepId) -> Result<()> {
        self.step_mut(step_id)?
            .transition(PreparedStepState::Prepared)
    }

    pub fn mark_dispatched(
        &mut self,
        step_id: HeterogeneousStepId,
        roles: &[DrainRole],
    ) -> Result<()> {
        if roles.is_empty() || roles.len() > 6 {
            return Err(LLMError::ModelError(
                "dispatched heterogeneous step must have 1..=6 drain obligations".into(),
            ));
        }
        if roles
            .iter()
            .enumerate()
            .any(|(index, role)| roles[..index].contains(role))
        {
            return Err(LLMError::ModelError(
                "dispatched heterogeneous step has duplicate drain roles".into(),
            ));
        }
        let step = self.step_mut(step_id)?;
        if !step.state.allows(PreparedStepState::Dispatched)
            || step.obligations.capacity() < roles.len()
            || step.terminal_roles.capacity() < roles.len()
        {
            return Err(LLMError::ModelError(
                "heterogeneous dispatch state or reserved obligation capacity is invalid".into(),
            ));
        }
        // All validation is complete; both extends remain within pre-reserved
        // capacity and the following transition is infallible.
        step.obligations
            .extend(roles.iter().copied().map(|role| DrainObligation {
                role,
                terminal: false,
            }));
        step.terminal_roles.extend_from_slice(roles);
        step.state = PreparedStepState::Dispatched;
        Ok(())
    }

    pub fn mark_terminal(&mut self, step_id: HeterogeneousStepId, role: DrainRole) -> Result<()> {
        let step = self.step_mut(step_id)?;
        if !matches!(
            step.state,
            PreparedStepState::Dispatched
                | PreparedStepState::PartiallyComplete
                | PreparedStepState::Draining
        ) {
            return Err(LLMError::ModelError(format!(
                "step {step_id} cannot accept a terminal obligation in {:?}",
                step.state
            )));
        }
        let obligation_index = step
            .obligations
            .iter()
            .position(|obligation| obligation.role == role)
            .ok_or_else(|| {
                LLMError::ModelError(format!("step {step_id} has no {role:?} drain obligation"))
            })?;
        if step.obligations[obligation_index].terminal {
            return Err(LLMError::ModelError(format!(
                "step {step_id} {role:?} obligation is already terminal"
            )));
        }
        // Mutation begins only after every fallible check.
        step.obligations[obligation_index].terminal = true;
        if step.state == PreparedStepState::Dispatched {
            step.state = PreparedStepState::PartiallyComplete;
        }
        step.lease.drained = step.all_terminal();
        Ok(())
    }

    pub fn mark_reduced(
        &mut self,
        step_id: HeterogeneousStepId,
        output_bf16_bits: &[u16],
    ) -> Result<()> {
        let step = self.step_mut(step_id)?;
        if step.publication_forbidden || !step.all_terminal() {
            return Err(LLMError::ModelError(
                "step cannot reduce before every result/drain is terminal".into(),
            ));
        }
        if step.state != PreparedStepState::PartiallyComplete {
            return Err(LLMError::ModelError(
                "step is not ready for rank-ordered reduction".into(),
            ));
        }
        if output_bf16_bits.len() != step.reduced_output_bf16_bits.len() {
            return Err(LLMError::ModelError(format!(
                "rank reduction output length {} != {}",
                output_bf16_bits.len(),
                step.reduced_output_bf16_bits.len()
            )));
        }
        step.reduced_output_bf16_bits
            .copy_from_slice(output_bf16_bits);
        step.state = PreparedStepState::Reduced;
        Ok(())
    }

    pub fn prepare_commit(
        &mut self,
        step_id: HeterogeneousStepId,
        image: SequenceCommitImage,
    ) -> Result<()> {
        let step = self
            .steps
            .get(&step_id)
            .ok_or_else(|| LLMError::ModelError(format!("unknown heterogeneous step {step_id}")))?;
        if step.state != PreparedStepState::Reduced
            || step.publication_forbidden
            || !step.lease.drained
        {
            return Err(LLMError::ModelError(
                "step cannot prepare publication before mandatory drain".into(),
            ));
        }
        let next_revision = step
            .lease
            .expected_revision
            .checked_add(1)
            .ok_or_else(|| LLMError::ModelError("request revision exhausted".into()))?;
        if image.next_revision != next_revision {
            return Err(LLMError::ModelError(
                "prepared commit image has a non-successor request revision".into(),
            ));
        }
        let private_length = usize::try_from(step.lease.private_length).map_err(|_| {
            LLMError::ModelError("private K/V length cannot be represented as usize".into())
        })?;
        if image.token_ids.len() != private_length {
            return Err(LLMError::ModelError(format!(
                "prepared token count {} does not match private K/V length {}",
                image.token_ids.len(),
                step.lease.private_length
            )));
        }
        let sequence = self
            .sequences
            .get(&step.lease.sequence_id)
            .ok_or_else(|| LLMError::ModelError("prepared commit sequence is missing".into()))?;
        sequence
            .visibility_epoch
            .checked_add(1)
            .ok_or_else(|| LLMError::ModelError("visibility epoch exhausted".into()))?;
        let step = self.step_mut(step_id)?;
        step.commit_image = Some(image);
        step.state = PreparedStepState::ReadyToCommit;
        Ok(())
    }

    pub fn cancel_step(
        &mut self,
        step_id: HeterogeneousStepId,
    ) -> Result<Option<TransactionTerminalRecord>> {
        let cancellation = error_record(
            HeterogeneousErrorKind::Cancelled,
            ErrorOwner::Coordinator,
            "heterogeneous step cancelled",
        );
        let pre_dispatch = {
            let step = self.step_mut(step_id)?;
            if step.errors.len() == step.errors.capacity() {
                return Err(LLMError::ModelError(
                    "heterogeneous step error capacity exhausted".into(),
                ));
            }
            let pre_dispatch = matches!(
                step.state,
                PreparedStepState::Reserved | PreparedStepState::Prepared
            );
            if !pre_dispatch
                && step.state != PreparedStepState::Draining
                && !step.state.allows(PreparedStepState::Draining)
            {
                return Err(LLMError::ModelError(format!(
                    "step {step_id} cannot be cancelled in {:?}",
                    step.state
                )));
            }
            step.cancelled = true;
            step.publication_forbidden = true;
            step.lease.committable = false;
            step.errors.push(cancellation);
            pre_dispatch
        };
        if pre_dispatch {
            return self.discard_pre_dispatch(step_id).map(Some);
        }
        let step = self.step_mut(step_id)?;
        if step.state != PreparedStepState::Draining {
            step.state = PreparedStepState::Draining;
        }
        Ok(None)
    }

    pub fn record_error(
        &mut self,
        step_id: HeterogeneousStepId,
        error: HeterogeneousErrorRecord,
    ) -> Result<Option<TransactionTerminalRecord>> {
        let pre_dispatch = {
            let step = self.step_mut(step_id)?;
            if step.errors.len() == step.errors.capacity() {
                return Err(LLMError::ModelError(
                    "heterogeneous step error capacity exhausted".into(),
                ));
            }
            let pre_dispatch = matches!(
                step.state,
                PreparedStepState::Reserved | PreparedStepState::Prepared
            );
            if !pre_dispatch
                && step.state != PreparedStepState::Draining
                && !step.state.allows(PreparedStepState::Draining)
            {
                return Err(LLMError::ModelError(format!(
                    "step {step_id} cannot record an execution error in {:?}",
                    step.state
                )));
            }
            step.errors.push(error);
            step.publication_forbidden = true;
            step.lease.committable = false;
            pre_dispatch
        };
        if pre_dispatch {
            return self.discard_pre_dispatch(step_id).map(Some);
        }
        let step = self.step_mut(step_id)?;
        if step.state != PreparedStepState::Draining {
            step.state = PreparedStepState::Draining;
        }
        Ok(None)
    }

    pub fn finalize_discard(
        &mut self,
        step_id: HeterogeneousStepId,
    ) -> Result<TransactionTerminalRecord> {
        let step = self
            .steps
            .get(&step_id)
            .ok_or_else(|| LLMError::ModelError(format!("unknown step {step_id}")))?;
        if !step.all_terminal() {
            return Err(LLMError::ModelError(
                "cannot discard while a CPU/CUDA drain obligation is live".into(),
            ));
        }
        if step.state != PreparedStepState::Draining {
            return Err(LLMError::ModelError(
                "only a draining step may be invalidated".into(),
            ));
        }
        // Release is all-or-none. If it fails, the step and active sequence
        // remain authoritative and the capacity stays quarantined.
        self.allocator
            .release_leased(step_id, &step.lease.new_blocks)?;
        let mut step = self.steps.remove(&step_id).expect("discard step retained");
        step.state = PreparedStepState::Invalidated;
        step.lease.invalidated = true;
        step.state = PreparedStepState::Discarded;
        self.active_by_sequence.remove(&step.lease.sequence_id);
        let fields = self.terminal_sequence_fields(step.lease.sequence_id);
        Ok(terminal_record(step, TransactionOutcome::Discarded, fields))
    }

    pub fn commit(&mut self, step_id: HeterogeneousStepId) -> Result<TransactionTerminalRecord> {
        self.commit_with_external_visibility(step_id, || Ok(()))
    }

    /// Commit one already-prepared external private state image inside the
    /// same exclusive publication section. The callback must itself be
    /// allocation-free and must expose no reader outside this coordinator's
    /// visibility epoch. All coordinator validation runs first; after a
    /// successful callback the remaining coordinator mutation is infallible
    /// and the visibility epoch is still the final store.
    pub fn commit_with_external_visibility<F>(
        &mut self,
        step_id: HeterogeneousStepId,
        publish_external: F,
    ) -> Result<TransactionTerminalRecord>
    where
        F: FnOnce() -> Result<()>,
    {
        let next_visibility_epoch = match self.validate_commit(step_id) {
            Ok(epoch) => epoch,
            Err((kind, message)) => {
                let discard_safe = self.steps.get(&step_id).is_some_and(|step| {
                    step.state == PreparedStepState::ReadyToCommit && step.all_terminal()
                });
                if !discard_safe {
                    return Err(LLMError::ModelError(message.into()));
                }
                let error = error_record(kind, ErrorOwner::Coordinator, message);
                let _ = self.record_error(step_id, error)?;
                return self.finalize_discard(step_id);
            }
        };

        publish_external()?;

        // From here to the epoch increment every operation is infallible and
        // allocation-free: remove, swaps, scalar stores, and validated state
        // changes only. Exclusive `&mut self` prevents any reader interleave.
        let mut step = self.steps.remove(&step_id).expect("commit validated step");
        let sequence_id = step.lease.sequence_id;
        let sequence = self
            .sequences
            .get_mut(&sequence_id)
            .expect("commit validated sequence");
        let mut image = step.commit_image.take().expect("commit validated image");
        self.allocator
            .commit_leased(step_id, sequence_id, &step.lease.new_blocks);
        std::mem::swap(
            &mut sequence.committed_block_table,
            &mut step.lease.private_block_table,
        );
        sequence.committed_length = step.lease.private_length;
        sequence.request_revision = image.next_revision;
        std::mem::swap(&mut sequence.token_ids, &mut image.token_ids);
        std::mem::swap(&mut sequence.output_image, &mut image.output_image);
        std::mem::swap(&mut sequence.evidence_image, &mut image.evidence_image);
        sequence.delivery_failure = None;
        step.lease.committable = false;
        step.state = PreparedStepState::Committed;
        self.active_by_sequence.remove(&sequence_id);

        // The one visibility epoch is deliberately the final publication
        // store. No state used by a committed reader changes after this line.
        sequence.visibility_epoch = next_visibility_epoch;
        let fields = (
            sequence.request_revision,
            sequence.visibility_epoch,
            sequence.placement_epoch,
        );
        Ok(terminal_record(step, TransactionOutcome::Committed, fields))
    }

    pub fn record_delivery_failure(
        &mut self,
        sequence_id: HeterogeneousSequenceId,
        message: String,
    ) -> Result<()> {
        let sequence = self.sequences.get_mut(&sequence_id).ok_or_else(|| {
            LLMError::ModelError(format!("sequence {sequence_id} is not registered"))
        })?;
        sequence.delivery_failure = Some(message);
        Ok(())
    }

    pub fn recycle_sequence(&mut self, sequence_id: HeterogeneousSequenceId) -> Result<()> {
        if self.active_by_sequence.contains_key(&sequence_id) {
            return Err(LLMError::ModelError(
                "cannot recycle a sequence with in-flight work".into(),
            ));
        }
        let sequence = self.sequences.get(&sequence_id).ok_or_else(|| {
            LLMError::ModelError(format!("sequence {sequence_id} is not registered"))
        })?;
        self.allocator
            .release_committed(sequence_id, &sequence.committed_block_table)?;
        self.sequences.remove(&sequence_id);
        Ok(())
    }

    pub fn begin_shutdown(&mut self) -> Result<Vec<TransactionTerminalRecord>> {
        self.admission_closed = true;
        let step_ids = self.steps.keys().copied().collect::<Vec<_>>();
        let mut immediate = Vec::new();
        for step_id in step_ids {
            if let Some(record) = self.cancel_step(step_id)? {
                immediate.push(record);
            }
        }
        Ok(immediate)
    }

    pub fn finish_shutdown(&mut self) -> Result<Vec<TransactionTerminalRecord>> {
        let step_ids = self.steps.keys().copied().collect::<Vec<_>>();
        if step_ids.iter().any(|step_id| {
            self.steps
                .get(step_id)
                .is_some_and(|step| !step.all_terminal())
        }) {
            return Err(LLMError::ModelError(
                "shutdown cannot finish while request-owned work is live".into(),
            ));
        }
        let mut terminal = Vec::with_capacity(step_ids.len());
        for step_id in step_ids {
            terminal.push(self.finalize_discard(step_id)?);
        }
        if !self.steps.is_empty() || !self.active_by_sequence.is_empty() {
            return Err(LLMError::ModelError(
                "shutdown retained an active heterogeneous step".into(),
            ));
        }
        Ok(terminal)
    }

    pub fn active_step_count(&self) -> usize {
        self.steps.len()
    }

    pub fn free_block_count(&self) -> usize {
        self.allocator.free_count()
    }

    fn validate_commit(
        &self,
        step_id: HeterogeneousStepId,
    ) -> std::result::Result<u64, (HeterogeneousErrorKind, &'static str)> {
        let publication = |message| (HeterogeneousErrorKind::Publication, message);
        let stale = |message| (HeterogeneousErrorKind::StaleRevision, message);
        let ownership = |message| (HeterogeneousErrorKind::Ownership, message);
        let step = self
            .steps
            .get(&step_id)
            .ok_or_else(|| publication("commit step is missing"))?;
        if step.state != PreparedStepState::ReadyToCommit {
            return Err(publication("step is not ready to commit"));
        }
        if step.publication_forbidden
            || step.cancelled
            || !step.lease.committable
            || !step.lease.drained
            || !step.all_terminal()
        {
            return Err(publication("step publication or drain invariant failed"));
        }
        let sequence = self
            .sequences
            .get(&step.lease.sequence_id)
            .ok_or_else(|| publication("commit sequence is missing"))?;
        if self.active_by_sequence.get(&step.lease.sequence_id) != Some(&step_id) {
            return Err(ownership("sequence in-flight identity changed"));
        }
        if sequence.request_revision != step.lease.expected_revision {
            return Err(stale("request revision changed before commit"));
        }
        if sequence.visibility_epoch != step.lease.expected_visibility_epoch {
            return Err(stale("visibility epoch changed before commit"));
        }
        if sequence.placement_epoch != step.lease.placement_epoch {
            return Err(stale("placement epoch changed before commit"));
        }
        if step.commit_image.is_none() {
            return Err(publication("commit image is missing"));
        }
        self.allocator
            .validate_private_table(
                step_id,
                step.lease.sequence_id,
                &step.lease.private_block_table,
                &step.lease.new_blocks,
            )
            .map_err(|_| ownership("K/V block generation or owner changed"))?;
        sequence
            .visibility_epoch
            .checked_add(1)
            .ok_or_else(|| publication("visibility epoch exhausted"))
    }

    fn discard_pre_dispatch(
        &mut self,
        step_id: HeterogeneousStepId,
    ) -> Result<TransactionTerminalRecord> {
        let step = self
            .steps
            .get(&step_id)
            .ok_or_else(|| LLMError::ModelError(format!("unknown step {step_id}")))?;
        if !matches!(
            step.state,
            PreparedStepState::Reserved | PreparedStepState::Prepared
        ) {
            return Err(LLMError::ModelError(
                "pre-dispatch discard called after dispatch".into(),
            ));
        }
        self.allocator
            .release_leased(step_id, &step.lease.new_blocks)?;
        let mut step = self
            .steps
            .remove(&step_id)
            .expect("pre-dispatch step retained");
        step.state = PreparedStepState::Discarded;
        step.lease.invalidated = true;
        step.lease.drained = true;
        self.active_by_sequence.remove(&step.lease.sequence_id);
        let fields = self.terminal_sequence_fields(step.lease.sequence_id);
        Ok(terminal_record(step, TransactionOutcome::Discarded, fields))
    }

    fn step_mut(&mut self, step_id: HeterogeneousStepId) -> Result<&mut PreparedHeterogeneousStep> {
        self.steps
            .get_mut(&step_id)
            .ok_or_else(|| LLMError::ModelError(format!("unknown heterogeneous step {step_id}")))
    }

    fn terminal_sequence_fields(&self, sequence_id: HeterogeneousSequenceId) -> (u64, u64, u64) {
        let sequence = self
            .sequences
            .get(&sequence_id)
            .expect("terminal step sequence remains registered");
        (
            sequence.request_revision,
            sequence.visibility_epoch,
            sequence.placement_epoch,
        )
    }
}

fn blocks_for(length: u32, block_size: u32) -> usize {
    if length == 0 {
        0
    } else {
        length.div_ceil(block_size) as usize
    }
}

fn error_record(
    kind: HeterogeneousErrorKind,
    owner: ErrorOwner,
    message: impl Into<String>,
) -> HeterogeneousErrorRecord {
    HeterogeneousErrorRecord {
        kind,
        owner,
        route_slot: None,
        message: message.into(),
    }
}

fn terminal_record(
    mut step: PreparedHeterogeneousStep,
    outcome: TransactionOutcome,
    sequence_fields: (u64, u64, u64),
) -> TransactionTerminalRecord {
    sort_errors_by_precedence(&mut step.errors);
    TransactionTerminalRecord {
        step_id: step.step_id,
        sequence_id: step.lease.sequence_id,
        outcome,
        final_state: step.state,
        request_revision: sequence_fields.0,
        visibility_epoch: sequence_fields.1,
        placement_epoch: sequence_fields.2,
        publication_forbidden: step.publication_forbidden,
        drained_roles: step.terminal_roles,
        errors: step.errors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_ROLES: [DrainRole; 6] = [
        DrainRole::LayerOwnerRouter,
        DrainRole::LayerOwnerExpert,
        DrainRole::LayerOwnerRelay,
        DrainRole::CpuExpert,
        DrainRole::RemoteGpuExpert,
        DrainRole::RankReduction,
    ];

    fn coordinator() -> HeterogeneousTransactionCoordinator {
        let mut coordinator = HeterogeneousTransactionCoordinator::new(4, 16, false).unwrap();
        coordinator
            .register_sequence(7, 3, 11, vec![10, 20, 30])
            .unwrap();
        coordinator
    }

    fn prepare_dispatched(
        coordinator: &mut HeterogeneousTransactionCoordinator,
    ) -> HeterogeneousStepId {
        let step = coordinator.reserve_step(7, 1, 11).unwrap();
        coordinator.mark_prepared(step).unwrap();
        coordinator.mark_dispatched(step, &ALL_ROLES).unwrap();
        step
    }

    fn drain_all(coordinator: &mut HeterogeneousTransactionCoordinator, step: u64) {
        for role in ALL_ROLES {
            coordinator.mark_terminal(step, role).unwrap();
        }
    }

    fn commit_image(revision: u64) -> SequenceCommitImage {
        SequenceCommitImage {
            next_revision: revision,
            token_ids: vec![10, 20, 30, 40],
            output_image: vec![1, 2, 3],
            evidence_image: vec![4, 5, 6],
        }
    }

    #[test]
    fn private_append_is_unreachable_until_epoch_advances_last() {
        let mut coordinator = coordinator();
        let before = coordinator.committed_view(7).unwrap().clone();
        let step = prepare_dispatched(&mut coordinator);
        let private = coordinator.private_kv_view(step).unwrap();
        assert_eq!(private.private_length, 4);
        assert_eq!(private.append_slot_mapping, [3]);
        assert_eq!(coordinator.committed_view(7).unwrap(), &before);
        assert!(coordinator.reserve_step(7, 1, 11).is_err());

        drain_all(&mut coordinator, step);
        coordinator.mark_reduced(step, &[0; 2_880]).unwrap();
        coordinator.prepare_commit(step, commit_image(1)).unwrap();
        assert_eq!(coordinator.committed_view(7).unwrap(), &before);
        let terminal = coordinator.commit(step).unwrap();
        assert_eq!(terminal.outcome, TransactionOutcome::Committed);
        let committed = coordinator.committed_view(7).unwrap();
        assert_eq!(committed.committed_length, 4);
        assert_eq!(committed.request_revision, 1);
        assert_eq!(committed.visibility_epoch, 1);
        assert_eq!(committed.token_ids, [10, 20, 30, 40]);
        assert_eq!(committed.output_image, [1, 2, 3]);
        assert_eq!(committed.evidence_image, [4, 5, 6]);
    }

    #[test]
    fn external_visibility_failure_leaves_ready_step_unpublished() {
        let mut coordinator = coordinator();
        let before = coordinator.committed_view(7).unwrap().clone();
        let step = prepare_dispatched(&mut coordinator);
        drain_all(&mut coordinator, step);
        coordinator.mark_reduced(step, &[0; 2_880]).unwrap();
        coordinator.prepare_commit(step, commit_image(1)).unwrap();
        let error = coordinator
            .commit_with_external_visibility(step, || {
                Err(LLMError::ModelError(
                    "injected external visibility failure".into(),
                ))
            })
            .unwrap_err();
        assert!(error.to_string().contains("external visibility"));
        assert_eq!(coordinator.committed_view(7).unwrap(), &before);
        assert_eq!(coordinator.active_step_count(), 1);

        assert!(coordinator.cancel_step(step).unwrap().is_none());
        let discarded = coordinator.finalize_discard(step).unwrap();
        assert_eq!(discarded.outcome, TransactionOutcome::Discarded);
        assert_eq!(coordinator.committed_view(7).unwrap(), &before);
        assert!(clean_second_commit(&mut coordinator));
    }

    #[test]
    fn cancellation_drains_before_reuse_and_dirty_tail_stays_private() {
        let mut coordinator = coordinator();
        let baseline = coordinator.committed_view(7).unwrap().clone();
        let free = coordinator.free_block_count();
        let step = prepare_dispatched(&mut coordinator);
        let dirty_slot = coordinator
            .private_kv_view(step)
            .unwrap()
            .append_slot_mapping[0];
        assert!(coordinator.cancel_step(step).unwrap().is_none());
        assert!(coordinator.finalize_discard(step).is_err());
        drain_all(&mut coordinator, step);
        let terminal = coordinator.finalize_discard(step).unwrap();
        assert_eq!(terminal.outcome, TransactionOutcome::Discarded);
        assert_eq!(coordinator.committed_view(7).unwrap(), &baseline);
        assert_eq!(coordinator.free_block_count(), free);

        let retry = coordinator.reserve_step(7, 1, 11).unwrap();
        assert_eq!(
            coordinator.private_kv_view(retry).unwrap().append_slot_mapping[0],
            dirty_slot,
            "the same dirty tail is safe only because committed length stayed old and the retry rewrites it"
        );
        coordinator.cancel_step(retry).unwrap().unwrap();
    }

    #[test]
    fn block_reservation_fault_rolls_back_and_generation_rejects_stale_state() {
        let mut coordinator = HeterogeneousTransactionCoordinator::new(2, 4, false).unwrap();
        coordinator.register_sequence(1, 0, 3, Vec::new()).unwrap();
        let free = coordinator.free_block_count();
        assert!(coordinator.reserve_step_inner(1, 4, 3, Some(1)).is_err());
        assert_eq!(coordinator.free_block_count(), free);
        assert_eq!(coordinator.active_step_count(), 0);

        let step = coordinator.reserve_step(1, 4, 3).unwrap();
        let stale = coordinator
            .private_kv_view(step)
            .unwrap()
            .private_block_table[0];
        coordinator.cancel_step(step).unwrap().unwrap();
        let retry = coordinator.reserve_step(1, 4, 3).unwrap();
        let fresh = coordinator
            .private_kv_view(retry)
            .unwrap()
            .private_block_table[0];
        assert_ne!(
            fresh, stale,
            "a released generation-tagged ticket must not alias"
        );
        coordinator.cancel_step(retry).unwrap().unwrap();
    }

    #[test]
    fn deterministic_error_precedence_is_timing_independent() {
        let errors = [
            error_record(
                HeterogeneousErrorKind::Cancelled,
                ErrorOwner::Coordinator,
                "cancelled",
            ),
            error_record(
                HeterogeneousErrorKind::CudaLaunch,
                ErrorOwner::RemoteGpu,
                "gpu1 kernel",
            ),
            error_record(HeterogeneousErrorKind::Cpu, ErrorOwner::Cpu, "cpu panic"),
            error_record(
                HeterogeneousErrorKind::Drain,
                ErrorOwner::LayerOwnerGpu,
                "secondary drain",
            ),
        ];
        for order in [[0_usize, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]] {
            let mut coordinator = coordinator();
            let step = prepare_dispatched(&mut coordinator);
            for index in order {
                coordinator
                    .record_error(step, errors[index].clone())
                    .unwrap();
            }
            drain_all(&mut coordinator, step);
            let terminal = coordinator.finalize_discard(step).unwrap();
            assert_eq!(terminal.errors[0].kind, HeterogeneousErrorKind::Cpu);
            assert_eq!(terminal.errors[0].owner, ErrorOwner::Cpu);
            assert_eq!(terminal.errors[1].owner, ErrorOwner::RemoteGpu);
            assert_eq!(terminal.errors[2].kind, HeterogeneousErrorKind::Cancelled);
            assert_eq!(terminal.errors[3].kind, HeterogeneousErrorKind::Drain);
        }
    }

    #[test]
    fn stale_commit_discards_without_epoch_or_output_publication() {
        let mut coordinator = coordinator();
        let step = prepare_dispatched(&mut coordinator);
        drain_all(&mut coordinator, step);
        coordinator.mark_reduced(step, &[0; 2_880]).unwrap();
        coordinator.prepare_commit(step, commit_image(1)).unwrap();
        coordinator.sequences.get_mut(&7).unwrap().request_revision = 99;
        let terminal = coordinator.commit(step).unwrap();
        assert_eq!(terminal.outcome, TransactionOutcome::Discarded);
        assert_eq!(
            terminal.errors[0].kind,
            HeterogeneousErrorKind::StaleRevision
        );
        let committed = coordinator.committed_view(7).unwrap();
        assert_eq!(committed.visibility_epoch, 0);
        assert!(committed.output_image.is_empty());
        assert!(committed.evidence_image.is_empty());
    }

    #[test]
    fn delivery_failure_after_commit_never_rolls_back_state() {
        let mut coordinator = coordinator();
        let step = prepare_dispatched(&mut coordinator);
        drain_all(&mut coordinator, step);
        coordinator.mark_reduced(step, &[0; 2_880]).unwrap();
        coordinator.prepare_commit(step, commit_image(1)).unwrap();
        coordinator.commit(step).unwrap();
        let before = coordinator.committed_view(7).unwrap().clone();
        coordinator
            .record_delivery_failure(7, "receiver closed".into())
            .unwrap();
        let after = coordinator.committed_view(7).unwrap();
        assert_eq!(after.committed_length, before.committed_length);
        assert_eq!(after.request_revision, before.request_revision);
        assert_eq!(after.visibility_epoch, before.visibility_epoch);
        assert_eq!(after.output_image, before.output_image);
        assert_eq!(after.delivery_failure.as_deref(), Some("receiver closed"));
    }

    #[test]
    fn shutdown_refuses_live_work_then_proves_zero_active_steps() {
        let mut coordinator = coordinator();
        let step = prepare_dispatched(&mut coordinator);
        assert!(coordinator.begin_shutdown().unwrap().is_empty());
        assert!(coordinator.reserve_step(7, 1, 11).is_err());
        assert!(coordinator.finish_shutdown().is_err());
        drain_all(&mut coordinator, step);
        let records = coordinator.finish_shutdown().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(coordinator.active_step_count(), 0);
        assert_eq!(records[0].outcome, TransactionOutcome::Discarded);
    }

    #[test]
    fn prefix_cache_is_rejected_for_first_proof() {
        assert!(HeterogeneousTransactionCoordinator::new(16, 16, true).is_err());
    }

    #[test]
    fn heterogeneous_transaction_invalid_calls_do_not_partially_mutate_state() {
        let mut coordinator = coordinator();
        let step = coordinator.reserve_step(7, 1, 11).unwrap();
        assert!(coordinator.mark_dispatched(step, &ALL_ROLES).is_err());
        let state = coordinator.steps.get(&step).unwrap();
        assert_eq!(state.state, PreparedStepState::Reserved);
        assert!(state.obligations.is_empty());
        assert!(state.terminal_roles.is_empty());

        coordinator.mark_prepared(step).unwrap();
        assert!(coordinator.mark_reduced(step, &[7; 2_880]).is_err());
        let state = coordinator.steps.get(&step).unwrap();
        assert_eq!(state.state, PreparedStepState::Prepared);
        assert!(state
            .reduced_output_bf16_bits
            .iter()
            .all(|value| *value == 0));
        assert!(coordinator.prepare_commit(step, commit_image(9)).is_err());
        let state = coordinator.steps.get(&step).unwrap();
        assert_eq!(state.state, PreparedStepState::Prepared);
        assert!(state.commit_image.is_none());
        coordinator.cancel_step(step).unwrap().unwrap();
    }

    #[test]
    fn heterogeneous_transaction_cleanup_failure_retains_authoritative_ownership() {
        let mut subject = coordinator();
        let step = subject.reserve_step(7, 2, 11).unwrap();
        subject.mark_prepared(step).unwrap();
        subject.mark_dispatched(step, &ALL_ROLES).unwrap();
        subject.cancel_step(step).unwrap();
        drain_all(&mut subject, step);
        let block = subject.steps[&step].lease.new_blocks[0];
        subject.allocator.entries[block.block_id as usize].generation += 1;
        assert!(subject.finalize_discard(step).is_err());
        assert_eq!(subject.active_step_count(), 1);
        assert_eq!(subject.active_by_sequence.get(&7), Some(&step));
        assert_eq!(subject.steps[&step].state, PreparedStepState::Draining);

        let mut recycle = coordinator();
        let block = recycle.sequences[&7].committed_block_table[0];
        recycle.allocator.entries[block.block_id as usize].generation += 1;
        assert!(recycle.recycle_sequence(7).is_err());
        assert!(recycle.sequences.contains_key(&7));
    }

    #[test]
    fn heterogeneous_transaction_identity_counters_fail_closed_on_overflow() {
        let mut step_overflow = coordinator();
        let free = step_overflow.free_block_count();
        step_overflow.next_step_id = u64::MAX;
        assert!(step_overflow.reserve_step(7, 1, 11).is_err());
        assert_eq!(step_overflow.active_step_count(), 0);
        assert_eq!(step_overflow.free_block_count(), free);

        let mut revision = coordinator();
        revision.sequences.get_mut(&7).unwrap().request_revision = u64::MAX;
        let step = prepare_dispatched(&mut revision);
        drain_all(&mut revision, step);
        revision.mark_reduced(step, &[0; 2_880]).unwrap();
        assert!(revision.prepare_commit(step, commit_image(0)).is_err());
        assert_eq!(revision.steps[&step].state, PreparedStepState::Reduced);
        assert!(revision.steps[&step].commit_image.is_none());

        let mut visibility = coordinator();
        visibility.sequences.get_mut(&7).unwrap().visibility_epoch = u64::MAX;
        let step = prepare_dispatched(&mut visibility);
        drain_all(&mut visibility, step);
        visibility.mark_reduced(step, &[0; 2_880]).unwrap();
        assert!(visibility.prepare_commit(step, commit_image(1)).is_err());
        assert_eq!(visibility.steps[&step].state, PreparedStepState::Reduced);
        assert!(visibility.steps[&step].commit_image.is_none());

        let mut generation = HeterogeneousTransactionCoordinator::new(4, 1, false).unwrap();
        generation.register_sequence(1, 0, 1, Vec::new()).unwrap();
        generation.allocator.entries[0].generation = u64::MAX;
        assert!(generation.reserve_step(1, 1, 1).is_err());
        assert_eq!(generation.active_step_count(), 0);
        assert_eq!(generation.free_block_count(), 1);
    }

    #[test]
    fn malformed_token_lengths_and_slot_geometry_fail_before_publication() {
        assert!(HeterogeneousTransactionCoordinator::new(u32::MAX, 2, false).is_err());

        let mut registration = HeterogeneousTransactionCoordinator::new(4, 16, false).unwrap();
        let free = registration.free_block_count();
        assert!(registration
            .register_sequence(7, 3, 11, vec![10, 20])
            .is_err());
        assert_eq!(registration.free_block_count(), free);
        assert_eq!(registration.active_step_count(), 0);
        assert!(registration.committed_view(7).is_none());

        let mut publication = coordinator();
        let step = prepare_dispatched(&mut publication);
        drain_all(&mut publication, step);
        publication.mark_reduced(step, &[0; 2_880]).unwrap();
        let before = publication.committed_view(7).unwrap().clone();
        assert!(publication
            .prepare_commit(
                step,
                SequenceCommitImage {
                    next_revision: 1,
                    token_ids: vec![10, 20, 30],
                    output_image: vec![1],
                    evidence_image: vec![2],
                },
            )
            .is_err());
        assert_eq!(publication.steps[&step].state, PreparedStepState::Reduced);
        assert!(publication.steps[&step].commit_image.is_none());
        assert_eq!(publication.committed_view(7).unwrap(), &before);
        assert!(publication.cancel_step(step).unwrap().is_none());
        publication.finalize_discard(step).unwrap();
        assert_eq!(publication.committed_view(7).unwrap(), &before);
        assert_eq!(publication.active_step_count(), 0);
    }

    #[derive(Serialize)]
    struct StaleIdentityCaseEvidence {
        name: &'static str,
        primary_error: HeterogeneousErrorKind,
        capacity_quarantined_until_repair: bool,
        authoritative_state_unchanged_by_discard: bool,
        clean_second_step_committed: bool,
    }

    #[derive(Serialize)]
    struct StaleIdentityEvidence {
        schema: &'static str,
        captured_unix_seconds: u64,
        repository_head: String,
        source_fingerprint_sha256: String,
        cases: Vec<StaleIdentityCaseEvidence>,
        passed: bool,
    }

    #[test]
    fn heterogeneous_transaction_stale_identity_cleanup_matrix_and_clean_retry() {
        let mut cases = Vec::with_capacity(5);

        for (name, mutation) in [
            ("stale_request_revision", 0_u8),
            ("stale_visibility_epoch", 1_u8),
            ("stale_placement_epoch", 2_u8),
        ] {
            let mut subject = coordinator();
            let baseline_free = subject.free_block_count();
            let step = ready_step(&mut subject, 1);
            match mutation {
                0 => subject.sequences.get_mut(&7).unwrap().request_revision = 41,
                1 => subject.sequences.get_mut(&7).unwrap().visibility_epoch = 41,
                2 => subject.sequences.get_mut(&7).unwrap().placement_epoch = 41,
                _ => unreachable!(),
            }
            let authoritative_after_external_change = subject.committed_view(7).unwrap().clone();
            let terminal = subject.commit(step).unwrap();
            assert_eq!(terminal.outcome, TransactionOutcome::Discarded);
            assert_eq!(
                terminal.errors[0].kind,
                HeterogeneousErrorKind::StaleRevision
            );
            assert_eq!(
                subject.committed_view(7).unwrap(),
                &authoritative_after_external_change
            );
            assert_eq!(subject.free_block_count(), baseline_free);
            assert_eq!(subject.active_step_count(), 0);
            let clean = clean_second_commit(&mut subject);
            cases.push(StaleIdentityCaseEvidence {
                name,
                primary_error: HeterogeneousErrorKind::StaleRevision,
                capacity_quarantined_until_repair: false,
                authoritative_state_unchanged_by_discard: true,
                clean_second_step_committed: clean,
            });
        }

        // A stale generation is detected before publication. The failed
        // release retains the step and its capacity rather than returning a
        // block whose identity is uncertain. Repair below is test-only and
        // models an independently proven allocator recovery operation.
        let mut stale_block = coordinator();
        let baseline_view = stale_block.committed_view(7).unwrap().clone();
        let baseline_free = stale_block.free_block_count();
        let step = ready_step(&mut stale_block, 2);
        let block = stale_block.steps[&step].lease.new_blocks[0];
        stale_block.allocator.entries[block.block_id as usize].generation += 1;
        assert!(stale_block.commit(step).is_err());
        assert_eq!(stale_block.active_step_count(), 1);
        assert!(stale_block.free_block_count() < baseline_free);
        assert_eq!(
            stale_block.steps[&step].errors[0].kind,
            HeterogeneousErrorKind::Ownership
        );
        stale_block.allocator.entries[block.block_id as usize].generation = block.generation;
        let terminal = stale_block.finalize_discard(step).unwrap();
        assert_eq!(terminal.errors[0].kind, HeterogeneousErrorKind::Ownership);
        assert_eq!(stale_block.committed_view(7).unwrap(), &baseline_view);
        assert_eq!(stale_block.free_block_count(), baseline_free);
        let clean = clean_second_commit(&mut stale_block);
        cases.push(StaleIdentityCaseEvidence {
            name: "stale_block_generation",
            primary_error: HeterogeneousErrorKind::Ownership,
            capacity_quarantined_until_repair: true,
            authoritative_state_unchanged_by_discard: true,
            clean_second_step_committed: clean,
        });

        // A cleanup failure cannot replace an earlier execution error. The
        // authoritative step and block stay quarantined until release can be
        // proven safe, after which a clean second request must commit.
        let mut cleanup = coordinator();
        let baseline_view = cleanup.committed_view(7).unwrap().clone();
        let baseline_free = cleanup.free_block_count();
        let step = cleanup.reserve_step(7, 2, 11).unwrap();
        cleanup.mark_prepared(step).unwrap();
        cleanup.mark_dispatched(step, &ALL_ROLES).unwrap();
        cleanup
            .record_error(
                step,
                error_record(HeterogeneousErrorKind::Cpu, ErrorOwner::Cpu, "CPU primary"),
            )
            .unwrap();
        drain_all(&mut cleanup, step);
        let block = cleanup.steps[&step].lease.new_blocks[0];
        cleanup.allocator.entries[block.block_id as usize].generation += 1;
        assert!(cleanup.finalize_discard(step).is_err());
        assert_eq!(cleanup.active_step_count(), 1);
        assert!(cleanup.free_block_count() < baseline_free);
        assert_eq!(
            cleanup.steps[&step].errors[0].kind,
            HeterogeneousErrorKind::Cpu
        );
        cleanup.allocator.entries[block.block_id as usize].generation = block.generation;
        let terminal = cleanup.finalize_discard(step).unwrap();
        assert_eq!(terminal.errors[0].kind, HeterogeneousErrorKind::Cpu);
        assert_eq!(cleanup.committed_view(7).unwrap(), &baseline_view);
        assert_eq!(cleanup.free_block_count(), baseline_free);
        let clean = clean_second_commit(&mut cleanup);
        cases.push(StaleIdentityCaseEvidence {
            name: "cleanup_free_failure_retains_primary",
            primary_error: HeterogeneousErrorKind::Cpu,
            capacity_quarantined_until_repair: true,
            authoritative_state_unchanged_by_discard: true,
            clean_second_step_committed: clean,
        });

        assert!(cases.iter().all(|case| case.clean_second_step_committed));
        if let Some(path) = std::env::var_os("GPT_OSS_H5_STALE_EVIDENCE") {
            let evidence = StaleIdentityEvidence {
                schema: "gpt-oss-rs.heterogeneous-h5-stale-identity/v1",
                captured_unix_seconds: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                repository_head: required_evidence_env("GPT_OSS_H5_REPO_HEAD"),
                source_fingerprint_sha256: required_evidence_env("GPT_OSS_H5_SOURCE_FINGERPRINT"),
                cases,
                passed: true,
            };
            write_evidence_json(std::path::Path::new(&path), &evidence);
        }
    }

    fn ready_step(
        coordinator: &mut HeterogeneousTransactionCoordinator,
        append_tokens: u32,
    ) -> HeterogeneousStepId {
        let placement_epoch = coordinator.committed_view(7).unwrap().placement_epoch;
        let step = coordinator
            .reserve_step(7, append_tokens, placement_epoch)
            .unwrap();
        coordinator.mark_prepared(step).unwrap();
        coordinator.mark_dispatched(step, &ALL_ROLES).unwrap();
        drain_all(coordinator, step);
        coordinator.mark_reduced(step, &[0; 2_880]).unwrap();
        let revision = coordinator.committed_view(7).unwrap().request_revision + 1;
        let private_length = coordinator.private_kv_view(step).unwrap().private_length as usize;
        let mut image = commit_image(revision);
        image.token_ids.resize(private_length, 40);
        coordinator.prepare_commit(step, image).unwrap();
        step
    }

    fn clean_second_commit(coordinator: &mut HeterogeneousTransactionCoordinator) -> bool {
        let before = coordinator.committed_view(7).unwrap().clone();
        let step = ready_step(coordinator, 1);
        let terminal = coordinator.commit(step).unwrap();
        let after = coordinator.committed_view(7).unwrap();
        terminal.outcome == TransactionOutcome::Committed
            && after.request_revision == before.request_revision + 1
            && after.visibility_epoch == before.visibility_epoch + 1
            && after.committed_length == before.committed_length + 1
            && coordinator.active_step_count() == 0
    }

    fn required_evidence_env(name: &str) -> String {
        std::env::var(name)
            .unwrap_or_else(|_| panic!("{name} is required when writing H5 evidence"))
    }

    fn write_evidence_json(path: &std::path::Path, value: &impl Serialize) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut bytes = serde_json::to_vec_pretty(value).unwrap();
        bytes.push(b'\n');
        std::fs::write(path, bytes).unwrap();
    }
}
