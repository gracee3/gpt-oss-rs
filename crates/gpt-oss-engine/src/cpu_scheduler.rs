//! Canonical native-CPU sequence table and ID-only reservation scheduler.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use gpt_oss_core::prelude::{LLMError, RequestId, Result, SamplingParams, SequenceId, TokenId};
use gpt_oss_model_runner::CpuSequenceModelState;

use crate::output::SequenceOutputState;
use crate::worker::CpuGenerationState;

/// Canonical lifecycle of one native-CPU sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuSequenceLifecycle {
    Waiting,
    Runnable,
    InFlight { step_id: u64 },
    Finished,
    Cancelled,
}

/// The single authoritative record for one CPU request/sequence.
///
/// Initial CPU batching supports exactly one sequence per request. Model,
/// generation, scheduler progress, and output state all live in this record;
/// scheduler queues contain only its `sequence_id`.
#[derive(Debug)]
pub struct CpuSequenceRecord {
    request_id: RequestId,
    sequence_id: SequenceId,
    arrival_order: u64,
    arrival_time: Instant,
    prompt: String,
    prompt_token_ids: Vec<TokenId>,
    sampling_params: SamplingParams,
    prompt_tokens_computed: usize,
    revision: u64,
    lifecycle: CpuSequenceLifecycle,
    model_state: Option<CpuSequenceModelState>,
    generation: CpuGenerationState,
    output: SequenceOutputState,
}

impl CpuSequenceRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: RequestId,
        sequence_id: SequenceId,
        arrival_order: u64,
        prompt: String,
        prompt_token_ids: Vec<TokenId>,
        sampling_params: SamplingParams,
        model_state: CpuSequenceModelState,
    ) -> Result<Self> {
        if prompt_token_ids.is_empty() {
            return Err(LLMError::SchedulerError(
                "CPU sequence prompt must not be empty".into(),
            ));
        }
        let generation = CpuGenerationState::new(
            request_id,
            sequence_id,
            &prompt_token_ids,
            sampling_params.seed,
        );
        Ok(Self {
            request_id,
            sequence_id,
            arrival_order,
            arrival_time: Instant::now(),
            prompt,
            prompt_token_ids,
            sampling_params,
            prompt_tokens_computed: 0,
            revision: 0,
            lifecycle: CpuSequenceLifecycle::Waiting,
            model_state: Some(model_state),
            generation,
            output: SequenceOutputState::new(),
        })
    }

    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    pub const fn sequence_id(&self) -> SequenceId {
        self.sequence_id
    }

    pub const fn arrival_order(&self) -> u64 {
        self.arrival_order
    }

    pub const fn arrival_time(&self) -> Instant {
        self.arrival_time
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub fn prompt_token_ids(&self) -> &[TokenId] {
        &self.prompt_token_ids
    }

    pub fn sampling_params(&self) -> &SamplingParams {
        &self.sampling_params
    }

    pub const fn prompt_tokens_computed(&self) -> usize {
        self.prompt_tokens_computed
    }

    pub const fn revision(&self) -> u64 {
        self.revision
    }

    pub const fn lifecycle(&self) -> CpuSequenceLifecycle {
        self.lifecycle
    }

    pub fn generation(&self) -> &CpuGenerationState {
        &self.generation
    }

    pub fn output(&self) -> &SequenceOutputState {
        &self.output
    }

    pub fn model_state(&self) -> Result<&CpuSequenceModelState> {
        self.model_state.as_ref().ok_or_else(|| {
            LLMError::SchedulerError(format!(
                "CPU sequence {} model state is temporarily unavailable",
                self.sequence_id
            ))
        })
    }

    pub(crate) fn model_state_mut(&mut self) -> Result<&mut CpuSequenceModelState> {
        self.model_state.as_mut().ok_or_else(|| {
            LLMError::SchedulerError(format!(
                "CPU sequence {} model state is temporarily unavailable",
                self.sequence_id
            ))
        })
    }

    pub(crate) fn take_model_state(&mut self) -> Result<CpuSequenceModelState> {
        self.model_state.take().ok_or_else(|| {
            LLMError::SchedulerError(format!(
                "CPU sequence {} model state is temporarily unavailable",
                self.sequence_id
            ))
        })
    }

    pub(crate) fn restore_model_state(&mut self, state: CpuSequenceModelState) -> Result<()> {
        if self.model_state.replace(state).is_some() {
            return Err(LLMError::SchedulerError(format!(
                "CPU sequence {} model state was restored twice",
                self.sequence_id
            )));
        }
        Ok(())
    }

    pub(crate) fn generation_mut(&mut self) -> &mut CpuGenerationState {
        &mut self.generation
    }

    pub(crate) fn output_mut(&mut self) -> &mut SequenceOutputState {
        &mut self.output
    }

    pub(crate) fn advance_prompt(&mut self, rows: usize) -> Result<()> {
        let next = self
            .prompt_tokens_computed
            .checked_add(rows)
            .ok_or_else(|| LLMError::SchedulerError("CPU prompt progress overflows".into()))?;
        if next > self.prompt_token_ids.len() {
            return Err(LLMError::SchedulerError(format!(
                "CPU sequence {} prompt progress exceeds prompt length",
                self.sequence_id
            )));
        }
        self.prompt_tokens_computed = next;
        Ok(())
    }

    pub(crate) fn advance_revision(&mut self) -> Result<()> {
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| LLMError::SchedulerError("CPU sequence revision overflows".into()))?;
        Ok(())
    }

    pub(crate) fn set_lifecycle(&mut self, lifecycle: CpuSequenceLifecycle) {
        self.lifecycle = lifecycle;
    }

    fn is_prefill(&self) -> bool {
        self.prompt_tokens_computed < self.prompt_token_ids.len()
    }

    fn decode_position(&self) -> Result<usize> {
        self.prompt_tokens_computed
            .checked_add(self.output.token_ids.len())
            .and_then(|position| position.checked_sub(1))
            .ok_or_else(|| {
                LLMError::SchedulerError(format!(
                    "CPU sequence {} has no sampled decode input",
                    self.sequence_id
                ))
            })
    }

    #[cfg(test)]
    fn synthetic(
        request_id: u64,
        sequence_id: u64,
        arrival_order: u64,
        prompt_token_ids: Vec<TokenId>,
    ) -> Self {
        let request_id = RequestId(request_id);
        let sequence_id = SequenceId(sequence_id);
        let sampling_params = SamplingParams {
            seed: Some(7),
            ..Default::default()
        };
        Self {
            request_id,
            sequence_id,
            arrival_order,
            arrival_time: Instant::now(),
            prompt: "prompt".into(),
            generation: CpuGenerationState::new(
                request_id,
                sequence_id,
                &prompt_token_ids,
                sampling_params.seed,
            ),
            prompt_token_ids,
            sampling_params,
            prompt_tokens_computed: 0,
            revision: 0,
            lifecycle: CpuSequenceLifecycle::Waiting,
            model_state: None,
            output: SequenceOutputState::new(),
        }
    }
}

/// Canonical CPU records keyed by sequence, with a one-to-one request index.
#[derive(Debug, Default)]
pub struct SequenceTable {
    records: HashMap<SequenceId, CpuSequenceRecord>,
    request_to_sequence: HashMap<RequestId, SequenceId>,
}

impl SequenceTable {
    pub fn insert(&mut self, record: CpuSequenceRecord) -> Result<()> {
        if self.records.contains_key(&record.sequence_id)
            || self.request_to_sequence.contains_key(&record.request_id)
        {
            return Err(LLMError::SchedulerError(format!(
                "duplicate CPU request {} or sequence {}",
                record.request_id, record.sequence_id
            )));
        }
        self.request_to_sequence
            .insert(record.request_id, record.sequence_id);
        self.records.insert(record.sequence_id, record);
        Ok(())
    }

    pub fn get(&self, sequence_id: SequenceId) -> Option<&CpuSequenceRecord> {
        self.records.get(&sequence_id)
    }

    pub fn get_mut(&mut self, sequence_id: SequenceId) -> Option<&mut CpuSequenceRecord> {
        self.records.get_mut(&sequence_id)
    }

    pub fn get_by_request(&self, request_id: RequestId) -> Option<&CpuSequenceRecord> {
        self.request_to_sequence
            .get(&request_id)
            .and_then(|sequence_id| self.records.get(sequence_id))
    }

    pub fn sequence_for_request(&self, request_id: RequestId) -> Option<SequenceId> {
        self.request_to_sequence.get(&request_id).copied()
    }

    pub fn remove_sequence(&mut self, sequence_id: SequenceId) -> Option<CpuSequenceRecord> {
        let record = self.records.remove(&sequence_id)?;
        self.request_to_sequence.remove(&record.request_id);
        Some(record)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// CPU scheduler limits. A zero prefill chunk means only the token budget
/// bounds each prompt chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuSchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_prefill_chunk: usize,
}

impl CpuSchedulerConfig {
    pub fn validate(self) -> Result<Self> {
        if self.max_num_seqs == 0 || self.max_num_batched_tokens == 0 {
            return Err(LLMError::ConfigError(
                "CPU scheduler sequence and token limits must be nonzero".into(),
            ));
        }
        Ok(self)
    }
}

/// Semantic phase of a reserved CPU input row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuScheduledPhase {
    Prefill,
    Decode,
}

/// Immutable CPU row selected by one reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuScheduledRow {
    pub batch_row: usize,
    pub sequence_id: SequenceId,
    pub token_id: TokenId,
    pub absolute_position: usize,
    pub phase: CpuScheduledPhase,
    pub logits_required: bool,
}

/// Immutable result of reserving one CPU iteration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuReservation {
    step_id: u64,
    rows: Vec<CpuScheduledRow>,
    sequence_revisions: Vec<(SequenceId, u64)>,
}

impl CpuReservation {
    pub const fn step_id(&self) -> u64 {
        self.step_id
    }

    pub fn rows(&self) -> &[CpuScheduledRow] {
        &self.rows
    }

    pub fn sequence_revisions(&self) -> &[(SequenceId, u64)] {
        &self.sequence_revisions
    }

    pub fn sequence_ids(&self) -> impl Iterator<Item = SequenceId> + '_ {
        self.sequence_revisions
            .iter()
            .map(|(sequence_id, _)| *sequence_id)
    }

    pub fn num_batched_tokens(&self) -> usize {
        self.rows.len()
    }
}

#[derive(Debug)]
struct CpuInFlight {
    step_id: u64,
    sequence_ids: Vec<SequenceId>,
}

/// Decode-first, bounded-prefill CPU scheduler with ID-only queues.
#[derive(Debug)]
pub struct CpuScheduler {
    config: CpuSchedulerConfig,
    waiting: VecDeque<SequenceId>,
    runnable: VecDeque<SequenceId>,
    in_flight: Option<CpuInFlight>,
    next_step_id: u64,
    budget_one_prefill_turn: bool,
}

impl CpuScheduler {
    pub fn new(config: CpuSchedulerConfig) -> Result<Self> {
        Ok(Self {
            config: config.validate()?,
            waiting: VecDeque::new(),
            runnable: VecDeque::new(),
            in_flight: None,
            next_step_id: 1,
            budget_one_prefill_turn: false,
        })
    }

    pub const fn config(&self) -> CpuSchedulerConfig {
        self.config
    }

    pub fn add_sequence(
        &mut self,
        table: &mut SequenceTable,
        record: CpuSequenceRecord,
    ) -> Result<()> {
        let sequence_id = record.sequence_id;
        table.insert(record)?;
        self.waiting.push_back(sequence_id);
        Ok(())
    }

    pub fn cancel_request(
        &mut self,
        table: &mut SequenceTable,
        request_id: RequestId,
    ) -> Result<bool> {
        let Some(sequence_id) = table.sequence_for_request(request_id) else {
            return Ok(false);
        };
        let is_in_flight = self
            .in_flight
            .as_ref()
            .is_some_and(|reservation| reservation.sequence_ids.contains(&sequence_id));
        let record = table.get_mut(sequence_id).ok_or_else(|| {
            LLMError::SchedulerError("CPU request index points to a missing sequence".into())
        })?;
        record.set_lifecycle(CpuSequenceLifecycle::Cancelled);
        record.advance_revision()?;
        if !is_in_flight {
            table.remove_sequence(sequence_id);
        }
        self.compact_queues(table);
        Ok(true)
    }

    pub fn reserve(&mut self, table: &mut SequenceTable) -> Result<Option<CpuReservation>> {
        if self.in_flight.is_some() {
            return Err(LLMError::SchedulerError(
                "CPU scheduler already has work in flight".into(),
            ));
        }
        self.compact_queues(table);
        self.admit_waiting(table)?;
        let runnable = self
            .runnable
            .iter()
            .copied()
            .filter(|sequence_id| {
                table
                    .get(*sequence_id)
                    .is_some_and(|record| record.lifecycle == CpuSequenceLifecycle::Runnable)
            })
            .collect::<Vec<_>>();
        let decodes = runnable
            .iter()
            .copied()
            .filter(|sequence_id| {
                table
                    .get(*sequence_id)
                    .is_some_and(|record| !record.is_prefill())
            })
            .collect::<Vec<_>>();
        let prefills = runnable
            .iter()
            .copied()
            .filter(|sequence_id| {
                table
                    .get(*sequence_id)
                    .is_some_and(CpuSequenceRecord::is_prefill)
            })
            .collect::<Vec<_>>();

        let mut rows = Vec::with_capacity(self.config.max_num_batched_tokens);
        let budget = self.config.max_num_batched_tokens;
        if !decodes.is_empty() && !prefills.is_empty() && budget == 1 {
            if self.budget_one_prefill_turn {
                self.push_prefill_rows(table, prefills[0], 1, &mut rows)?;
            } else {
                self.push_decode_row(table, decodes[0], &mut rows)?;
            }
            self.budget_one_prefill_turn = !self.budget_one_prefill_turn;
        } else if !decodes.is_empty() && !prefills.is_empty() {
            for &sequence_id in decodes.iter().take(budget - 1) {
                self.push_decode_row(table, sequence_id, &mut rows)?;
            }
            for &sequence_id in &prefills {
                let remaining = budget - rows.len();
                if remaining == 0 {
                    break;
                }
                self.push_prefill_rows(table, sequence_id, remaining, &mut rows)?;
            }
            for &sequence_id in decodes.iter().skip(
                rows.iter()
                    .filter(|row| row.phase == CpuScheduledPhase::Decode)
                    .count(),
            ) {
                if rows.len() == budget {
                    break;
                }
                self.push_decode_row(table, sequence_id, &mut rows)?;
            }
        } else if !decodes.is_empty() {
            for &sequence_id in decodes.iter().take(budget) {
                self.push_decode_row(table, sequence_id, &mut rows)?;
            }
        } else {
            for &sequence_id in &prefills {
                let remaining = budget - rows.len();
                if remaining == 0 {
                    break;
                }
                self.push_prefill_rows(table, sequence_id, remaining, &mut rows)?;
            }
        }

        if rows.is_empty() {
            return Ok(None);
        }
        for (batch_row, row) in rows.iter_mut().enumerate() {
            row.batch_row = batch_row;
        }
        let mut seen = HashSet::new();
        let sequence_ids = rows
            .iter()
            .filter_map(|row| seen.insert(row.sequence_id).then_some(row.sequence_id))
            .collect::<Vec<_>>();
        let step_id = self.next_step_id;
        self.next_step_id = self
            .next_step_id
            .checked_add(1)
            .ok_or_else(|| LLMError::SchedulerError("CPU step ID overflows".into()))?;
        let mut sequence_revisions = Vec::with_capacity(sequence_ids.len());
        for &sequence_id in &sequence_ids {
            let record = table.get_mut(sequence_id).ok_or_else(|| {
                LLMError::SchedulerError("reserved CPU sequence disappeared".into())
            })?;
            sequence_revisions.push((sequence_id, record.revision));
            record.set_lifecycle(CpuSequenceLifecycle::InFlight { step_id });
        }
        self.in_flight = Some(CpuInFlight {
            step_id,
            sequence_ids,
        });
        Ok(Some(CpuReservation {
            step_id,
            rows,
            sequence_revisions,
        }))
    }

    pub fn release(&mut self, table: &mut SequenceTable, step_id: u64) -> Result<()> {
        let in_flight = self.take_in_flight(step_id)?;
        for sequence_id in in_flight.sequence_ids {
            let lifecycle = table.get(sequence_id).map(CpuSequenceRecord::lifecycle);
            match lifecycle {
                Some(CpuSequenceLifecycle::InFlight { step_id: current }) if current == step_id => {
                    table
                        .get_mut(sequence_id)
                        .expect("record was just observed")
                        .set_lifecycle(CpuSequenceLifecycle::Runnable);
                }
                Some(CpuSequenceLifecycle::Cancelled | CpuSequenceLifecycle::Finished) => {
                    table.remove_sequence(sequence_id);
                }
                Some(_) => {
                    return Err(LLMError::SchedulerError(format!(
                        "CPU sequence {sequence_id} has stale in-flight lifecycle"
                    )));
                }
                None => {}
            }
        }
        self.compact_queues(table);
        Ok(())
    }

    pub fn complete(&mut self, table: &mut SequenceTable, step_id: u64) -> Result<()> {
        self.release(table, step_id)
    }

    pub fn has_unfinished(&self, table: &SequenceTable) -> bool {
        !table.is_empty()
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_runnable(&self) -> usize {
        self.runnable.len()
    }

    pub fn has_in_flight(&self) -> bool {
        self.in_flight.is_some()
    }

    fn push_decode_row(
        &self,
        table: &SequenceTable,
        sequence_id: SequenceId,
        rows: &mut Vec<CpuScheduledRow>,
    ) -> Result<()> {
        let record = table
            .get(sequence_id)
            .ok_or_else(|| LLMError::SchedulerError("decode CPU sequence disappeared".into()))?;
        let token_id = record.generation.last_generated().ok_or_else(|| {
            LLMError::SchedulerError(format!(
                "CPU sequence {sequence_id} has no decode input token"
            ))
        })?;
        rows.push(CpuScheduledRow {
            batch_row: 0,
            sequence_id,
            token_id,
            absolute_position: record.decode_position()?,
            phase: CpuScheduledPhase::Decode,
            logits_required: true,
        });
        Ok(())
    }

    fn push_prefill_rows(
        &self,
        table: &SequenceTable,
        sequence_id: SequenceId,
        budget: usize,
        rows: &mut Vec<CpuScheduledRow>,
    ) -> Result<()> {
        let record = table
            .get(sequence_id)
            .ok_or_else(|| LLMError::SchedulerError("prefill CPU sequence disappeared".into()))?;
        let remaining = record
            .prompt_token_ids
            .len()
            .saturating_sub(record.prompt_tokens_computed);
        let configured = if self.config.max_prefill_chunk == 0 {
            remaining
        } else {
            remaining.min(self.config.max_prefill_chunk)
        };
        let chunk = configured.min(budget);
        for offset in 0..chunk {
            let prompt_index = record.prompt_tokens_computed + offset;
            rows.push(CpuScheduledRow {
                batch_row: 0,
                sequence_id,
                token_id: record.prompt_token_ids[prompt_index],
                absolute_position: prompt_index,
                phase: CpuScheduledPhase::Prefill,
                logits_required: prompt_index + 1 == record.prompt_token_ids.len(),
            });
        }
        Ok(())
    }

    fn admit_waiting(&mut self, table: &mut SequenceTable) -> Result<()> {
        let mut active = self
            .runnable
            .iter()
            .filter(|sequence_id| {
                table.get(**sequence_id).is_some_and(|record| {
                    matches!(
                        record.lifecycle,
                        CpuSequenceLifecycle::Runnable | CpuSequenceLifecycle::InFlight { .. }
                    )
                })
            })
            .count();
        while active < self.config.max_num_seqs {
            let Some(sequence_id) = self.waiting.pop_front() else {
                break;
            };
            let Some(record) = table.get_mut(sequence_id) else {
                continue;
            };
            if record.lifecycle != CpuSequenceLifecycle::Waiting {
                continue;
            }
            record.set_lifecycle(CpuSequenceLifecycle::Runnable);
            self.runnable.push_back(sequence_id);
            active += 1;
        }
        Ok(())
    }

    fn take_in_flight(&mut self, step_id: u64) -> Result<CpuInFlight> {
        let in_flight = self.in_flight.take().ok_or_else(|| {
            LLMError::SchedulerError("CPU scheduler has no reservation to release".into())
        })?;
        if in_flight.step_id != step_id {
            self.in_flight = Some(in_flight);
            return Err(LLMError::SchedulerError(format!(
                "stale CPU reservation {step_id}"
            )));
        }
        Ok(in_flight)
    }

    fn compact_queues(&mut self, table: &SequenceTable) {
        self.waiting.retain(|sequence_id| {
            table
                .get(*sequence_id)
                .is_some_and(|record| record.lifecycle == CpuSequenceLifecycle::Waiting)
        });
        self.runnable.retain(|sequence_id| {
            table.get(*sequence_id).is_some_and(|record| {
                matches!(
                    record.lifecycle,
                    CpuSequenceLifecycle::Runnable | CpuSequenceLifecycle::InFlight { .. }
                )
            })
        });
    }
}

#[cfg(test)]
mod tests {
    use rand::RngCore;

    use super::*;

    fn scheduler(max_num_seqs: usize, tokens: usize, chunk: usize) -> CpuScheduler {
        CpuScheduler::new(CpuSchedulerConfig {
            max_num_seqs,
            max_num_batched_tokens: tokens,
            max_prefill_chunk: chunk,
        })
        .unwrap()
    }

    fn add(
        scheduler: &mut CpuScheduler,
        table: &mut SequenceTable,
        request: u64,
        prompt: &[TokenId],
    ) {
        scheduler
            .add_sequence(
                table,
                CpuSequenceRecord::synthetic(request, request, request, prompt.to_vec()),
            )
            .unwrap();
    }

    fn commit_prompt_only(
        scheduler: &mut CpuScheduler,
        table: &mut SequenceTable,
        reservation: &CpuReservation,
    ) {
        for &(sequence_id, _) in reservation.sequence_revisions() {
            let rows = reservation
                .rows()
                .iter()
                .filter(|row| {
                    row.sequence_id == sequence_id && row.phase == CpuScheduledPhase::Prefill
                })
                .count();
            let record = table.get_mut(sequence_id).unwrap();
            record.advance_prompt(rows).unwrap();
            record.advance_revision().unwrap();
        }
        scheduler.complete(table, reservation.step_id()).unwrap();
    }

    fn make_decode(record: &mut CpuSequenceRecord, token: TokenId) {
        record.prompt_tokens_computed = record.prompt_token_ids.len();
        record.generation.set_last_generated(token);
        record.output.token_ids.push(token);
    }

    #[test]
    fn reservation_changes_no_progress_model_generation_rng_or_output() {
        let mut scheduler = scheduler(1, 2, 0);
        let mut table = SequenceTable::default();
        add(&mut scheduler, &mut table, 1, &[10, 11, 12]);
        let record = table.get(SequenceId(1)).unwrap();
        let before_progress = record.prompt_tokens_computed;
        let before_revision = record.revision;
        let before_position = record
            .model_state
            .as_ref()
            .map(CpuSequenceModelState::position);
        let before_output = record.output.token_ids.clone();
        let mut before_rng = record.generation.rng().clone();

        let reservation = scheduler.reserve(&mut table).unwrap().unwrap();
        assert_eq!(reservation.num_batched_tokens(), 2);
        let record = table.get(SequenceId(1)).unwrap();
        assert_eq!(record.prompt_tokens_computed, before_progress);
        assert_eq!(record.revision, before_revision);
        assert_eq!(
            record
                .model_state
                .as_ref()
                .map(CpuSequenceModelState::position),
            before_position
        );
        assert_eq!(record.output.token_ids, before_output);
        let mut after_rng = record.generation.rng().clone();
        assert_eq!(before_rng.next_u64(), after_rng.next_u64());

        scheduler
            .release(&mut table, reservation.step_id())
            .unwrap();
        assert_eq!(
            table.get(SequenceId(1)).unwrap().lifecycle,
            CpuSequenceLifecycle::Runnable
        );
    }

    #[test]
    fn chunk_and_token_budgets_preserve_fcfs_prefill_order() {
        let mut scheduler = scheduler(3, 5, 2);
        let mut table = SequenceTable::default();
        add(&mut scheduler, &mut table, 1, &[10, 11, 12, 13]);
        add(&mut scheduler, &mut table, 2, &[20, 21, 22, 23]);
        add(&mut scheduler, &mut table, 3, &[30, 31, 32, 33]);

        let reservation = scheduler.reserve(&mut table).unwrap().unwrap();
        assert_eq!(reservation.num_batched_tokens(), 5);
        assert_eq!(
            reservation
                .rows()
                .iter()
                .map(|row| (row.sequence_id, row.token_id))
                .collect::<Vec<_>>(),
            vec![
                (SequenceId(1), 10),
                (SequenceId(1), 11),
                (SequenceId(2), 20),
                (SequenceId(2), 21),
                (SequenceId(3), 30),
            ]
        );
        assert!(reservation.rows().iter().all(|row| !row.logits_required));
    }

    #[test]
    fn mixed_batches_are_decode_first_and_reserve_oldest_prefill_progress() {
        let mut scheduler = scheduler(3, 4, 0);
        let mut table = SequenceTable::default();
        add(&mut scheduler, &mut table, 1, &[10]);
        add(&mut scheduler, &mut table, 2, &[20]);
        add(&mut scheduler, &mut table, 3, &[30, 31, 32]);
        scheduler.admit_waiting(&mut table).unwrap();
        make_decode(table.get_mut(SequenceId(1)).unwrap(), 101);
        make_decode(table.get_mut(SequenceId(2)).unwrap(), 102);

        let reservation = scheduler.reserve(&mut table).unwrap().unwrap();
        assert_eq!(
            reservation
                .rows()
                .iter()
                .map(|row| (row.sequence_id, row.phase))
                .collect::<Vec<_>>(),
            vec![
                (SequenceId(1), CpuScheduledPhase::Decode),
                (SequenceId(2), CpuScheduledPhase::Decode),
                (SequenceId(3), CpuScheduledPhase::Prefill),
                (SequenceId(3), CpuScheduledPhase::Prefill),
            ]
        );
    }

    #[test]
    fn budget_one_alternates_mixed_decode_and_prefill_turns() {
        let mut scheduler = scheduler(2, 1, 0);
        let mut table = SequenceTable::default();
        add(&mut scheduler, &mut table, 1, &[10]);
        add(&mut scheduler, &mut table, 2, &[20, 21, 22]);
        scheduler.admit_waiting(&mut table).unwrap();
        make_decode(table.get_mut(SequenceId(1)).unwrap(), 101);

        let first = scheduler.reserve(&mut table).unwrap().unwrap();
        assert_eq!(first.rows()[0].phase, CpuScheduledPhase::Decode);
        scheduler.release(&mut table, first.step_id()).unwrap();
        let second = scheduler.reserve(&mut table).unwrap().unwrap();
        assert_eq!(second.rows()[0].phase, CpuScheduledPhase::Prefill);
        commit_prompt_only(&mut scheduler, &mut table, &second);
        let third = scheduler.reserve(&mut table).unwrap().unwrap();
        assert_eq!(third.rows()[0].phase, CpuScheduledPhase::Decode);
    }

    #[test]
    fn waiting_runnable_and_inflight_cancellation_use_canonical_identity() {
        let mut scheduler = scheduler(1, 1, 0);
        let mut table = SequenceTable::default();
        add(&mut scheduler, &mut table, 1, &[10]);
        add(&mut scheduler, &mut table, 2, &[20]);
        assert!(scheduler.cancel_request(&mut table, RequestId(2)).unwrap());
        assert!(table.get_by_request(RequestId(2)).is_none());

        let reservation = scheduler.reserve(&mut table).unwrap().unwrap();
        assert!(scheduler.cancel_request(&mut table, RequestId(1)).unwrap());
        assert_eq!(
            table.get(SequenceId(1)).unwrap().lifecycle,
            CpuSequenceLifecycle::Cancelled
        );
        scheduler
            .release(&mut table, reservation.step_id())
            .unwrap();
        assert!(table.is_empty());
        assert!(!scheduler.has_unfinished(&table));
    }

    #[test]
    fn stale_release_does_not_destroy_the_current_reservation() {
        let mut scheduler = scheduler(1, 1, 0);
        let mut table = SequenceTable::default();
        add(&mut scheduler, &mut table, 1, &[10]);
        let reservation = scheduler.reserve(&mut table).unwrap().unwrap();
        assert!(scheduler
            .release(&mut table, reservation.step_id() + 1)
            .is_err());
        assert!(scheduler.has_in_flight());
        scheduler
            .release(&mut table, reservation.step_id())
            .unwrap();
    }
}
