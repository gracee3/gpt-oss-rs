//! Reserve/execute/commit engine for native multi-request CPU inference.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use gpt_oss_core::prelude::{
    LLMError, RequestId, RequestOutput, Result, SamplingParams, SequenceId,
};
use gpt_oss_model_runner::sampling::Sampler;
use gpt_oss_model_runner::{
    CpuExecutionContext, CpuModel, CpuSequenceModelState, CpuStepBatch, CpuStepRow, PreparedCpuStep,
};
use gpt_oss_tokenizer::Tokenizer;

use crate::config::EngineConfig;
use crate::cpu_scheduler::{
    CpuReservation, CpuScheduledPhase, CpuScheduler, CpuSchedulerConfig, CpuSequenceLifecycle,
    CpuSequenceRecord, SequenceTable,
};
use crate::memory::{
    CpuKvGeometry, GrantId, MemoryClass, MemoryEstimate, ReservationLedger, ReservationLimits,
};
use crate::output::{OutputProcessor, SequenceOutputState};
use crate::service::CommittedEvent;
use crate::worker::CpuGenerationState;

#[derive(Debug, Clone)]
struct CpuPreparedModelRow {
    sequence_id: SequenceId,
    absolute_position: usize,
    logits: Option<Vec<f32>>,
}

trait CpuPreparedModel: Send {
    fn rows(&self) -> &[CpuPreparedModelRow];
    fn commit(
        self: Box<Self>,
        retained: &HashSet<SequenceId>,
        table: &mut SequenceTable,
    ) -> Result<()>;
    fn discard(self: Box<Self>) {}
}

trait CpuBatchForward: Send {
    fn vocab_size(&self) -> usize;
    fn new_sequence_state(&self, context_cap: usize) -> Result<Option<CpuSequenceModelState>>;
    fn prepare(
        &mut self,
        batch: &CpuStepBatch,
        table: &SequenceTable,
    ) -> Result<Box<dyn CpuPreparedModel>>;
}

struct NativeCpuBatchForward {
    model: Arc<CpuModel>,
    execution: CpuExecutionContext,
}

impl NativeCpuBatchForward {
    fn new(model: Arc<CpuModel>) -> Self {
        Self {
            model,
            execution: CpuExecutionContext::new(),
        }
    }
}

struct NativePreparedModel {
    prepared: PreparedCpuStep,
    rows: Vec<CpuPreparedModelRow>,
}

impl CpuPreparedModel for NativePreparedModel {
    fn rows(&self) -> &[CpuPreparedModelRow] {
        &self.rows
    }

    fn commit(
        self: Box<Self>,
        retained: &HashSet<SequenceId>,
        table: &mut SequenceTable,
    ) -> Result<()> {
        let NativePreparedModel { prepared, .. } = *self;
        let mut states = HashMap::with_capacity(retained.len());
        for &sequence_id in retained {
            let state = match table
                .get_mut(sequence_id)
                .ok_or_else(|| {
                    LLMError::SchedulerError(format!(
                        "retained CPU sequence {sequence_id} disappeared"
                    ))
                })
                .and_then(CpuSequenceRecord::take_model_state)
            {
                Ok(state) => state,
                Err(error) => {
                    restore_model_states(table, states)?;
                    return Err(error);
                }
            };
            states.insert(sequence_id, state);
        }

        let result = prepared
            .retain_sequences(retained)
            .commit_states(&mut states)
            .map(|_| ());
        let restore = restore_model_states(table, states);
        result.and(restore)
    }
}

fn restore_model_states(
    table: &mut SequenceTable,
    states: HashMap<SequenceId, CpuSequenceModelState>,
) -> Result<()> {
    for (sequence_id, state) in states {
        table
            .get_mut(sequence_id)
            .ok_or_else(|| {
                LLMError::SchedulerError(format!(
                    "CPU sequence {sequence_id} disappeared while its state was borrowed"
                ))
            })?
            .restore_model_state(state)?;
    }
    Ok(())
}

impl CpuBatchForward for NativeCpuBatchForward {
    fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }

    fn new_sequence_state(&self, context_cap: usize) -> Result<Option<CpuSequenceModelState>> {
        self.model.new_sequence_state(context_cap).map(Some)
    }

    fn prepare(
        &mut self,
        batch: &CpuStepBatch,
        table: &SequenceTable,
    ) -> Result<Box<dyn CpuPreparedModel>> {
        let mut seen = HashSet::new();
        let sequence_ids = batch
            .rows()
            .iter()
            .filter_map(|row| seen.insert(row.sequence_id).then_some(row.sequence_id))
            .collect::<Vec<_>>();
        let states = sequence_ids
            .iter()
            .map(|&sequence_id| {
                let state = table
                    .get(sequence_id)
                    .ok_or_else(|| {
                        LLMError::SchedulerError(format!(
                            "scheduled CPU sequence {sequence_id} disappeared"
                        ))
                    })?
                    .model_state()?;
                Ok((sequence_id, state))
            })
            .collect::<Result<Vec<_>>>()?;
        let prepared = self
            .model
            .prepare_step(&mut self.execution, batch, &states)?;
        let rows = prepared
            .rows()
            .iter()
            .map(|row| CpuPreparedModelRow {
                sequence_id: row.sequence_id,
                absolute_position: row.absolute_position,
                logits: row.logits().map(<[f32]>::to_vec),
            })
            .collect();
        Ok(Box::new(NativePreparedModel { prepared, rows }))
    }
}

#[derive(Clone)]
struct StagedCpuSequence {
    generation: CpuGenerationState,
    output: SequenceOutputState,
}

/// Prepared CPU model and sampling work awaiting a cancellation/revision
/// recheck and one publish operation.
pub struct PreparedCpuIteration {
    reservation: CpuReservation,
    model: Box<dyn CpuPreparedModel>,
    staged: HashMap<SequenceId, StagedCpuSequence>,
}

impl PreparedCpuIteration {
    pub const fn step_id(&self) -> u64 {
        self.reservation.step_id()
    }

    pub fn reservation(&self) -> &CpuReservation {
        &self.reservation
    }
}

/// Result published by one successful CPU iteration.
#[derive(Debug, Default)]
pub struct CpuBatchCommitResult {
    /// Suffix-only compatibility outputs. Prompt metadata is present exactly
    /// once; generated text/tokens/logprobs are never cumulative.
    pub outputs: Vec<RequestOutput>,
    /// Canonical ordered events ready for byte-charged delivery.
    pub events: Vec<(RequestId, CommittedEvent)>,
    pub cancelled_requests: Vec<RequestId>,
}

#[derive(Debug, Clone, Default)]
struct PublishedCursor {
    metadata_published: bool,
    text_bytes: usize,
    token_count: usize,
    logprob_count: usize,
    cumulative_logprob: f32,
}

/// Synchronous owner of all native-CPU request and execution state.
pub struct CpuBatchEngine {
    config: EngineConfig,
    forward: Box<dyn CpuBatchForward>,
    scheduler: CpuScheduler,
    table: SequenceTable,
    sampler: Sampler,
    tokenizer: Tokenizer,
    terminal_token_ids: Vec<u32>,
    decoded_token_byte_bound: usize,
    next_sequence_id: u64,
    next_arrival_order: u64,
    published: HashMap<SequenceId, PublishedCursor>,
    reservations: Option<ReservationLedger>,
    request_grants: HashMap<RequestId, GrantId>,
    kv_geometry: Option<CpuKvGeometry>,
    shutdown: bool,
}

impl CpuBatchEngine {
    pub fn max_num_seqs(&self) -> usize {
        self.config.scheduler.max_num_seqs
    }

    pub fn new(config: EngineConfig, model: Arc<CpuModel>, tokenizer: Tokenizer) -> Result<Self> {
        Self::new_with_request_budget(config, model, tokenizer, None)
    }

    pub fn new_with_request_budget(
        config: EngineConfig,
        model: Arc<CpuModel>,
        tokenizer: Tokenizer,
        request_budget_bytes: Option<u128>,
    ) -> Result<Self> {
        let model_config = model.config();
        let decoded_token_byte_bound = tokenizer.max_decoded_token_bytes()?;
        let kv_geometry = CpuKvGeometry {
            scalar_bytes: 2,
            kv_heads: model_config.num_key_value_heads as u128,
            head_dim: model_config.head_dim as u128,
            full_layers: model_config
                .layer_types
                .iter()
                .filter(|kind| kind.as_str() == "full_attention")
                .count() as u128,
            sliding_layers: model_config
                .layer_types
                .iter()
                .filter(|kind| kind.as_str() != "full_attention")
                .count() as u128,
            sliding_window: model_config.sliding_window as u128,
        };
        let worst = estimate_cpu_request(
            &config,
            kv_geometry,
            2 * 1024 * 1024,
            config.model.max_model_len,
            20,
            decoded_token_byte_bound,
        )?;
        let worst_total = worst.total().map_err(grant_error)?;
        let global_budget = match request_budget_bytes {
            Some(bytes) => bytes,
            None => worst_total
                .checked_mul(config.scheduler.max_num_seqs as u128)
                .ok_or_else(|| LLMError::MemoryError("CPU request budget overflows".into()))?,
        };
        let delivery_limit = (config.scheduler.max_num_seqs as u128)
            .checked_mul(1024 * 1024)
            .ok_or_else(|| LLMError::MemoryError("CPU delivery budget overflows".into()))?;
        let limits =
            ReservationLimits::bounded(config.scheduler.max_num_seqs, worst_total, global_budget)
                .with_class_limit(MemoryClass::Delivery, delivery_limit);
        let ledger = ReservationLedger::new(limits).map_err(grant_error)?;
        let mut engine = Self::from_forward_with_decoded_bound(
            config,
            Box::new(NativeCpuBatchForward::new(model)),
            tokenizer,
            decoded_token_byte_bound,
        )?;
        engine.kv_geometry = Some(kv_geometry);
        engine.reservations = Some(ledger);
        Ok(engine)
    }

    #[cfg(test)]
    fn from_forward(
        config: EngineConfig,
        forward: Box<dyn CpuBatchForward>,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        let decoded_token_byte_bound = tokenizer.max_decoded_token_bytes()?;
        Self::from_forward_with_decoded_bound(config, forward, tokenizer, decoded_token_byte_bound)
    }

    fn from_forward_with_decoded_bound(
        config: EngineConfig,
        forward: Box<dyn CpuBatchForward>,
        tokenizer: Tokenizer,
        decoded_token_byte_bound: usize,
    ) -> Result<Self> {
        let scheduler = CpuScheduler::new(CpuSchedulerConfig {
            max_num_seqs: config.scheduler.max_num_seqs,
            max_num_batched_tokens: config.scheduler.max_num_batched_tokens,
            max_prefill_chunk: config.scheduler.max_prefill_chunk,
        })?;
        let terminal_token_ids = tokenizer.terminal_token_ids().to_vec();
        Ok(Self {
            config,
            forward,
            scheduler,
            table: SequenceTable::default(),
            sampler: Sampler::new(),
            tokenizer,
            terminal_token_ids,
            decoded_token_byte_bound,
            next_sequence_id: 0,
            next_arrival_order: 0,
            published: HashMap::new(),
            reservations: None,
            request_grants: HashMap::new(),
            kv_geometry: None,
            shutdown: false,
        })
    }

    pub fn add_request(
        &mut self,
        request_id: RequestId,
        prompt: String,
        sampling_params: SamplingParams,
    ) -> Result<SequenceId> {
        let prompt_token_ids = self.tokenizer.encode(&prompt)?;
        self.add_tokenized_request(request_id, prompt, prompt_token_ids, sampling_params)
    }

    pub fn add_tokenized_request(
        &mut self,
        request_id: RequestId,
        prompt: String,
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
    ) -> Result<SequenceId> {
        if self.shutdown {
            return Err(LLMError::SchedulerError("CPU engine is shut down".into()));
        }
        if sampling_params.best_of != 1 || sampling_params.use_beam_search {
            return Err(LLMError::ConfigError(
                "CPU batching supports one candidate and rejects best-of and beam search".into(),
            ));
        }
        if prompt_token_ids.is_empty() {
            return Err(LLMError::TokenizerError(
                "CPU prompt produced zero tokens".into(),
            ));
        }
        if prompt_token_ids
            .len()
            .checked_add(sampling_params.max_tokens)
            .is_none_or(|total| total > self.config.model.max_model_len)
        {
            return Err(LLMError::ConfigError(format!(
                "CPU prompt plus max_tokens exceeds context cap {}",
                self.config.model.max_model_len
            )));
        }
        let memory_estimate = self
            .kv_geometry
            .map(|geometry| {
                let reachable_context = prompt_token_ids
                    .len()
                    .checked_add(sampling_params.max_tokens)
                    .ok_or_else(|| {
                        LLMError::MemoryError("request context estimate overflows".into())
                    })?;
                estimate_cpu_request(
                    &self.config,
                    geometry,
                    prompt.len(),
                    reachable_context,
                    sampling_params.logprobs.unwrap_or(0),
                    self.decoded_token_byte_bound,
                )
            })
            .transpose()?;
        let memory_grant = match (&mut self.reservations, memory_estimate) {
            (Some(ledger), Some(estimate)) => {
                let classes = estimate.by_class.keys().copied().collect::<Vec<_>>();
                match ledger.grant(request_id, estimate) {
                    Ok(grant) => {
                        for class in classes {
                            crate::telemetry::metrics::record_reservation(
                                crate::telemetry::metrics::ReservationEvent::Grant,
                                class,
                                crate::telemetry::metrics::ResultClass::Accepted,
                                ledger.reserved(class),
                            );
                        }
                        Some(grant)
                    }
                    Err(error) => {
                        for class in classes {
                            crate::telemetry::metrics::record_reservation(
                                crate::telemetry::metrics::ReservationEvent::Reject,
                                class,
                                crate::telemetry::metrics::ResultClass::Rejected,
                                ledger.reserved(class),
                            );
                        }
                        return Err(grant_error(error));
                    }
                }
            }
            _ => None,
        };
        let sequence_id = SequenceId(self.next_sequence_id);
        self.next_sequence_id = self
            .next_sequence_id
            .checked_add(1)
            .ok_or_else(|| LLMError::SchedulerError("CPU sequence ID overflows".into()))?;
        let arrival_order = self.next_arrival_order;
        self.next_arrival_order = self
            .next_arrival_order
            .checked_add(1)
            .ok_or_else(|| LLMError::SchedulerError("CPU arrival order overflows".into()))?;
        let admission = (|| {
            let model_state = self
                .forward
                .new_sequence_state(self.config.model.max_model_len)?;
            let record = CpuSequenceRecord::new_with_optional_state(
                request_id,
                sequence_id,
                arrival_order,
                prompt,
                prompt_token_ids,
                sampling_params,
                model_state,
            )?;
            self.scheduler.add_sequence(&mut self.table, record)
        })();
        if let Err(error) = admission {
            if let (Some(ledger), Some(grant)) = (&mut self.reservations, &memory_grant) {
                let _ = ledger.release(grant.id);
            }
            return Err(error);
        }
        if let Some(grant) = memory_grant {
            self.reservations
                .as_mut()
                .expect("native memory grant has a ledger")
                .activate(grant.id)
                .map_err(grant_error)?;
            self.request_grants.insert(request_id, grant.id);
        }
        self.published
            .insert(sequence_id, PublishedCursor::default());
        Ok(sequence_id)
    }

    pub fn cancel_request(&mut self, request_id: RequestId) -> Result<bool> {
        let sequence_id = self.table.sequence_for_request(request_id);
        let cancelled = self.scheduler.cancel_request(&mut self.table, request_id)?;
        if cancelled {
            if let Some(sequence_id) = sequence_id {
                self.published.remove(&sequence_id);
            }
            if self.table.get_by_request(request_id).is_none() {
                self.release_request_grant(request_id)?;
            }
        }
        Ok(cancelled)
    }

    pub fn reserve(&mut self) -> Result<Option<CpuReservation>> {
        if self.shutdown {
            return Ok(None);
        }
        let reservation = self.scheduler.reserve(&mut self.table)?;
        if let Some(reservation) = &reservation {
            metrics::histogram!(
                crate::telemetry::metrics::SCHEDULED_ROWS,
                "backend" => crate::telemetry::metrics::BackendClass::Cpu.as_str()
            )
            .record(reservation.rows().len() as f64);
        }
        Ok(reservation)
    }

    pub fn execute(&mut self, reservation: CpuReservation) -> Result<PreparedCpuIteration> {
        let step_id = reservation.step_id();
        let result = self.execute_inner(reservation);
        if result.is_err() {
            self.scheduler.release(&mut self.table, step_id)?;
        }
        result
    }

    fn execute_inner(&mut self, reservation: CpuReservation) -> Result<PreparedCpuIteration> {
        let batch = CpuStepBatch::new(
            reservation
                .rows()
                .iter()
                .map(|row| {
                    CpuStepRow::new(
                        row.sequence_id,
                        row.token_id,
                        row.absolute_position,
                        row.logits_required,
                    )
                })
                .collect(),
        )?;
        let model = self.forward.prepare(&batch, &self.table)?;
        let mut staged = HashMap::new();
        for scheduled in reservation.rows().iter().filter(|row| row.logits_required) {
            let prepared = model
                .rows()
                .iter()
                .find(|row| {
                    row.sequence_id == scheduled.sequence_id
                        && row.absolute_position == scheduled.absolute_position
                })
                .ok_or_else(|| {
                    LLMError::ModelError(format!(
                        "CPU prepared step omitted logits row {}:{}",
                        scheduled.sequence_id, scheduled.absolute_position
                    ))
                })?;
            let logits = prepared.logits.as_deref().ok_or_else(|| {
                LLMError::ModelError(format!(
                    "CPU prepared row {}:{} has no logits",
                    scheduled.sequence_id, scheduled.absolute_position
                ))
            })?;
            let record = self.table.get(scheduled.sequence_id).ok_or_else(|| {
                LLMError::SchedulerError("CPU sequence disappeared during execution".into())
            })?;
            let mut generation = record.generation().clone();
            let past_tokens = generation.past_tokens().to_vec();
            let sampled = self.sampler.sample(
                logits,
                self.forward.vocab_size(),
                record.sampling_params(),
                &past_tokens,
                generation.rng_mut(),
            )?;
            generation.set_last_generated(sampled.token_id);
            generation.push_past_token(sampled.token_id);
            let decoded = self.tokenizer.decode(&[sampled.token_id])?;
            let mut output = record.output().clone();
            OutputProcessor::process_token(
                &mut output,
                sampled.token_id,
                sampled.logprob,
                (!sampled.top_logprobs.is_empty()).then_some(sampled.top_logprobs),
                &decoded,
                record.sampling_params(),
                &self.terminal_token_ids,
            );
            staged.insert(
                scheduled.sequence_id,
                StagedCpuSequence { generation, output },
            );
        }
        Ok(PreparedCpuIteration {
            reservation,
            model,
            staged,
        })
    }

    pub fn commit(&mut self, prepared: PreparedCpuIteration) -> Result<CpuBatchCommitResult> {
        let PreparedCpuIteration {
            reservation,
            model,
            mut staged,
        } = prepared;
        let step_id = reservation.step_id();
        let mut retained = HashSet::new();
        let mut cancelled_requests = Vec::new();
        let mut cancelled_sequences = Vec::new();
        for &(sequence_id, expected_revision) in reservation.sequence_revisions() {
            let Some(record) = self.table.get(sequence_id) else {
                model.discard();
                self.scheduler.release(&mut self.table, step_id)?;
                return Err(LLMError::SchedulerError(format!(
                    "reserved CPU sequence {sequence_id} disappeared before commit"
                )));
            };
            if record.lifecycle() == CpuSequenceLifecycle::Cancelled {
                cancelled_requests.push(record.request_id());
                cancelled_sequences.push(sequence_id);
                continue;
            }
            if record.lifecycle() != (CpuSequenceLifecycle::InFlight { step_id })
                || record.revision() != expected_revision
            {
                model.discard();
                self.scheduler.release(&mut self.table, step_id)?;
                return Err(LLMError::SchedulerError(format!(
                    "stale CPU reservation {step_id} for sequence {sequence_id}"
                )));
            }
            if record.revision() == u64::MAX {
                model.discard();
                self.scheduler.release(&mut self.table, step_id)?;
                return Err(LLMError::SchedulerError(
                    "CPU sequence revision overflows".into(),
                ));
            }
            retained.insert(sequence_id);
        }
        if let Err(error) = self.validate_commit_rows(&reservation, &retained, &staged) {
            model.discard();
            self.scheduler.release(&mut self.table, step_id)?;
            return Err(error);
        }

        if retained.is_empty() {
            model.discard();
        } else if let Err(error) = model.commit(&retained, &mut self.table) {
            self.scheduler.release(&mut self.table, step_id)?;
            return Err(error);
        }

        let mut outputs = Vec::new();
        let mut events = Vec::new();
        let mut finished_sequences = Vec::new();
        for &sequence_id in reservation.sequence_ids().collect::<Vec<_>>().iter() {
            if !retained.contains(&sequence_id) {
                continue;
            }
            let prefill_rows = reservation
                .rows()
                .iter()
                .filter(|row| {
                    row.sequence_id == sequence_id && row.phase == CpuScheduledPhase::Prefill
                })
                .count();
            let has_sample = staged.contains_key(&sequence_id);
            let record = self
                .table
                .get_mut(sequence_id)
                .expect("retained CPU sequence was validated");
            record
                .advance_prompt(prefill_rows)
                .expect("CPU prompt progress was validated");
            if let Some(staged) = staged.remove(&sequence_id) {
                *record.generation_mut() = staged.generation;
                *record.output_mut() = staged.output;
            }
            record
                .advance_revision()
                .expect("CPU revision capacity was validated");
            if record.output().is_finished() {
                record.set_lifecycle(CpuSequenceLifecycle::Finished);
            }
            if has_sample {
                let cursor = self
                    .published
                    .get_mut(&sequence_id)
                    .expect("admitted CPU sequence has a publication cursor");
                let output = OutputProcessor::build_request_delta(
                    record.request_id(),
                    record.prompt(),
                    record.prompt_token_ids(),
                    record.output(),
                    record.sampling_params(),
                    cursor.metadata_published,
                    cursor.text_bytes,
                    cursor.token_count,
                    cursor.logprob_count,
                    cursor.cumulative_logprob,
                )?;
                cursor.metadata_published = true;
                cursor.text_bytes = cursor
                    .text_bytes
                    .checked_add(output.outputs[0].text.len())
                    .ok_or_else(|| {
                        LLMError::SchedulerError("published text cursor overflows".into())
                    })?;
                cursor.token_count = record.output().token_ids.len();
                cursor.logprob_count = record.output().logprobs.len();
                cursor.cumulative_logprob = record.output().cumulative_logprob;

                let request_id = record.request_id();
                let completion = &output.outputs[0];
                events.push((
                    request_id,
                    CommittedEvent::Delta {
                        choice: completion.index as u32,
                        text: completion.text.clone(),
                        token_ids: completion.token_ids.clone(),
                        logprobs: completion.logprobs.clone(),
                    },
                ));
                if output.finished {
                    finished_sequences.push(sequence_id);
                    events.push((
                        request_id,
                        CommittedEvent::Usage {
                            committed_prompt: record.prompt_token_ids().len() as u64,
                            committed_completion: record.output().token_ids.len() as u64,
                        },
                    ));
                    events.push((
                        request_id,
                        CommittedEvent::Finish {
                            choice: 0,
                            reason: record
                                .output()
                                .finish_reason
                                .expect("finished output has a reason"),
                        },
                    ));
                    events.push((request_id, CommittedEvent::Done));
                }
                outputs.push(output);
            }
        }
        self.scheduler.complete(&mut self.table, step_id)?;
        for sequence_id in cancelled_sequences.into_iter().chain(finished_sequences) {
            self.published.remove(&sequence_id);
        }
        let terminal_requests = cancelled_requests
            .iter()
            .copied()
            .chain(
                outputs
                    .iter()
                    .filter(|output| output.finished)
                    .map(|output| output.request_id),
            )
            .collect::<Vec<_>>();
        for request_id in terminal_requests {
            self.release_request_grant(request_id)?;
        }
        Ok(CpuBatchCommitResult {
            outputs,
            events,
            cancelled_requests,
        })
    }

    pub fn discard(&mut self, prepared: PreparedCpuIteration) -> Result<()> {
        let PreparedCpuIteration {
            reservation, model, ..
        } = prepared;
        model.discard();
        self.scheduler
            .release(&mut self.table, reservation.step_id())
    }

    fn validate_commit_rows(
        &self,
        reservation: &CpuReservation,
        retained: &HashSet<SequenceId>,
        staged: &HashMap<SequenceId, StagedCpuSequence>,
    ) -> Result<()> {
        for &sequence_id in retained {
            let record = self.table.get(sequence_id).ok_or_else(|| {
                LLMError::SchedulerError("retained CPU sequence disappeared".into())
            })?;
            let rows = reservation
                .rows()
                .iter()
                .filter(|row| row.sequence_id == sequence_id)
                .collect::<Vec<_>>();
            let logits_required = rows.iter().any(|row| row.logits_required);
            if logits_required != staged.contains_key(&sequence_id) {
                return Err(LLMError::SchedulerError(format!(
                    "CPU logits/sample staging mismatch for sequence {sequence_id}"
                )));
            }
            let mut next_prefill = record.prompt_tokens_computed();
            for row in rows {
                match row.phase {
                    CpuScheduledPhase::Prefill => {
                        if row.absolute_position != next_prefill
                            || record.prompt_token_ids().get(next_prefill) != Some(&row.token_id)
                        {
                            return Err(LLMError::SchedulerError(format!(
                                "stale CPU prefill row for sequence {sequence_id}"
                            )));
                        }
                        next_prefill += 1;
                    }
                    CpuScheduledPhase::Decode => {
                        if row.absolute_position != record.decode_position()? {
                            return Err(LLMError::SchedulerError(format!(
                                "stale CPU decode row for sequence {sequence_id}"
                            )));
                        }
                    }
                }
            }
            if next_prefill > record.prompt_token_ids().len() {
                return Err(LLMError::SchedulerError(format!(
                    "CPU prompt commit exceeds sequence {sequence_id}"
                )));
            }
        }
        Ok(())
    }

    pub fn has_unfinished(&self) -> bool {
        self.scheduler.has_unfinished(&self.table)
    }

    pub fn table(&self) -> &SequenceTable {
        &self.table
    }

    pub fn shutdown(&mut self) -> Result<()> {
        self.shutdown = true;
        let request_ids = self
            .table
            .sequence_ids()
            .into_iter()
            .filter_map(|sequence_id| {
                self.table
                    .get(sequence_id)
                    .map(CpuSequenceRecord::request_id)
            })
            .collect::<Vec<_>>();
        for request_id in request_ids {
            self.cancel_request(request_id)?;
        }
        if let Some(ledger) = &mut self.reservations {
            ledger.release_all().map_err(grant_error)?;
        }
        self.request_grants.clear();
        Ok(())
    }

    pub fn reservation_ledger(&self) -> Option<&ReservationLedger> {
        self.reservations.as_ref()
    }

    pub fn tokenizer_clone(&self) -> Tokenizer {
        self.tokenizer.clone()
    }

    fn release_request_grant(&mut self, request_id: RequestId) -> Result<()> {
        if let Some(id) = self.request_grants.remove(&request_id) {
            let ledger = self
                .reservations
                .as_mut()
                .expect("request grant has a ledger");
            let classes = ledger
                .grant_snapshot(id)
                .map(|grant| grant.granted.by_class.keys().copied().collect::<Vec<_>>())
                .unwrap_or_default();
            ledger.release(id).map_err(grant_error)?;
            for class in classes {
                crate::telemetry::metrics::record_reservation(
                    crate::telemetry::metrics::ReservationEvent::Release,
                    class,
                    crate::telemetry::metrics::ResultClass::Completed,
                    ledger.reserved(class),
                );
            }
        }
        Ok(())
    }
}

fn estimate_cpu_request(
    config: &EngineConfig,
    geometry: CpuKvGeometry,
    prompt_bytes: usize,
    reachable_context: usize,
    logprobs: usize,
    decoded_token_byte_bound: usize,
) -> Result<MemoryEstimate> {
    let context = reachable_context as u128;
    let mut estimate = MemoryEstimate::new();
    estimate
        .checked_add(MemoryClass::Request, prompt_bytes as u128)
        .map_err(grant_error)?;
    estimate
        .checked_add(
            MemoryClass::KvCache,
            geometry.logical_bytes(context).map_err(grant_error)?,
        )
        .map_err(grant_error)?;
    estimate
        .checked_add(
            MemoryClass::StagedKv,
            geometry
                .staged_bytes(context.min(config.scheduler.max_num_batched_tokens as u128))
                .map_err(grant_error)?,
        )
        .map_err(grant_error)?;
    let token_vectors = context
        .checked_mul(std::mem::size_of::<u32>() as u128)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| LLMError::MemoryError("token vector estimate overflows".into()))?;
    estimate
        .checked_add(MemoryClass::TokenVectors, token_vectors)
        .map_err(grant_error)?;
    let logprob_bytes = context
        .checked_mul(logprobs as u128)
        .and_then(|value| value.checked_mul(8))
        .ok_or_else(|| LLMError::MemoryError("logprob estimate overflows".into()))?;
    let decoded_bytes = context
        .checked_mul(decoded_token_byte_bound as u128)
        .ok_or_else(|| LLMError::MemoryError("decoded byte estimate overflows".into()))?;
    estimate
        .checked_add(
            MemoryClass::GenerationState,
            logprob_bytes
                .checked_add(decoded_bytes)
                .and_then(|bytes| bytes.checked_add(4096))
                .ok_or_else(|| LLMError::MemoryError("generation estimate overflows".into()))?,
        )
        .map_err(grant_error)?;
    estimate
        .checked_add(MemoryClass::Delivery, 1024 * 1024)
        .map_err(grant_error)?;
    Ok(estimate)
}

fn grant_error(error: crate::memory::GrantFailure) -> LLMError {
    LLMError::MemoryError(format!("CPU logical reservation failed: {error}"))
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::service::{ServiceState, StableFailureCode};
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::Tokenizer as HfTokenizer;

    use super::*;

    #[derive(Debug, Default)]
    struct FakeCalls {
        batches: Vec<Vec<CpuStepRow>>,
        commits: usize,
        discards: usize,
        fail_prepare: bool,
        bad_logits: bool,
        prepare_delay_ms: u64,
    }

    struct FakeForward {
        calls: Arc<Mutex<FakeCalls>>,
    }

    struct FakePrepared {
        rows: Vec<CpuPreparedModelRow>,
        calls: Arc<Mutex<FakeCalls>>,
    }

    impl CpuPreparedModel for FakePrepared {
        fn rows(&self) -> &[CpuPreparedModelRow] {
            &self.rows
        }

        fn commit(
            self: Box<Self>,
            _retained: &HashSet<SequenceId>,
            _table: &mut SequenceTable,
        ) -> Result<()> {
            self.calls.lock().unwrap().commits += 1;
            Ok(())
        }

        fn discard(self: Box<Self>) {
            self.calls.lock().unwrap().discards += 1;
        }
    }

    impl CpuBatchForward for FakeForward {
        fn vocab_size(&self) -> usize {
            5
        }

        fn new_sequence_state(&self, _context_cap: usize) -> Result<Option<CpuSequenceModelState>> {
            Ok(None)
        }

        fn prepare(
            &mut self,
            batch: &CpuStepBatch,
            _table: &SequenceTable,
        ) -> Result<Box<dyn CpuPreparedModel>> {
            let mut calls = self.calls.lock().unwrap();
            if calls.fail_prepare {
                calls.fail_prepare = false;
                return Err(LLMError::ModelError("injected CPU model failure".into()));
            }
            calls.batches.push(batch.rows().to_vec());
            let bad_logits = calls.bad_logits;
            calls.bad_logits = false;
            let prepare_delay_ms = calls.prepare_delay_ms;
            drop(calls);
            if prepare_delay_ms != 0 {
                std::thread::sleep(std::time::Duration::from_millis(prepare_delay_ms));
            }
            let rows = batch
                .rows()
                .iter()
                .map(|row| CpuPreparedModelRow {
                    sequence_id: row.sequence_id,
                    absolute_position: row.absolute_position,
                    logits: row.logits_required.then(|| {
                        if bad_logits {
                            vec![0.0; 3]
                        } else if row.token_id % 2 == 0 {
                            vec![0.0, 5.0, 1.0, 2.0, 3.0]
                        } else {
                            vec![0.0, 1.0, 5.0, 2.0, 3.0]
                        }
                    }),
                })
                .collect();
            Ok(Box::new(FakePrepared {
                rows,
                calls: self.calls.clone(),
            }))
        }
    }

    fn tokenizer() -> Tokenizer {
        let vocab = HashMap::from([
            ("hello".to_string(), 0),
            ("world".to_string(), 1),
            ("[UNK]".to_string(), 2),
        ]);
        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = HfTokenizer::new(bpe);
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        tokenizer.save(&path, false).unwrap();
        Tokenizer::from_file(&path).unwrap()
    }

    fn engine(
        max_num_seqs: usize,
        max_num_batched_tokens: usize,
        max_prefill_chunk: usize,
    ) -> (CpuBatchEngine, Arc<Mutex<FakeCalls>>) {
        let mut config = EngineConfig::default();
        config.device.device = "cpu".into();
        config.model.max_model_len = 32;
        config.scheduler.max_num_seqs = max_num_seqs;
        config.scheduler.max_num_batched_tokens = max_num_batched_tokens;
        config.scheduler.max_prefill_chunk = max_prefill_chunk;
        let calls = Arc::new(Mutex::new(FakeCalls::default()));
        let engine = CpuBatchEngine::from_forward(
            config,
            Box::new(FakeForward {
                calls: calls.clone(),
            }),
            tokenizer(),
        )
        .unwrap();
        (engine, calls)
    }

    fn params(max_tokens: usize) -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            seed: Some(9),
            max_tokens,
            ..Default::default()
        }
    }

    #[test]
    fn multi_request_prefill_and_decode_commit_in_request_order() {
        let (mut engine, calls) = engine(2, 4, 0);
        engine
            .add_tokenized_request(RequestId(1), "a".into(), vec![10], params(2))
            .unwrap();
        engine
            .add_tokenized_request(RequestId(2), "b".into(), vec![11], params(2))
            .unwrap();
        let reservation = engine.reserve().unwrap().unwrap();
        assert_eq!(reservation.rows().len(), 2);
        let prepared = engine.execute(reservation).unwrap();
        let first = engine.commit(prepared).unwrap();
        assert_eq!(
            first
                .outputs
                .iter()
                .map(|output| output.request_id)
                .collect::<Vec<_>>(),
            vec![RequestId(1), RequestId(2)]
        );
        assert!(first.outputs.iter().all(|output| !output.finished));

        let reservation = engine.reserve().unwrap().unwrap();
        assert!(reservation
            .rows()
            .iter()
            .all(|row| row.phase == CpuScheduledPhase::Decode));
        let prepared = engine.execute(reservation).unwrap();
        let second = engine.commit(prepared).unwrap();
        assert!(second.outputs.iter().all(|output| output.finished));
        assert!(!engine.has_unfinished());
        assert_eq!(calls.lock().unwrap().commits, 2);
    }

    #[test]
    fn intermediate_prompt_chunks_commit_without_sampling_or_output() {
        let (mut engine, calls) = engine(1, 2, 2);
        engine
            .add_tokenized_request(RequestId(1), "a".into(), vec![10, 11, 12], params(1))
            .unwrap();
        let reservation = engine.reserve().unwrap().unwrap();
        assert!(reservation.rows().iter().all(|row| !row.logits_required));
        let prepared = engine.execute(reservation).unwrap();
        let committed = engine.commit(prepared).unwrap();
        assert!(committed.outputs.is_empty());
        assert_eq!(
            engine
                .table()
                .get_by_request(RequestId(1))
                .unwrap()
                .prompt_tokens_computed(),
            2
        );
        assert_eq!(calls.lock().unwrap().commits, 1);
    }

    #[test]
    fn final_multirow_prompt_chunk_samples_only_its_last_row() {
        let (mut engine, calls) = engine(1, 4, 0);
        engine
            .add_tokenized_request(RequestId(1), "a".into(), vec![10, 11, 12], params(1))
            .unwrap();
        let reservation = engine.reserve().unwrap().unwrap();
        assert_eq!(
            reservation
                .rows()
                .iter()
                .map(|row| row.logits_required)
                .collect::<Vec<_>>(),
            vec![false, false, true]
        );
        let prepared = engine.execute(reservation).unwrap();
        let committed = engine.commit(prepared).unwrap();
        assert_eq!(committed.outputs.len(), 1);
        assert!(committed.outputs[0].finished);
        assert_eq!(calls.lock().unwrap().commits, 1);
    }

    #[test]
    fn model_and_sampling_failures_release_without_progress() {
        let (mut engine, calls) = engine(1, 2, 0);
        engine
            .add_tokenized_request(RequestId(1), "a".into(), vec![10], params(1))
            .unwrap();
        calls.lock().unwrap().fail_prepare = true;
        let reservation = engine.reserve().unwrap().unwrap();
        assert!(engine.execute(reservation).is_err());
        let record = engine.table().get_by_request(RequestId(1)).unwrap();
        assert_eq!(record.prompt_tokens_computed(), 0);
        assert!(record.output().token_ids.is_empty());

        calls.lock().unwrap().bad_logits = true;
        let reservation = engine.reserve().unwrap().unwrap();
        assert!(engine.execute(reservation).is_err());
        let record = engine.table().get_by_request(RequestId(1)).unwrap();
        assert_eq!(record.prompt_tokens_computed(), 0);
        assert!(record.output().token_ids.is_empty());
        assert_eq!(calls.lock().unwrap().commits, 0);
    }

    #[test]
    fn inflight_cancellation_discards_one_sequence_beside_success() {
        let (mut engine, calls) = engine(2, 2, 0);
        engine
            .add_tokenized_request(RequestId(1), "a".into(), vec![10], params(2))
            .unwrap();
        engine
            .add_tokenized_request(RequestId(2), "b".into(), vec![11], params(2))
            .unwrap();
        let reservation = engine.reserve().unwrap().unwrap();
        let prepared = engine.execute(reservation).unwrap();
        engine.cancel_request(RequestId(1)).unwrap();
        let committed = engine.commit(prepared).unwrap();
        assert_eq!(committed.cancelled_requests, vec![RequestId(1)]);
        assert_eq!(committed.outputs.len(), 1);
        assert_eq!(committed.outputs[0].request_id, RequestId(2));
        assert!(engine.table().get_by_request(RequestId(1)).is_none());
        assert_eq!(
            engine
                .table()
                .get_by_request(RequestId(2))
                .unwrap()
                .prompt_tokens_computed(),
            1
        );
        assert_eq!(calls.lock().unwrap().commits, 1);
    }

    #[test]
    fn stale_reservation_releases_every_sequence_without_commit() {
        let (mut engine, calls) = engine(2, 2, 0);
        engine
            .add_tokenized_request(RequestId(1), "a".into(), vec![10], params(2))
            .unwrap();
        engine
            .add_tokenized_request(RequestId(2), "b".into(), vec![11], params(2))
            .unwrap();
        let reservation = engine.reserve().unwrap().unwrap();
        let prepared = engine.execute(reservation).unwrap();
        engine
            .table
            .get_mut(SequenceId(1))
            .unwrap()
            .advance_revision()
            .unwrap();
        assert!(engine.commit(prepared).is_err());
        assert_eq!(calls.lock().unwrap().commits, 0);
        assert_eq!(
            engine
                .table()
                .get(SequenceId(0))
                .unwrap()
                .prompt_tokens_computed(),
            0
        );
        assert_eq!(
            engine
                .table()
                .get(SequenceId(1))
                .unwrap()
                .prompt_tokens_computed(),
            0
        );
    }

    #[test]
    fn best_of_and_beam_are_rejected_at_admission() {
        let (mut engine, _) = engine(2, 2, 0);
        let mut best_of = params(2);
        best_of.best_of = 2;
        assert!(engine
            .add_tokenized_request(RequestId(1), "a".into(), vec![10], best_of)
            .is_err());
        let mut beam = params(2);
        beam.use_beam_search = true;
        assert!(engine
            .add_tokenized_request(RequestId(2), "b".into(), vec![11], beam)
            .is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn async_engine_preserves_each_concurrent_request_stream_order() {
        let (engine, _) = engine(2, 2, 0);
        let engine = crate::AsyncCpuBatchEngine::new(engine);
        let (first, second) = tokio::join!(
            engine.generate("hello".into(), params(2)),
            engine.generate("world".into(), params(2)),
        );
        let mut first = first.unwrap();
        let mut second = second.unwrap();
        let first_id = first.request_id;
        let second_id = second.request_id;
        let mut first_outputs = Vec::new();
        let mut second_outputs = Vec::new();
        while let Some(output) = first.events.recv().await {
            first_outputs.push(output);
        }
        while let Some(output) = second.events.recv().await {
            second_outputs.push(output);
        }
        assert_ne!(first_id, second_id);
        for outputs in [&first_outputs, &second_outputs] {
            assert_eq!(
                outputs
                    .iter()
                    .filter_map(|event| match event {
                        CommittedEvent::Delta { token_ids, .. } => Some(token_ids.len()),
                        _ => None,
                    })
                    .sum::<usize>(),
                2
            );
            assert!(outputs.iter().any(|event| matches!(
                event,
                CommittedEvent::Usage {
                    committed_completion: 2,
                    ..
                }
            )));
            assert!(outputs
                .iter()
                .any(|event| matches!(event, CommittedEvent::Finish { .. })));
            // Adjacent deltas may be coalesced, but Done is always last.
            assert!(matches!(outputs.last(), Some(CommittedEvent::Done)));
        }
        engine.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn disconnected_inflight_stream_is_cancelled_before_commit() {
        let (engine, calls) = engine(1, 1, 0);
        calls.lock().unwrap().prepare_delay_ms = 50;
        let engine = crate::AsyncCpuBatchEngine::new(engine);
        let request = engine.generate("hello".into(), params(2)).await.unwrap();
        drop(request);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert_eq!(calls.lock().unwrap().commits, 0);
        engine.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn owner_progresses_without_client_reads_and_releases_delivery_ownership() {
        let (engine, calls) = engine(1, 1, 0);
        let engine = crate::AsyncCpuBatchEngine::new(engine);
        let mut request = engine.generate("hello".into(), params(2)).await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if calls.lock().unwrap().commits >= 2 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let mut events = Vec::new();
        while let Some(event) = request.events.recv().await {
            events.push(event);
        }
        assert!(matches!(events.last(), Some(CommittedEvent::Done)));
        assert_eq!(engine.queued_delivery_bytes(), 0);
        engine.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn disconnect_after_first_commit_allows_recovery_request_and_clean_shutdown() {
        let (engine, calls) = engine(1, 1, 0);
        let engine = crate::AsyncCpuBatchEngine::new(engine);
        let mut first = engine.generate("hello".into(), params(3)).await.unwrap();
        assert!(matches!(
            first.events.recv().await,
            Some(CommittedEvent::Delta { .. })
        ));
        drop(first);
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if engine.available_request_slots() == 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let mut recovery = engine.generate("world".into(), params(1)).await.unwrap();
        let mut events = Vec::new();
        while let Some(event) = recovery.events.recv().await {
            events.push(event);
        }
        assert!(matches!(events.last(), Some(CommittedEvent::Done)));
        assert!(calls.lock().unwrap().commits >= 2);
        assert_eq!(engine.queued_delivery_bytes(), 0);
        engine.shutdown().await.unwrap();
        assert_eq!(engine.lifecycle().status().state, ServiceState::Stopped);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn draining_closes_admission_finishes_active_work_and_joins_owner() {
        let (engine, _) = engine(1, 1, 0);
        let engine = crate::AsyncCpuBatchEngine::new(engine);
        let mut active = engine.generate("hello".into(), params(2)).await.unwrap();
        engine.begin_shutdown().unwrap();
        let rejected = engine
            .generate("world".into(), params(1))
            .await
            .unwrap_err();
        assert_eq!(rejected.code, StableFailureCode::Draining);
        let mut events = Vec::new();
        while let Some(event) = active.events.recv().await {
            events.push(event);
        }
        assert!(matches!(events.last(), Some(CommittedEvent::Done)));
        engine.shutdown().await.unwrap();
        assert_eq!(engine.lifecycle().status().state, ServiceState::Stopped);
        assert_eq!(engine.queued_delivery_bytes(), 0);
    }
}
