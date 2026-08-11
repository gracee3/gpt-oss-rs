//! Batch-one native CPU executor for GPT-OSS.

use std::path::Path;
use std::sync::Arc;

use gpt_oss_core::prelude::{LLMError, RequestId, Result, SequenceId, TokenId};
use gpt_oss_cpu_kernels::KernelPath;
use gpt_oss_model_runner::sampling::Sampler;
use gpt_oss_model_runner::{
    CpuExecutionContext, CpuModel, CpuModelRunner, CpuSequenceModelState, CpuStepBatch, CpuStepRow,
    PreparedCpuStep,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::engine::{Executor, ExecutorInput, SamplerOutput};

trait CpuForward: Send {
    fn vocab_size(&self) -> usize;
    fn prepare_prefill(
        &mut self,
        sequence_id: SequenceId,
        token_ids: &[TokenId],
    ) -> Result<Vec<f32>>;
    fn prepare_decode(&mut self, sequence_id: SequenceId, token_id: TokenId) -> Result<Vec<f32>>;
    fn commit(&mut self) -> Result<()>;
    fn discard(&mut self);
    fn reset(&mut self) -> Result<()>;
    fn abort(&mut self) -> Result<()>;
    fn remove(&mut self) -> Result<()>;
}

struct NativePendingStep {
    sequence_id: SequenceId,
    prepared: PreparedCpuStep,
}

struct NativeCpuForward {
    model: Arc<CpuModel>,
    state: CpuSequenceModelState,
    execution: CpuExecutionContext,
    pending: Option<NativePendingStep>,
}

impl NativeCpuForward {
    fn from_runner(runner: CpuModelRunner) -> Self {
        let (model, state, execution) = runner.into_parts();
        Self {
            model,
            state,
            execution,
            pending: None,
        }
    }

    fn store_prepared(
        &mut self,
        sequence_id: SequenceId,
        prepared: PreparedCpuStep,
    ) -> Result<Vec<f32>> {
        let logits = prepared
            .rows()
            .iter()
            .rev()
            .find_map(|row| row.logits())
            .ok_or_else(|| LLMError::ModelError("CPU prepared step has no logits".into()))?
            .to_vec();
        self.pending = Some(NativePendingStep {
            sequence_id,
            prepared,
        });
        Ok(logits)
    }
}

impl CpuForward for NativeCpuForward {
    fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }

    fn prepare_prefill(
        &mut self,
        sequence_id: SequenceId,
        token_ids: &[TokenId],
    ) -> Result<Vec<f32>> {
        if self.pending.is_some() {
            return Err(LLMError::ModelError(
                "CPU forward already has a prepared step".into(),
            ));
        }
        let rows = token_ids
            .iter()
            .enumerate()
            .map(|(offset, &token_id)| {
                CpuStepRow::new(
                    sequence_id,
                    token_id,
                    self.state.position() + offset,
                    offset + 1 == token_ids.len(),
                )
            })
            .collect();
        let batch = CpuStepBatch::new(rows)?;
        let prepared =
            self.model
                .prepare_step(&mut self.execution, &batch, &[(sequence_id, &self.state)])?;
        self.store_prepared(sequence_id, prepared)
    }

    fn prepare_decode(&mut self, sequence_id: SequenceId, token_id: TokenId) -> Result<Vec<f32>> {
        if self.pending.is_some() {
            return Err(LLMError::ModelError(
                "CPU forward already has a prepared step".into(),
            ));
        }
        let batch = CpuStepBatch::single(CpuStepRow::new(
            sequence_id,
            token_id,
            self.state.position(),
            true,
        ));
        let prepared =
            self.model
                .prepare_step(&mut self.execution, &batch, &[(sequence_id, &self.state)])?;
        self.store_prepared(sequence_id, prepared)
    }

    fn commit(&mut self) -> Result<()> {
        let pending = self.pending.take().ok_or_else(|| {
            LLMError::ModelError("CPU forward has no prepared step to commit".into())
        })?;
        pending
            .prepared
            .commit(&mut [(pending.sequence_id, &mut self.state)])?;
        Ok(())
    }

    fn discard(&mut self) {
        self.pending.take();
    }

    fn reset(&mut self) -> Result<()> {
        self.discard();
        self.state.reset()
    }

    fn abort(&mut self) -> Result<()> {
        self.discard();
        self.state.abort()
    }

    fn remove(&mut self) -> Result<()> {
        self.discard();
        self.state = self.model.new_sequence_state(self.state.context_cap())?;
        Ok(())
    }
}

/// Sampling state committed alongside the model's per-sequence KV state.
#[derive(Clone)]
pub struct CpuGenerationState {
    request_id: RequestId,
    sequence_id: SequenceId,
    last_generated: Option<TokenId>,
    past_tokens: Vec<TokenId>,
    rng: StdRng,
}

impl CpuGenerationState {
    pub fn request_id(&self) -> RequestId {
        self.request_id
    }

    pub fn sequence_id(&self) -> SequenceId {
        self.sequence_id
    }

    pub fn last_generated(&self) -> Option<TokenId> {
        self.last_generated
    }

    pub fn past_tokens(&self) -> &[TokenId] {
        &self.past_tokens
    }
}

/// Real CPU executor owning a native [`CpuModelRunner`] and the sampling state
/// for the single active request.
pub struct CpuWorker {
    runner: Box<dyn CpuForward>,
    sampler: Sampler,
    state: Option<CpuGenerationState>,
    shutdown: bool,
}

impl CpuWorker {
    pub fn load(
        snapshot: impl AsRef<Path>,
        repack_root: impl AsRef<Path>,
        kernel_path: KernelPath,
        threads: usize,
        context_cap: usize,
    ) -> Result<Self> {
        let runner =
            CpuModelRunner::load(snapshot, repack_root, kernel_path, threads, context_cap)?;
        tracing::info!(
            requested_path = %kernel_path,
            compatibility_path = %runner.kernel_path(),
            dispatch_plan = %runner.kernel_dispatch_plan(),
            mxfp4_gemv = %runner.kernel_dispatch_plan().mxfp4_gemv(),
            mxfp4_layout = %runner.kernel_dispatch_plan().mxfp4_weight_layout(),
            "resolved CPU kernel dispatch"
        );
        Ok(Self::from_runner(runner))
    }

    pub fn from_runner(runner: CpuModelRunner) -> Self {
        Self {
            runner: Box::new(NativeCpuForward::from_runner(runner)),
            sampler: Sampler::new(),
            state: None,
            shutdown: false,
        }
    }

    #[cfg(test)]
    fn from_forward(runner: impl CpuForward + 'static) -> Self {
        Self {
            runner: Box::new(runner),
            sampler: Sampler::new(),
            state: None,
            shutdown: false,
        }
    }

    pub fn abort_sequence(&mut self, sequence_id: SequenceId) -> Result<bool> {
        if self
            .state
            .as_ref()
            .is_none_or(|state| state.sequence_id != sequence_id)
        {
            return Ok(false);
        }
        self.runner.abort()?;
        self.state = None;
        Ok(true)
    }

    pub fn reset_sequence(&mut self, sequence_id: SequenceId) -> Result<bool> {
        if self
            .state
            .as_ref()
            .is_none_or(|state| state.sequence_id != sequence_id)
        {
            return Ok(false);
        }
        self.runner.reset()?;
        self.state = None;
        Ok(true)
    }

    pub fn remove_sequence(&mut self, sequence_id: SequenceId) -> Result<bool> {
        if self
            .state
            .as_ref()
            .is_none_or(|state| state.sequence_id != sequence_id)
        {
            return Ok(false);
        }
        self.runner.remove()?;
        self.state = None;
        Ok(true)
    }

    pub fn shutdown(&mut self) -> Result<()> {
        if self.shutdown {
            return Ok(());
        }
        self.runner.abort()?;
        self.state = None;
        self.shutdown = true;
        Ok(())
    }
}

impl Executor for CpuWorker {
    fn execute_model(&mut self, input: ExecutorInput) -> Result<Vec<SamplerOutput>> {
        if self.shutdown {
            return Err(LLMError::SchedulerError("CPU worker is shut down".into()));
        }
        if input.seq_group_metadata.len() != 1 {
            return Err(LLMError::SchedulerError(format!(
                "CPU backend requires exactly one scheduled request, got {}",
                input.seq_group_metadata.len()
            )));
        }
        let metadata = &input.seq_group_metadata[0];
        if metadata.seq_data.len() != 1
            || metadata.sampling_params.best_of != 1
            || metadata.sampling_params.use_beam_search
        {
            return Err(LLMError::ConfigError(
                "CPU backend supports one sequence and does not support best-of or beam search"
                    .into(),
            ));
        }
        let (&seq_id, sequence) =
            metadata.seq_data.iter().next().ok_or_else(|| {
                LLMError::SchedulerError("CPU request has no sequence data".into())
            })?;

        let continuing = self.state.as_ref().is_some_and(|state| {
            state.request_id == metadata.request_id && state.sequence_id == seq_id
        });
        let mut staged_generation = if continuing {
            self.state
                .clone()
                .ok_or_else(|| LLMError::SchedulerError("CPU state disappeared".into()))?
        } else {
            self.runner.remove()?;
            self.state = None;
            let prompt = &sequence.prompt_token_ids;
            if prompt.is_empty() {
                return Err(LLMError::SchedulerError(
                    "CPU request has an empty prompt".into(),
                ));
            }
            CpuGenerationState {
                request_id: metadata.request_id,
                sequence_id: seq_id,
                last_generated: None,
                past_tokens: prompt.clone(),
                rng: StdRng::seed_from_u64(
                    metadata.sampling_params.seed.unwrap_or_else(rand::random),
                ),
            }
        };
        let logits = if continuing {
            let token = self
                .state
                .as_ref()
                .and_then(|state| state.last_generated)
                .ok_or_else(|| {
                    LLMError::SchedulerError("CPU decode has no prior sampled token".into())
                })?;
            self.runner.prepare_decode(seq_id, token)?
        } else {
            let prompt = &sequence.prompt_token_ids;
            self.runner.prepare_prefill(seq_id, prompt)?
        };

        let sampled = match self.sampler.sample(
            &logits,
            self.runner.vocab_size(),
            &metadata.sampling_params,
            &staged_generation.past_tokens,
            &mut staged_generation.rng,
        ) {
            Ok(sampled) => sampled,
            Err(error) => {
                self.runner.discard();
                return Err(error);
            }
        };
        staged_generation.last_generated = Some(sampled.token_id);
        staged_generation.past_tokens.push(sampled.token_id);
        if let Err(error) = self.runner.commit() {
            self.runner.discard();
            return Err(error);
        }
        self.state = Some(staged_generation);
        tracing::debug!(
            request_id = %metadata.request_id,
            seq_id = %seq_id,
            token_id = sampled.token_id,
            logprob = sampled.logprob,
            "CPU sampled token"
        );

        Ok(vec![SamplerOutput {
            seq_id,
            token_id: sampled.token_id,
            logprob: sampled.logprob,
            top_logprobs: (!sampled.top_logprobs.is_empty()).then_some(sampled.top_logprobs),
        }])
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use gpt_oss_core::prelude::{SamplingParams, SequenceId};
    use gpt_oss_engine::sequence::{SequenceData, SequenceGroupMetadata};

    use super::*;

    #[derive(Default)]
    struct Calls {
        prefills: Vec<Vec<TokenId>>,
        decodes: Vec<TokenId>,
        commits: usize,
        discards: usize,
        resets: usize,
        aborts: usize,
        removes: usize,
    }

    enum FakePending {
        Prefill(Vec<TokenId>),
        Decode(TokenId),
    }

    struct FakeForward {
        calls: Arc<Mutex<Calls>>,
        pending: Option<FakePending>,
        bad_logits_remaining: usize,
    }

    impl FakeForward {
        fn new(calls: Arc<Mutex<Calls>>) -> Self {
            Self {
                calls,
                pending: None,
                bad_logits_remaining: 0,
            }
        }

        fn with_bad_logits(calls: Arc<Mutex<Calls>>, count: usize) -> Self {
            Self {
                calls,
                pending: None,
                bad_logits_remaining: count,
            }
        }

        fn logits(&mut self, normal: Vec<f32>) -> Vec<f32> {
            if self.bad_logits_remaining == 0 {
                normal
            } else {
                self.bad_logits_remaining -= 1;
                vec![0.0; 3]
            }
        }
    }

    impl CpuForward for FakeForward {
        fn vocab_size(&self) -> usize {
            4
        }

        fn prepare_prefill(
            &mut self,
            _sequence_id: SequenceId,
            token_ids: &[TokenId],
        ) -> Result<Vec<f32>> {
            assert!(self.pending.is_none());
            self.pending = Some(FakePending::Prefill(token_ids.to_vec()));
            Ok(self.logits(vec![0.0, 1.0, 4.0, 3.0]))
        }

        fn prepare_decode(
            &mut self,
            _sequence_id: SequenceId,
            token_id: TokenId,
        ) -> Result<Vec<f32>> {
            assert!(self.pending.is_none());
            self.pending = Some(FakePending::Decode(token_id));
            Ok(self.logits(vec![0.0, 5.0, 2.0, 1.0]))
        }

        fn commit(&mut self) -> Result<()> {
            let pending = self.pending.take().unwrap();
            let mut calls = self.calls.lock().unwrap();
            match pending {
                FakePending::Prefill(tokens) => calls.prefills.push(tokens),
                FakePending::Decode(token) => calls.decodes.push(token),
            }
            calls.commits += 1;
            Ok(())
        }

        fn discard(&mut self) {
            self.pending.take();
            self.calls.lock().unwrap().discards += 1;
        }

        fn reset(&mut self) -> Result<()> {
            self.pending.take();
            self.calls.lock().unwrap().resets += 1;
            Ok(())
        }

        fn abort(&mut self) -> Result<()> {
            self.pending.take();
            self.calls.lock().unwrap().aborts += 1;
            Ok(())
        }

        fn remove(&mut self) -> Result<()> {
            self.pending.take();
            self.calls.lock().unwrap().removes += 1;
            Ok(())
        }
    }

    fn input(request: u64, seq: u64, prompt: &[TokenId]) -> ExecutorInput {
        let mut seq_data = HashMap::new();
        seq_data.insert(
            SequenceId(seq),
            SequenceData {
                prompt_token_ids: prompt.to_vec(),
                output_token_ids: Vec::new(),
                cumulative_logprob: 0.0,
            },
        );
        ExecutorInput {
            seq_group_metadata: vec![SequenceGroupMetadata {
                request_id: RequestId(request),
                is_prompt: true,
                seq_data,
                sampling_params: SamplingParams {
                    temperature: 0.0,
                    seed: Some(7),
                    ..Default::default()
                },
                block_tables: HashMap::new(),
            }],
        }
    }

    #[test]
    fn prefills_then_decodes_last_sample_and_resets_for_new_request() {
        let calls = Arc::new(Mutex::new(Calls::default()));
        let mut worker = CpuWorker::from_forward(FakeForward::new(calls.clone()));

        let first = worker.execute_model(input(1, 10, &[1, 3])).unwrap();
        assert_eq!(first[0].token_id, 2);
        let second = worker.execute_model(input(1, 10, &[1, 3])).unwrap();
        assert_eq!(second[0].token_id, 1);
        worker.execute_model(input(2, 11, &[3, 2])).unwrap();

        let calls = calls.lock().unwrap();
        assert_eq!(calls.prefills, vec![vec![1, 3], vec![3, 2]]);
        assert_eq!(calls.decodes, vec![2]);
        assert_eq!(calls.commits, 3);
        assert_eq!(calls.removes, 2);
    }

    #[test]
    fn rejects_batching_and_best_of() {
        let calls = Arc::new(Mutex::new(Calls::default()));
        let mut worker = CpuWorker::from_forward(FakeForward::new(calls));
        let mut batched = input(1, 10, &[1]);
        batched
            .seq_group_metadata
            .push(input(2, 11, &[2]).seq_group_metadata.pop().unwrap());
        assert!(worker.execute_model(batched).is_err());

        let mut best_of = input(1, 10, &[1]);
        best_of.seq_group_metadata[0].sampling_params.best_of = 2;
        assert!(worker.execute_model(best_of).is_err());
    }

    #[test]
    fn sampling_failure_discards_model_and_generation_state() {
        let calls = Arc::new(Mutex::new(Calls::default()));
        let mut worker = CpuWorker::from_forward(FakeForward::with_bad_logits(calls.clone(), 1));

        assert!(worker.execute_model(input(1, 10, &[1, 3])).is_err());
        assert!(worker.state.is_none());

        let retried = worker.execute_model(input(1, 10, &[1, 3])).unwrap();
        assert_eq!(retried[0].token_id, 2);
        let state = worker.state.as_ref().unwrap();
        assert_eq!(state.past_tokens(), &[1, 3, 2]);

        let calls = calls.lock().unwrap();
        assert_eq!(calls.prefills, vec![vec![1, 3]]);
        assert_eq!(calls.commits, 1);
        assert_eq!(calls.discards, 1);
    }

    #[test]
    fn lifecycle_operations_are_explicit_and_id_scoped() {
        let calls = Arc::new(Mutex::new(Calls::default()));
        let mut worker = CpuWorker::from_forward(FakeForward::new(calls.clone()));

        worker.execute_model(input(1, 10, &[1])).unwrap();
        assert!(!worker.reset_sequence(SequenceId(99)).unwrap());
        assert!(worker.reset_sequence(SequenceId(10)).unwrap());

        worker.execute_model(input(2, 20, &[2])).unwrap();
        assert!(!worker.abort_sequence(SequenceId(99)).unwrap());
        assert!(worker.abort_sequence(SequenceId(20)).unwrap());

        worker.execute_model(input(3, 30, &[3])).unwrap();
        assert!(worker.remove_sequence(SequenceId(30)).unwrap());
        worker.shutdown().unwrap();
        worker.shutdown().unwrap();
        assert!(worker.execute_model(input(4, 40, &[1])).is_err());

        let calls = calls.lock().unwrap();
        assert_eq!(calls.resets, 1);
        assert_eq!(calls.aborts, 2);
        assert_eq!(calls.removes, 4);
    }
}
