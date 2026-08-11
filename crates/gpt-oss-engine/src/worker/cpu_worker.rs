//! Batch-one native CPU executor for GPT-OSS.

use std::path::Path;

use gpt_oss_core::prelude::{LLMError, RequestId, Result, TokenId};
use gpt_oss_cpu_kernels::KernelPath;
use gpt_oss_model_runner::sampling::Sampler;
use gpt_oss_model_runner::CpuModelRunner;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::engine::{Executor, ExecutorInput, SamplerOutput};

trait CpuForward: Send {
    fn vocab_size(&self) -> usize;
    fn prefill(&mut self, token_ids: &[TokenId]) -> Result<Vec<f32>>;
    fn decode(&mut self, token_id: TokenId) -> Result<Vec<f32>>;
}

impl CpuForward for CpuModelRunner {
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }

    fn prefill(&mut self, token_ids: &[TokenId]) -> Result<Vec<f32>> {
        CpuModelRunner::prefill(self, token_ids)
    }

    fn decode(&mut self, token_id: TokenId) -> Result<Vec<f32>> {
        CpuModelRunner::decode(self, token_id)
    }
}

struct CpuSequenceState {
    request_id: RequestId,
    last_generated: Option<TokenId>,
    past_tokens: Vec<TokenId>,
    rng: StdRng,
}

/// Real CPU executor owning a native [`CpuModelRunner`] and the sampling state
/// for the single active request.
pub struct CpuWorker {
    runner: Box<dyn CpuForward>,
    sampler: Sampler,
    state: Option<CpuSequenceState>,
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
            "resolved CPU kernel dispatch"
        );
        Ok(Self::from_runner(runner))
    }

    pub fn from_runner(runner: CpuModelRunner) -> Self {
        Self {
            runner: Box::new(runner),
            sampler: Sampler::new(),
            state: None,
        }
    }

    #[cfg(test)]
    fn from_forward(runner: impl CpuForward + 'static) -> Self {
        Self {
            runner: Box::new(runner),
            sampler: Sampler::new(),
            state: None,
        }
    }
}

impl Executor for CpuWorker {
    fn execute_model(&mut self, input: ExecutorInput) -> Result<Vec<SamplerOutput>> {
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

        let continuing = self
            .state
            .as_ref()
            .is_some_and(|state| state.request_id == metadata.request_id);
        let logits = if continuing {
            let token = self
                .state
                .as_ref()
                .and_then(|state| state.last_generated)
                .ok_or_else(|| {
                    LLMError::SchedulerError("CPU decode has no prior sampled token".into())
                })?;
            self.runner.decode(token)?
        } else {
            let prompt = &sequence.prompt_token_ids;
            if prompt.is_empty() {
                return Err(LLMError::SchedulerError(
                    "CPU request has an empty prompt".into(),
                ));
            }
            let logits = self.runner.prefill(prompt)?;
            self.state = Some(CpuSequenceState {
                request_id: metadata.request_id,
                last_generated: None,
                past_tokens: prompt.clone(),
                rng: StdRng::seed_from_u64(
                    metadata.sampling_params.seed.unwrap_or_else(rand::random),
                ),
            });
            logits
        };

        let state = self
            .state
            .as_mut()
            .ok_or_else(|| LLMError::SchedulerError("CPU sampling state is missing".into()))?;
        let sampled = self.sampler.sample(
            &logits,
            self.runner.vocab_size(),
            &metadata.sampling_params,
            &state.past_tokens,
            &mut state.rng,
        )?;
        state.last_generated = Some(sampled.token_id);
        state.past_tokens.push(sampled.token_id);
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
    }

    struct FakeForward {
        calls: Arc<Mutex<Calls>>,
    }

    impl CpuForward for FakeForward {
        fn vocab_size(&self) -> usize {
            4
        }

        fn prefill(&mut self, token_ids: &[TokenId]) -> Result<Vec<f32>> {
            self.calls.lock().unwrap().prefills.push(token_ids.to_vec());
            Ok(vec![0.0, 1.0, 4.0, 3.0])
        }

        fn decode(&mut self, token_id: TokenId) -> Result<Vec<f32>> {
            self.calls.lock().unwrap().decodes.push(token_id);
            Ok(vec![0.0, 5.0, 2.0, 1.0])
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
        let mut worker = CpuWorker::from_forward(FakeForward {
            calls: calls.clone(),
        });

        let first = worker.execute_model(input(1, 10, &[1, 3])).unwrap();
        assert_eq!(first[0].token_id, 2);
        let second = worker.execute_model(input(1, 10, &[1, 3])).unwrap();
        assert_eq!(second[0].token_id, 1);
        worker.execute_model(input(2, 11, &[3, 2])).unwrap();

        let calls = calls.lock().unwrap();
        assert_eq!(calls.prefills, vec![vec![1, 3], vec![3, 2]]);
        assert_eq!(calls.decodes, vec![2]);
    }

    #[test]
    fn rejects_batching_and_best_of() {
        let calls = Arc::new(Mutex::new(Calls::default()));
        let mut worker = CpuWorker::from_forward(FakeForward { calls });
        let mut batched = input(1, 10, &[1]);
        batched
            .seq_group_metadata
            .push(input(2, 11, &[2]).seq_group_metadata.pop().unwrap());
        assert!(worker.execute_model(batched).is_err());

        let mut best_of = input(1, 10, &[1]);
        best_of.seq_group_metadata[0].sampling_params.best_of = 2;
        assert!(worker.execute_model(best_of).is_err());
    }
}
