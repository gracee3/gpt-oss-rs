use crate::trace::TraceSummary;
use gpt_oss_core::{prelude::SamplingParams, types::Dtype};
use gpt_oss_model_runner::Sampler;
use gpt_oss_reference::{
    ReferenceExecutor, ReferenceExecutorConfig, ReferenceInput, ReferenceOutput,
};
use gpt_oss_runtime_plan::{plan_request, ExecutionPlan, PlanRequest, RuntimeMode};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use gpt_oss_model_runner::{
    bridge::AttentionMetadata as BridgeAttentionMetadata, ModelInput, ModelRunner,
};

pub trait ConformanceBackend {
    fn name(&self) -> &str;
    fn run(&self, case: &ConformanceCase) -> ExecutionSample;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConformanceCase {
    pub name: String,
    pub inputs: Vec<u32>,
    pub is_prefill: bool,
    pub seq_start_pos: u32,
    pub sampling_params: SamplingParams,
    pub past_tokens: Vec<u32>,
    pub seed: u64,
}

impl ConformanceCase {
    pub fn synthetic(name: impl Into<String>, inputs: Vec<u32>) -> Self {
        Self {
            name: name.into(),
            inputs,
            is_prefill: false,
            seq_start_pos: 0,
            sampling_params: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            past_tokens: Vec::new(),
            seed: 0,
        }
    }

    pub fn prefill(name: impl Into<String>, inputs: Vec<u32>) -> Self {
        Self {
            name: name.into(),
            inputs,
            is_prefill: true,
            seq_start_pos: 0,
            sampling_params: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            past_tokens: Vec::new(),
            seed: 0,
        }
    }

    pub fn decode(
        name: impl Into<String>,
        seq_start_pos: u32,
        inputs: Vec<u32>,
    ) -> Self {
        Self {
            name: name.into(),
            inputs,
            is_prefill: false,
            seq_start_pos,
            sampling_params: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            past_tokens: Vec::new(),
            seed: 0,
        }
    }

    pub fn with_sampling(
        mut self,
        sampling_params: SamplingParams,
        past_tokens: Vec<u32>,
        seed: u64,
    ) -> Self {
        self.sampling_params = sampling_params;
        self.past_tokens = past_tokens;
        self.seed = seed;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionSample {
    pub logits: Vec<f32>,
    pub tokens: Vec<u32>,
    pub trace: TraceSummary,
    pub plan: Option<ExecutionPlan>,
}

#[derive(Clone, Debug)]
pub struct ObservedLogitsCase {
    pub logits: Vec<f32>,
    pub sampling_params: SamplingParams,
    pub past_tokens: Vec<u32>,
    pub seed: u64,
    pub trace: TraceSummary,
    pub plan: Option<ExecutionPlan>,
}

#[derive(Clone, Debug, Default)]
pub struct SampledLogitsBackend {
    name: String,
    cases: HashMap<String, ObservedLogitsCase>,
}

impl SampledLogitsBackend {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            cases: HashMap::new(),
        }
    }

    pub fn with_case(mut self, name: impl Into<String>, case: ObservedLogitsCase) -> Self {
        self.cases.insert(name.into(), case);
        self
    }
}

#[derive(Clone, Debug)]
pub struct PlaceholderBackend {
    name: String,
    trace_suffix: String,
}

impl PlaceholderBackend {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            trace_suffix: String::new(),
        }
    }

    pub fn with_trace_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.trace_suffix = suffix.into();
        self
    }
}

impl ConformanceBackend for PlaceholderBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, case: &ConformanceCase) -> ExecutionSample {
        let logits = case
            .inputs
            .iter()
            .enumerate()
            .map(|(idx, value)| (*value as f32) + (idx as f32) * 0.25)
            .collect::<Vec<_>>();
        let trace = TraceSummary::synthetic(
            format!("{}{}", case.name, self.trace_suffix),
            case.inputs.clone(),
        );
        let sampled = sample_tokens_from_logits(
            &logits,
            &case.sampling_params,
            &case.past_tokens,
            case.seed,
        );

        ExecutionSample {
            logits,
            tokens: sampled,
            trace,
            plan: None,
        }
    }
}

impl ConformanceBackend for SampledLogitsBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, case: &ConformanceCase) -> ExecutionSample {
        let observed = self
            .cases
            .get(&case.name)
            .unwrap_or_else(|| panic!("missing observed logits case '{}'", case.name));

        ExecutionSample {
            logits: observed.logits.clone(),
            tokens: sample_tokens_from_logits(
                &observed.logits,
                &observed.sampling_params,
                &observed.past_tokens,
                observed.seed,
            ),
            trace: observed.trace.clone(),
            plan: observed.plan.clone(),
        }
    }
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct ModelRunnerGreedyBackend {
    name: String,
    runner: Arc<ModelRunner>,
    runtime_mode: RuntimeMode,
    model_name: String,
    block_size: usize,
    traced_num_local_experts: usize,
    traced_num_experts_per_tok: usize,
    traced_moe_layer_indices: Vec<usize>,
    graph_enabled: bool,
    graph_max_batch_size: usize,
    graph_padded_batch_size: Option<usize>,
}

#[cfg(test)]
impl ModelRunnerGreedyBackend {
    pub(crate) fn new(name: impl Into<String>, runner: Arc<ModelRunner>) -> Self {
        Self {
            name: name.into(),
            runner,
            runtime_mode: RuntimeMode::Trusted,
            model_name: "openai/gpt-oss-20b".into(),
            block_size: 16,
            traced_num_local_experts: 0,
            traced_num_experts_per_tok: 0,
            traced_moe_layer_indices: Vec::new(),
            graph_enabled: true,
            graph_max_batch_size: 32,
            graph_padded_batch_size: Some(8),
        }
    }

    pub(crate) fn with_traced_moe(
        mut self,
        num_local_experts: usize,
        num_experts_per_tok: usize,
        moe_layer_indices: Vec<usize>,
    ) -> Self {
        self.traced_num_local_experts = num_local_experts;
        self.traced_num_experts_per_tok = num_experts_per_tok;
        self.traced_moe_layer_indices = moe_layer_indices;
        self
    }

    pub(crate) fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    fn plan_for_case(&self, case: &ConformanceCase) -> ExecutionPlan {
        plan_request(&PlanRequest::new(
            self.runtime_mode,
            self.model_name.clone(),
            case.is_prefill,
            true,
            self.graph_enabled,
            self.graph_max_batch_size,
            self.graph_padded_batch_size,
            self.runner.config.dtype,
        ))
        .expect("model runner conformance backend should be plannable")
    }
}

#[cfg(test)]
impl ConformanceBackend for ModelRunnerGreedyBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, case: &ConformanceCase) -> ExecutionSample {
        let input = model_input_from_case(case);
        let plan = self.plan_for_case(case);
        let logits_batch = self
            .runner
            .execute_model(input)
            .expect("model runner backend should produce logits");
        let vocab_size = self.runner.config.vocab_size;
        let logits = logits_batch.data[logits_batch.data.len() - vocab_size..].to_vec();

        ExecutionSample {
            tokens: sample_tokens_from_logits(
                &logits,
                &case.sampling_params,
                &case.past_tokens,
                case.seed,
            ),
            logits,
            trace: TraceSummary::from_observed_case_with_plan(
                format!("{}:{}", self.name, case.name),
                &plan,
                case.is_prefill,
                case.seq_start_pos,
                case.inputs.len(),
                self.block_size,
                &self.runner.config.layer_types,
                self.traced_num_local_experts,
                self.traced_num_experts_per_tok,
                &self.traced_moe_layer_indices,
                self.runner.config.num_layers,
            ),
            plan: Some(plan),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlannedReferenceBackend {
    name: String,
    config: PlannedReferenceBackendConfig,
    reference: ReferenceExecutor,
}

#[derive(Clone, Debug)]
pub struct PlannedReferenceBackendConfig {
    pub runtime_mode: RuntimeMode,
    pub model_name: String,
    pub greedy_only: bool,
    pub graph_enabled: bool,
    pub graph_max_batch_size: usize,
    pub graph_padded_batch_size: Option<usize>,
    pub dtype: Dtype,
    pub reference: ReferenceExecutorConfig,
}

impl PlannedReferenceBackend {
    pub fn new(name: impl Into<String>, config: PlannedReferenceBackendConfig) -> Self {
        let reference = ReferenceExecutor::new(config.reference.clone());
        Self {
            name: name.into(),
            config,
            reference,
        }
    }
}

impl ConformanceBackend for PlannedReferenceBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, case: &ConformanceCase) -> ExecutionSample {
        let plan = plan_request(&PlanRequest::new(
            self.config.runtime_mode,
            self.config.model_name.clone(),
            case.is_prefill,
            self.config.greedy_only,
            self.config.graph_enabled,
            self.config.graph_max_batch_size,
            self.config.graph_padded_batch_size,
            self.config.dtype,
        ))
        .expect("planner should accept the configured reference backend");

        let reference_output: ReferenceOutput = self
            .reference
            .forward(ReferenceInput {
                tokens: case.inputs.clone(),
                phase: if case.is_prefill {
                    gpt_oss_reference::ReferencePhase::Prefill
                } else {
                    gpt_oss_reference::ReferencePhase::Decode
                },
                seq_start_pos: case.seq_start_pos,
            })
            .expect("reference executor should accept the configured case");

        let trace = TraceSummary::from_reference(
            format!("{}:{}", self.name, case.name),
            &plan,
            &reference_output.trace,
        );
        let tokens = sample_tokens_from_logits(
            &reference_output.logits,
            &case.sampling_params,
            &case.past_tokens,
            case.seed,
        );

        ExecutionSample {
            logits: reference_output.logits,
            tokens,
            trace,
            plan: Some(plan),
        }
    }
}

fn sample_tokens_from_logits(
    logits: &[f32],
    sampling_params: &SamplingParams,
    past_tokens: &[u32],
    seed: u64,
) -> Vec<u32> {
    let sampler = Sampler::new();
    let mut rng = StdRng::seed_from_u64(seed);
    let sample = sampler
        .sample(logits, logits.len(), sampling_params, past_tokens, &mut rng)
        .expect("conformance sampling should produce a deterministic sample");
    vec![sample.token_id]
}

#[cfg(test)]
fn model_input_from_case(case: &ConformanceCase) -> ModelInput {
    let token_count = case.inputs.len() as u32;
    let context_len = case.seq_start_pos + token_count;
    ModelInput {
        token_ids: case.inputs.clone(),
        position_ids: (0..case.inputs.len())
            .map(|idx| case.seq_start_pos + idx as u32)
            .collect(),
        attention_metadata: BridgeAttentionMetadata {
            slot_mapping: (0..case.inputs.len() as u32).collect(),
            context_lens: vec![context_len],
            block_tables: vec![vec![0]],
            query_lens: vec![if case.is_prefill { token_count } else { 1 }],
            max_context_len: context_len,
        },
        is_prefill: case.is_prefill,
    }
}
