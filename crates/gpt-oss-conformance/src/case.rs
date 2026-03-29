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

pub trait ConformanceBackend {
    fn name(&self) -> &str;
    fn run(&self, case: &ConformanceCase) -> ExecutionSample;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceCase {
    pub name: String,
    pub inputs: Vec<u32>,
    pub is_prefill: bool,
    pub seq_start_pos: u32,
}

impl ConformanceCase {
    pub fn synthetic(name: impl Into<String>, inputs: Vec<u32>) -> Self {
        Self {
            name: name.into(),
            inputs,
            is_prefill: false,
            seq_start_pos: 0,
        }
    }

    pub fn prefill(name: impl Into<String>, inputs: Vec<u32>) -> Self {
        Self {
            name: name.into(),
            inputs,
            is_prefill: true,
            seq_start_pos: 0,
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
        }
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
        let tokens = case
            .inputs
            .iter()
            .map(|value| value % 11)
            .collect::<Vec<_>>();
        let trace = TraceSummary::synthetic(
            format!("{}{}", case.name, self.trace_suffix),
            case.inputs.clone(),
        );

        ExecutionSample {
            logits,
            tokens,
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
        let sampler = Sampler::new();
        let mut rng = StdRng::seed_from_u64(observed.seed);
        let sample = sampler
            .sample(
                &observed.logits,
                observed.logits.len(),
                &observed.sampling_params,
                &observed.past_tokens,
                &mut rng,
            )
            .expect("sampled logits backend should produce a deterministic sample");

        ExecutionSample {
            logits: observed.logits.clone(),
            tokens: vec![sample.token_id],
            trace: observed.trace.clone(),
            plan: observed.plan.clone(),
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

        let tokens = case
            .inputs
            .iter()
            .map(|value| value % 11)
            .collect::<Vec<_>>();
        let trace = TraceSummary::from_reference(
            format!("{}:{}", self.name, case.name),
            &plan,
            &reference_output.trace,
        );

        ExecutionSample {
            logits: reference_output.logits,
            tokens,
            trace,
            plan: Some(plan),
        }
    }
}
