use crate::trace::TraceSummary;
use gpt_oss_core::types::Dtype;
use gpt_oss_reference::{
    ReferenceExecutor, ReferenceExecutorConfig, ReferenceInput, ReferenceOutput,
};
use gpt_oss_runtime_plan::{plan_request, ExecutionPlan, PlanRequest, RuntimeMode};
use serde::{Deserialize, Serialize};

pub trait ConformanceBackend {
    fn name(&self) -> &str;
    fn run(&self, case: &ConformanceCase) -> ExecutionSample;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceCase {
    pub name: String,
    pub inputs: Vec<u32>,
    pub is_prefill: bool,
}

impl ConformanceCase {
    pub fn synthetic(name: impl Into<String>, inputs: Vec<u32>) -> Self {
        Self {
            name: name.into(),
            inputs,
            is_prefill: false,
        }
    }

    pub fn prefill(name: impl Into<String>, inputs: Vec<u32>) -> Self {
        Self {
            name: name.into(),
            inputs,
            is_prefill: true,
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
        let reference = ReferenceExecutor::new(config.reference);
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
