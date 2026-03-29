use serde::{Deserialize, Serialize};
use thiserror::Error;

use gpt_oss_moe_semantics::MoeSemanticsError;

use crate::case::{ConformanceCase, MoeCaseMetadata};
use crate::report::{compare_execution_samples, ComparisonReport};
use crate::trace::{ExecutionSample, TraceSummary};

#[derive(Debug, Error)]
pub enum ConformanceError {
    #[error("{0}")]
    MissingMoeCase(String),
    #[error("{0}")]
    Backend(String),
    #[error(transparent)]
    Moe(#[from] MoeSemanticsError),
}

pub trait ConformanceBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn run(&self, case: &ConformanceCase) -> Result<ExecutionSample, ConformanceError>;
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SemanticMoeBackend;

impl ConformanceBackend for SemanticMoeBackend {
    fn name(&self) -> &'static str {
        "semantic-moe"
    }

    fn run(&self, case: &ConformanceCase) -> Result<ExecutionSample, ConformanceError> {
        let moe = case
            .moe
            .as_ref()
            .ok_or_else(|| ConformanceError::MissingMoeCase(case.name.clone()))?;
        let trace = run_semantic_case(moe)?;
        Ok(ExecutionSample {
            backend: self.name().to_string(),
            case_name: case.name.clone(),
            trace,
        })
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PlaceholderMoeBackend;

impl ConformanceBackend for PlaceholderMoeBackend {
    fn name(&self) -> &'static str {
        "placeholder-moe"
    }

    fn run(&self, case: &ConformanceCase) -> Result<ExecutionSample, ConformanceError> {
        let moe = case
            .moe
            .as_ref()
            .ok_or_else(|| ConformanceError::MissingMoeCase(case.name.clone()))?;
        let trace = run_placeholder_case(moe)?;
        Ok(ExecutionSample {
            backend: self.name().to_string(),
            case_name: case.name.clone(),
            trace,
        })
    }
}

pub struct ConformanceHarness<Expected, Actual> {
    expected: Expected,
    actual: Actual,
}

impl<Expected, Actual> ConformanceHarness<Expected, Actual> {
    pub fn new(expected: Expected, actual: Actual) -> Self {
        Self { expected, actual }
    }
}

impl<Expected, Actual> ConformanceHarness<Expected, Actual>
where
    Expected: ConformanceBackend,
    Actual: ConformanceBackend,
{
    pub fn compare(&self, case: &ConformanceCase) -> Result<ComparisonReport, ConformanceError> {
        let expected = self.expected.run(case)?;
        let actual = self.actual.run(case)?;
        let comparison = compare_execution_samples(&expected, &actual);
        Ok(ComparisonReport::from_single(comparison))
    }
}

fn run_semantic_case(moe: &MoeCaseMetadata) -> Result<TraceSummary, ConformanceError> {
    let trace = moe.contract.evaluate(&moe.router_logits)?;
    Ok(TraceSummary::from_moe_trace(&trace))
}

fn run_placeholder_case(moe: &MoeCaseMetadata) -> Result<TraceSummary, ConformanceError> {
    let mut trace = moe.contract.evaluate(&moe.router_logits)?;
    for token in &mut trace.tokens {
        token.selected_experts.reverse();
        token.normalized_weights.reverse();
        token.routed_experts.reverse();
        for (rank, routed) in token.routed_experts.iter_mut().enumerate() {
            routed.rank = rank;
        }
    }
    Ok(TraceSummary::from_moe_trace(&trace))
}
