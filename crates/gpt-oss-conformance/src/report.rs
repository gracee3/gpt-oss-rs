use serde::{Deserialize, Serialize};

use crate::trace::{ExecutionSample, MoeTraceSummary, TraceSummary};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParityOutcome {
    Match,
    Mismatch {
        field: String,
        left: String,
        right: String,
    },
}

impl ParityOutcome {
    pub fn is_match(&self) -> bool {
        matches!(self, Self::Match)
    }

    pub fn mismatch(
        field: impl Into<String>,
        left: impl Into<String>,
        right: impl Into<String>,
    ) -> Self {
        Self::Mismatch {
            field: field.into(),
            left: left.into(),
            right: right.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeTokenComparison {
    pub token_index: usize,
    pub router_logits: ParityOutcome,
    pub selected_experts: ParityOutcome,
    pub normalized_weights: ParityOutcome,
    pub weight_sum: ParityOutcome,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeComparison {
    pub contract: ParityOutcome,
    pub storage: ParityOutcome,
    pub tokens: Vec<MoeTokenComparison>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunComparison {
    pub case_name: String,
    pub expected_backend: String,
    pub actual_backend: String,
    pub outcome: ParityOutcome,
    pub moe: Option<MoeComparison>,
    pub expected: TraceSummary,
    pub actual: TraceSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub comparisons: Vec<RunComparison>,
    pub overall: ParityOutcome,
}

impl ComparisonReport {
    pub fn from_single(comparison: RunComparison) -> Self {
        let overall = if comparison.outcome.is_match()
            && comparison
                .moe
                .as_ref()
                .map(|moe| {
                    moe.contract.is_match()
                        && moe.storage.is_match()
                        && moe.tokens.iter().all(|token| {
                            token.router_logits.is_match()
                                && token.selected_experts.is_match()
                                && token.normalized_weights.is_match()
                                && token.weight_sum.is_match()
                        })
                })
                .unwrap_or(true)
        {
            ParityOutcome::Match
        } else {
            ParityOutcome::mismatch("overall", "match", "mismatch")
        };

        Self {
            comparisons: vec![comparison],
            overall,
        }
    }
}

pub fn compare_execution_samples(
    expected: &ExecutionSample,
    actual: &ExecutionSample,
) -> RunComparison {
    let trace = compare_trace_summaries(&expected.trace, &actual.trace);
    let moe = compare_moe_summaries(expected.trace.moe.as_ref(), actual.trace.moe.as_ref());

    RunComparison {
        case_name: expected.case_name.clone(),
        expected_backend: expected.backend.clone(),
        actual_backend: actual.backend.clone(),
        outcome: trace,
        moe,
        expected: expected.trace.clone(),
        actual: actual.trace.clone(),
    }
}

fn compare_trace_summaries(expected: &TraceSummary, actual: &TraceSummary) -> ParityOutcome {
    if expected.events == actual.events && expected.moe == actual.moe {
        ParityOutcome::Match
    } else {
        ParityOutcome::mismatch("trace", format!("{expected:?}"), format!("{actual:?}"))
    }
}

fn compare_moe_summaries(
    expected: Option<&MoeTraceSummary>,
    actual: Option<&MoeTraceSummary>,
) -> Option<MoeComparison> {
    match (expected, actual) {
        (None, None) => None,
        (Some(expected), Some(actual)) => Some(MoeComparison {
            contract: compare_values("moe.contract", &expected.contract, &actual.contract),
            storage: compare_values("moe.storage", &expected.storage, &actual.storage),
            tokens: compare_tokens(expected, actual),
        }),
        (Some(expected), None) => Some(MoeComparison {
            contract: ParityOutcome::mismatch(
                "moe.contract",
                format!("{:?}", expected.contract),
                "missing",
            ),
            storage: ParityOutcome::mismatch(
                "moe.storage",
                format!("{:?}", expected.storage),
                "missing",
            ),
            tokens: expected
                .tokens
                .iter()
                .map(|token| missing_token_expected(token))
                .collect(),
        }),
        (None, Some(actual)) => Some(MoeComparison {
            contract: ParityOutcome::mismatch(
                "moe.contract",
                "missing",
                format!("{:?}", actual.contract),
            ),
            storage: ParityOutcome::mismatch(
                "moe.storage",
                "missing",
                format!("{:?}", actual.storage),
            ),
            tokens: actual.tokens.iter().map(missing_token_actual).collect(),
        }),
    }
}

fn compare_tokens(expected: &MoeTraceSummary, actual: &MoeTraceSummary) -> Vec<MoeTokenComparison> {
    let mut tokens: Vec<MoeTokenComparison> = expected
        .tokens
        .iter()
        .zip(actual.tokens.iter())
        .map(|(left, right)| MoeTokenComparison {
            token_index: left.token_index,
            router_logits: compare_float_vec(
                format!("moe.tokens[{}].router_logits", left.token_index),
                &left.router_logits,
                &right.router_logits,
            ),
            selected_experts: compare_values(
                format!("moe.tokens[{}].selected_experts", left.token_index),
                &left.selected_experts,
                &right.selected_experts,
            ),
            normalized_weights: compare_float_vec(
                format!("moe.tokens[{}].normalized_weights", left.token_index),
                &left.normalized_weights,
                &right.normalized_weights,
            ),
            weight_sum: compare_float(
                format!("moe.tokens[{}].weight_sum", left.token_index),
                left.weight_sum,
                right.weight_sum,
            ),
        })
        .collect();

    if expected.tokens.len() != actual.tokens.len() {
        let token_index = expected.tokens.len().min(actual.tokens.len());
        tokens.push(MoeTokenComparison {
            token_index,
            router_logits: ParityOutcome::mismatch(
                format!("moe.tokens[{token_index}].router_logits"),
                expected.tokens.len().to_string(),
                actual.tokens.len().to_string(),
            ),
            selected_experts: ParityOutcome::mismatch(
                format!("moe.tokens[{token_index}].selected_experts"),
                expected.tokens.len().to_string(),
                actual.tokens.len().to_string(),
            ),
            normalized_weights: ParityOutcome::mismatch(
                format!("moe.tokens[{token_index}].normalized_weights"),
                expected.tokens.len().to_string(),
                actual.tokens.len().to_string(),
            ),
            weight_sum: ParityOutcome::mismatch(
                format!("moe.tokens[{token_index}].weight_sum"),
                expected.tokens.len().to_string(),
                actual.tokens.len().to_string(),
            ),
        });
    }

    tokens
}

fn missing_token_expected(token: &crate::trace::MoeTokenSummary) -> MoeTokenComparison {
    MoeTokenComparison {
        token_index: token.token_index,
        router_logits: ParityOutcome::mismatch(
            format!("moe.tokens[{}].router_logits", token.token_index),
            format!("{:?}", token.router_logits),
            "missing",
        ),
        selected_experts: ParityOutcome::mismatch(
            format!("moe.tokens[{}].selected_experts", token.token_index),
            format!("{:?}", token.selected_experts),
            "missing",
        ),
        normalized_weights: ParityOutcome::mismatch(
            format!("moe.tokens[{}].normalized_weights", token.token_index),
            format!("{:?}", token.normalized_weights),
            "missing",
        ),
        weight_sum: ParityOutcome::mismatch(
            format!("moe.tokens[{}].weight_sum", token.token_index),
            token.weight_sum.to_string(),
            "missing",
        ),
    }
}

fn missing_token_actual(token: &crate::trace::MoeTokenSummary) -> MoeTokenComparison {
    MoeTokenComparison {
        token_index: token.token_index,
        router_logits: ParityOutcome::mismatch(
            format!("moe.tokens[{}].router_logits", token.token_index),
            "missing",
            format!("{:?}", token.router_logits),
        ),
        selected_experts: ParityOutcome::mismatch(
            format!("moe.tokens[{}].selected_experts", token.token_index),
            "missing",
            format!("{:?}", token.selected_experts),
        ),
        normalized_weights: ParityOutcome::mismatch(
            format!("moe.tokens[{}].normalized_weights", token.token_index),
            "missing",
            format!("{:?}", token.normalized_weights),
        ),
        weight_sum: ParityOutcome::mismatch(
            format!("moe.tokens[{}].weight_sum", token.token_index),
            "missing",
            token.weight_sum.to_string(),
        ),
    }
}

fn compare_values<T: PartialEq + std::fmt::Debug>(
    field: impl Into<String>,
    left: &T,
    right: &T,
) -> ParityOutcome {
    if left == right {
        ParityOutcome::Match
    } else {
        ParityOutcome::mismatch(field, format!("{left:?}"), format!("{right:?}"))
    }
}

fn compare_float_vec(field: impl Into<String>, left: &[f32], right: &[f32]) -> ParityOutcome {
    let field = field.into();
    if left.len() != right.len() {
        return ParityOutcome::mismatch(field, format!("{left:?}"), format!("{right:?}"));
    }

    if left
        .iter()
        .zip(right.iter())
        .all(|(l, r)| compare_float_value(*l, *r))
    {
        ParityOutcome::Match
    } else {
        ParityOutcome::mismatch(field, format!("{left:?}"), format!("{right:?}"))
    }
}

fn compare_float(field: impl Into<String>, left: f32, right: f32) -> ParityOutcome {
    let field = field.into();
    if compare_float_value(left, right) {
        ParityOutcome::Match
    } else {
        ParityOutcome::mismatch(field, left.to_string(), right.to_string())
    }
}

fn compare_float_value(left: f32, right: f32) -> bool {
    (left - right).abs() <= 1e-6
}
