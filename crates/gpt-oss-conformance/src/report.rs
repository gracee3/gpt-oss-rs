use crate::{case::ExecutionSample, trace::TraceSummary};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParityOutcome {
    Match,
    Mismatch,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunComparison {
    pub diffs: Vec<String>,
}

impl RunComparison {
    pub fn exact() -> Self {
        Self { diffs: Vec::new() }
    }

    pub fn diff_count(&self) -> usize {
        self.diffs.len()
    }

    pub fn is_exact(&self) -> bool {
        self.diffs.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub outcome: ParityOutcome,
    pub comparison: RunComparison,
    pub expected: ExecutionSample,
    pub observed: ExecutionSample,
}

impl fmt::Display for ComparisonReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "outcome={:?} diff_count={} expected_frames={} observed_frames={}",
            self.outcome,
            self.comparison.diff_count(),
            self.expected.trace.frames.len(),
            self.observed.trace.frames.len()
        )
    }
}

pub fn compare_samples(expected: &ExecutionSample, observed: &ExecutionSample) -> RunComparison {
    let mut diffs = Vec::new();

    if expected.logits != observed.logits {
        diffs.push("logits differ".to_string());
    }
    if expected.tokens != observed.tokens {
        diffs.push("tokens differ".to_string());
    }
    if expected.trace != observed.trace {
        diffs.push(trace_diff(&expected.trace, &observed.trace));
    }

    RunComparison { diffs }
}

fn trace_diff(expected: &TraceSummary, observed: &TraceSummary) -> String {
    format!(
        "trace mismatch: expected_label={} observed_label={} expected_frames={} observed_frames={}",
        expected.label,
        observed.label,
        expected.frames.len(),
        observed.frames.len()
    )
}
