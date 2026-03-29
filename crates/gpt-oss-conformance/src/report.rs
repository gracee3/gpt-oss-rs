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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContinuityReport {
    pub outcome: ParityOutcome,
    pub comparison: RunComparison,
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
    if expected.plan != observed.plan {
        diffs.push("plans differ".to_string());
    }
    if expected.trace != observed.trace {
        diffs.extend(trace_diffs(&expected.trace, &observed.trace));
    }

    RunComparison { diffs }
}

fn trace_diffs(expected: &TraceSummary, observed: &TraceSummary) -> Vec<String> {
    let mut diffs = Vec::new();

    if expected.label != observed.label {
        diffs.push(format!(
            "trace label differs: expected={} observed={}",
            expected.label, observed.label
        ));
    }

    for stage in ["reference_phase", "seq_start_pos"] {
        let expected_value = expected.find_event_payload(stage);
        let observed_value = observed.find_event_payload(stage);
        if expected_value != observed_value {
            diffs.push(format!(
                "{stage} differs: expected={} observed={}",
                expected_value.unwrap_or("missing"),
                observed_value.unwrap_or("missing")
            ));
        }
    }

    if expected.frames.len() != observed.frames.len() {
        diffs.push(format!(
            "trace frame count differs: expected={} observed={}",
            expected.frames.len(),
            observed.frames.len()
        ));
    }

    if diffs.is_empty() {
        diffs.push(format!(
            "trace mismatch: expected_label={} observed_label={} expected_frames={} observed_frames={}",
            expected.label,
            observed.label,
            expected.frames.len(),
            observed.frames.len()
        ));
    }

    diffs
}

pub fn compare_prefill_decode_continuity(
    prefill: &ExecutionSample,
    decode: &ExecutionSample,
) -> RunComparison {
    let mut diffs = Vec::new();

    match prefill.trace.find_event_payload("reference_phase") {
        Some("Prefill") => {}
        Some(other) => diffs.push(format!("prefill phase mismatch: expected Prefill got {other}")),
        None => diffs.push("prefill phase missing from trace".to_string()),
    }

    match decode.trace.find_event_payload("reference_phase") {
        Some("Decode") => {}
        Some(other) => diffs.push(format!("decode phase mismatch: expected Decode got {other}")),
        None => diffs.push("decode phase missing from trace".to_string()),
    }

    let expected_start = prefill.tokens.len().to_string();
    match decode.trace.find_event_payload("seq_start_pos") {
        Some(value) if value == expected_start => {}
        Some(value) => diffs.push(format!(
            "decode seq_start_pos mismatch: expected {expected_start} got {value}"
        )),
        None => diffs.push("decode seq_start_pos missing from trace".to_string()),
    }

    RunComparison { diffs }
}
