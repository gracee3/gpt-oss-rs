use crate::{
    case::{ConformanceBackend, ConformanceCase, ExecutionSample},
    report::{
        compare_prefill_decode_continuity, compare_samples, ComparisonReport, ContinuityReport,
        ParityOutcome, RunComparison,
    },
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessConfig {
    pub strict_trace_matching: bool,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            strict_trace_matching: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ConformanceHarness {
    pub config: HarnessConfig,
}

impl ConformanceHarness {
    pub fn compare(
        &self,
        case: &ConformanceCase,
        expected_backend: &impl ConformanceBackend,
        observed_backend: &impl ConformanceBackend,
    ) -> ComparisonReport {
        let expected = expected_backend.run(case);
        let observed = observed_backend.run(case);
        let comparison = compare_samples(&expected, &observed);
        let outcome = if comparison.is_exact() {
            ParityOutcome::Match
        } else {
            ParityOutcome::Mismatch
        };

        ComparisonReport {
            outcome,
            comparison,
            expected,
            observed,
        }
    }

    pub fn compare_samples(
        &self,
        expected: &ExecutionSample,
        observed: &ExecutionSample,
    ) -> RunComparison {
        let comparison = compare_samples(expected, observed);
        if self.config.strict_trace_matching && !comparison.is_exact() {
            return comparison;
        }
        comparison
    }

    pub fn compare_prefill_decode_continuity(
        &self,
        prefill_case: &ConformanceCase,
        decode_case: &ConformanceCase,
        backend: &impl ConformanceBackend,
    ) -> ContinuityReport {
        let prefill = backend.run(prefill_case);
        let decode = backend.run(decode_case);
        let comparison = compare_prefill_decode_continuity(
            &prefill,
            &decode,
            prefill_case.seq_start_pos + prefill_case.inputs.len() as u32,
        );
        let outcome = if comparison.is_exact() {
            ParityOutcome::Match
        } else {
            ParityOutcome::Mismatch
        };

        ContinuityReport { outcome, comparison }
    }
}
