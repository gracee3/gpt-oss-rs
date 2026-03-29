use crate::{
    case::{ConformanceCase, ExecutionSample, PlaceholderBackend},
    report::{compare_samples, ComparisonReport, ParityOutcome, RunComparison},
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
        expected_backend: &PlaceholderBackend,
        observed_backend: &PlaceholderBackend,
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
}
