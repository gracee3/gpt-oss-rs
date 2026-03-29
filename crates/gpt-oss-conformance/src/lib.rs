mod case;
mod harness;
mod report;
mod trace;

pub use case::{ConformanceCase, ExecutionSample, PlaceholderBackend};
pub use harness::{ConformanceHarness, HarnessConfig};
pub use report::{ComparisonReport, ParityOutcome, RunComparison};
pub use trace::{TraceEvent, TraceFrame, TraceSummary};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_runs_compare_deterministically() {
        let case = ConformanceCase::synthetic("single_block_decode", vec![1, 2, 3, 4]);
        let harness = ConformanceHarness::default();
        let report = harness.compare(
            &case,
            &PlaceholderBackend::new("planner"),
            &PlaceholderBackend::new("reference"),
        );

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
        assert_eq!(report.expected.trace.frames.len(), 1);
        assert_eq!(report.observed.trace.frames.len(), 1);
        assert_eq!(report.expected.logits, report.observed.logits);
        assert_eq!(report.expected.tokens, report.observed.tokens);
    }

    #[test]
    fn placeholder_runs_detect_mismatch() {
        let case = ConformanceCase::synthetic("single_block_decode", vec![9, 8, 7]);
        let harness = ConformanceHarness::default();
        let report = harness.compare(
            &case,
            &PlaceholderBackend::new("planner"),
            &PlaceholderBackend::new("reference-mismatch").with_trace_suffix("-x"),
        );

        assert_eq!(report.outcome, ParityOutcome::Mismatch);
        assert!(report.comparison.diff_count() > 0);
        assert!(!report.comparison.is_exact());
    }
}
