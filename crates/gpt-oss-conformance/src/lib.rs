#![forbid(unsafe_code)]

mod case;
mod harness;
mod report;
mod trace;

pub use case::{ConformanceCase, MoeCaseMetadata};
pub use harness::{
    ConformanceBackend, ConformanceError, ConformanceHarness, PlaceholderMoeBackend,
    SemanticMoeBackend,
};
pub use report::{
    ComparisonReport, MoeComparison, MoeTokenComparison, ParityOutcome, RunComparison,
};
pub use trace::{ExecutionSample, MoeTokenSummary, MoeTraceSummary, TraceEvent, TraceSummary};

#[cfg(test)]
mod tests {
    use gpt_oss_moe_semantics::{ExpertStorageKind, MoeContract};

    use crate::{ConformanceCase, ConformanceHarness, PlaceholderMoeBackend, SemanticMoeBackend};

    fn case() -> ConformanceCase {
        ConformanceCase::moe(
            "moe-router-topk-normalization",
            MoeContract {
                num_local_experts: 4,
                num_experts_per_tok: 2,
                renormalize: true,
                storage: ExpertStorageKind::Mxfp4BlocksAndScales,
            },
            vec![vec![0.1, 1.4, 1.4, -0.2], vec![0.7, 0.2, 1.3, 0.0]],
        )
    }

    #[test]
    fn semantic_backend_matches_itself() {
        let harness = ConformanceHarness::new(SemanticMoeBackend, SemanticMoeBackend);
        let report = harness.compare(&case()).expect("comparison should succeed");
        assert!(report.overall.is_match());
        assert!(report
            .comparisons
            .iter()
            .all(|comparison| comparison.outcome.is_match()));
        let moe = report.comparisons[0].moe.as_ref().expect("moe report");
        assert!(moe.contract.is_match());
        assert!(moe
            .tokens
            .iter()
            .all(|token| token.selected_experts.is_match()));
    }

    #[test]
    fn placeholder_backend_diverges_and_is_reported() {
        let harness = ConformanceHarness::new(SemanticMoeBackend, PlaceholderMoeBackend);
        let report = harness.compare(&case()).expect("comparison should succeed");
        assert!(!report.overall.is_match());
        let comparison = &report.comparisons[0];
        assert!(!comparison.outcome.is_match());
        let moe = comparison.moe.as_ref().expect("moe report");
        assert!(moe.tokens.iter().any(
            |token| !token.selected_experts.is_match() || !token.normalized_weights.is_match()
        ));
    }
}
