mod case;
mod harness;
mod report;
mod trace;

pub use case::{
    ConformanceBackend, ConformanceCase, ExecutionSample, PlaceholderBackend,
    PlannedReferenceBackend, PlannedReferenceBackendConfig,
};
pub use harness::{ConformanceHarness, HarnessConfig};
pub use report::{ComparisonReport, ParityOutcome, RunComparison};
pub use trace::{TraceEvent, TraceFrame, TraceSummary};

#[cfg(test)]
mod tests {
    use super::*;
    use gpt_oss_core::types::Dtype;
    use gpt_oss_runtime_plan::RuntimeMode;

    fn planned_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference",
            PlannedReferenceBackendConfig {
                runtime_mode: RuntimeMode::Trusted,
                model_name: "openai/gpt-oss-20b".to_string(),
                greedy_only: true,
                graph_enabled: true,
                graph_max_batch_size: 32,
                graph_padded_batch_size: Some(8),
                dtype: Dtype::Float16,
                reference: gpt_oss_reference::ReferenceExecutorConfig {
                    vocab_size: 8,
                    num_layers: 2,
                    block_size: 16,
                    layer_types: vec!["full_attention".into(), "sliding_attention".into()],
                    sliding_window: Some(4),
                    sink_tokens: 1,
                    num_local_experts: 4,
                    num_experts_per_tok: 2,
                    moe_layer_indices: vec![1],
                },
            },
        )
    }

    #[test]
    fn planned_reference_runs_compare_deterministically() {
        let case = ConformanceCase::prefill("single_block_decode", vec![1, 2, 3, 4]);
        let harness = ConformanceHarness::default();
        let backend = planned_backend();
        let report = harness.compare(&case, &backend, &backend);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
        assert_eq!(report.expected.plan, report.observed.plan);
        assert!(report.expected.plan.is_some());
        assert_eq!(
            report.expected.plan.as_ref().unwrap().backend_path,
            gpt_oss_runtime_plan::BackendPath::CudaEager
        );
    }

    #[test]
    fn placeholder_and_planned_reference_mixed_path_surface_diffs() {
        let case = ConformanceCase::synthetic("single_block_decode", vec![9, 8, 7]);
        let harness = ConformanceHarness::default();
        let planned = planned_backend();
        let placeholder = PlaceholderBackend::new("placeholder");
        let report = harness.compare(&case, &placeholder, &planned);

        assert_eq!(report.outcome, ParityOutcome::Mismatch);
        assert!(report.comparison.diff_count() > 0);
        assert!(report.expected.plan.is_none());
        assert!(report.observed.plan.is_some());
    }
}
