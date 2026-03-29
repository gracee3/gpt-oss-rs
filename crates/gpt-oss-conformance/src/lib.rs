mod case;
mod harness;
mod report;
mod trace;

pub use case::{
    ConformanceBackend, ConformanceCase, ExecutionSample, ObservedLogitsCase, PlaceholderBackend,
    PlannedReferenceBackend, PlannedReferenceBackendConfig, SampledLogitsBackend,
};
pub use harness::{ConformanceHarness, HarnessConfig};
pub use report::{compare_prefill_decode_continuity, ComparisonReport, ContinuityReport, ParityOutcome, RunComparison};
pub use trace::{TraceEvent, TraceFrame, TraceSummary};

#[cfg(test)]
mod tests {
    use super::*;
    use gpt_oss_core::{prelude::SamplingParams, types::Dtype};
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

    #[test]
    fn decode_case_surfaces_phase_boundary_trace_details() {
        let case = ConformanceCase::decode("decode_step", 4, vec![9]);
        let harness = ConformanceHarness::default();
        let backend = planned_backend();
        let report = harness.compare(&case, &backend, &backend);

        assert_eq!(report.outcome, ParityOutcome::Match);
        let plan_frame = &report.expected.trace.frames[0];
        assert!(plan_frame
            .events
            .iter()
            .any(|event| event.stage == "reference_phase" && event.payload == "Decode"));
        assert!(plan_frame
            .events
            .iter()
            .any(|event| event.stage == "seq_start_pos" && event.payload == "4"));
        assert!(report.expected.trace.frames.iter().skip(1).any(|frame| {
            frame.events.iter().any(|event| {
                event.stage == "layer" && event.payload.contains("positions=[4]")
            })
        }));
    }

    #[test]
    fn planned_reference_reports_prefill_decode_continuity() {
        let prefill = ConformanceCase::prefill("prefill_step", vec![1, 2, 3, 4]);
        let decode = ConformanceCase::decode("decode_step", 4, vec![9]);
        let harness = ConformanceHarness::default();
        let backend = planned_backend();

        let report = harness.compare_prefill_decode_continuity(&prefill, &decode, &backend);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn trace_diffs_call_out_phase_boundary_fields() {
        let backend = planned_backend();
        let harness = ConformanceHarness::default();
        let expected = backend.run(&ConformanceCase::decode("decode_step", 4, vec![9]));
        let observed = backend.run(&ConformanceCase::decode("decode_step", 5, vec![9]));

        let comparison = harness.compare_samples(&expected, &observed);

        assert!(!comparison.is_exact());
        assert!(comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("seq_start_pos differs")));
    }

    #[test]
    fn sampled_logits_backend_produces_deterministic_observed_sample() {
        let case = ConformanceCase::synthetic("observed-logits", vec![1, 2, 3]);
        let backend = SampledLogitsBackend::new("observed").with_case(
            case.name.clone(),
            ObservedLogitsCase {
                logits: vec![0.1, 3.0, 0.2],
                sampling_params: SamplingParams {
                    temperature: 0.0,
                    ..Default::default()
                },
                past_tokens: vec![1, 2],
                seed: 7,
                trace: TraceSummary::synthetic("observed-logits", vec![1, 2, 3]),
                plan: None,
            },
        );

        let first = backend.run(&case);
        let second = backend.run(&case);

        assert_eq!(first.tokens, vec![1]);
        assert_eq!(first.tokens, second.tokens);
        assert_eq!(first.logits, second.logits);
    }
}
