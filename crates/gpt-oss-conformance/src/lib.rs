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
    use crate::case::ModelRunnerGreedyBackend;
    use gpt_oss_core::{prelude::SamplingParams, types::Dtype};
    use gpt_oss_model_runner::{
        bridge::{
            AttentionMetadata, CacheEngine as BridgeCacheEngine, MockAttentionBackend,
            MockGpuAllocator, ModelWeights as BridgeModelWeights, WeightTensor,
        },
        ModelInput, ModelRunner, ModelRunnerConfig,
    };
    use gpt_oss_runtime_plan::RuntimeMode;
    use half::f16;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn planned_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference",
            PlannedReferenceBackendConfig {
                runtime_mode: RuntimeMode::Experimental,
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
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![1],
                },
            },
        )
    }

    fn dense_baseline_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-dense",
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
                    num_layers: 1,
                    block_size: 16,
                    layer_types: vec!["full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 0,
                    num_experts_per_tok: 0,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![],
                },
            },
        )
    }

    fn artifact_aligned_sliding_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-sliding-128",
            PlannedReferenceBackendConfig {
                runtime_mode: RuntimeMode::Experimental,
                model_name: "openai/gpt-oss-20b".to_string(),
                greedy_only: true,
                graph_enabled: false,
                graph_max_batch_size: 32,
                graph_padded_batch_size: None,
                dtype: Dtype::Float16,
                reference: gpt_oss_reference::ReferenceExecutorConfig {
                    vocab_size: 8,
                    num_layers: 1,
                    block_size: 16,
                    layer_types: vec!["sliding_attention".into()],
                    sliding_window: Some(128),
                    sink_tokens: 0,
                    num_local_experts: 0,
                    num_experts_per_tok: 0,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![],
                },
            },
        )
    }

    fn nonzero_dense_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-dense-nonzero",
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
                    num_layers: 1,
                    block_size: 16,
                    layer_types: vec!["full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 0,
                    num_experts_per_tok: 0,
                    token_embedding_rows: vec![
                        vec![0.0, 0.0, 0.0, 0.0],
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                    ],
                    final_norm_weight: vec![1.0, 1.0, 1.0, 1.0],
                    rms_norm_eps: 1e-5,
                    lm_head_rows: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                        vec![0.5, 0.0, 0.0, 0.5],
                    ],
                    expert_output_rows: Vec::new(),
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![],
                },
            },
        )
    }

    fn full_attention_moe_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-moe",
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
                    num_layers: 1,
                    block_size: 16,
                    layer_types: vec!["full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 1,
                    num_experts_per_tok: 1,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![0],
                },
            },
        )
    }

    fn nonzero_single_expert_moe_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-moe-nonzero",
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
                    num_layers: 1,
                    block_size: 16,
                    layer_types: vec!["full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 1,
                    num_experts_per_tok: 1,
                    token_embedding_rows: vec![
                        vec![0.0, 0.0, 0.0, 0.0],
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                    ],
                    final_norm_weight: vec![1.0, 1.0, 1.0, 1.0],
                    rms_norm_eps: 1e-5,
                    lm_head_rows: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                        vec![0.5, 0.0, 0.0, 0.5],
                    ],
                    expert_output_rows: vec![vec![1.0, 0.0, 0.0, 0.0]],
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![0],
                },
            },
        )
    }

    fn full_attention_two_expert_moe_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-moe-top2",
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
                    num_layers: 1,
                    block_size: 16,
                    layer_types: vec!["full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 2,
                    num_experts_per_tok: 2,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![0],
                },
            },
        )
    }

    fn nonzero_two_expert_moe_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-moe-top2-nonzero",
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
                    num_layers: 1,
                    block_size: 16,
                    layer_types: vec!["full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 2,
                    num_experts_per_tok: 2,
                    token_embedding_rows: vec![
                        vec![0.0, 0.0, 0.0, 0.0],
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                    ],
                    final_norm_weight: vec![1.0, 1.0, 1.0, 1.0],
                    rms_norm_eps: 1e-5,
                    lm_head_rows: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                        vec![0.5, 0.0, 0.0, 0.5],
                    ],
                    expert_output_rows: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                    ],
                    router_bias: vec![0.0, 0.0],
                    moe_layer_indices: vec![0],
                },
            },
        )
    }

    fn full_attention_three_expert_top2_moe_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-moe-3e-top2",
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
                    num_layers: 1,
                    block_size: 16,
                    layer_types: vec!["full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 3,
                    num_experts_per_tok: 2,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![0],
                },
            },
        )
    }

    fn biased_full_attention_three_expert_top2_moe_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-moe-3e-top2-biased",
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
                    num_layers: 1,
                    block_size: 16,
                    layer_types: vec!["full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 3,
                    num_experts_per_tok: 2,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: vec![0.0, 1.0, 2.0],
                    moe_layer_indices: vec![0],
                },
            },
        )
    }

    fn full_attention_three_expert_top4_requested_moe_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-moe-3e-top4req",
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
                    num_layers: 1,
                    block_size: 16,
                    layer_types: vec!["full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 3,
                    num_experts_per_tok: 4,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![0],
                },
            },
        )
    }

    fn two_layer_full_attention_moe_on_second_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-2layer-moe-second",
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
                    layer_types: vec!["full_attention".into(), "full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 1,
                    num_experts_per_tok: 1,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![1],
                },
            },
        )
    }

    fn nonzero_two_layer_full_attention_moe_on_second_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-2layer-moe-second-nonzero",
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
                    layer_types: vec!["full_attention".into(), "full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 1,
                    num_experts_per_tok: 1,
                    token_embedding_rows: vec![
                        vec![0.0, 0.0, 0.0, 0.0],
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                    ],
                    final_norm_weight: vec![1.0, 1.0, 1.0, 1.0],
                    rms_norm_eps: 1e-5,
                    lm_head_rows: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                        vec![0.5, 0.0, 0.0, 0.5],
                    ],
                    expert_output_rows: vec![vec![1.0, 0.0, 0.0, 0.0]],
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![1],
                },
            },
        )
    }

    fn two_layer_full_attention_moe_both_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-2layer-moe-both",
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
                    layer_types: vec!["full_attention".into(), "full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 2,
                    num_experts_per_tok: 2,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![0, 1],
                },
            },
        )
    }

    fn nonzero_two_layer_full_attention_moe_both_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-2layer-moe-both-nonzero",
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
                    layer_types: vec!["full_attention".into(), "full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 2,
                    num_experts_per_tok: 2,
                    token_embedding_rows: vec![
                        vec![0.0, 0.0, 0.0, 0.0],
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                    ],
                    final_norm_weight: vec![1.0, 1.0, 1.0, 1.0],
                    rms_norm_eps: 1e-5,
                    lm_head_rows: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                        vec![0.5, 0.0, 0.0, 0.5],
                    ],
                    expert_output_rows: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                    ],
                    router_bias: vec![0.0, 0.0],
                    moe_layer_indices: vec![0, 1],
                },
            },
        )
    }

    fn biased_two_layer_full_attention_moe_both_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-2layer-moe-both-biased",
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
                    layer_types: vec!["full_attention".into(), "full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 3,
                    num_experts_per_tok: 2,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: vec![0.0, 1.0, 2.0],
                    moe_layer_indices: vec![0, 1],
                },
            },
        )
    }

    fn two_layer_full_attention_moe_both_multiblock_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-2layer-moe-both-multiblock",
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
                    block_size: 2,
                    layer_types: vec!["full_attention".into(), "full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 2,
                    num_experts_per_tok: 2,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![0, 1],
                },
            },
        )
    }

    fn biased_two_layer_full_attention_moe_both_multiblock_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-2layer-moe-both-biased-multiblock",
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
                    block_size: 2,
                    layer_types: vec!["full_attention".into(), "full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 3,
                    num_experts_per_tok: 2,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: vec![0.0, 1.0, 2.0],
                    moe_layer_indices: vec![0, 1],
                },
            },
        )
    }

    fn three_layer_full_attention_middle_moe_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-3layer-moe-middle",
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
                    num_layers: 3,
                    block_size: 16,
                    layer_types: vec![
                        "full_attention".into(),
                        "full_attention".into(),
                        "full_attention".into(),
                    ],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 2,
                    num_experts_per_tok: 2,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![1],
                },
            },
        )
    }

    fn nonzero_three_layer_full_attention_middle_moe_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-3layer-moe-middle-nonzero",
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
                    num_layers: 3,
                    block_size: 16,
                    layer_types: vec![
                        "full_attention".into(),
                        "full_attention".into(),
                        "full_attention".into(),
                    ],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 2,
                    num_experts_per_tok: 2,
                    token_embedding_rows: vec![
                        vec![0.0, 0.0, 0.0, 0.0],
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                    ],
                    final_norm_weight: vec![1.0, 1.0, 1.0, 1.0],
                    rms_norm_eps: 1e-5,
                    lm_head_rows: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                        vec![0.5, 0.0, 0.0, 0.5],
                    ],
                    expert_output_rows: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                    ],
                    router_bias: vec![0.0, 0.0],
                    moe_layer_indices: vec![1],
                },
            },
        )
    }

    fn nonzero_two_layer_full_attention_moe_both_multiblock_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-2layer-moe-both-multiblock-nonzero",
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
                    block_size: 2,
                    layer_types: vec!["full_attention".into(), "full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 2,
                    num_experts_per_tok: 2,
                    token_embedding_rows: vec![
                        vec![0.0, 0.0, 0.0, 0.0],
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                    ],
                    final_norm_weight: vec![1.0, 1.0, 1.0, 1.0],
                    rms_norm_eps: 1e-5,
                    lm_head_rows: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                        vec![0.5, 0.0, 0.0, 0.5],
                    ],
                    expert_output_rows: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                    ],
                    router_bias: vec![0.0, 0.0],
                    moe_layer_indices: vec![0, 1],
                },
            },
        )
    }

    fn nonzero_biased_three_expert_top2_moe_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-moe-3e-top2-nonzero-biased",
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
                    num_layers: 1,
                    block_size: 16,
                    layer_types: vec!["full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 3,
                    num_experts_per_tok: 2,
                    token_embedding_rows: vec![
                        vec![0.0, 0.0, 0.0, 0.0],
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                    ],
                    final_norm_weight: vec![1.0, 1.0, 1.0, 1.0],
                    rms_norm_eps: 1e-5,
                    lm_head_rows: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                        vec![0.0, 0.0, 0.0, 1.0],
                        vec![0.5, 0.5, 0.0, 0.0],
                        vec![0.0, 0.5, 0.5, 0.0],
                        vec![0.0, 0.0, 0.5, 0.5],
                        vec![0.5, 0.0, 0.0, 0.5],
                    ],
                    expert_output_rows: vec![
                        vec![1.0, 0.0, 0.0, 0.0],
                        vec![0.0, 1.0, 0.0, 0.0],
                        vec![0.0, 0.0, 1.0, 0.0],
                    ],
                    router_bias: vec![0.0, 1.0, 2.0],
                    moe_layer_indices: vec![0],
                },
            },
        )
    }

    fn multi_block_dense_backend() -> PlannedReferenceBackend {
        PlannedReferenceBackend::new(
            "planned-reference-dense-multiblock",
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
                    num_layers: 1,
                    block_size: 2,
                    layer_types: vec!["full_attention".into()],
                    sliding_window: None,
                    sink_tokens: 0,
                    num_local_experts: 0,
                    num_experts_per_tok: 0,
                    token_embedding_rows: Vec::new(),
                    final_norm_weight: Vec::new(),
                    rms_norm_eps: 1e-5,
                    lm_head_rows: Vec::new(),
                    expert_output_rows: Vec::new(),
                    router_bias: Vec::new(),
                    moe_layer_indices: vec![],
                },
            },
        )
    }

    fn tensor(name: &str, vals: &[f32], shape: &[usize]) -> (String, WeightTensor) {
        (
            name.to_string(),
            WeightTensor {
                name: name.to_string(),
                data: vals.iter().map(|&v| f16::from_f32(v)).collect(),
                shape: shape.to_vec(),
            },
        )
    }

    fn runner_config() -> ModelRunnerConfig {
        runner_config_with_moe(1, 1)
    }

    fn runner_config_with_moe(
        num_local_experts: usize,
        num_experts_per_tok: usize,
    ) -> ModelRunnerConfig {
        runner_config_with_layers(1, vec!["full_attention".into()], num_local_experts, num_experts_per_tok)
    }

    fn runner_config_with_layers(
        num_layers: usize,
        layer_types: Vec<String>,
        num_local_experts: usize,
        num_experts_per_tok: usize,
    ) -> ModelRunnerConfig {
        ModelRunnerConfig {
            tensor_parallel_rank: 0,
            tensor_parallel_size: 1,
            num_layers,
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            intermediate_size: 2,
            vocab_size: 8,
            max_position: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            partial_rotary_factor: 1.0,
            attn_logit_softcapping: 0.0,
            attention_bias: false,
            sliding_window: None,
            layer_types,
            num_local_experts,
            num_experts_per_tok,
            dtype: Dtype::Float16,
            architecture: "GptOssForCausalLM".into(),
        }
    }

    fn runner_weights() -> BridgeModelWeights {
        runner_weights_with_experts(1)
    }

    fn runner_weights_with_router_bias(
        num_local_experts: usize,
        router_bias: &[f32],
    ) -> BridgeModelWeights {
        let mut weights = runner_weights_with_experts(num_local_experts);
        weights.tensors.insert(
            "model.layers.0.mlp.router.bias".to_string(),
            WeightTensor {
                name: "model.layers.0.mlp.router.bias".to_string(),
                data: router_bias.iter().map(|&v| f16::from_f32(v)).collect(),
                shape: vec![router_bias.len()],
            },
        );
        weights
    }

    fn runner_weights_with_experts(num_local_experts: usize) -> BridgeModelWeights {
        runner_weights_with_layers(1, num_local_experts, &[0])
    }

    fn runner_weights_with_layers_and_router_bias(
        num_layers: usize,
        num_local_experts: usize,
        moe_layers: &[usize],
        router_bias_layers: &[(usize, &[f32])],
    ) -> BridgeModelWeights {
        let mut weights = runner_weights_with_layers(num_layers, num_local_experts, moe_layers);
        for (layer_index, router_bias) in router_bias_layers {
            let name = format!("model.layers.{layer_index}.mlp.router.bias");
            weights.tensors.insert(
                name.clone(),
                WeightTensor {
                    name,
                    data: router_bias.iter().map(|&v| f16::from_f32(v)).collect(),
                    shape: vec![router_bias.len()],
                },
            );
        }
        weights
    }

    fn runner_weights_with_layers(
        num_layers: usize,
        num_local_experts: usize,
        moe_layers: &[usize],
    ) -> BridgeModelWeights {
        let mut tensors = HashMap::new();
        tensors.extend([
            tensor("model.embed_tokens.weight", &[0.0; 32], &[8, 4]),
            tensor("model.norm.weight", &[1.0; 4], &[4]),
            tensor("lm_head.weight", &[0.0; 32], &[8, 4]),
        ]);

        for layer_index in 0..num_layers {
            let prefix = format!("model.layers.{layer_index}");
            tensors.extend([
                tensor(&format!("{prefix}.input_layernorm.weight"), &[1.0; 4], &[4]),
                tensor(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    &[1.0; 4],
                    &[4],
                ),
                tensor(&format!("{prefix}.self_attn.q_proj.weight"), &[0.0; 16], &[4, 4]),
                tensor(&format!("{prefix}.self_attn.k_proj.weight"), &[0.0; 16], &[4, 4]),
                tensor(&format!("{prefix}.self_attn.v_proj.weight"), &[0.0; 16], &[4, 4]),
                tensor(&format!("{prefix}.self_attn.o_proj.weight"), &[0.0; 16], &[4, 4]),
                tensor(&format!("{prefix}.self_attn.sinks"), &[0.0; 2], &[2]),
            ]);

            if moe_layers.contains(&layer_index) || num_local_experts > 0 {
                tensors.extend([
                    tensor(
                        &format!("{prefix}.mlp.router.weight"),
                        &vec![0.0; num_local_experts * 4],
                        &[num_local_experts, 4],
                    ),
                    tensor(
                        &format!("{prefix}.mlp.router.bias"),
                        &vec![0.0; num_local_experts],
                        &[num_local_experts],
                    ),
                    tensor(
                        &format!("{prefix}.mlp.experts.gate_up_proj"),
                        &vec![0.0; num_local_experts * 16],
                        &[num_local_experts, 4, 4],
                    ),
                    tensor(
                        &format!("{prefix}.mlp.experts.gate_up_proj_bias"),
                        &vec![0.0; num_local_experts * 4],
                        &[num_local_experts, 4],
                    ),
                    tensor(
                        &format!("{prefix}.mlp.experts.down_proj"),
                        &vec![0.0; num_local_experts * 8],
                        &[num_local_experts, 2, 4],
                    ),
                    tensor(
                        &format!("{prefix}.mlp.experts.down_proj_bias"),
                        &vec![0.0; num_local_experts * 4],
                        &[num_local_experts, 4],
                    ),
                ]);
            }
        }

        BridgeModelWeights { tensors }
    }

    fn runner_weights_with_nonzero_dense_signal() -> BridgeModelWeights {
        let mut weights = runner_weights();
        weights.tensors.insert(
            "model.embed_tokens.weight".to_string(),
            tensor(
                "model.embed_tokens.weight",
                &[
                    0.0, 0.0, 0.0, 0.0, // token 0
                    1.0, 0.0, 0.0, 0.0, // token 1
                    0.0, 1.0, 0.0, 0.0, // token 2
                    0.0, 0.0, 1.0, 0.0, // token 3
                    0.0, 0.0, 0.0, 1.0, // token 4
                    0.5, 0.5, 0.0, 0.0, // token 5
                    0.0, 0.5, 0.5, 0.0, // token 6
                    0.0, 0.0, 0.5, 0.5, // token 7
                ],
                &[8, 4],
            )
            .1,
        );
        weights.tensors.insert(
            "lm_head.weight".to_string(),
            tensor(
                "lm_head.weight",
                &[
                    1.0, 0.0, 0.0, 0.0, // vocab 0
                    0.0, 1.0, 0.0, 0.0, // vocab 1
                    0.0, 0.0, 1.0, 0.0, // vocab 2
                    0.0, 0.0, 0.0, 1.0, // vocab 3
                    0.5, 0.5, 0.0, 0.0, // vocab 4
                    0.0, 0.5, 0.5, 0.0, // vocab 5
                    0.0, 0.0, 0.5, 0.5, // vocab 6
                    0.5, 0.0, 0.0, 0.5, // vocab 7
                ],
                &[8, 4],
            )
            .1,
        );
        weights
    }

    fn runner_weights_with_nonzero_single_expert_output() -> BridgeModelWeights {
        let mut weights = runner_weights_with_nonzero_dense_signal();
        weights.tensors.insert(
            "model.layers.0.mlp.experts.down_proj_bias".to_string(),
            tensor(
                "model.layers.0.mlp.experts.down_proj_bias",
                &[1.0, 0.0, 0.0, 0.0],
                &[1, 4],
            )
            .1,
        );
        weights
    }

    fn runner_weights_with_nonzero_two_expert_output() -> BridgeModelWeights {
        let mut weights = runner_weights_with_layers(1, 2, &[0]);
        weights.tensors.insert(
            "model.embed_tokens.weight".to_string(),
            tensor(
                "model.embed_tokens.weight",
                &[
                    0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                    0.5, 0.5, 0.0, 0.0,
                    0.0, 0.5, 0.5, 0.0,
                    0.0, 0.0, 0.5, 0.5,
                ],
                &[8, 4],
            )
            .1,
        );
        weights.tensors.insert(
            "lm_head.weight".to_string(),
            tensor(
                "lm_head.weight",
                &[
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                    0.5, 0.5, 0.0, 0.0,
                    0.0, 0.5, 0.5, 0.0,
                    0.0, 0.0, 0.5, 0.5,
                    0.5, 0.0, 0.0, 0.5,
                ],
                &[8, 4],
            )
            .1,
        );
        weights.tensors.insert(
            "model.layers.0.mlp.experts.down_proj_bias".to_string(),
            tensor(
                "model.layers.0.mlp.experts.down_proj_bias",
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                &[2, 4],
            )
            .1,
        );
        weights
    }

    fn runner_weights_with_layer_expert_down_proj_bias(
        num_layers: usize,
        num_local_experts: usize,
        moe_layers: &[usize],
        layer_expert_biases: &[(usize, &[f32])],
    ) -> BridgeModelWeights {
        let mut weights = runner_weights_with_layers(num_layers, num_local_experts, moe_layers);
        weights.tensors.insert(
            "model.embed_tokens.weight".to_string(),
            tensor(
                "model.embed_tokens.weight",
                &[
                    0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                    0.5, 0.5, 0.0, 0.0,
                    0.0, 0.5, 0.5, 0.0,
                    0.0, 0.0, 0.5, 0.5,
                ],
                &[8, 4],
            )
            .1,
        );
        weights.tensors.insert(
            "lm_head.weight".to_string(),
            tensor(
                "lm_head.weight",
                &[
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0,
                    0.5, 0.5, 0.0, 0.0,
                    0.0, 0.5, 0.5, 0.0,
                    0.0, 0.0, 0.5, 0.5,
                    0.5, 0.0, 0.0, 0.5,
                ],
                &[8, 4],
            )
            .1,
        );
        for (layer_index, bias) in layer_expert_biases {
            let name = format!("model.layers.{layer_index}.mlp.experts.down_proj_bias");
            weights.tensors.insert(
                name.clone(),
                tensor(&name, bias, &[num_local_experts, 4]).1,
            );
        }
        weights
    }

    fn runner_input() -> ModelInput {
        ModelInput {
            token_ids: vec![1, 2],
            position_ids: vec![0, 1],
            attention_metadata: AttentionMetadata {
                slot_mapping: vec![0, 1],
                context_lens: vec![2],
                block_tables: vec![vec![0]],
                query_lens: vec![1],
                max_context_len: 2,
            },
            is_prefill: true,
        }
    }

    fn real_model_runner_logits_case(name: &str) -> ObservedLogitsCase {
        let runner = ModelRunner::new(
            runner_weights(),
            runner_config(),
            Box::new(MockAttentionBackend),
            Arc::new(BridgeCacheEngine::new(1, 64)),
            MockGpuAllocator::new(1 << 20),
        )
        .expect("test model runner");
        let output = runner.execute_model(runner_input()).expect("runner logits");
        let vocab_size = runner.config.vocab_size;
        let last_row_start = output.data.len() - vocab_size;
        let logits = output.data[last_row_start..].to_vec();

        ObservedLogitsCase {
            logits,
            sampling_params: SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            past_tokens: vec![1, 2],
            seed: 11,
            trace: TraceSummary::synthetic(format!("{name}:model-runner"), vec![1, 2]),
            plan: None,
        }
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
    fn artifact_aligned_sliding_decode_case_is_plannable_and_diagnosable() {
        let case = ConformanceCase::decode("sliding_decode_128", 128, vec![9]);
        let harness = ConformanceHarness::default();
        let backend = artifact_aligned_sliding_backend();
        let report = harness.compare(&case, &backend, &backend);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
        assert_eq!(
            report.expected.plan.as_ref().unwrap().backend_path,
            gpt_oss_runtime_plan::BackendPath::CudaEager
        );
        assert_eq!(
            report.expected.plan.as_ref().unwrap().runtime_mode,
            RuntimeMode::Experimental
        );
        let layer_frame = report
            .expected
            .trace
            .frames
            .iter()
            .find(|frame| frame.label.ends_with("layer-0"))
            .expect("sliding layer frame");
        assert!(layer_frame.events.iter().any(|event| {
            event.stage == "attention"
                && event.payload.contains("Sliding/128")
                && event.payload.contains("visible=[1, 2, 3")
                && event.payload.contains(", 128]")
        }));
    }

    #[test]
    fn artifact_aligned_sliding_prefill_decode_continuity_matches() {
        let prefill = ConformanceCase::prefill(
            "sliding_prefill_128",
            (0..128).map(|token| token as u32).collect(),
        );
        let decode = ConformanceCase::decode("sliding_decode_128", 128, vec![999]);
        let harness = ConformanceHarness::default();
        let backend = artifact_aligned_sliding_backend();

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

    #[test]
    fn sampled_logits_backend_accepts_real_model_runner_logits() {
        let case = ConformanceCase::synthetic("model-runner-logits", vec![1, 2]);
        let backend = SampledLogitsBackend::new("observed")
            .with_case(case.name.clone(), real_model_runner_logits_case(&case.name));

        let sample = backend.run(&case);

        assert_eq!(sample.logits.len(), 8);
        assert_eq!(sample.tokens.len(), 1);
    }

    #[test]
    fn nonzero_dense_full_attention_parity_matches() {
        let case = ConformanceCase::prefill("nonzero-dense-frontier", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_nonzero_dense_signal(),
                runner_config(),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner);
        let reference = nonzero_dense_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn model_runner_backend_is_deterministic() {
        let case = ConformanceCase::prefill("model-runner-logits", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights(),
                runner_config(),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let backend = ModelRunnerGreedyBackend::new("model-runner", runner);

        let first = backend.run(&case);
        let second = backend.run(&case);

        assert_eq!(first.logits, second.logits);
        assert_eq!(first.tokens, second.tokens);
    }

    #[test]
    fn dense_full_attention_parity_no_longer_differs_on_logits() {
        let case = ConformanceCase::prefill("dense-baseline", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights(),
                runner_config(),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner);
        let reference = dense_baseline_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("logits differ")));
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn dense_full_attention_decode_parity_no_longer_differs_on_logits() {
        let case = ConformanceCase::decode("dense-decode-baseline", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights(),
                runner_config(),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner);
        let reference = dense_baseline_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("logits differ")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("plans differ")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("trace frame count differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("frame 1 label differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event layer differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event attention differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event cache differs")));
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn dense_full_attention_prefill_parity_no_longer_differs_on_logits_or_plan() {
        let case = ConformanceCase::prefill("dense-baseline", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights(),
                runner_config(),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner);
        let reference = dense_baseline_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("logits differ")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("plans differ")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("trace frame count differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("frame 1 label differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event layer differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event attention differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event cache differs")));
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn single_expert_full_attention_moe_parity_no_longer_differs_on_logits() {
        let case = ConformanceCase::prefill("single-expert-moe", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights(),
                runner_config(),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            1,
            1,
            vec![0],
        );
        let reference = full_attention_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("logits differ")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("plans differ")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("trace frame count differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("frame 1 label differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event layer differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event attention differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event cache differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event moe differs")));
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn single_expert_full_attention_moe_decode_parity_no_longer_differs_on_logits() {
        let case = ConformanceCase::decode("single-expert-moe-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights(),
                runner_config(),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            1,
            1,
            vec![0],
        );
        let reference = full_attention_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("logits differ")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("plans differ")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("trace frame count differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("frame 1 label differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event layer differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event attention differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event cache differs")));
        assert!(!report
            .comparison
            .diffs
            .iter()
            .any(|diff| diff.contains("event moe differs")));
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn nonzero_single_expert_moe_output_parity_matches() {
        let case = ConformanceCase::prefill("single-expert-moe-nonzero", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_nonzero_single_expert_output(),
                runner_config(),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            1,
            1,
            vec![0],
        );
        let reference = nonzero_single_expert_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn nonzero_single_expert_moe_output_decode_parity_matches() {
        let case = ConformanceCase::decode("single-expert-moe-nonzero-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_nonzero_single_expert_output(),
                runner_config(),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            1,
            1,
            vec![0],
        );
        let reference = nonzero_single_expert_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn nonzero_two_layer_second_layer_moe_output_parity_matches() {
        let case = ConformanceCase::decode("two-layer-second-moe-nonzero-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layer_expert_down_proj_bias(2, 1, &[1], &[(1, &[1.0, 0.0, 0.0, 0.0])]),
                runner_config_with_layers(
                    2,
                    vec!["full_attention".into(), "full_attention".into()],
                    1,
                    1,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            1,
            1,
            vec![1],
        );
        let reference = nonzero_two_layer_full_attention_moe_on_second_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn nonzero_two_expert_moe_output_parity_matches() {
        let case = ConformanceCase::prefill("two-expert-moe-nonzero", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_nonzero_two_expert_output(),
                runner_config_with_moe(2, 2),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            2,
            2,
            vec![0],
        );
        let reference = nonzero_two_expert_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn nonzero_two_expert_moe_output_decode_parity_matches() {
        let case = ConformanceCase::decode("two-expert-moe-nonzero-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_nonzero_two_expert_output(),
                runner_config_with_moe(2, 2),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            2,
            2,
            vec![0],
        );
        let reference = nonzero_two_expert_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn nonzero_two_layer_both_layers_moe_output_decode_parity_matches() {
        let case = ConformanceCase::decode("two-layer-both-moe-nonzero-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layer_expert_down_proj_bias(
                    2,
                    2,
                    &[0, 1],
                    &[
                        (0, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                        (1, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                    ],
                ),
                runner_config_with_layers(
                    2,
                    vec!["full_attention".into(), "full_attention".into()],
                    2,
                    2,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            2,
            2,
            vec![0, 1],
        );
        let reference = nonzero_two_layer_full_attention_moe_both_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn nonzero_two_layer_both_layers_moe_output_prefill_parity_matches() {
        let case = ConformanceCase::prefill("two-layer-both-moe-nonzero-prefill", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layer_expert_down_proj_bias(
                    2,
                    2,
                    &[0, 1],
                    &[
                        (0, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                        (1, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                    ],
                ),
                runner_config_with_layers(
                    2,
                    vec!["full_attention".into(), "full_attention".into()],
                    2,
                    2,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            2,
            2,
            vec![0, 1],
        );
        let reference = nonzero_two_layer_full_attention_moe_both_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn nonzero_three_layer_middle_moe_output_decode_parity_matches() {
        let case = ConformanceCase::decode("three-layer-middle-moe-nonzero-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layer_expert_down_proj_bias(
                    3,
                    2,
                    &[1],
                    &[(1, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])],
                ),
                runner_config_with_layers(
                    3,
                    vec![
                        "full_attention".into(),
                        "full_attention".into(),
                        "full_attention".into(),
                    ],
                    2,
                    2,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            2,
            2,
            vec![1],
        );
        let reference = nonzero_three_layer_full_attention_middle_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn nonzero_three_layer_middle_moe_output_prefill_parity_matches() {
        let case = ConformanceCase::prefill("three-layer-middle-moe-nonzero-prefill", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layer_expert_down_proj_bias(
                    3,
                    2,
                    &[1],
                    &[(1, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])],
                ),
                runner_config_with_layers(
                    3,
                    vec![
                        "full_attention".into(),
                        "full_attention".into(),
                        "full_attention".into(),
                    ],
                    2,
                    2,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            2,
            2,
            vec![1],
        );
        let reference = nonzero_three_layer_full_attention_middle_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn nonzero_two_layer_both_layers_moe_multiblock_prefill_parity_matches() {
        let case =
            ConformanceCase::prefill("two-layer-both-moe-multiblock-nonzero-prefill", vec![1, 2, 3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layer_expert_down_proj_bias(
                    2,
                    2,
                    &[0, 1],
                    &[
                        (0, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                        (1, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                    ],
                ),
                runner_config_with_layers(
                    2,
                    vec!["full_attention".into(), "full_attention".into()],
                    2,
                    2,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner)
            .with_traced_moe(2, 2, vec![0, 1])
            .with_block_size(2);
        let reference = nonzero_two_layer_full_attention_moe_both_multiblock_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn nonzero_biased_three_expert_top2_moe_decode_parity_matches() {
        let case = ConformanceCase::decode("three-expert-top2-moe-nonzero-biased-decode", 2, vec![3]);
        let mut weights = runner_weights_with_layer_expert_down_proj_bias(
            1,
            3,
            &[0],
            &[(0, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])],
        );
        weights.tensors.insert(
            "model.layers.0.mlp.router.bias".to_string(),
            tensor(
                "model.layers.0.mlp.router.bias",
                &[0.0, 1.0, 2.0],
                &[3],
            )
            .1,
        );
        let runner = Arc::new(
            ModelRunner::new(
                weights,
                runner_config_with_moe(3, 2),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner)
            .with_traced_moe(3, 2, vec![0])
            .with_traced_router_bias(vec![0.0, 1.0, 2.0]);
        let reference = nonzero_biased_three_expert_top2_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn nonzero_biased_three_expert_top2_moe_prefill_parity_matches() {
        let case = ConformanceCase::prefill("three-expert-top2-moe-nonzero-biased-prefill", vec![1, 2]);
        let mut weights = runner_weights_with_layer_expert_down_proj_bias(
            1,
            3,
            &[0],
            &[(0, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])],
        );
        weights.tensors.insert(
            "model.layers.0.mlp.router.bias".to_string(),
            tensor(
                "model.layers.0.mlp.router.bias",
                &[0.0, 1.0, 2.0],
                &[3],
            )
            .1,
        );
        let runner = Arc::new(
            ModelRunner::new(
                weights,
                runner_config_with_moe(3, 2),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner)
            .with_traced_moe(3, 2, vec![0])
            .with_traced_router_bias(vec![0.0, 1.0, 2.0]);
        let reference = nonzero_biased_three_expert_top2_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn multi_block_full_attention_prefill_parity_matches() {
        let case = ConformanceCase::prefill("dense-multiblock", vec![1, 2, 3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights(),
                runner_config(),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_block_size(2);
        let reference = multi_block_dense_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn two_expert_top2_full_attention_moe_decode_parity_matches() {
        let case = ConformanceCase::decode("two-expert-moe-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_experts(2),
                runner_config_with_moe(2, 2),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            2,
            2,
            vec![0],
        );
        let reference = full_attention_two_expert_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn two_expert_top2_full_attention_moe_prefill_parity_matches() {
        let case = ConformanceCase::prefill("two-expert-moe-prefill", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_experts(2),
                runner_config_with_moe(2, 2),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            2,
            2,
            vec![0],
        );
        let reference = full_attention_two_expert_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn three_expert_top2_full_attention_moe_decode_parity_matches() {
        let case = ConformanceCase::decode("three-expert-top2-moe-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_experts(3),
                runner_config_with_moe(3, 2),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            3,
            2,
            vec![0],
        );
        let reference = full_attention_three_expert_top2_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn three_expert_top2_full_attention_moe_prefill_parity_matches() {
        let case = ConformanceCase::prefill("three-expert-top2-moe-prefill", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_experts(3),
                runner_config_with_moe(3, 2),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            3,
            2,
            vec![0],
        );
        let reference = full_attention_three_expert_top2_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn three_expert_top4_requested_full_attention_moe_decode_parity_matches() {
        let case = ConformanceCase::decode("three-expert-top4req-moe-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_experts(3),
                runner_config_with_moe(3, 4),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            3,
            4,
            vec![0],
        );
        let reference = full_attention_three_expert_top4_requested_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn two_layer_full_attention_second_layer_moe_decode_parity_matches() {
        let case = ConformanceCase::decode("two-layer-second-moe-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layers(2, 1, &[1]),
                runner_config_with_layers(
                    2,
                    vec!["full_attention".into(), "full_attention".into()],
                    1,
                    1,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            1,
            1,
            vec![1],
        );
        let reference = two_layer_full_attention_moe_on_second_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn two_layer_full_attention_second_layer_moe_prefill_parity_matches() {
        let case = ConformanceCase::prefill("two-layer-second-moe-prefill", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layers(2, 1, &[1]),
                runner_config_with_layers(
                    2,
                    vec!["full_attention".into(), "full_attention".into()],
                    1,
                    1,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            1,
            1,
            vec![1],
        );
        let reference = two_layer_full_attention_moe_on_second_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn two_layer_full_attention_both_layers_moe_decode_parity_matches() {
        let case = ConformanceCase::decode("two-layer-both-moe-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layers(2, 2, &[0, 1]),
                runner_config_with_layers(
                    2,
                    vec!["full_attention".into(), "full_attention".into()],
                    2,
                    2,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            2,
            2,
            vec![0, 1],
        );
        let reference = two_layer_full_attention_moe_both_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn two_layer_full_attention_both_layers_moe_prefill_parity_matches() {
        let case = ConformanceCase::prefill("two-layer-both-moe-prefill", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layers(2, 2, &[0, 1]),
                runner_config_with_layers(
                    2,
                    vec!["full_attention".into(), "full_attention".into()],
                    2,
                    2,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            2,
            2,
            vec![0, 1],
        );
        let reference = two_layer_full_attention_moe_both_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn biased_two_layer_full_attention_both_layers_moe_decode_parity_matches() {
        let case = ConformanceCase::decode("two-layer-both-moe-biased-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layers_and_router_bias(
                    2,
                    3,
                    &[0, 1],
                    &[(0, &[0.0, 1.0, 2.0]), (1, &[0.0, 1.0, 2.0])],
                ),
                runner_config_with_layers(
                    2,
                    vec!["full_attention".into(), "full_attention".into()],
                    3,
                    2,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner)
            .with_traced_moe(3, 2, vec![0, 1])
            .with_traced_router_bias(vec![0.0, 1.0, 2.0]);
        let reference = biased_two_layer_full_attention_moe_both_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn biased_two_layer_full_attention_both_layers_moe_prefill_parity_matches() {
        let case = ConformanceCase::prefill("two-layer-both-moe-biased-prefill", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layers_and_router_bias(
                    2,
                    3,
                    &[0, 1],
                    &[(0, &[0.0, 1.0, 2.0]), (1, &[0.0, 1.0, 2.0])],
                ),
                runner_config_with_layers(
                    2,
                    vec!["full_attention".into(), "full_attention".into()],
                    3,
                    2,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner)
            .with_traced_moe(3, 2, vec![0, 1])
            .with_traced_router_bias(vec![0.0, 1.0, 2.0]);
        let reference = biased_two_layer_full_attention_moe_both_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn two_layer_full_attention_both_layers_moe_multiblock_prefill_parity_matches() {
        let case = ConformanceCase::prefill("two-layer-both-moe-multiblock-prefill", vec![1, 2, 3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layers(2, 2, &[0, 1]),
                runner_config_with_layers(
                    2,
                    vec!["full_attention".into(), "full_attention".into()],
                    2,
                    2,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner)
            .with_traced_moe(2, 2, vec![0, 1])
            .with_block_size(2);
        let reference = two_layer_full_attention_moe_both_multiblock_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn biased_two_layer_full_attention_both_layers_moe_multiblock_prefill_parity_matches() {
        let case =
            ConformanceCase::prefill("two-layer-both-moe-biased-multiblock-prefill", vec![1, 2, 3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layers_and_router_bias(
                    2,
                    3,
                    &[0, 1],
                    &[(0, &[0.0, 1.0, 2.0]), (1, &[0.0, 1.0, 2.0])],
                ),
                runner_config_with_layers(
                    2,
                    vec!["full_attention".into(), "full_attention".into()],
                    3,
                    2,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner)
            .with_traced_moe(3, 2, vec![0, 1])
            .with_traced_router_bias(vec![0.0, 1.0, 2.0])
            .with_block_size(2);
        let reference = biased_two_layer_full_attention_moe_both_multiblock_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn three_layer_full_attention_middle_moe_decode_parity_matches() {
        let case = ConformanceCase::decode("three-layer-middle-moe-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layers(3, 2, &[1]),
                runner_config_with_layers(
                    3,
                    vec![
                        "full_attention".into(),
                        "full_attention".into(),
                        "full_attention".into(),
                    ],
                    2,
                    2,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            2,
            2,
            vec![1],
        );
        let reference = three_layer_full_attention_middle_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn three_layer_full_attention_middle_moe_prefill_parity_matches() {
        let case = ConformanceCase::prefill("three-layer-middle-moe-prefill", vec![1, 2]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_layers(3, 2, &[1]),
                runner_config_with_layers(
                    3,
                    vec![
                        "full_attention".into(),
                        "full_attention".into(),
                        "full_attention".into(),
                    ],
                    2,
                    2,
                ),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            2,
            2,
            vec![1],
        );
        let reference = three_layer_full_attention_middle_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);

        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }

    #[test]
    fn biased_three_expert_top2_full_attention_moe_decode_parity_matches() {
        let case = ConformanceCase::decode("biased-three-expert-top2-moe-decode", 2, vec![3]);
        let runner = Arc::new(
            ModelRunner::new(
                runner_weights_with_router_bias(3, &[0.0, 1.0, 2.0]),
                runner_config_with_moe(3, 2),
                Box::new(MockAttentionBackend),
                Arc::new(BridgeCacheEngine::new(1, 64)),
                MockGpuAllocator::new(1 << 20),
            )
            .expect("test model runner"),
        );
        let observed = ModelRunnerGreedyBackend::new("model-runner", runner).with_traced_moe(
            3,
            2,
            vec![0],
        )
        .with_traced_router_bias(vec![0.0, 1.0, 2.0]);
        let reference = biased_full_attention_three_expert_top2_moe_backend();
        let harness = ConformanceHarness::default();

        let report = harness.compare(&case, &reference, &observed);
        assert_eq!(report.outcome, ParityOutcome::Match);
        assert_eq!(report.comparison.diff_count(), 0);
    }
}
