#![forbid(unsafe_code)]
//! Native reference scaffold for GPT-OSS semantics.
//!
//! This crate is intentionally small at first: it defines a deterministic
//! single-block executor boundary that can grow into a full oracle backend.

mod attention;
mod cache;
mod executor;
mod moe;
mod trace;

pub use attention::{AttentionMode, AttentionTrace};
pub use cache::{CacheLayout, CacheModelError, CacheState, CacheVisibility};
pub use executor::{
    ReferenceError, ReferenceExecutor, ReferenceExecutorConfig, ReferenceInput, ReferenceOutput,
};
pub use moe::{MoeMode, MoeTrace};
pub use trace::{LayerTrace, ReferencePhase, ReferenceTrace};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_single_block_placeholder_is_deterministic() {
        let executor = ReferenceExecutor::new(ReferenceExecutorConfig {
            vocab_size: 8,
            num_layers: 2,
            block_size: 16,
            layer_types: vec!["full_attention".into(), "sliding_attention".into()],
            sliding_window: Some(2),
            sink_tokens: 1,
            num_local_experts: 4,
            num_experts_per_tok: 2,
            moe_layer_indices: vec![1],
        });

        let output = executor
            .forward(ReferenceInput {
                tokens: vec![1u32, 2u32],
                phase: ReferencePhase::Prefill,
                seq_start_pos: 0,
            })
            .expect("placeholder forward");

        assert_eq!(output.logits.len(), 8);
        assert_eq!(output.trace.layers.len(), 2);
        assert_eq!(output.trace.cache.blocks.len(), 1);
        assert_eq!(output.trace.attention.mode, AttentionMode::Sink);
        assert_eq!(output.trace.moe.mode, MoeMode::SparseTopK);
        assert!(output.logits.iter().any(|logit| *logit != 0.0));
        assert_eq!(output.trace.layers[1].attention.visible_tokens, vec![0, 1]);
        assert_eq!(output.trace.layers[1].moe.experts_invoked, 4);
        assert_eq!(output.trace.phase, ReferencePhase::Prefill);
        assert_eq!(output.trace.layers[1].position_ids, vec![0, 1]);
    }

    #[test]
    fn forward_supports_multi_block_cache_layout() {
        let executor = ReferenceExecutor::new(ReferenceExecutorConfig {
            vocab_size: 4,
            num_layers: 1,
            block_size: 2,
            layer_types: vec!["full_attention".into()],
            sliding_window: None,
            sink_tokens: 0,
            num_local_experts: 0,
            num_experts_per_tok: 0,
            moe_layer_indices: Vec::new(),
        });

        let output = executor
            .forward(ReferenceInput {
                tokens: vec![0, 1, 2],
                phase: ReferencePhase::Prefill,
                seq_start_pos: 0,
            })
            .expect("multi-block forward");

        assert_eq!(output.trace.cache.blocks.len(), 2);
        assert_eq!(output.trace.layers[0].attention.visible_tokens, vec![0, 1, 2]);
        assert_eq!(output.logits, vec![0.0; 4]);
    }

    #[test]
    fn invalid_layer_type_shape_is_rejected() {
        let executor = ReferenceExecutor::new(ReferenceExecutorConfig {
            vocab_size: 4,
            num_layers: 2,
            block_size: 2,
            layer_types: vec!["full_attention".into()],
            sliding_window: None,
            sink_tokens: 0,
            num_local_experts: 0,
            num_experts_per_tok: 0,
            moe_layer_indices: Vec::new(),
        });

        let err = executor
            .forward(ReferenceInput {
                tokens: vec![0, 1],
                phase: ReferencePhase::Prefill,
                seq_start_pos: 0,
            })
            .expect_err("expected layer type validation failure");

        assert!(matches!(err, ReferenceError::LayerTypesLengthMismatch { .. }));
    }

    #[test]
    fn decode_tracks_absolute_positions_and_cache_alignment() {
        let executor = ReferenceExecutor::new(ReferenceExecutorConfig {
            vocab_size: 4,
            num_layers: 2,
            block_size: 4,
            layer_types: vec!["full_attention".into(), "sliding_attention".into()],
            sliding_window: Some(2),
            sink_tokens: 1,
            num_local_experts: 2,
            num_experts_per_tok: 1,
            moe_layer_indices: vec![1],
        });

        let output = executor
            .forward(ReferenceInput {
                tokens: vec![9],
                phase: ReferencePhase::Decode,
                seq_start_pos: 4,
            })
            .expect("decode forward");

        assert_eq!(output.trace.phase, ReferencePhase::Decode);
        assert_eq!(output.trace.seq_start_pos, 4);
        assert_eq!(output.trace.layers[0].position_ids, vec![4]);
        assert_eq!(output.trace.cache.blocks.len(), 1);
        assert_eq!(output.trace.cache_layout.seq_start_pos, 4);
        assert_eq!(output.trace.layers[1].attention.visible_tokens, vec![0, 3, 4]);
    }
}
