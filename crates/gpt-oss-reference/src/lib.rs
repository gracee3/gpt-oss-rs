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
pub use trace::{LayerTrace, ReferenceTrace};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_single_block_placeholder_is_deterministic() {
        let executor = ReferenceExecutor::new(ReferenceExecutorConfig {
            vocab_size: 8,
            num_layers: 2,
            block_size: 16,
        });

        let output = executor
            .forward(ReferenceInput {
                tokens: vec![1u32, 2u32],
            })
            .expect("placeholder forward");

        assert_eq!(output.logits.len(), 8);
        assert_eq!(output.trace.layers.len(), 2);
        assert_eq!(output.trace.cache.blocks.len(), 1);
        assert_eq!(output.trace.attention.mode, AttentionMode::Full);
        assert_eq!(output.trace.moe.mode, MoeMode::DenseOnly);
    }

    #[test]
    fn forward_rejects_inputs_larger_than_one_block() {
        let executor = ReferenceExecutor::new(ReferenceExecutorConfig {
            vocab_size: 4,
            num_layers: 1,
            block_size: 2,
        });

        let err = executor
            .forward(ReferenceInput {
                tokens: vec![0, 1, 2],
            })
            .expect_err("expected a single-block rejection");

        assert!(matches!(err, ReferenceError::InputTooLarge { .. }));
    }
}
