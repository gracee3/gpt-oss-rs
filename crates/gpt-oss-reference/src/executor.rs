use crate::{
    attention::AttentionTrace,
    cache::{CacheLayout, CacheModelError, CacheState, CacheVisibility},
    moe::MoeTrace,
    trace::{LayerTrace, ReferenceTrace},
};
use gpt_oss_core::types::TokenId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Config for the reference executor scaffold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceExecutorConfig {
    pub vocab_size: usize,
    pub num_layers: usize,
    pub block_size: usize,
}

/// Input to the reference executor scaffold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceInput {
    pub tokens: Vec<TokenId>,
}

/// Output of the reference executor scaffold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceOutput {
    pub logits: Vec<f32>,
    pub trace: ReferenceTrace,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ReferenceError {
    #[error(
        "input exceeds the single-block scaffold limit: {tokens} tokens > {block_size} block size"
    )]
    InputTooLarge { tokens: usize, block_size: usize },
    #[error(transparent)]
    Cache(#[from] CacheModelError),
}

/// Deterministic native reference executor scaffold.
#[derive(Debug, Clone)]
pub struct ReferenceExecutor {
    config: ReferenceExecutorConfig,
}

impl ReferenceExecutor {
    pub fn new(config: ReferenceExecutorConfig) -> Self {
        Self { config }
    }

    pub fn forward(&self, input: ReferenceInput) -> Result<ReferenceOutput, ReferenceError> {
        if input.tokens.len() > self.config.block_size {
            return Err(ReferenceError::InputTooLarge {
                tokens: input.tokens.len(),
                block_size: self.config.block_size,
            });
        }

        let cache_layout = CacheLayout::new(self.config.block_size, CacheVisibility::Full, 0);
        let cache = CacheState::from_tokens(&input.tokens, &cache_layout)?;

        let layer_count = self.config.num_layers;
        let token_count = input.tokens.len();
        let layers = (0..layer_count)
            .map(|layer_index| LayerTrace {
                layer_index,
                input_tokens: token_count,
                output_tokens: token_count,
            })
            .collect();

        let trace = ReferenceTrace {
            layers,
            attention: AttentionTrace::full(token_count),
            moe: MoeTrace::dense_only(),
            cache_layout,
            cache,
        };

        Ok(ReferenceOutput {
            logits: vec![0.0; self.config.vocab_size],
            trace,
        })
    }
}
