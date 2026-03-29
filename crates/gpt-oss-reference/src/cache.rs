use gpt_oss_core::types::TokenId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Cache visibility mode used by the reference scaffold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheVisibility {
    Full,
    Sliding,
    Sink,
}

/// Logical cache layout for the single-block placeholder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheLayout {
    pub block_size: usize,
    pub visibility: CacheVisibility,
    pub seq_start_pos: usize,
}

/// Mutable cache state used by the reference scaffold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CacheState {
    pub blocks: Vec<Vec<TokenId>>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CacheModelError {
    #[error("block size must be greater than zero")]
    InvalidBlockSize,
    #[error("sequence start position must not exceed the current token count")]
    InvalidSequenceStart,
}

impl CacheLayout {
    pub fn new(block_size: usize, visibility: CacheVisibility, seq_start_pos: usize) -> Self {
        Self {
            block_size,
            visibility,
            seq_start_pos,
        }
    }

    pub fn validate(&self, token_count: usize) -> Result<(), CacheModelError> {
        if self.block_size == 0 {
            return Err(CacheModelError::InvalidBlockSize);
        }
        if self.seq_start_pos > token_count {
            return Err(CacheModelError::InvalidSequenceStart);
        }
        Ok(())
    }
}

impl CacheState {
    pub fn from_tokens(tokens: &[TokenId], layout: &CacheLayout) -> Result<Self, CacheModelError> {
        layout.validate(tokens.len())?;
        let blocks = tokens
            .chunks(layout.block_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        Ok(Self { blocks })
    }
}

