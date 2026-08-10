use gpt_oss_core::types::TokenId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Cache visibility mode used by the reference scaffold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheVisibility {
    Full,
    Sliding {
        window_tokens: usize,
    },
    Sink {
        sink_tokens: usize,
        window_tokens: usize,
    },
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
}

impl CacheLayout {
    pub fn new(block_size: usize, visibility: CacheVisibility, seq_start_pos: usize) -> Self {
        Self {
            block_size,
            visibility,
            seq_start_pos,
        }
    }

    pub fn validate(&self, _token_count: usize) -> Result<(), CacheModelError> {
        if self.block_size == 0 {
            return Err(CacheModelError::InvalidBlockSize);
        }
        Ok(())
    }
}

impl CacheState {
    pub fn from_tokens(tokens: &[TokenId], layout: &CacheLayout) -> Result<Self, CacheModelError> {
        layout.validate(tokens.len())?;
        let mut blocks = Vec::new();
        let mut current_block = Vec::new();
        let mut current_block_index = layout.seq_start_pos / layout.block_size;

        for (token_offset, token) in tokens.iter().enumerate() {
            let absolute_pos = layout.seq_start_pos + token_offset;
            let block_index = absolute_pos / layout.block_size;
            if block_index != current_block_index && !current_block.is_empty() {
                blocks.push(std::mem::take(&mut current_block));
                current_block_index = block_index;
            }
            current_block.push(*token);
        }

        if !current_block.is_empty() {
            blocks.push(current_block);
        }
        Ok(Self { blocks })
    }
}
