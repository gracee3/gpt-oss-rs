#![forbid(unsafe_code)]
//! Backend-independent KV-cache semantics.
//!
//! This crate models cache layout and visibility rules without depending on
//! device storage details. The current runtime can continue to use its existing
//! GPU-backed cache implementations while future planner work relies on these
//! types for legality and invariants.

use std::collections::HashSet;
use std::ops::Range;

use gpt_oss_core::prelude::{BlockId, SequenceId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CacheModelError {
    #[error("block size must be greater than zero")]
    InvalidBlockSize,
    #[error("layer index {layer_index} is out of bounds for {num_layers} layers")]
    InvalidLayerIndex {
        layer_index: usize,
        num_layers: usize,
    },
    #[error("sequence {sequence_id} has duplicate block ids in its block table")]
    DuplicateBlockIds { sequence_id: SequenceId },
    #[error(
        "sequence {sequence_id} block table length {actual} does not match required {expected}"
    )]
    BlockTableLengthMismatch {
        sequence_id: SequenceId,
        actual: usize,
        expected: usize,
    },
    #[error("sequence {sequence_id} has a zero-length span but a non-empty block table")]
    EmptySpanWithBlocks { sequence_id: SequenceId },
    #[error("sequence ids must be unique within a cache state view")]
    DuplicateSequenceId { sequence_id: SequenceId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheVisibility {
    Full,
    Sliding {
        window_tokens: usize,
    },
    SinkSliding {
        sink_tokens: usize,
        window_tokens: usize,
    },
}

impl CacheVisibility {
    pub fn visible_token_offsets(&self, query_offset: usize, total_tokens: usize) -> Vec<usize> {
        if total_tokens == 0 {
            return Vec::new();
        }

        let last = query_offset.min(total_tokens.saturating_sub(1));
        match *self {
            CacheVisibility::Full => (0..=last).collect(),
            CacheVisibility::Sliding { window_tokens } => {
                if window_tokens == 0 {
                    return Vec::new();
                }
                let start = last.saturating_add(1).saturating_sub(window_tokens);
                (start..=last).collect()
            }
            CacheVisibility::SinkSliding {
                sink_tokens,
                window_tokens,
            } => {
                if sink_tokens == 0 && window_tokens == 0 {
                    return Vec::new();
                }
                let mut out = Vec::new();
                let sink_end = sink_tokens.min(total_tokens).min(last.saturating_add(1));
                out.extend(0..sink_end);

                if window_tokens == 0 {
                    return out;
                }
                let tail_start = last.saturating_add(1).saturating_sub(window_tokens);
                for idx in tail_start..=last {
                    if !out.contains(&idx) {
                        out.push(idx);
                    }
                }
                out
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerCacheSpec {
    pub layer_index: usize,
    pub visibility: CacheVisibility,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheLayoutSpec {
    pub num_layers: usize,
    pub block_size: usize,
    pub layers: Vec<LayerCacheSpec>,
}

impl CacheLayoutSpec {
    pub fn new(num_layers: usize, block_size: usize) -> Self {
        let layers = (0..num_layers)
            .map(|layer_index| LayerCacheSpec {
                layer_index,
                visibility: CacheVisibility::Full,
            })
            .collect();
        Self {
            num_layers,
            block_size,
            layers,
        }
    }

    pub fn with_layer_visibility(
        mut self,
        layer_index: usize,
        visibility: CacheVisibility,
    ) -> Result<Self, CacheModelError> {
        self.validate_layer_index(layer_index)?;
        self.layers[layer_index].visibility = visibility;
        Ok(self)
    }

    pub fn layer(&self, layer_index: usize) -> Option<&LayerCacheSpec> {
        self.layers.get(layer_index)
    }

    pub fn validate(&self) -> Result<(), CacheModelError> {
        if self.block_size == 0 {
            return Err(CacheModelError::InvalidBlockSize);
        }
        if self.layers.len() != self.num_layers {
            return Err(CacheModelError::InvalidLayerIndex {
                layer_index: self.layers.len(),
                num_layers: self.num_layers,
            });
        }
        Ok(())
    }

    fn validate_layer_index(&self, layer_index: usize) -> Result<(), CacheModelError> {
        if layer_index >= self.num_layers {
            return Err(CacheModelError::InvalidLayerIndex {
                layer_index,
                num_layers: self.num_layers,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceCacheState {
    pub sequence_id: SequenceId,
    pub seq_start_pos: u32,
    pub token_count: usize,
    pub block_table: Vec<BlockId>,
}

impl SequenceCacheState {
    pub fn new(
        sequence_id: SequenceId,
        seq_start_pos: u32,
        token_count: usize,
        block_table: Vec<BlockId>,
        block_size: usize,
    ) -> Result<Self, CacheModelError> {
        let state = Self {
            sequence_id,
            seq_start_pos,
            token_count,
            block_table,
        };
        state.validate(block_size)?;
        Ok(state)
    }

    pub fn end_pos_exclusive(&self) -> u32 {
        self.seq_start_pos + self.token_count as u32
    }

    pub fn required_block_range(&self, block_size: usize) -> Range<usize> {
        if self.token_count == 0 {
            let start_block = self.seq_start_pos as usize / block_size;
            return start_block..start_block;
        }
        let start_block = self.seq_start_pos as usize / block_size;
        let end_pos = self.end_pos_exclusive() as usize;
        let end_block = end_pos.saturating_sub(1) / block_size;
        start_block..(end_block + 1)
    }

    pub fn required_block_count(&self, block_size: usize) -> usize {
        if self.token_count == 0 {
            return 0;
        }
        let range = self.required_block_range(block_size);
        range.end - range.start
    }

    pub fn append_tokens(&self, additional_tokens: usize) -> Self {
        let mut next = self.clone();
        next.token_count += additional_tokens;
        next
    }

    pub fn position_ids(&self) -> Vec<u32> {
        (0..self.token_count)
            .map(|idx| self.seq_start_pos + idx as u32)
            .collect()
    }

    pub fn slot_mapping(&self, block_size: usize) -> Result<Vec<u32>, CacheModelError> {
        self.validate(block_size)?;
        let base_block = self.seq_start_pos as usize / block_size;
        let mut out = Vec::with_capacity(self.token_count);
        for token_idx in 0..self.token_count {
            let absolute_pos = self.seq_start_pos as usize + token_idx;
            let block_idx = absolute_pos / block_size - base_block;
            let block_offset = absolute_pos % block_size;
            let physical_block = self.block_table[block_idx].0 as usize;
            out.push((physical_block * block_size) as u32 + block_offset as u32);
        }
        Ok(out)
    }

    pub fn validate(&self, block_size: usize) -> Result<(), CacheModelError> {
        if block_size == 0 {
            return Err(CacheModelError::InvalidBlockSize);
        }
        let expected = self.required_block_count(block_size);
        let actual = self.block_table.len();
        if self.token_count == 0 {
            if actual != 0 {
                return Err(CacheModelError::EmptySpanWithBlocks {
                    sequence_id: self.sequence_id,
                });
            }
            return Ok(());
        }
        if expected != actual {
            return Err(CacheModelError::BlockTableLengthMismatch {
                sequence_id: self.sequence_id,
                actual,
                expected,
            });
        }
        let mut seen = HashSet::new();
        for block_id in &self.block_table {
            if !seen.insert(*block_id) {
                return Err(CacheModelError::DuplicateBlockIds {
                    sequence_id: self.sequence_id,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheStateView {
    pub layout: CacheLayoutSpec,
    pub sequences: Vec<SequenceCacheState>,
}

impl CacheStateView {
    pub fn new(
        layout: CacheLayoutSpec,
        sequences: Vec<SequenceCacheState>,
    ) -> Result<Self, CacheModelError> {
        layout.validate()?;
        let view = Self { layout, sequences };
        view.validate()?;
        Ok(view)
    }

    pub fn validate(&self) -> Result<(), CacheModelError> {
        self.layout.validate()?;
        let mut seen = HashSet::new();
        for sequence in &self.sequences {
            if !seen.insert(sequence.sequence_id) {
                return Err(CacheModelError::DuplicateSequenceId {
                    sequence_id: sequence.sequence_id,
                });
            }
            sequence.validate(self.layout.block_size)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_layout_defaults_to_full_visibility() {
        let layout = CacheLayoutSpec::new(3, 16);
        assert_eq!(layout.num_layers, 3);
        assert_eq!(layout.block_size, 16);
        assert!(matches!(
            layout.layer(0).unwrap().visibility,
            CacheVisibility::Full
        ));
        assert!(layout.validate().is_ok());
    }

    #[test]
    fn sequence_slot_mapping_uses_absolute_positions() {
        let state = SequenceCacheState::new(SequenceId(7), 2, 5, vec![BlockId(10), BlockId(11)], 4)
            .unwrap();

        assert_eq!(state.position_ids(), vec![2, 3, 4, 5, 6]);
        assert_eq!(state.required_block_range(4), 0..2);
        assert_eq!(state.slot_mapping(4).unwrap(), vec![42, 43, 44, 45, 46]);
    }

    #[test]
    fn append_keeps_start_position_and_extends_length() {
        let state = SequenceCacheState::new(SequenceId(1), 0, 4, vec![BlockId(0)], 4).unwrap();

        let next = state.append_tokens(2);
        assert_eq!(next.seq_start_pos, 0);
        assert_eq!(next.token_count, 6);
    }

    #[test]
    fn visibility_modes_are_explicit() {
        assert_eq!(
            CacheVisibility::Full.visible_token_offsets(3, 8),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            CacheVisibility::Sliding { window_tokens: 3 }.visible_token_offsets(5, 8),
            vec![3, 4, 5]
        );
        assert_eq!(
            CacheVisibility::SinkSliding {
                sink_tokens: 2,
                window_tokens: 3,
            }
            .visible_token_offsets(5, 8),
            vec![0, 1, 3, 4, 5]
        );
    }

    #[test]
    fn cache_state_view_rejects_duplicate_sequences() {
        let layout = CacheLayoutSpec::new(1, 4);
        let seq = SequenceCacheState::new(SequenceId(1), 0, 4, vec![BlockId(0)], 4).unwrap();
        let err = CacheStateView::new(layout, vec![seq.clone(), seq]).unwrap_err();
        assert!(matches!(err, CacheModelError::DuplicateSequenceId { .. }));
    }
}
