use crate::{AttentionTrace, CacheLayout, CacheState, MoeTrace};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferencePhase {
    Prefill,
    Decode,
}

/// Per-layer trace for the reference scaffold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerTrace {
    pub layer_index: usize,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub position_ids: Vec<u32>,
    pub attention: AttentionTrace,
    pub moe: MoeTrace,
}

/// Reference execution trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceTrace {
    pub phase: ReferencePhase,
    pub seq_start_pos: u32,
    pub layers: Vec<LayerTrace>,
    pub attention: AttentionTrace,
    pub moe: MoeTrace,
    pub cache_layout: CacheLayout,
    pub cache: CacheState,
}
