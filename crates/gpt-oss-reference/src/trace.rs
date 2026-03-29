use crate::{AttentionTrace, CacheLayout, CacheState, MoeTrace};
use serde::{Deserialize, Serialize};

/// Per-layer trace for the reference scaffold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerTrace {
    pub layer_index: usize,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub attention: AttentionTrace,
    pub moe: MoeTrace,
}

/// Reference execution trace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceTrace {
    pub layers: Vec<LayerTrace>,
    pub attention: AttentionTrace,
    pub moe: MoeTrace,
    pub cache_layout: CacheLayout,
    pub cache: CacheState,
}
