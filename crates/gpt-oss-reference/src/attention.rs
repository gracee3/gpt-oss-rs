use serde::{Deserialize, Serialize};

/// High-level attention mode used by the reference scaffold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionMode {
    Full,
    Sliding,
    Sink,
}

/// Trace payload for the attention stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionTrace {
    pub mode: AttentionMode,
    pub attended_tokens: usize,
}

impl AttentionTrace {
    pub fn full(attended_tokens: usize) -> Self {
        Self {
            mode: AttentionMode::Full,
            attended_tokens,
        }
    }
}
