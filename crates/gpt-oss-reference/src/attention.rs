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
    pub visible_tokens: Vec<usize>,
}

impl AttentionTrace {
    pub fn full(visible_tokens: Vec<usize>) -> Self {
        Self {
            mode: AttentionMode::Full,
            attended_tokens: visible_tokens.len(),
            visible_tokens,
        }
    }

    pub fn sliding(visible_tokens: Vec<usize>) -> Self {
        Self {
            mode: AttentionMode::Sliding,
            attended_tokens: visible_tokens.len(),
            visible_tokens,
        }
    }

    pub fn sink(visible_tokens: Vec<usize>) -> Self {
        Self {
            mode: AttentionMode::Sink,
            attended_tokens: visible_tokens.len(),
            visible_tokens,
        }
    }
}
