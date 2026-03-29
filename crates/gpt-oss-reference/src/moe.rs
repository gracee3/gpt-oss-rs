use serde::{Deserialize, Serialize};

/// High-level MoE mode used by the reference scaffold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoeMode {
    DenseOnly,
    SparsePlaceholder,
}

/// Trace payload for the MoE stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeTrace {
    pub mode: MoeMode,
    pub experts_invoked: usize,
}

impl MoeTrace {
    pub fn dense_only() -> Self {
        Self {
            mode: MoeMode::DenseOnly,
            experts_invoked: 0,
        }
    }
}

