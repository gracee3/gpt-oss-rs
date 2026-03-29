use serde::{Deserialize, Serialize};

/// High-level MoE mode used by the reference scaffold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoeMode {
    DenseOnly,
    SparseTopK,
}

/// Trace payload for the MoE stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeTrace {
    pub mode: MoeMode,
    pub experts_invoked: usize,
    pub selected_experts: Vec<Vec<usize>>,
    pub normalized_weights: Vec<Vec<f32>>,
}

impl MoeTrace {
    pub fn dense_only() -> Self {
        Self {
            mode: MoeMode::DenseOnly,
            experts_invoked: 0,
            selected_experts: Vec::new(),
            normalized_weights: Vec::new(),
        }
    }

    pub fn sparse(routes: &[Vec<(usize, f32)>]) -> Self {
        let experts_invoked = routes.iter().map(Vec::len).sum();
        let selected_experts = routes
            .iter()
            .map(|route| route.iter().map(|(expert, _)| *expert).collect())
            .collect();
        let normalized_weights = routes
            .iter()
            .map(|route| route.iter().map(|(_, weight)| *weight).collect())
            .collect();
        Self {
            mode: MoeMode::SparseTopK,
            experts_invoked,
            selected_experts,
            normalized_weights,
        }
    }
}
