//! Canonical semantic model for GPT-OSS MoE routing and storage boundaries.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, MoeSemanticError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum MoeSemanticError {
    #[error("gpt-oss moe requires hidden_size > 0")]
    HiddenSizeZero,
    #[error("gpt-oss moe requires num_local_experts > 0")]
    NumLocalExpertsZero,
    #[error("gpt-oss moe requires num_experts_per_tok > 0")]
    NumExpertsPerTokZero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopKSelection {
    StableDescendingByLogitThenIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingNormalization {
    Softmax,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpertOrdering {
    GroupedByExpertIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccumulationMode {
    WeightedSum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpertStorageBoundary {
    Unquantized,
    Mxfp4BlocksAndScales,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouterInputSpec {
    pub hidden_size: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub has_router_bias: bool,
}

impl RouterInputSpec {
    pub fn new(
        hidden_size: usize,
        num_local_experts: usize,
        num_experts_per_tok: usize,
        has_router_bias: bool,
    ) -> Result<Self> {
        if hidden_size == 0 {
            return Err(MoeSemanticError::HiddenSizeZero);
        }
        if num_local_experts == 0 {
            return Err(MoeSemanticError::NumLocalExpertsZero);
        }
        if num_experts_per_tok == 0 {
            return Err(MoeSemanticError::NumExpertsPerTokZero);
        }

        Ok(Self {
            hidden_size,
            num_local_experts,
            num_experts_per_tok,
            has_router_bias,
        })
    }

    pub fn effective_top_k(&self) -> usize {
        self.num_experts_per_tok.min(self.num_local_experts)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopKSpec {
    pub requested: usize,
    pub effective: usize,
    pub selection: TopKSelection,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingWeightsSpec {
    pub normalization: RoutingNormalization,
    pub renormalize_selected: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertOrderingSpec {
    pub ordering: ExpertOrdering,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccumulationSpec {
    pub mode: AccumulationMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeSemanticSpec {
    pub router_input: RouterInputSpec,
    pub top_k: TopKSpec,
    pub routing_weights: RoutingWeightsSpec,
    pub expert_ordering: ExpertOrderingSpec,
    pub accumulation: AccumulationSpec,
    pub storage_boundary: ExpertStorageBoundary,
}

impl MoeSemanticSpec {
    pub fn from_router_input(
        router_input: RouterInputSpec,
        storage_boundary: ExpertStorageBoundary,
    ) -> Result<Self> {
        let effective_top_k = router_input.effective_top_k();
        Ok(Self {
            top_k: TopKSpec {
                requested: router_input.num_experts_per_tok,
                effective: effective_top_k,
                selection: TopKSelection::StableDescendingByLogitThenIndex,
            },
            routing_weights: RoutingWeightsSpec {
                normalization: RoutingNormalization::Softmax,
                renormalize_selected: true,
            },
            expert_ordering: ExpertOrderingSpec {
                ordering: ExpertOrdering::GroupedByExpertIndex,
            },
            accumulation: AccumulationSpec {
                mode: AccumulationMode::WeightedSum,
            },
            router_input,
            storage_boundary,
        })
    }

    pub fn effective_top_k(&self) -> usize {
        self.top_k.effective
    }
}

/// Return the indices of the top-k largest logits with a stable index tiebreak.
pub fn stable_top_k_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(left_idx, left_val), (right_idx, right_val)| {
        right_val
            .partial_cmp(left_val)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_idx.cmp(right_idx))
    });
    indexed.into_iter().take(k).map(|(idx, _)| idx).collect()
}

/// Numerically stable softmax used by the MoE routing contract.
pub fn softmax_weights(vals: &[f32]) -> Vec<f32> {
    if vals.is_empty() {
        return vec![];
    }

    let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = vals.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

pub fn route_top_k(logits: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let top_indices = stable_top_k_indices(logits, top_k);
    let top_logits: Vec<f32> = top_indices.iter().map(|&idx| logits[idx]).collect();
    let weights = softmax_weights(&top_logits);
    top_indices
        .into_iter()
        .zip(weights)
        .collect::<Vec<(usize, f32)>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_input_validates_positive_dimensions() {
        assert!(RouterInputSpec::new(0, 1, 1, true).is_err());
        assert!(RouterInputSpec::new(4, 0, 1, true).is_err());
        assert!(RouterInputSpec::new(4, 1, 0, true).is_err());
        assert!(RouterInputSpec::new(4, 2, 1, true).is_ok());
    }

    #[test]
    fn stable_top_k_uses_index_tiebreak() {
        let logits = [1.0, 1.0, 0.5, 1.0];
        assert_eq!(stable_top_k_indices(&logits, 2), vec![0, 1]);
    }

    #[test]
    fn routing_weights_sum_to_one() {
        let weights = softmax_weights(&[1.0, 2.0, 3.0]);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn spec_tracks_requested_and_effective_top_k() {
        let router_input = RouterInputSpec::new(4, 2, 4, true).unwrap();
        let spec =
            MoeSemanticSpec::from_router_input(router_input, ExpertStorageBoundary::Unquantized)
                .unwrap();

        assert_eq!(spec.top_k.requested, 4);
        assert_eq!(spec.top_k.effective, 2);
        assert!(matches!(
            spec.routing_weights.normalization,
            RoutingNormalization::Softmax
        ));
        assert!(matches!(
            spec.expert_ordering.ordering,
            ExpertOrdering::GroupedByExpertIndex
        ));
        assert!(matches!(
            spec.accumulation.mode,
            AccumulationMode::WeightedSum
        ));
    }

    #[test]
    fn route_top_k_returns_weighted_indices() {
        let routed = route_top_k(&[0.0, 2.0, 1.0], 2);
        assert_eq!(routed.len(), 2);
        assert_eq!(routed[0].0, 1);
        assert_eq!(routed[1].0, 2);
        let sum: f32 = routed.iter().map(|(_, weight)| weight).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
