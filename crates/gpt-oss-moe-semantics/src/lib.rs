#![forbid(unsafe_code)]

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpertStorageKind {
    Unquantized,
    Mxfp4BlocksAndScales,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeContract {
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub renormalize: bool,
    pub storage: ExpertStorageKind,
}

impl MoeContract {
    pub fn evaluate(&self, router_logits: &[Vec<f32>]) -> Result<MoeTrace, MoeSemanticsError> {
        if router_logits.is_empty() {
            return Ok(MoeTrace {
                contract: self.clone(),
                tokens: Vec::new(),
            });
        }

        for (token_index, logits) in router_logits.iter().enumerate() {
            if logits.len() != self.num_local_experts {
                return Err(MoeSemanticsError::ExpertCountMismatch {
                    token_index,
                    expected: self.num_local_experts,
                    actual: logits.len(),
                });
            }
        }

        let top_k = self.num_experts_per_tok.min(self.num_local_experts);
        let mut tokens = Vec::with_capacity(router_logits.len());

        for (token_index, logits) in router_logits.iter().enumerate() {
            let selected = select_top_k(logits, top_k);
            let top_logits: Vec<f32> = selected.iter().map(|&index| logits[index]).collect();
            let weights = if self.renormalize {
                softmax(&top_logits)
            } else {
                exp_normalized(&top_logits)
            };
            let weight_sum = weights.iter().copied().sum();

            let routed_experts = selected
                .iter()
                .enumerate()
                .map(|(rank, &expert_index)| RoutedExpert {
                    expert_index,
                    rank,
                    weight: weights[rank],
                })
                .collect();

            tokens.push(MoeTokenTrace {
                token_index,
                router_logits: logits.clone(),
                selected_experts: selected,
                normalized_weights: weights,
                weight_sum,
                routed_experts,
            });
        }

        Ok(MoeTrace {
            contract: self.clone(),
            tokens,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutedExpert {
    pub expert_index: usize,
    pub rank: usize,
    pub weight: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeTokenTrace {
    pub token_index: usize,
    pub router_logits: Vec<f32>,
    pub selected_experts: Vec<usize>,
    pub normalized_weights: Vec<f32>,
    pub weight_sum: f32,
    pub routed_experts: Vec<RoutedExpert>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeTrace {
    pub contract: MoeContract,
    pub tokens: Vec<MoeTokenTrace>,
}

#[derive(Debug, Error, PartialEq)]
pub enum MoeSemanticsError {
    #[error("token {token_index} had {actual} router logits but expected {expected}")]
    ExpertCountMismatch {
        token_index: usize,
        expected: usize,
        actual: usize,
    },
}

fn select_top_k(logits: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(left_index, left_value), (right_index, right_value)| {
        match right_value
            .partial_cmp(left_value)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => left_index.cmp(right_index),
            other => other,
        }
    });
    indexed
        .into_iter()
        .take(k)
        .map(|(index, _)| index)
        .collect()
}

fn softmax(vals: &[f32]) -> Vec<f32> {
    if vals.is_empty() {
        return Vec::new();
    }

    let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = vals.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; vals.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

fn exp_normalized(vals: &[f32]) -> Vec<f32> {
    if vals.is_empty() {
        return Vec::new();
    }

    let max_val = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = vals.iter().map(|&v| (v - max_val).exp()).collect();
    exps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_top_k_prefers_lower_index_on_ties() {
        let contract = MoeContract {
            num_local_experts: 4,
            num_experts_per_tok: 2,
            renormalize: true,
            storage: ExpertStorageKind::Unquantized,
        };
        let trace = contract
            .evaluate(&[vec![1.0, 2.0, 2.0, 0.5]])
            .expect("contract should evaluate");
        let token = &trace.tokens[0];
        assert_eq!(token.selected_experts, vec![1, 2]);
        assert_eq!(token.routed_experts[0].rank, 0);
        assert_eq!(token.routed_experts[1].rank, 1);
    }

    #[test]
    fn normalized_weights_sum_to_one() {
        let contract = MoeContract {
            num_local_experts: 3,
            num_experts_per_tok: 2,
            renormalize: true,
            storage: ExpertStorageKind::Mxfp4BlocksAndScales,
        };
        let trace = contract
            .evaluate(&[vec![0.1, 0.9, 0.4]])
            .expect("contract should evaluate");
        let token = &trace.tokens[0];
        assert!((token.weight_sum - 1.0).abs() < 1e-6);
        assert_eq!(token.selected_experts, vec![1, 2]);
        assert!(token.normalized_weights[0] > token.normalized_weights[1]);
    }

    #[test]
    fn missing_router_width_is_rejected() {
        let contract = MoeContract {
            num_local_experts: 2,
            num_experts_per_tok: 1,
            renormalize: true,
            storage: ExpertStorageKind::Missing,
        };
        let err = contract.evaluate(&[vec![0.0]]).unwrap_err();
        assert!(matches!(err, MoeSemanticsError::ExpertCountMismatch { .. }));
    }
}
