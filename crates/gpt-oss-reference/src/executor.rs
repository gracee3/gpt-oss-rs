use crate::{
    attention::AttentionTrace,
    cache::{CacheLayout, CacheModelError, CacheState, CacheVisibility},
    moe::MoeTrace,
    trace::{LayerTrace, ReferencePhase, ReferenceTrace},
};
use gpt_oss_core::types::TokenId;
use gpt_oss_moe_semantics::route_top_k;
use half::f16;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Config for the reference executor scaffold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceExecutorConfig {
    pub vocab_size: usize,
    pub num_layers: usize,
    pub block_size: usize,
    #[serde(default)]
    pub layer_types: Vec<String>,
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub sink_tokens: usize,
    #[serde(default)]
    pub num_local_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub token_embedding_rows: Vec<Vec<f32>>,
    #[serde(default)]
    pub final_norm_weight: Vec<f32>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub lm_head_rows: Vec<Vec<f32>>,
    #[serde(default)]
    pub expert_output_rows: Vec<Vec<f32>>,
    #[serde(default)]
    pub router_bias: Vec<f32>,
    #[serde(default)]
    pub moe_layer_indices: Vec<usize>,
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

/// Input to the reference executor scaffold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceInput {
    pub tokens: Vec<TokenId>,
    pub phase: ReferencePhase,
    #[serde(default)]
    pub seq_start_pos: u32,
}

/// Output of the reference executor scaffold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceOutput {
    pub logits: Vec<f32>,
    pub trace: ReferenceTrace,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ReferenceError {
    #[error("layer_types length mismatch: expected 0 or {expected}, got {actual}")]
    LayerTypesLengthMismatch { expected: usize, actual: usize },
    #[error(transparent)]
    Cache(#[from] CacheModelError),
}

/// Deterministic native reference executor scaffold.
#[derive(Debug, Clone)]
pub struct ReferenceExecutor {
    config: ReferenceExecutorConfig,
}

impl ReferenceExecutor {
    pub fn new(config: ReferenceExecutorConfig) -> Self {
        Self { config }
    }

    pub fn forward(&self, input: ReferenceInput) -> Result<ReferenceOutput, ReferenceError> {
        if !self.config.layer_types.is_empty()
            && self.config.layer_types.len() != self.config.num_layers
        {
            return Err(ReferenceError::LayerTypesLengthMismatch {
                expected: self.config.num_layers,
                actual: self.config.layer_types.len(),
            });
        }

        let cache_layout = CacheLayout::new(
            self.config.block_size,
            self.cache_visibility(),
            input.seq_start_pos as usize,
        );
        let cache = CacheState::from_tokens(&input.tokens, &cache_layout)?;

        let token_count = input.tokens.len();
        if self.has_explicit_dense_projection() {
            return self.forward_explicit_dense(input, cache_layout, cache);
        }

        let zero_logit_baseline = self.uses_zero_logit_baseline();
        let mut hidden = if zero_logit_baseline {
            vec![0.0; token_count]
        } else {
            input
                .tokens
                .iter()
                .enumerate()
                .map(|(index, token)| {
                    *token as f32 + ((input.seq_start_pos as usize + index) as f32 * 0.125)
                })
                .collect::<Vec<_>>()
        };
        let mut layers = Vec::with_capacity(self.config.num_layers);
        let mut aggregate_moe_routes = Vec::new();
        let position_ids = (0..token_count)
            .map(|offset| input.seq_start_pos + offset as u32)
            .collect::<Vec<_>>();
        let total_sequence_tokens = input.seq_start_pos as usize + token_count;

        for layer_index in 0..self.config.num_layers {
            let layer_type = self.layer_type(layer_index);
            let query_index = total_sequence_tokens.saturating_sub(1);
            let layer_attention =
                self.attention_trace_for(layer_type, query_index, total_sequence_tokens);
            let mut next_hidden = hidden.clone();
            let mut layer_routes = Vec::new();

            for token_index in 0..token_count {
                let absolute_pos = input.seq_start_pos as usize + token_index;
                let visible = self.visible_tokens(layer_type, absolute_pos, total_sequence_tokens);
                let attention_signal = if zero_logit_baseline {
                    0.0
                } else {
                    self.synthetic_attention_signal(&visible, hidden[token_index])
                };
                let moe_signal = if self.layer_has_moe(layer_index) {
                    let logits = self.router_logits(attention_signal, hidden[token_index]);
                    let routes = route_top_k(&logits, self.effective_top_k());
                    layer_routes.push(routes);
                    if zero_logit_baseline {
                        0.0
                    } else {
                        layer_routes
                            .last()
                            .into_iter()
                            .flatten()
                            .map(|(expert, weight)| (*expert as f32 + 1.0) * *weight)
                            .sum::<f32>()
                    }
                } else {
                    0.0
                };
                next_hidden[token_index] = hidden[token_index]
                    + attention_signal
                    + moe_signal
                    + if zero_logit_baseline {
                        0.0
                    } else {
                        layer_index as f32 * 0.01
                    };
            }

            hidden = next_hidden;
            aggregate_moe_routes.extend(layer_routes.iter().cloned());
            let layer_moe = if self.layer_has_moe(layer_index) {
                MoeTrace::sparse(&layer_routes)
            } else {
                MoeTrace::dense_only()
            };
            layers.push(LayerTrace {
                layer_index,
                input_tokens: token_count,
                output_tokens: token_count,
                position_ids: position_ids.clone(),
                attention: layer_attention,
                moe: layer_moe,
            });
        }

        let attention = layers
            .last()
            .map(|layer| layer.attention.clone())
            .unwrap_or_else(|| AttentionTrace::full(Vec::new()));
        let moe = if aggregate_moe_routes.is_empty() {
            MoeTrace::dense_only()
        } else {
            MoeTrace::sparse(&aggregate_moe_routes)
        };

        let trace = ReferenceTrace {
            phase: input.phase,
            seq_start_pos: input.seq_start_pos,
            layers,
            attention,
            moe,
            cache_layout,
            cache,
        };

        Ok(ReferenceOutput {
            logits: self.project_logits(&hidden),
            trace,
        })
    }

    fn layer_type(&self, layer_index: usize) -> &str {
        if self.config.layer_types.is_empty() {
            "full_attention"
        } else {
            &self.config.layer_types[layer_index]
        }
    }

    fn layer_has_moe(&self, layer_index: usize) -> bool {
        self.effective_top_k() > 0 && self.config.moe_layer_indices.contains(&layer_index)
    }

    fn effective_top_k(&self) -> usize {
        self.config
            .num_experts_per_tok
            .min(self.config.num_local_experts)
    }

    fn cache_visibility(&self) -> CacheVisibility {
        if self.config.sink_tokens > 0 {
            CacheVisibility::Sink {
                sink_tokens: self.config.sink_tokens,
                window_tokens: self.config.sliding_window.unwrap_or(self.config.block_size),
            }
        } else if let Some(window_tokens) = self.config.sliding_window {
            CacheVisibility::Sliding { window_tokens }
        } else {
            CacheVisibility::Full
        }
    }

    fn uses_zero_logit_baseline(&self) -> bool {
        if self.has_explicit_dense_projection() {
            return false;
        }
        if self.config.sink_tokens > 0 || self.config.sliding_window.is_some() {
            return false;
        }

        let full_attention_only = self
            .config
            .layer_types
            .iter()
            .all(|layer| layer == "full_attention" || layer == "global_attention");

        if !full_attention_only {
            return false;
        }

        self.effective_top_k() == 0
            || (self.config.num_local_experts > 0
                && self.effective_top_k() > 0
                && self.config.moe_layer_indices.iter().all(|layer| {
                    matches!(self.layer_type(*layer), "full_attention" | "global_attention")
                }))
    }

    fn attention_trace_for(
        &self,
        layer_type: &str,
        query_index: usize,
        token_count: usize,
    ) -> AttentionTrace {
        let visible = self.visible_tokens(layer_type, query_index, token_count);
        if self.config.sink_tokens > 0 && layer_type != "full_attention" && layer_type != "global_attention" {
            AttentionTrace::sink(visible)
        } else if layer_type == "sliding_attention" || layer_type == "local_attention" {
            AttentionTrace::sliding(visible)
        } else {
            AttentionTrace::full(visible)
        }
    }

    fn visible_tokens(&self, layer_type: &str, query_index: usize, token_count: usize) -> Vec<usize> {
        if token_count == 0 {
            return Vec::new();
        }

        let last = query_index.min(token_count - 1);
        if layer_type == "sliding_attention" || layer_type == "local_attention" {
            let mut visible = Vec::new();
            let sink_end = self.config.sink_tokens.min(token_count).min(last + 1);
            visible.extend(0..sink_end);

            let window = self.config.sliding_window.unwrap_or(token_count);
            let start = last.saturating_add(1).saturating_sub(window);
            for index in start..=last {
                if !visible.contains(&index) {
                    visible.push(index);
                }
            }
            visible
        } else {
            (0..=last).collect()
        }
    }

    fn synthetic_attention_signal(&self, visible: &[usize], hidden_value: f32) -> f32 {
        if visible.is_empty() {
            return 0.0;
        }
        let positional_mean = visible.iter().map(|idx| *idx as f32).sum::<f32>() / visible.len() as f32;
        hidden_value + positional_mean * 0.01
    }

    fn synthetic_router_logits(&self, attention_signal: f32, hidden_value: f32) -> Vec<f32> {
        (0..self.config.num_local_experts)
            .map(|expert_index| {
                let synthetic = attention_signal * ((expert_index + 1) as f32 * 0.1)
                    + hidden_value * (((expert_index % 3) + 1) as f32 * 0.05)
                    - expert_index as f32 * 0.02;
                synthetic + self.config.router_bias.get(expert_index).copied().unwrap_or(0.0)
            })
            .collect()
    }

    fn router_logits(&self, attention_signal: f32, hidden_value: f32) -> Vec<f32> {
        if self.config.router_bias.len() == self.config.num_local_experts
            && (self.uses_zero_logit_baseline() || self.has_explicit_dense_projection())
        {
            return self.config.router_bias.clone();
        }

        if self.uses_zero_logit_baseline() {
            if self.config.router_bias.len() == self.config.num_local_experts {
                return self.config.router_bias.clone();
            }
            return vec![0.0; self.config.num_local_experts];
        }

        self.synthetic_router_logits(attention_signal, hidden_value)
    }

    fn project_logits(&self, hidden: &[f32]) -> Vec<f32> {
        let last_hidden = hidden.last().copied().unwrap_or(0.0);
        (0..self.config.vocab_size)
            .map(|vocab_index| {
                let basis = (vocab_index % 7 + 1) as f32;
                last_hidden * basis
            })
            .collect()
    }

    fn has_explicit_dense_projection(&self) -> bool {
        !self.config.token_embedding_rows.is_empty()
            && !self.config.final_norm_weight.is_empty()
            && !self.config.lm_head_rows.is_empty()
    }

    fn forward_explicit_dense(
        &self,
        input: ReferenceInput,
        cache_layout: CacheLayout,
        cache: CacheState,
    ) -> Result<ReferenceOutput, ReferenceError> {
        let token_count = input.tokens.len();
        let mut hidden = input
            .tokens
            .iter()
            .map(|token| self.embedding_row(*token))
            .collect::<Vec<_>>();
        let mut layers = Vec::with_capacity(self.config.num_layers);
        let position_ids = (0..token_count)
            .map(|offset| input.seq_start_pos + offset as u32)
            .collect::<Vec<_>>();
        let total_sequence_tokens = input.seq_start_pos as usize + token_count;

        for layer_index in 0..self.config.num_layers {
            let layer_type = self.layer_type(layer_index);
            let query_index = total_sequence_tokens.saturating_sub(1);
            let layer_attention =
                self.attention_trace_for(layer_type, query_index, total_sequence_tokens);
            let layer_moe = if self.layer_has_moe(layer_index) {
                let routes = hidden
                    .iter()
                    .map(|row| {
                        route_top_k(
                            &self.router_logits(0.0, row.first().copied().unwrap_or(0.0)),
                            self.effective_top_k(),
                        )
                    })
                    .collect::<Vec<_>>();
                if !self.config.expert_output_rows.is_empty() {
                    for (token_index, token_routes) in routes.iter().enumerate() {
                        for (expert_index, weight) in token_routes {
                            if let Some(expert_row) =
                                self.config.expert_output_rows.get(*expert_index)
                            {
                                for (hidden_value, bias) in
                                    hidden[token_index].iter_mut().zip(expert_row.iter())
                                {
                                    *hidden_value += bias * *weight;
                                }
                            }
                        }
                    }
                }
                MoeTrace::sparse(&routes)
            } else {
                MoeTrace::dense_only()
            };
            layers.push(LayerTrace {
                layer_index,
                input_tokens: token_count,
                output_tokens: token_count,
                position_ids: position_ids.clone(),
                attention: layer_attention,
                moe: layer_moe,
            });
        }

        let attention = layers
            .last()
            .map(|layer| layer.attention.clone())
            .unwrap_or_else(|| AttentionTrace::full(Vec::new()));
        let moe = layers
            .last()
            .map(|layer| layer.moe.clone())
            .unwrap_or_else(MoeTrace::dense_only);
        let trace = ReferenceTrace {
            phase: input.phase,
            seq_start_pos: input.seq_start_pos,
            layers,
            attention,
            moe,
            cache_layout,
            cache,
        };

        let normalized = hidden
            .iter_mut()
            .map(|row| self.rms_norm(row))
            .collect::<Vec<_>>();
        let logits = normalized
            .last()
            .map(|row| self.project_explicit_logits(row))
            .unwrap_or_else(|| vec![0.0; self.config.vocab_size]);

        Ok(ReferenceOutput { logits, trace })
    }

    fn embedding_row(&self, token: TokenId) -> Vec<f32> {
        self.config
            .token_embedding_rows
            .get(token as usize)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.config.final_norm_weight.len()])
    }

    fn rms_norm(&self, row: &[f32]) -> Vec<f32> {
        if row.is_empty() {
            return Vec::new();
        }
        let sum_sq = row.iter().map(|value| value * value).sum::<f32>();
        let inv_rms = (sum_sq / row.len() as f32 + self.config.rms_norm_eps)
            .sqrt()
            .recip();
        row.iter()
            .zip(self.config.final_norm_weight.iter())
            .map(|(value, weight)| f16::from_f32(value * inv_rms * weight).to_f32())
            .collect()
    }

    fn project_explicit_logits(&self, row: &[f32]) -> Vec<f32> {
        self.config
            .lm_head_rows
            .iter()
            .map(|weights| {
                row.iter()
                    .zip(weights.iter())
                    .map(|(value, weight)| value * weight)
                    .sum::<f32>()
            })
            .collect()
    }
}
