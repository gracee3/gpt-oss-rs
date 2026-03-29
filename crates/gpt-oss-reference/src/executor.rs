use crate::{
    attention::AttentionTrace,
    cache::{CacheLayout, CacheModelError, CacheState, CacheVisibility},
    moe::MoeTrace,
    trace::{LayerTrace, ReferencePhase, ReferenceTrace},
};
use gpt_oss_core::types::TokenId;
use gpt_oss_moe_semantics::route_top_k;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Config for the reference executor scaffold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    pub moe_layer_indices: Vec<usize>,
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
        let dense_zero_baseline = self.uses_dense_zero_baseline();
        let mut hidden = if dense_zero_baseline {
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
                let attention_signal = if dense_zero_baseline {
                    0.0
                } else {
                    self.synthetic_attention_signal(&visible, hidden[token_index])
                };
                let moe_signal = if self.layer_has_moe(layer_index) {
                    let logits = self.synthetic_router_logits(attention_signal, hidden[token_index]);
                    let routes = route_top_k(&logits, self.effective_top_k());
                    let signal = routes
                        .iter()
                        .map(|(expert, weight)| (*expert as f32 + 1.0) * *weight)
                        .sum::<f32>();
                    layer_routes.push(routes);
                    signal
                } else {
                    0.0
                };
                next_hidden[token_index] = hidden[token_index]
                    + attention_signal
                    + moe_signal
                    + (layer_index as f32 * 0.01);
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

    fn uses_dense_zero_baseline(&self) -> bool {
        self.config.sink_tokens == 0
            && self.config.sliding_window.is_none()
            && self.effective_top_k() == 0
            && self
                .config
                .layer_types
                .iter()
                .all(|layer| layer == "full_attention" || layer == "global_attention")
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
                attention_signal * ((expert_index + 1) as f32 * 0.1)
                    + hidden_value * (((expert_index % 3) + 1) as f32 * 0.05)
                    - expert_index as f32 * 0.02
            })
            .collect()
    }

    fn project_logits(&self, hidden: &[f32]) -> Vec<f32> {
        (0..self.config.vocab_size)
            .map(|vocab_index| {
                hidden
                    .iter()
                    .enumerate()
                    .map(|(token_index, value)| {
                        let basis = ((vocab_index + token_index) % 7 + 1) as f32;
                        value * basis
                    })
                    .sum::<f32>()
            })
            .collect()
    }
}
