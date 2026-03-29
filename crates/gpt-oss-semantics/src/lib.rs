//! Canonical GPT-OSS semantic model types.

use gpt_oss_core::types::{Dtype, SequenceId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Result alias for semantic model construction and validation.
pub type Result<T> = std::result::Result<T, SemanticError>;

/// Errors raised when constructing or validating GPT-OSS semantic metadata.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SemanticError {
    #[error("gpt-oss layer_types length mismatch: expected {expected}, got {got}")]
    LayerTypesLengthMismatch { expected: usize, got: usize },
    #[error("unsupported gpt-oss layer type: {layer_type}")]
    UnsupportedLayerType { layer_type: String },
}

/// High-level attention mode for a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionKind {
    Full,
    Sliding,
}

impl AttentionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            AttentionKind::Full => "full_attention",
            AttentionKind::Sliding => "sliding_attention",
        }
    }
}

/// Sink semantics associated with a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SinkBehavior {
    Disabled,
    Available,
}

/// Semantic description of attention for a single layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionSpec {
    pub kind: AttentionKind,
    pub sliding_window: Option<usize>,
    pub sink_behavior: SinkBehavior,
}

impl AttentionSpec {
    pub fn full() -> Self {
        Self {
            kind: AttentionKind::Full,
            sliding_window: None,
            sink_behavior: SinkBehavior::Disabled,
        }
    }

    pub fn sliding(sliding_window: Option<usize>) -> Self {
        Self {
            kind: AttentionKind::Sliding,
            sliding_window,
            sink_behavior: SinkBehavior::Available,
        }
    }

    pub fn from_layer_type(layer_type: &str, sliding_window: Option<usize>) -> Result<Self> {
        match layer_type {
            "full_attention" | "global_attention" => Ok(Self::full()),
            "sliding_attention" | "local_attention" => Ok(Self::sliding(sliding_window)),
            other => Err(SemanticError::UnsupportedLayerType {
                layer_type: other.to_string(),
            }),
        }
    }
}

/// How the model stores and combines MoE expert weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpertStorageSpec {
    Unquantized,
    QuantizedMxfp4,
    Missing,
}

/// Semantic description of the MoE routing contract for a layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeSpec {
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub storage: ExpertStorageSpec,
}

impl MoeSpec {
    pub fn enabled(&self) -> bool {
        self.num_local_experts > 0 && self.num_experts_per_tok > 0
    }
}

/// Semantic description of a single GPT-OSS transformer layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerSpec {
    pub index: usize,
    pub layer_type: String,
    pub attention: AttentionSpec,
    pub moe: MoeSpec,
}

/// Canonical semantic model of a GPT-OSS checkpoint/config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSpec {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub attn_logit_softcapping: f32,
    pub attention_bias: bool,
    pub sliding_window: Option<usize>,
    pub dtype: Dtype,
    pub architecture: String,
    pub layers: Vec<LayerSpec>,
}

impl ModelSpec {
    pub fn layer(&self, index: usize) -> Option<&LayerSpec> {
        self.layers.get(index)
    }

    pub fn num_sliding_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| matches!(layer.attention.kind, AttentionKind::Sliding))
            .count()
    }

    pub fn num_moe_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.moe.enabled())
            .count()
    }
}

/// Semantic state for one sequence in a request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceState {
    pub sequence_id: SequenceId,
    pub prompt_len: usize,
    pub generated_len: usize,
}

impl SequenceState {
    pub fn total_len(&self) -> usize {
        self.prompt_len + self.generated_len
    }
}

/// Semantic step kind for the execution trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepKind {
    Prefill,
    Decode,
}

/// Semantic state for a request step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepState {
    pub kind: StepKind,
    pub sequences: Vec<SequenceState>,
}

/// High-level execution trace rooted in the semantic model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub model: ModelSpec,
    pub steps: Vec<StepState>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpt_oss_core::types::SequenceId;

    #[test]
    fn parses_layer_attention_modes() {
        let full = AttentionSpec::from_layer_type("full_attention", None).unwrap();
        assert!(matches!(full.kind, AttentionKind::Full));
        assert!(matches!(full.sink_behavior, SinkBehavior::Disabled));

        let sliding = AttentionSpec::from_layer_type("sliding_attention", Some(128)).unwrap();
        assert!(matches!(sliding.kind, AttentionKind::Sliding));
        assert!(matches!(sliding.sink_behavior, SinkBehavior::Available));
        assert_eq!(sliding.sliding_window, Some(128));
    }

    #[test]
    fn sequence_state_reports_total_length() {
        let state = SequenceState {
            sequence_id: SequenceId(7),
            prompt_len: 12,
            generated_len: 3,
        };

        assert_eq!(state.total_len(), 15);
    }
}
