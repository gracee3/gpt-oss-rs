//! ModelRunner: orchestrates the forward pass.

use std::sync::Arc;

use gpt_oss_core::types::Dtype;
use gpt_oss_semantics::{
    AttentionSpec, ExpertStorageSpec, LayerSpec, ModelSpec, MoeSpec, SemanticError,
};
use tracing::debug;

use crate::architectures::{create_model, Architecture};
use crate::bridge::{
    AttentionBackend, CacheEngine, GpuAllocator, GpuBuffer, LLMError, ModelWeights, Result,
};
use crate::input::ModelInput;

/// Static configuration for the model runner, derived from the model config.
#[derive(Debug, Clone)]
pub struct ModelRunnerConfig {
    pub tensor_parallel_rank: usize,
    pub tensor_parallel_size: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position: usize,
    pub initial_context_length: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: f32,
    pub rope_ntk_alpha: f32,
    pub rope_ntk_beta: f32,
    pub rope_scaling_truncate: bool,
    pub partial_rotary_factor: f32,
    pub attn_logit_softcapping: f32,
    pub attention_bias: bool,
    pub sliding_window: Option<usize>,
    pub layer_types: Vec<String>,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub dtype: Dtype,
    pub architecture: String,
}

impl ModelRunnerConfig {
    /// Build the canonical GPT-OSS semantic model description for this runner config.
    pub fn semantic_model_spec(&self) -> gpt_oss_semantics::Result<ModelSpec> {
        let layer_types = if self.layer_types.is_empty() {
            vec!["full_attention".to_string(); self.num_layers]
        } else {
            self.layer_types.clone()
        };

        if layer_types.len() != self.num_layers {
            return Err(SemanticError::LayerTypesLengthMismatch {
                expected: self.num_layers,
                got: layer_types.len(),
            });
        }

        let layers = layer_types
            .iter()
            .enumerate()
            .map(|(index, layer_type)| {
                let attention = AttentionSpec::from_layer_type(layer_type, self.sliding_window)?;
                Ok(LayerSpec {
                    index,
                    layer_type: layer_type.clone(),
                    attention,
                    moe: MoeSpec {
                        num_local_experts: self.num_local_experts,
                        num_experts_per_tok: self.num_experts_per_tok,
                        storage: ExpertStorageSpec::Unquantized,
                    },
                })
            })
            .collect::<gpt_oss_semantics::Result<Vec<_>>>()?;

        Ok(ModelSpec {
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            max_position: self.max_position,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            partial_rotary_factor: self.partial_rotary_factor,
            attn_logit_softcapping: self.attn_logit_softcapping,
            attention_bias: self.attention_bias,
            sliding_window: self.sliding_window,
            dtype: self.dtype,
            architecture: self.architecture.clone(),
            layers,
        })
    }
}

/// Drives the transformer forward pass: embed -> layers -> LM head -> logits.
pub struct ModelRunner {
    pub config: ModelRunnerConfig,
    model: Box<dyn Architecture>,
    attention: Box<dyn AttentionBackend>,
    cache: Arc<CacheEngine>,
    #[allow(dead_code)]
    gpu: Arc<dyn GpuAllocator>,
}

impl ModelRunner {
    pub fn new(
        weights: ModelWeights,
        config: ModelRunnerConfig,
        attention: Box<dyn AttentionBackend>,
        cache: Arc<CacheEngine>,
        gpu: Arc<dyn GpuAllocator>,
    ) -> Result<Self> {
        debug!(arch = %config.architecture, "creating model runner");
        let model = create_model(&config.architecture, weights, &config)?;
        Ok(Self {
            config,
            model,
            attention,
            cache,
            gpu,
        })
    }

    /// Execute a single forward pass, returning logits [batch, vocab].
    pub fn execute_model(&self, input: ModelInput) -> Result<GpuBuffer<f32>> {
        debug!(
            num_tokens = input.num_tokens(),
            is_prefill = input.is_prefill,
            "execute_model"
        );

        if input.token_ids.is_empty() {
            return Err(LLMError::ModelError("empty input".into()));
        }

        self.model
            .forward(&input, &self.cache, self.attention.as_ref())
    }
}
