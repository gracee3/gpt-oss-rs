//! Model architecture implementations.

pub mod gpt_oss;
mod shared;

use crate::bridge::{AttentionBackend, CacheEngine, GpuBuffer, LLMError, ModelWeights, Result};
use crate::input::ModelInput;
use crate::runner::ModelRunnerConfig;

/// Trait for a causal LM architecture.
pub trait Architecture: Send + Sync {
    /// Run the full forward pass: embed -> layers -> LM head -> logits.
    fn forward(
        &self,
        input: &ModelInput,
        cache: &CacheEngine,
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>>;
}

/// Factory function to instantiate a model architecture by name.
pub fn create_model(
    architecture: &str,
    weights: ModelWeights,
    config: &ModelRunnerConfig,
) -> Result<Box<dyn Architecture>> {
    match architecture {
        "GptOssForCausalLM" => Ok(Box::new(gpt_oss::GptOssForCausalLM::new(weights, config)?)),
        other => Err(LLMError::ModelError(format!(
            "unsupported architecture for this fork: {other}. Only GptOssForCausalLM is supported."
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::ModelRunnerConfig;
    use gpt_oss_core::types::Dtype;

    fn test_config() -> ModelRunnerConfig {
        ModelRunnerConfig {
            tensor_parallel_rank: 0,
            tensor_parallel_size: 1,
            num_layers: 1,
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            vocab_size: 32,
            max_position: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling_factor: 1.0,
            rope_initial_context_length: 128,
            rope_ntk_alpha: 1.0,
            rope_ntk_beta: 32.0,
            partial_rotary_factor: 1.0,
            attn_logit_softcapping: 0.0,
            attention_bias: false,
            sliding_window: None,
            layer_types: vec!["full_attention".into()],
            num_local_experts: 2,
            num_experts_per_tok: 1,
            dtype: Dtype::Float16,
            architecture: "GptOssForCausalLM".into(),
        }
    }

    #[test]
    fn rejects_non_gpt_oss_architectures() {
        let err = match create_model("LlamaForCausalLM", ModelWeights::default(), &test_config()) {
            Ok(_) => panic!("non-GPT-OSS architectures should be rejected"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("Only GptOssForCausalLM is supported"));
    }

    #[test]
    fn accepts_gpt_oss_architecture() {
        create_model("GptOssForCausalLM", ModelWeights::default(), &test_config()).unwrap();
    }
}
