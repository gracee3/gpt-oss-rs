//! Worker configuration.

use gpt_oss_core::prelude::{LLMError, Result};
use gpt_oss_core::types::Dtype;
use gpt_oss_runtime_plan::RuntimeMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorParallelDims {
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
}

/// Configuration for a single GPU worker instance.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Model name or repo id used for trust-tier and planner decisions.
    pub model_name: String,
    /// Runtime trust tier for this worker.
    pub runtime_mode: RuntimeMode,
    /// GPU device ordinal this worker owns.
    pub device_id: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV attention heads.
    pub num_kv_heads: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Intermediate (FFN) size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_model_len: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Tokens per KV cache block.
    pub block_size: usize,
    /// Fraction of GPU memory to use for KV cache.
    pub gpu_memory_utilization: f32,
    /// Tensor parallel rank of this worker.
    pub rank: usize,
    /// Tensor parallel world size.
    pub tensor_parallel_size: usize,
    /// Pipeline parallel world size.
    pub pipeline_parallel_size: usize,
    /// Model architecture name (for this fork, `GptOssForCausalLM`).
    pub architecture: String,
    /// Data type for model weights and compute.
    pub dtype: Dtype,
    /// RoPE theta parameter.
    pub rope_theta: f32,
    /// YaRN scaling factor. 1.0 disables scaling.
    pub rope_scaling_factor: f32,
    /// Original context length used to derive YaRN ramp bounds.
    pub rope_initial_context_length: usize,
    /// YaRN low-frequency bound parameter.
    pub rope_ntk_alpha: f32,
    /// YaRN high-frequency bound parameter.
    pub rope_ntk_beta: f32,
    /// Fraction of head_dim that gets RoPE (Phi: 0.5, others: 1.0).
    pub partial_rotary_factor: f32,
    /// Soft-capping value for attention logits (Gemma 2). 0.0 = disabled.
    pub attn_logit_softcapping: f32,
    /// Whether attention projections include biases.
    pub attention_bias: bool,
    /// Sliding-window size for architectures that alternate local/global attention.
    pub sliding_window: Option<usize>,
    /// Per-layer attention mode names from the HF config.
    pub layer_types: Vec<String>,
    /// Number of MoE experts (Mixtral: 8, DeepSeek: 64). 0 = dense.
    pub num_local_experts: usize,
    /// Number of experts activated per token (Mixtral: 2, DeepSeek: 6).
    pub num_experts_per_tok: usize,
    /// KV cache data type: "auto", "fp8", "fp8_e4m3".
    pub kv_cache_dtype: String,
    /// Enable prefix caching.
    pub enable_prefix_caching: bool,
}

impl WorkerConfig {
    /// Clone this worker config for a specific TP rank / device.
    pub fn for_rank(&self, rank: usize, device_id: usize) -> Result<Self> {
        if rank >= self.tensor_parallel_size {
            return Err(LLMError::ConfigError(format!(
                "rank {} >= tensor_parallel_size {}",
                rank, self.tensor_parallel_size
            )));
        }

        let mut cfg = self.clone();
        cfg.rank = rank;
        cfg.device_id = device_id;
        Ok(cfg)
    }

    /// Per-rank dimensions after tensor parallel sharding.
    pub fn tensor_parallel_dims(&self) -> Result<TensorParallelDims> {
        if self.tensor_parallel_size <= 1 {
            return Ok(TensorParallelDims {
                num_attention_heads: self.num_attention_heads,
                num_kv_heads: self.num_kv_heads,
                intermediate_size: self.intermediate_size,
            });
        }

        if self.rank >= self.tensor_parallel_size {
            return Err(LLMError::ConfigError(format!(
                "rank {} >= tensor_parallel_size {}",
                self.rank, self.tensor_parallel_size
            )));
        }

        let tp = self.tensor_parallel_size;
        let shard = |label: &str, total: usize| -> Result<usize> {
            if total % tp != 0 {
                return Err(LLMError::ConfigError(format!(
                    "{}={} is not divisible by tensor_parallel_size={}",
                    label, total, tp
                )));
            }
            Ok(total / tp)
        };

        Ok(TensorParallelDims {
            num_attention_heads: shard("num_attention_heads", self.num_attention_heads)?,
            num_kv_heads: shard("num_kv_heads", self.num_kv_heads)?,
            intermediate_size: shard("intermediate_size", self.intermediate_size)?,
        })
    }

    /// Build a `ModelRunnerConfig` from this worker config.
    pub fn model_runner_config(&self) -> Result<gpt_oss_model_runner::ModelRunnerConfig> {
        let tp_dims = self.tensor_parallel_dims()?;
        Ok(gpt_oss_model_runner::ModelRunnerConfig {
            tensor_parallel_rank: self.rank,
            tensor_parallel_size: self.tensor_parallel_size,
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            num_heads: tp_dims.num_attention_heads,
            num_kv_heads: tp_dims.num_kv_heads,
            head_dim: self.head_dim,
            intermediate_size: tp_dims.intermediate_size,
            vocab_size: self.vocab_size,
            max_position: self.max_model_len,
            rms_norm_eps: self.rms_norm_eps,
            dtype: self.dtype,
            architecture: self.architecture.clone(),
            rope_theta: self.rope_theta,
            rope_scaling_factor: self.rope_scaling_factor,
            rope_initial_context_length: self.rope_initial_context_length,
            rope_ntk_alpha: self.rope_ntk_alpha,
            rope_ntk_beta: self.rope_ntk_beta,
            partial_rotary_factor: self.partial_rotary_factor,
            attn_logit_softcapping: self.attn_logit_softcapping,
            attention_bias: self.attention_bias,
            sliding_window: self.sliding_window,
            layer_types: self.layer_types.clone(),
            num_local_experts: self.num_local_experts,
            num_experts_per_tok: self.num_experts_per_tok,
        })
    }

    /// Build a `CacheConfig` from this worker config.
    pub fn cache_config(&self) -> Result<gpt_oss_model_runner::kv_cache::CacheConfig> {
        let tp_dims = self.tensor_parallel_dims()?;
        Ok(gpt_oss_model_runner::kv_cache::CacheConfig::new(
            self.num_layers,
            tp_dims.num_kv_heads,
            self.head_dim,
            self.block_size,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> WorkerConfig {
        WorkerConfig {
            model_name: "openai/gpt-oss-20b".into(),
            runtime_mode: RuntimeMode::Experimental,
            device_id: 0,
            num_layers: 24,
            num_kv_heads: 8,
            head_dim: 128,
            hidden_size: 1024,
            num_attention_heads: 16,
            intermediate_size: 4096,
            vocab_size: 32000,
            max_model_len: 8192,
            rms_norm_eps: 1e-5,
            block_size: 16,
            gpu_memory_utilization: 0.9,
            rank: 0,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            architecture: "GptOssForCausalLM".into(),
            dtype: Dtype::Float16,
            rope_theta: 10000.0,
            rope_scaling_factor: 1.0,
            rope_initial_context_length: 4096,
            rope_ntk_alpha: 1.0,
            rope_ntk_beta: 32.0,
            partial_rotary_factor: 1.0,
            attn_logit_softcapping: 0.0,
            attention_bias: false,
            sliding_window: None,
            layer_types: Vec::new(),
            num_local_experts: 32,
            num_experts_per_tok: 4,
            kv_cache_dtype: "auto".into(),
            enable_prefix_caching: false,
        }
    }

    #[test]
    fn tensor_parallel_dims_stay_global_for_single_rank() {
        let cfg = make_config();
        let dims = cfg.tensor_parallel_dims().unwrap();
        assert_eq!(
            dims,
            TensorParallelDims {
                num_attention_heads: 16,
                num_kv_heads: 8,
                intermediate_size: 4096,
            }
        );
    }

    #[test]
    fn tensor_parallel_dims_are_sharded_per_rank() {
        let mut cfg = make_config();
        cfg.tensor_parallel_size = 4;
        cfg.rank = 2;
        let dims = cfg.tensor_parallel_dims().unwrap();
        assert_eq!(
            dims,
            TensorParallelDims {
                num_attention_heads: 4,
                num_kv_heads: 2,
                intermediate_size: 1024,
            }
        );
    }

    #[test]
    fn tensor_parallel_dims_require_divisible_shapes() {
        let mut cfg = make_config();
        cfg.tensor_parallel_size = 3;
        let err = cfg.tensor_parallel_dims().unwrap_err().to_string();
        assert!(err.contains("num_attention_heads"));
    }

    #[test]
    fn for_rank_updates_rank_and_device() {
        let mut cfg = make_config();
        cfg.tensor_parallel_size = 2;
        let rank1 = cfg.for_rank(1, 3).unwrap();
        assert_eq!(rank1.rank, 1);
        assert_eq!(rank1.device_id, 3);
    }
}
