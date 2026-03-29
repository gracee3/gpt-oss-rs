//! Runtime policy helpers for model-family-specific serving behavior.

use gpt_oss_engine::RuntimeMode;

/// Practical default context cap for GPT-OSS on 24 GB consumer GPUs.
pub const GPT_OSS_CONSUMER_MAX_MODEL_LEN: usize = 8192;
/// Default KV/cache allocation target for the consumer GPT-OSS profile.
pub const GPT_OSS_CONSUMER_GPU_MEMORY_UTILIZATION: f32 = 0.90;
/// VRAM threshold used to identify 24 GB-class consumer cards.
pub const GPT_OSS_CONSUMER_MAX_VRAM_BYTES: usize = 24 * 1024 * 1024 * 1024;

/// Backend path selected for serving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeBackendPath {
    /// CUDA-backed execution.
    Cuda,
    /// CPU/mock fallback execution.
    Mock,
}

impl RuntimeBackendPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Mock => "mock",
        }
    }
}

/// Resolved runtime decision for a request or server session.
#[derive(Debug, Clone)]
pub struct RuntimeDecision {
    pub runtime_mode: RuntimeMode,
    pub backend_path: RuntimeBackendPath,
    pub reason: String,
}

impl RuntimeDecision {
    pub fn backend_label(&self) -> &'static str {
        self.backend_path.as_str()
    }

    pub fn summary(&self) -> String {
        format!(
            "mode={:?}, backend={}, reason={}",
            self.runtime_mode,
            self.backend_label(),
            self.reason
        )
    }
}

/// Whether the model name targets the GPT-OSS family.
pub fn is_gpt_oss_model(model_name: &str) -> bool {
    let model_name = model_name.to_ascii_lowercase();
    model_name.contains("gpt-oss") || model_name.contains("gpt_oss")
}

fn validate_cuda_gpt_oss_runtime(
    max_model_len: usize,
    tensor_parallel_size: usize,
    primary_gpu_total_memory: Option<usize>,
    allow_long_context_override: bool,
) -> Result<(), String> {
    if tensor_parallel_size > 1 {
        return Err(format!(
            "gpt-oss CUDA serving does not support tensor_parallel_size={} yet: the live engine still executes rank 0 only",
            tensor_parallel_size
        ));
    }

    if allow_long_context_override {
        return Ok(());
    }

    if let Some(total_memory) = primary_gpu_total_memory {
        if total_memory <= GPT_OSS_CONSUMER_MAX_VRAM_BYTES
            && max_model_len > GPT_OSS_CONSUMER_MAX_MODEL_LEN
        {
            return Err(format!(
                "gpt-oss on 24 GB-class GPUs defaults to max_model_len <= {}; requested {}. Lower --max-model-len or set GPT_OSS_RS_ALLOW_LONG_CONTEXT=1 to override.",
                GPT_OSS_CONSUMER_MAX_MODEL_LEN,
                max_model_len
            ));
        }
    }

    Ok(())
}

/// Resolve the runtime path for GPT-OSS serving and fail unsupported trusted-mode combinations early.
///
/// In `trusted` mode, GPT-OSS requires a CUDA-capable GPU and the selected CUDA
/// path must satisfy the existing phase-boundary constraints.
/// In `experimental` mode, the existing mock fallback remains available.
pub fn validate_gpt_oss_runtime(
    model_name: &str,
    runtime_mode: RuntimeMode,
    gpu_available: bool,
    max_model_len: usize,
    tensor_parallel_size: usize,
    primary_gpu_total_memory: Option<usize>,
    allow_long_context_override: bool,
) -> Result<RuntimeDecision, String> {
    if !is_gpt_oss_model(model_name) {
        let backend_path = if gpu_available {
            RuntimeBackendPath::Cuda
        } else {
            RuntimeBackendPath::Mock
        };
        let reason = match backend_path {
            RuntimeBackendPath::Cuda => {
                "non-GPT-OSS model using CUDA backend because a GPU is available"
            }
            RuntimeBackendPath::Mock => {
                "non-GPT-OSS model using mock backend because no GPU was detected"
            }
        };
        return Ok(RuntimeDecision {
            runtime_mode,
            backend_path,
            reason: reason.into(),
        });
    }

    if runtime_mode == RuntimeMode::Trusted {
        if !gpu_available {
            return Err(
                "trusted GPT-OSS mode requires a CUDA-capable GPU; no GPU was detected".into(),
            );
        }

        validate_cuda_gpt_oss_runtime(
            max_model_len,
            tensor_parallel_size,
            primary_gpu_total_memory,
            allow_long_context_override,
        )?;

        return Ok(RuntimeDecision {
            runtime_mode,
            backend_path: RuntimeBackendPath::Cuda,
            reason: "trusted GPT-OSS mode selected the CUDA backend after passing planning checks"
                .into(),
        });
    }

    if gpu_available {
        validate_cuda_gpt_oss_runtime(
            max_model_len,
            tensor_parallel_size,
            primary_gpu_total_memory,
            allow_long_context_override,
        )?;

        Ok(RuntimeDecision {
            runtime_mode,
            backend_path: RuntimeBackendPath::Cuda,
            reason: "experimental GPT-OSS mode selected the CUDA backend".into(),
        })
    } else {
        Ok(RuntimeDecision {
            runtime_mode,
            backend_path: RuntimeBackendPath::Mock,
            reason: "experimental GPT-OSS mode fell back to the mock backend because no GPU was detected".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpt_oss_engine::RuntimeMode;

    #[test]
    fn detects_gpt_oss_model_names() {
        assert!(is_gpt_oss_model("openai/gpt-oss-20b"));
        assert!(is_gpt_oss_model("OPENAI/GPT_OSS_20B"));
        assert!(!is_gpt_oss_model("/models/local-checkpoint"));
    }

    #[test]
    fn rejects_tensor_parallel_for_gpt_oss_runtime() {
        let err = validate_gpt_oss_runtime(
            "openai/gpt-oss-20b",
            RuntimeMode::Experimental,
            true,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN,
            2,
            Some(GPT_OSS_CONSUMER_MAX_VRAM_BYTES),
            false,
        )
        .unwrap_err();
        assert!(err.contains("tensor_parallel_size=2"));
    }

    #[test]
    fn rejects_long_context_on_24gb_cards_without_override() {
        let err = validate_gpt_oss_runtime(
            "openai/gpt-oss-20b",
            RuntimeMode::Experimental,
            true,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN + 1,
            1,
            Some(GPT_OSS_CONSUMER_MAX_VRAM_BYTES),
            false,
        )
        .unwrap_err();
        assert!(err.contains("max_model_len"));
        assert!(err.contains("GPT_OSS_RS_ALLOW_LONG_CONTEXT=1"));
    }

    #[test]
    fn allows_long_context_override() {
        let decision = validate_gpt_oss_runtime(
            "openai/gpt-oss-20b",
            RuntimeMode::Experimental,
            true,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN + 4096,
            1,
            Some(GPT_OSS_CONSUMER_MAX_VRAM_BYTES),
            true,
        )
        .unwrap();
        assert_eq!(decision.backend_path, RuntimeBackendPath::Cuda);
    }

    #[test]
    fn ignores_non_gpt_oss_names() {
        let decision = validate_gpt_oss_runtime(
            "/models/local-checkpoint",
            RuntimeMode::Experimental,
            false,
            32768,
            4,
            Some(8),
            false,
        )
        .unwrap();
        assert_eq!(decision.backend_path, RuntimeBackendPath::Mock);
    }

    #[test]
    fn trusted_mode_rejects_missing_gpu_for_gpt_oss() {
        let err = validate_gpt_oss_runtime(
            "openai/gpt-oss-20b",
            RuntimeMode::Trusted,
            false,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN,
            1,
            Some(GPT_OSS_CONSUMER_MAX_VRAM_BYTES),
            false,
        )
        .unwrap_err();
        assert!(err.contains("trusted GPT-OSS mode requires a CUDA-capable GPU"));
    }

    #[test]
    fn experimental_mode_falls_back_to_mock_when_no_gpu_is_available() {
        let decision = validate_gpt_oss_runtime(
            "openai/gpt-oss-20b",
            RuntimeMode::Experimental,
            false,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN,
            1,
            None,
            false,
        )
        .unwrap();
        assert_eq!(decision.backend_path, RuntimeBackendPath::Mock);
        assert!(decision.reason.contains("mock backend"));
    }
}
