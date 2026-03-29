//! Runtime policy helpers for model-family-specific serving behavior.

/// Practical default context cap for GPT-OSS on 24 GB consumer GPUs.
pub const GPT_OSS_CONSUMER_MAX_MODEL_LEN: usize = 8192;
/// Default KV/cache allocation target for the consumer GPT-OSS profile.
pub const GPT_OSS_CONSUMER_GPU_MEMORY_UTILIZATION: f32 = 0.90;
/// VRAM threshold used to identify 24 GB-class consumer cards.
pub const GPT_OSS_CONSUMER_MAX_VRAM_BYTES: usize = 24 * 1024 * 1024 * 1024;

/// Whether the model name targets the GPT-OSS family.
pub fn is_gpt_oss_model(model_name: &str) -> bool {
    let model_name = model_name.to_ascii_lowercase();
    model_name.contains("gpt-oss") || model_name.contains("gpt_oss")
}

/// Validate runtime assumptions for GPT-OSS serving.
///
/// This keeps the current fork honest:
/// - the live CUDA engine is still single-rank, so TP > 1 must fail clearly
/// - 24 GB cards default to an 8k context budget unless the user opts out
pub fn validate_gpt_oss_runtime(
    model_name: &str,
    max_model_len: usize,
    tensor_parallel_size: usize,
    primary_gpu_total_memory: Option<usize>,
    allow_long_context_override: bool,
) -> Result<(), String> {
    if !is_gpt_oss_model(model_name) {
        return Ok(());
    }

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
                "gpt-oss on 24 GB-class GPUs defaults to max_model_len <= {}; requested {}. Lower --max-model-len or set RVLLM_ALLOW_LONG_CONTEXT=1 to override.",
                GPT_OSS_CONSUMER_MAX_MODEL_LEN,
                max_model_len
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_gpt_oss_model_names() {
        assert!(is_gpt_oss_model("openai/gpt-oss-20b"));
        assert!(is_gpt_oss_model("OPENAI/GPT_OSS_20B"));
        assert!(!is_gpt_oss_model("Qwen/Qwen2.5-7B-Instruct"));
    }

    #[test]
    fn rejects_tensor_parallel_for_gpt_oss_runtime() {
        let err = validate_gpt_oss_runtime(
            "openai/gpt-oss-20b",
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
            GPT_OSS_CONSUMER_MAX_MODEL_LEN + 1,
            1,
            Some(GPT_OSS_CONSUMER_MAX_VRAM_BYTES),
            false,
        )
        .unwrap_err();
        assert!(err.contains("max_model_len"));
        assert!(err.contains("RVLLM_ALLOW_LONG_CONTEXT=1"));
    }

    #[test]
    fn allows_long_context_override() {
        validate_gpt_oss_runtime(
            "openai/gpt-oss-20b",
            GPT_OSS_CONSUMER_MAX_MODEL_LEN + 4096,
            1,
            Some(GPT_OSS_CONSUMER_MAX_VRAM_BYTES),
            true,
        )
        .unwrap();
    }

    #[test]
    fn ignores_non_gpt_oss_models() {
        validate_gpt_oss_runtime("meta-llama/Llama-3.1-8B", 32768, 4, Some(8), false).unwrap();
    }
}
