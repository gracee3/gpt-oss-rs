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
    /// Native batched GPT-OSS CPU execution.
    Cpu,
    /// Native CPU execution with bounded Xe expert projections.
    CpuXe,
    /// Explicit test-only mock execution.
    Mock,
}

impl RuntimeBackendPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
            Self::CpuXe => "cpu_xe",
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
    let normalized = model_name.to_ascii_lowercase();
    if normalized.contains("gpt-oss") || normalized.contains("gpt_oss") {
        return true;
    }

    let config_path = std::path::Path::new(model_name).join("config.json");
    std::fs::read(&config_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|config| config.get("architectures")?.as_array().cloned())
        .is_some_and(|architectures| {
            architectures
                .iter()
                .any(|architecture| architecture.as_str() == Some("GptOssForCausalLM"))
        })
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

/// Resolve explicit or automatic device selection and reject unsupported
/// model/backend combinations before any weights are loaded.
#[allow(clippy::too_many_arguments)]
pub fn validate_gpt_oss_runtime(
    model_name: &str,
    runtime_mode: RuntimeMode,
    requested_device: &str,
    gpu_available: bool,
    max_model_len: usize,
    tensor_parallel_size: usize,
    pipeline_parallel_size: usize,
    max_num_seqs: usize,
    primary_gpu_total_memory: Option<usize>,
    allow_long_context_override: bool,
) -> Result<RuntimeDecision, String> {
    validate_gpt_oss_runtime_with_xe(
        model_name,
        runtime_mode,
        requested_device,
        gpu_available,
        false,
        max_model_len,
        tensor_parallel_size,
        pipeline_parallel_size,
        max_num_seqs,
        primary_gpu_total_memory,
        allow_long_context_override,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn validate_gpt_oss_runtime_with_xe(
    model_name: &str,
    runtime_mode: RuntimeMode,
    requested_device: &str,
    gpu_available: bool,
    xe_auto_available: bool,
    max_model_len: usize,
    tensor_parallel_size: usize,
    pipeline_parallel_size: usize,
    max_num_seqs: usize,
    primary_gpu_total_memory: Option<usize>,
    allow_long_context_override: bool,
) -> Result<RuntimeDecision, String> {
    let is_gpt_oss = is_gpt_oss_model(model_name);
    let backend_path = match requested_device {
        "auto" if is_gpt_oss && xe_auto_available => RuntimeBackendPath::CpuXe,
        "auto" if is_gpt_oss => RuntimeBackendPath::Cpu,
        "auto" => {
            return Err("automatic serving only supports GPT-OSS models; select an explicit experimental backend for other model families".into())
        }
        "cuda" if runtime_mode == RuntimeMode::Trusted => {
            return Err("trusted mode rejects CUDA serving; use --runtime-mode experimental".into())
        }
        "cuda" if gpu_available => RuntimeBackendPath::Cuda,
        "cuda" => return Err("CUDA was requested but no usable CUDA device was detected".into()),
        "cpu" if is_gpt_oss => RuntimeBackendPath::Cpu,
        "cpu" => return Err("native CPU serving only supports GPT-OSS models".into()),
        "xe" if !cfg!(feature = "xe") => {
            return Err("Xe was requested but this server was built without the 'xe' feature".into())
        }
        "xe" if is_gpt_oss => RuntimeBackendPath::CpuXe,
        "xe" => return Err("CPU+Xe serving only supports GPT-OSS models".into()),
        "mock" => RuntimeBackendPath::Mock,
        other => {
            return Err(format!(
                "unknown device '{other}': expected auto, cpu, xe, cuda, or mock"
            ))
        }
    };

    match backend_path {
        RuntimeBackendPath::Cuda => {
            if is_gpt_oss {
                validate_cuda_gpt_oss_runtime(
                    max_model_len,
                    tensor_parallel_size,
                    primary_gpu_total_memory,
                    allow_long_context_override,
                )?;
            }
            Ok(RuntimeDecision {
                runtime_mode,
                backend_path,
                reason: "explicit experimental CUDA execution selected".into(),
            })
        }
        RuntimeBackendPath::Cpu | RuntimeBackendPath::CpuXe => {
            if runtime_mode == RuntimeMode::Trusted {
                return Err(
                    "trusted GPT-OSS CPU serving is blocked until the final i7 conformance gate"
                        .into(),
                );
            }
            if tensor_parallel_size != 1 || pipeline_parallel_size != 1 {
                return Err(format!(
                    "CPU serving requires tensor_parallel_size=1 and pipeline_parallel_size=1; got {tensor_parallel_size} and {pipeline_parallel_size}"
                ));
            }
            Ok(RuntimeDecision {
                runtime_mode,
                backend_path,
                reason: if backend_path == RuntimeBackendPath::CpuXe && requested_device == "auto" {
                    "auto selected the promoted GPT-OSS CPU+Xe hybrid backend".into()
                } else if backend_path == RuntimeBackendPath::CpuXe {
                    "explicit GPT-OSS CPU+Xe hybrid execution selected".into()
                } else if max_num_seqs > 1 {
                    format!(
                        "experimental native GPT-OSS CPU batching selected with max_num_seqs={max_num_seqs}"
                    )
                } else if requested_device == "auto" {
                    "auto selected the native GPT-OSS CPU backend".into()
                } else {
                    "explicit native GPT-OSS CPU execution selected".into()
                },
            })
        }
        RuntimeBackendPath::Mock => {
            if runtime_mode == RuntimeMode::Trusted && is_gpt_oss {
                return Err("trusted GPT-OSS serving rejects the mock backend".into());
            }
            Ok(RuntimeDecision {
                runtime_mode,
                backend_path,
                reason: "explicit test-only mock execution selected".into(),
            })
        }
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
    fn detects_local_gpt_oss_snapshot_from_config() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(
            temp.path().join("config.json"),
            br#"{"architectures":["GptOssForCausalLM"]}"#,
        )
        .unwrap();
        assert!(is_gpt_oss_model(temp.path().to_str().unwrap()));
    }

    #[test]
    fn rejects_tensor_parallel_for_gpt_oss_runtime() {
        let err = validate_gpt_oss_runtime(
            "openai/gpt-oss-20b",
            RuntimeMode::Experimental,
            "cuda",
            true,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN,
            2,
            1,
            256,
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
            "cuda",
            true,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN + 1,
            1,
            1,
            256,
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
            "cuda",
            true,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN + 4096,
            1,
            1,
            256,
            Some(GPT_OSS_CONSUMER_MAX_VRAM_BYTES),
            true,
        )
        .unwrap();
        assert_eq!(decision.backend_path, RuntimeBackendPath::Cuda);
    }

    #[test]
    fn explicit_mock_is_the_only_no_cuda_non_gpt_oss_fallback() {
        let decision = validate_gpt_oss_runtime(
            "/models/local-checkpoint",
            RuntimeMode::Experimental,
            "mock",
            false,
            32768,
            4,
            2,
            256,
            Some(8),
            false,
        )
        .unwrap();
        assert_eq!(decision.backend_path, RuntimeBackendPath::Mock);

        let error = validate_gpt_oss_runtime(
            "/models/local-checkpoint",
            RuntimeMode::Experimental,
            "auto",
            false,
            32768,
            1,
            1,
            1,
            None,
            false,
        )
        .unwrap_err();
        assert!(error.contains("only supports GPT-OSS"));
    }

    #[test]
    fn trusted_mode_rejects_cpu_first_auto_for_gpt_oss() {
        let err = validate_gpt_oss_runtime(
            "openai/gpt-oss-20b",
            RuntimeMode::Trusted,
            "auto",
            false,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN,
            1,
            1,
            1,
            Some(GPT_OSS_CONSUMER_MAX_VRAM_BYTES),
            false,
        )
        .unwrap_err();
        assert!(err.contains("final i7 conformance gate"));
    }

    #[test]
    fn experimental_auto_uses_real_cpu_with_or_without_a_gpu() {
        for gpu_available in [false, true] {
            let decision = validate_gpt_oss_runtime(
                "openai/gpt-oss-20b",
                RuntimeMode::Experimental,
                "auto",
                gpu_available,
                GPT_OSS_CONSUMER_MAX_MODEL_LEN,
                1,
                1,
                1,
                Some(GPT_OSS_CONSUMER_MAX_VRAM_BYTES),
                false,
            )
            .unwrap();
            assert_eq!(decision.backend_path, RuntimeBackendPath::Cpu);
            assert!(decision.reason.contains("native GPT-OSS CPU backend"));
        }
    }

    #[test]
    fn promoted_auto_selects_cpu_xe_without_considering_cuda() {
        for gpu_available in [false, true] {
            let decision = validate_gpt_oss_runtime_with_xe(
                "openai/gpt-oss-20b",
                RuntimeMode::Experimental,
                "auto",
                gpu_available,
                true,
                GPT_OSS_CONSUMER_MAX_MODEL_LEN,
                1,
                1,
                4,
                Some(GPT_OSS_CONSUMER_MAX_VRAM_BYTES),
                false,
            )
            .unwrap();
            assert_eq!(decision.backend_path, RuntimeBackendPath::CpuXe);
            assert_eq!(decision.backend_label(), "cpu_xe");
        }
    }

    #[cfg(feature = "xe")]
    #[test]
    fn explicit_xe_is_experimental_only_and_requires_cpu_topology() {
        let decision = validate_gpt_oss_runtime_with_xe(
            "openai/gpt-oss-20b",
            RuntimeMode::Experimental,
            "xe",
            false,
            false,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN,
            1,
            1,
            4,
            None,
            false,
        )
        .unwrap();
        assert_eq!(decision.backend_path, RuntimeBackendPath::CpuXe);
        assert!(decision.reason.contains("explicit GPT-OSS CPU+Xe"));

        let trusted = validate_gpt_oss_runtime_with_xe(
            "openai/gpt-oss-20b",
            RuntimeMode::Trusted,
            "xe",
            false,
            false,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN,
            1,
            1,
            4,
            None,
            false,
        )
        .unwrap_err();
        assert!(trusted.contains("final i7 conformance gate"));
    }

    #[cfg(not(feature = "xe"))]
    #[test]
    fn featureless_build_rejects_explicit_xe() {
        let error = validate_gpt_oss_runtime_with_xe(
            "openai/gpt-oss-20b",
            RuntimeMode::Experimental,
            "xe",
            false,
            false,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN,
            1,
            1,
            4,
            None,
            false,
        )
        .unwrap_err();
        assert!(error.contains("without the 'xe' feature"));
    }

    #[test]
    fn explicit_cuda_is_experimental_only() {
        let decision = validate_gpt_oss_runtime(
            "openai/gpt-oss-20b",
            RuntimeMode::Experimental,
            "cuda",
            true,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN,
            1,
            1,
            256,
            Some(GPT_OSS_CONSUMER_MAX_VRAM_BYTES),
            false,
        )
        .unwrap();
        assert_eq!(decision.backend_path, RuntimeBackendPath::Cuda);
        assert!(decision.reason.contains("experimental CUDA"));

        let error = validate_gpt_oss_runtime(
            "openai/gpt-oss-20b",
            RuntimeMode::Trusted,
            "cuda",
            true,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN,
            1,
            1,
            256,
            Some(GPT_OSS_CONSUMER_MAX_VRAM_BYTES),
            false,
        )
        .unwrap_err();
        assert!(error.contains("trusted mode rejects CUDA"));
        assert!(error.contains("--runtime-mode experimental"));
    }

    #[test]
    fn auto_rejects_non_gpt_oss_even_when_a_gpu_is_available() {
        for gpu_available in [false, true] {
            let error = validate_gpt_oss_runtime(
                "other/model",
                RuntimeMode::Experimental,
                "auto",
                gpu_available,
                2048,
                1,
                1,
                1,
                None,
                false,
            )
            .unwrap_err();
            assert!(error.contains("automatic serving only supports GPT-OSS"));
        }
    }

    #[test]
    fn cpu_rejects_parallelism_but_allows_explicit_request_batching() {
        for (tp, pp) in [(2, 1), (1, 2)] {
            assert!(validate_gpt_oss_runtime(
                "openai/gpt-oss-20b",
                RuntimeMode::Experimental,
                "cpu",
                false,
                GPT_OSS_CONSUMER_MAX_MODEL_LEN,
                tp,
                pp,
                2,
                None,
                false,
            )
            .is_err());
        }
        let decision = validate_gpt_oss_runtime(
            "openai/gpt-oss-20b",
            RuntimeMode::Experimental,
            "cpu",
            false,
            GPT_OSS_CONSUMER_MAX_MODEL_LEN,
            1,
            1,
            2,
            None,
            false,
        )
        .unwrap();
        assert!(decision
            .reason
            .contains("experimental native GPT-OSS CPU batching"));
    }
}
