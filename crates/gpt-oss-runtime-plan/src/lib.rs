#![forbid(unsafe_code)]
//! Pure runtime planning for GPT-OSS execution paths.
//!
//! This crate decides which coarse backend path is legal for a request. It
//! does not execute kernels or own cache storage.

use clap::ValueEnum;
use gpt_oss_core::types::Dtype;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Runtime trust tier for GPT-OSS serving.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeMode {
    /// Preserve current fallback behavior while the architecture is still migrating.
    Experimental,
    /// Reject combinations that have not been proven safe by conformance.
    Trusted,
}

impl Default for RuntimeMode {
    fn default() -> Self {
        Self::Experimental
    }
}

/// Coarse backend path selected for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendPath {
    CudaEager,
    CudaGraph,
    Mock,
}

/// Request phase used by the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestKind {
    Prefill,
    Decode,
}

/// Graph replay legality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphPolicy {
    Forbidden { reason: String },
    Eligible { padded_batch_size: usize },
}

/// Planned output-readback mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputPolicy {
    Logits,
    GreedyTokens,
}

/// Pure input to the planner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanRequest {
    pub runtime_mode: RuntimeMode,
    pub model_name: String,
    pub is_prefill: bool,
    pub greedy_only: bool,
    pub graph_enabled: bool,
    pub graph_max_batch_size: usize,
    pub graph_padded_batch_size: Option<usize>,
    pub dtype: Dtype,
}

impl PlanRequest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        runtime_mode: RuntimeMode,
        model_name: impl Into<String>,
        is_prefill: bool,
        greedy_only: bool,
        graph_enabled: bool,
        graph_max_batch_size: usize,
        graph_padded_batch_size: Option<usize>,
        dtype: Dtype,
    ) -> Self {
        Self {
            runtime_mode,
            model_name: model_name.into(),
            is_prefill,
            greedy_only,
            graph_enabled,
            graph_max_batch_size,
            graph_padded_batch_size,
            dtype,
        }
    }
}

/// The planner output used by the worker routing seam.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub runtime_mode: RuntimeMode,
    pub model_name: String,
    pub request_kind: RequestKind,
    pub backend_path: BackendPath,
    pub graph_policy: GraphPolicy,
    pub output_policy: OutputPolicy,
    pub dtype: Dtype,
    pub reason: String,
}

/// Planner failure.
#[derive(Debug, Error, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanError {
    #[error("invalid plan request: {0}")]
    InvalidRequest(String),
}

/// GPT-OSS model-family detection shared by the planner and startup gate.
pub fn is_gpt_oss_model(model_name: &str) -> bool {
    let model_name = model_name.to_ascii_lowercase();
    model_name.contains("gpt-oss") || model_name.contains("gpt_oss")
}

/// Decide the coarse execution path for a request.
pub fn plan_request(request: &PlanRequest) -> Result<ExecutionPlan, PlanError> {
    if request.model_name.trim().is_empty() {
        return Err(PlanError::InvalidRequest(
            "model_name must not be empty".into(),
        ));
    }

    let request_kind = if request.is_prefill {
        RequestKind::Prefill
    } else {
        RequestKind::Decode
    };
    let output_policy = if request.greedy_only {
        OutputPolicy::GreedyTokens
    } else {
        OutputPolicy::Logits
    };
    let is_gpt_oss = is_gpt_oss_model(&request.model_name);

    let graph_policy = if request.is_prefill {
        GraphPolicy::Forbidden {
            reason: "prefill uses the eager execution body in this phase".into(),
        }
    } else if !request.greedy_only {
        GraphPolicy::Forbidden {
            reason: "graph replay is only enabled for greedy decode in this phase".into(),
        }
    } else if !request.graph_enabled {
        GraphPolicy::Forbidden {
            reason: "graph runner is disabled".into(),
        }
    } else if let Some(padded_batch_size) = request.graph_padded_batch_size {
        if padded_batch_size > request.graph_max_batch_size {
            GraphPolicy::Forbidden {
                reason: format!(
                    "padded batch {} exceeds graph max {}",
                    padded_batch_size, request.graph_max_batch_size
                ),
            }
        } else if request.runtime_mode == RuntimeMode::Trusted && is_gpt_oss {
            GraphPolicy::Forbidden {
                reason: "trusted GPT-OSS mode rejects graph replay until parity is proven".into(),
            }
        } else {
            GraphPolicy::Eligible { padded_batch_size }
        }
    } else {
        GraphPolicy::Forbidden {
            reason: "decode batch is not graph-compatible".into(),
        }
    };

    let backend_path = match &graph_policy {
        GraphPolicy::Eligible { .. } => BackendPath::CudaGraph,
        GraphPolicy::Forbidden { .. } => BackendPath::CudaEager,
    };

    let reason = match &graph_policy {
        GraphPolicy::Eligible { padded_batch_size } => {
            format!("selected CUDA graph for decode with padded_batch_size={padded_batch_size}")
        }
        GraphPolicy::Forbidden { reason } => format!("selected CUDA eager: {reason}"),
    };

    Ok(ExecutionPlan {
        runtime_mode: request.runtime_mode,
        model_name: request.model_name.clone(),
        request_kind,
        backend_path,
        graph_policy,
        output_policy,
        dtype: request.dtype,
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trusted_gpt_oss_decode_forbids_graph_replay() {
        let plan = plan_request(&PlanRequest::new(
            RuntimeMode::Trusted,
            "openai/gpt-oss-20b",
            false,
            true,
            true,
            32,
            Some(8),
            Dtype::Float16,
        ))
        .unwrap();

        assert_eq!(plan.backend_path, BackendPath::CudaEager);
        assert!(matches!(plan.graph_policy, GraphPolicy::Forbidden { .. }));
    }

    #[test]
    fn experimental_decode_uses_graph_when_eligible() {
        let plan = plan_request(&PlanRequest::new(
            RuntimeMode::Experimental,
            "openai/gpt-oss-20b",
            false,
            true,
            true,
            32,
            Some(8),
            Dtype::Float16,
        ))
        .unwrap();

        assert_eq!(plan.backend_path, BackendPath::CudaGraph);
        assert!(matches!(
            plan.graph_policy,
            GraphPolicy::Eligible {
                padded_batch_size: 8
            }
        ));
    }

    #[test]
    fn prefill_uses_eager_path() {
        let plan = plan_request(&PlanRequest::new(
            RuntimeMode::Experimental,
            "openai/gpt-oss-20b",
            true,
            true,
            true,
            32,
            Some(8),
            Dtype::Float16,
        ))
        .unwrap();

        assert_eq!(plan.backend_path, BackendPath::CudaEager);
        assert!(matches!(plan.graph_policy, GraphPolicy::Forbidden { .. }));
    }
}
