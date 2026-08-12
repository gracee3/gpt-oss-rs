//! Health check endpoint.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use std::sync::Arc;

use crate::server::AppState;

/// GET /health -- simple liveness check.
pub async fn health_check() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// GET /ready -- admission readiness plus only sanitized identity/snapshot facts.
pub async fn ready_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let status = state.lifecycle.status();
    if status.state == gpt_oss_engine::ServiceState::Ready {
        (
            StatusCode::OK,
            axum::Json(serde_json::json!({
                "state": status.state.as_str(),
                "model": status.served_model_id,
                "runtime_snapshot_sha256": status.runtime_snapshot_sha256,
            })),
        )
            .into_response()
    } else {
        crate::error::ApiError::from(gpt_oss_engine::StableFailure::unavailable(status.state))
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn health_returns_ok() {
        let resp = health_check().await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
