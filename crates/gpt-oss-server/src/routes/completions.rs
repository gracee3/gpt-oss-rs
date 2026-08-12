//! Completion endpoint: POST /v1/completions

use std::sync::Arc;

use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tracing::info;

use crate::error::ApiError;
use crate::server::{
    ensure_non_streaming_size, AppState, BoundedSseSender, RequestOutputAccumulator,
};
use crate::types::request::CompletionRequest;
use crate::types::response::CompletionResponse;
use crate::types::streaming::{format_sse_data, CompletionStreamChunk, SSE_DONE};

/// POST /v1/completions -- text completion (streaming or non-streaming).
pub async fn create_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    req.validate()?;
    if req
        .logprobs
        .is_some_and(|value| value > state.limits.max_logprobs)
    {
        return Err(ApiError::from(gpt_oss_engine::StableFailure::new(
            gpt_oss_engine::StableFailureCode::UnsupportedOption,
            gpt_oss_engine::FailurePhase::Validation,
            false,
            format!("logprobs exceeds maximum {}", state.limits.max_logprobs),
        )));
    }

    if req.model != state.model_name {
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found, available: {}",
            req.model, state.model_name
        )));
    }

    let sampling_params = req.to_sampling_params();

    info!(
        model = %req.model,
        stream = req.stream,
        max_tokens = req.max_tokens,
        runtime = %state.runtime_decision.summary(),
        "completion request"
    );

    if req.stream {
        let stream_id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let model = state.model_name.clone();

        let mut output_stream = state
            .engine()
            .await
            .map_err(ApiError::from)?
            .generate(req.prompt, sampling_params)
            .await
            .map_err(ApiError::from)?;

        let (raw_tx, rx) =
            tokio::sync::mpsc::channel::<Result<String, std::convert::Infallible>>(1);
        let tx = BoundedSseSender::new(raw_tx, state.limits.max_stream_event_bytes);
        tokio::spawn(async move {
            loop {
                match output_stream.recv().await {
                    Ok(Some(output)) => {
                        let mut events = String::new();
                        for co in &output.outputs {
                            let finish = co.finish_reason.and_then(finish_reason);
                            let chunk = CompletionStreamChunk::new(
                                &stream_id, &model, co.index, &co.text, finish,
                            );
                            events.push_str(&format_sse_data(&chunk));
                        }
                        if output.finished {
                            events.push_str(SSE_DONE);
                        }
                        if tx.send(Ok(events)).await.is_err() || output.finished {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(failure) => {
                        let event = serde_json::json!({
                            "error": {
                                "code": failure.code.as_str(),
                                "message": failure.message,
                                "retryable": failure.retryable,
                            }
                        });
                        let _ = tx.send(Ok(format!("data: {event}\n\n{SSE_DONE}"))).await;
                        break;
                    }
                }
            }
        });

        let body = axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
        Ok(Response::builder()
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .body(body)
            .unwrap()
            .into_response())
    } else {
        // Non-streaming: collect all outputs from the stream until finished.
        let mut output_stream = state
            .engine()
            .await
            .map_err(ApiError::from)?
            .generate(req.prompt, sampling_params)
            .await
            .map_err(ApiError::from)?;

        let mut accumulator = RequestOutputAccumulator::default();
        while let Some(output) = output_stream.recv().await.map_err(ApiError::from)? {
            let finished = output.finished;
            accumulator.push(output).map_err(ApiError::from)?;
            if finished {
                break;
            }
        }

        let output = accumulator
            .finish()
            .ok_or_else(|| ApiError::Internal("engine produced no output".into()))?;

        let resp = CompletionResponse::from_request_output(&output, &state.model_name);
        ensure_non_streaming_size(&resp, state.limits.max_non_streaming_bytes)?;
        Ok(Json(resp).into_response())
    }
}

fn finish_reason(reason: gpt_oss_core::prelude::FinishReason) -> Option<String> {
    match reason {
        gpt_oss_core::prelude::FinishReason::Stop => Some("stop".into()),
        gpt_oss_core::prelude::FinishReason::Length => Some("length".into()),
        gpt_oss_core::prelude::FinishReason::Abort => None,
    }
}
