//! Chat completion endpoint: POST /v1/chat/completions

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio_stream::StreamExt;
use tracing::info;

use crate::error::ApiError;
use crate::protocol_stream::{
    visible_text_from_protocol_messages, StreamedChatChoiceState,
};
use crate::routes::tools::ToolChoice;
use crate::runtime_policy::is_gpt_oss_model;
use crate::server::AppState;
use crate::types::request::ChatCompletionRequest;
use crate::types::response::{ChatChoice, ChatCompletionResponse, Usage};
use crate::types::streaming::{
    format_sse_data, ChatCompletionStreamChunk, SSE_DONE,
};

/// POST /v1/chat/completions -- chat completion (streaming or non-streaming).
pub async fn create_chat_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    req.validate()?;

    if req.model != state.model_name {
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found, available: {}",
            req.model, state.model_name
        )));
    }

    let sampling_params = req.to_sampling_params();

    // Check if tools are active
    let tools_active = req.tools.as_ref().map_or(false, |t| !t.is_empty())
        && !matches!(req.tool_choice.as_ref(), Some(ToolChoice::Mode(m)) if m == "none");

    let tool_defs: Vec<gpt_oss_tokenizer::ToolDefinition> = req
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .map(|t| gpt_oss_tokenizer::ToolDefinition {
                    tool_type: t.tool_type.clone(),
                    function: gpt_oss_tokenizer::FunctionDefinition {
                        name: t.function.name.clone(),
                        description: t.function.description.clone(),
                        parameters: t
                            .function
                            .parameters
                            .as_ref()
                            .and_then(|p| serde_json::from_value(p.clone()).ok()),
                    },
                })
                .collect()
        })
        .unwrap_or_default();

    let protocol = gpt_oss_tokenizer::HarmonyProtocol::gpt_oss()
        .map_err(|e| ApiError::Internal(format!("harmony init error: {}", e)))?;
    let protocol_messages: Vec<gpt_oss_tokenizer::ProtocolMessage> = req
        .messages
        .iter()
        .map(|m| gpt_oss_tokenizer::ProtocolMessage::new(&m.role, &m.content))
        .collect();
    let prompt = protocol
        .render_prompt(&protocol_messages, None, &tool_defs)
        .map(|rendered| rendered.text)
        .map_err(|e| ApiError::Internal(format!("harmony render error: {}", e)))?;

    info!(
        model = %req.model,
        stream = req.stream,
        messages = req.messages.len(),
        runtime = %state.runtime_decision.summary(),
        "chat completion request"
    );

    if req.stream {
        let stream_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let model = state.model_name.clone();
        let is_gpt_oss = is_gpt_oss_model(&state.model_name);
        let tools_active_for_stream = tools_active;

        let (_request_id, mut output_stream) = state
            .engine
            .generate(prompt, sampling_params)
            .await
            .map_err(ApiError::from)?;

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::convert::Infallible>>(32);
        let stream_id_clone = stream_id.clone();
        let model_clone = model.clone();

        tokio::spawn(async move {
            if tx
                .send(Ok(format_sse_data(&ChatCompletionStreamChunk::role_chunk(
                    &stream_id_clone,
                    &model_clone,
                ))))
                .await
                .is_err()
            {
                return;
            }

            let mut choice_states: HashMap<usize, StreamedChatChoiceState> = HashMap::new();

            while let Some(output) = output_stream.next().await {
                let mut events = String::new();
                for co in &output.outputs {
                    let finish = co.finish_reason.map(|r| match r {
                        gpt_oss_core::prelude::FinishReason::Stop => "stop".to_string(),
                        gpt_oss_core::prelude::FinishReason::Length => "length".to_string(),
                        gpt_oss_core::prelude::FinishReason::Abort => "stop".to_string(),
                    });

                    if is_gpt_oss {
                        let state = match choice_states.entry(co.index) {
                            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                            std::collections::hash_map::Entry::Vacant(entry) => {
                                match StreamedChatChoiceState::new() {
                                    Ok(state) => entry.insert(state),
                                    Err(_) => return,
                                }
                            }
                        };

                        let (content_delta, tool_call_deltas) = match state.ingest(
                            &stream_id_clone,
                            co.index,
                            &co.token_ids,
                            output.finished,
                        ) {
                            Ok(value) => value,
                            Err(_) => return,
                        };

                        if let Some(delta) = content_delta {
                            let chunk = ChatCompletionStreamChunk::content_chunk(
                                &stream_id_clone,
                                &model_clone,
                                co.index,
                                &delta,
                                None,
                            );
                            events.push_str(&format_sse_data(&chunk));
                        }

                        if tools_active_for_stream && !tool_call_deltas.is_empty() {
                            let chunk = ChatCompletionStreamChunk::tool_call_chunk(
                                &stream_id_clone,
                                &model_clone,
                                co.index,
                                tool_call_deltas,
                                None,
                            );
                            events.push_str(&format_sse_data(&chunk));
                        }
                    } else if finish.is_none() {
                        let chunk = ChatCompletionStreamChunk::content_chunk(
                            &stream_id_clone,
                            &model_clone,
                            co.index,
                            &co.text,
                            None,
                        );
                        events.push_str(&format_sse_data(&chunk));
                    }

                    if let Some(reason) = finish {
                        let finish_reason = if is_gpt_oss
                            && tools_active_for_stream
                            && choice_states
                                .get(&co.index)
                                .is_some_and(|state| state.has_tool_calls())
                        {
                            "tool_calls"
                        } else {
                            reason.as_str()
                        };
                        let chunk = ChatCompletionStreamChunk::finish_chunk(
                            &stream_id_clone,
                            &model_clone,
                            co.index,
                            finish_reason,
                        );
                        events.push_str(&format_sse_data(&chunk));
                    }
                }
                if output.finished {
                    events.push_str(SSE_DONE);
                }
                if tx.send(Ok(events)).await.is_err() {
                    return;
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
        // Non-streaming: collect all outputs until finished.
        let (_request_id, mut output_stream) = state
            .engine
            .generate(prompt, sampling_params)
            .await
            .map_err(ApiError::from)?;

        let mut last_output = None;
        while let Some(output) = output_stream.next().await {
            if output.finished {
                last_output = Some(output);
                break;
            }
            last_output = Some(output);
        }

        let output =
            last_output.ok_or_else(|| ApiError::Internal("engine produced no output".into()))?;

        if is_gpt_oss_model(&state.model_name) && !tools_active {
            let protocol = gpt_oss_tokenizer::HarmonyProtocol::gpt_oss()
                .map_err(|e| ApiError::Internal(format!("harmony init error: {}", e)))?;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let mut total_completion = 0usize;
            let choices: Vec<ChatChoice> = output
                .outputs
                .iter()
                .map(|co| {
                    total_completion += co.token_ids.len();
                    let parsed = protocol
                        .parse_completion_tokens(&co.token_ids)
                        .unwrap_or_default();
                    let content =
                        visible_text_from_protocol_messages(&parsed).unwrap_or_else(|| co.text.clone());
                    ChatChoice {
                        message: crate::types::request::ChatMessage {
                            role: "assistant".to_string(),
                            content,
                        },
                        index: co.index,
                        finish_reason: co.finish_reason.map(|r| match r {
                            gpt_oss_core::prelude::FinishReason::Stop => "stop".to_string(),
                            gpt_oss_core::prelude::FinishReason::Length => "length".to_string(),
                            gpt_oss_core::prelude::FinishReason::Abort => "stop".to_string(),
                        }),
                    }
                })
                .collect();
            let resp = ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                object: "chat.completion".to_string(),
                created: now,
                model: state.model_name.clone(),
                choices,
                usage: Usage {
                    prompt_tokens: output.prompt_token_ids.len(),
                    completion_tokens: total_completion,
                    total_tokens: output.prompt_token_ids.len() + total_completion,
                },
            };
            return Ok(Json(resp).into_response());
        }

        if tools_active {
            // Parse output for tool calls
            let resp_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let mut total_completion = 0usize;
            let choices: Vec<crate::routes::tools::ToolChatChoice> = output
                .outputs
                .iter()
                .map(|co| {
                    total_completion += co.token_ids.len();
                    let finish_reason_val = co.finish_reason.map(|r| match r {
                        gpt_oss_core::prelude::FinishReason::Stop => "stop",
                        gpt_oss_core::prelude::FinishReason::Length => "length",
                        gpt_oss_core::prelude::FinishReason::Abort => "stop",
                    });
                    if is_gpt_oss_model(&state.model_name) {
                        let protocol = gpt_oss_tokenizer::HarmonyProtocol::gpt_oss()
                            .map_err(|e| ApiError::Internal(format!("harmony init error: {}", e)))
                            .ok();
                        if let Some(protocol) = protocol {
                            let parsed = protocol
                                .parse_completion_tokens(&co.token_ids)
                                .unwrap_or_default();
                            let tool_calls: Vec<crate::routes::tools::ResponseToolCall> = parsed
                                .iter()
                                .filter_map(|message| {
                                    let recipient = message.recipient.as_ref()?;
                                    Some(crate::routes::tools::ResponseToolCall {
                                        id: format!("{}_{}_{}", resp_id, co.index, recipient),
                                        call_type: "function".to_string(),
                                        function: crate::routes::tools::ResponseFunctionCall {
                                            name: recipient
                                                .strip_prefix("functions.")
                                                .unwrap_or(recipient.as_str())
                                                .to_string(),
                                            arguments: message.content.clone(),
                                        },
                                    })
                                })
                                .collect();
                            let content = visible_text_from_protocol_messages(&parsed);
                            return crate::routes::tools::ToolChatChoice {
                                index: co.index,
                                message: crate::routes::tools::ToolChatMessage {
                                    role: "assistant".to_string(),
                                    content,
                                    tool_calls: if tool_calls.is_empty() {
                                        None
                                    } else {
                                        Some(tool_calls)
                                    },
                                },
                                finish_reason: Some(
                                    if parsed.iter().any(|message| message.recipient.is_some()) {
                                        "tool_calls".to_string()
                                    } else {
                                        finish_reason_val.unwrap_or("stop").to_string()
                                    },
                                ),
                            };
                        }
                    }
                    let call_prefix = format!("{}_{}_", resp_id, co.index);
                    match gpt_oss_tokenizer::parse_tool_calls(&co.text, &call_prefix) {
                        gpt_oss_tokenizer::ToolParseResult::ToolCalls { prefix_text, calls } => {
                            let tool_calls: Vec<crate::routes::tools::ResponseToolCall> = calls
                                .into_iter()
                                .map(|tc| crate::routes::tools::ResponseToolCall {
                                    id: tc.id,
                                    call_type: "function".to_string(),
                                    function: crate::routes::tools::ResponseFunctionCall {
                                        name: tc.name,
                                        arguments: tc.arguments,
                                    },
                                })
                                .collect();
                            let content = if prefix_text.is_empty() {
                                None
                            } else {
                                Some(prefix_text)
                            };
                            crate::routes::tools::ToolChatChoice {
                                index: co.index,
                                message: crate::routes::tools::ToolChatMessage {
                                    role: "assistant".to_string(),
                                    content,
                                    tool_calls: Some(tool_calls),
                                },
                                finish_reason: Some("tool_calls".to_string()),
                            }
                        }
                        gpt_oss_tokenizer::ToolParseResult::PlainText(text) => {
                            crate::routes::tools::ToolChatChoice {
                                index: co.index,
                                message: crate::routes::tools::ToolChatMessage {
                                    role: "assistant".to_string(),
                                    content: Some(text),
                                    tool_calls: None,
                                },
                                finish_reason: finish_reason_val.map(|s| s.to_string()),
                            }
                        }
                    }
                })
                .collect();

            let resp = crate::routes::tools::ChatCompletionToolResponse {
                id: resp_id,
                object: "chat.completion".to_string(),
                created: now,
                model: state.model_name.clone(),
                choices,
                usage: crate::types::response::Usage {
                    prompt_tokens: output.prompt_token_ids.len(),
                    completion_tokens: total_completion,
                    total_tokens: output.prompt_token_ids.len() + total_completion,
                },
            };
            Ok(Json(resp).into_response())
        } else {
            let resp = ChatCompletionResponse::from_request_output(&output, &state.model_name);
            Ok(Json(resp).into_response())
        }
    }
}
