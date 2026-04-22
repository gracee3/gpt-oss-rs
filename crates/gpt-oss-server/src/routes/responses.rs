//! Responses API routes: unified text generation with stored follow-up turns.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use tokio::sync::RwLock;
use tokio_stream::StreamExt;
use tracing::info;

use crate::error::ApiError;
use crate::protocol_stream::{
    apply_gpt_oss_sampling_policy, load_harmony_protocol, new_gpt_oss_stream_parser,
    parse_gpt_oss_completion, visible_text_from_protocol_messages,
};
use crate::runtime_policy::is_gpt_oss_model;
use crate::server::AppState;
use crate::types::request::ChatMessage;
use crate::types::responses::{
    CreateResponseRequest, ResponseFunctionCallItem, ResponseFunctionCallOutputItem,
    ResponseInputItem, ResponseInputItemsList, ResponseObject, ResponseOutputItem,
    ResponseOutputMessage, ResponseToolChoice, ResponseUsage,
};

#[derive(Debug, Clone)]
pub enum StoredConversationItem {
    Input(ResponseInputItem),
    Output(ResponseOutputItem),
    Protocol(gpt_oss_tokenizer::ProtocolMessage),
}

#[derive(Debug, Clone)]
pub struct StoredResponse {
    pub response: ResponseObject,
    pub input_items: Vec<ResponseInputItem>,
    pub conversation_items: Vec<StoredConversationItem>,
}

pub type SharedResponseStore = Arc<RwLock<HashMap<String, StoredResponse>>>;

/// POST /v1/responses -- create a unified response.
pub async fn create_response(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateResponseRequest>,
) -> Result<Response, ApiError> {
    req.validate()?;

    if req.model != state.model_name {
        return Err(ApiError::ModelNotFound(format!(
            "model '{}' not found, available: {}",
            req.model, state.model_name
        )));
    }

    let input_items = req.normalize_input_items()?;
    let mut conversation_items = if let Some(previous_response_id) = &req.previous_response_id {
        let store = state.response_store.read().await;
        let stored = store.get(previous_response_id).ok_or_else(|| {
            ApiError::InvalidRequest(format!(
                "previous_response_id '{}' was not found or is not stored",
                previous_response_id
            ))
        })?;
        stored.conversation_items.clone()
    } else {
        Vec::new()
    };
    conversation_items.extend(
        input_items
            .iter()
            .cloned()
            .map(StoredConversationItem::Input),
    );

    let protocol_messages = render_conversation_protocol_items(&conversation_items);
    if protocol_messages.is_empty() {
        return Err(ApiError::InvalidRequest(
            "input must not be empty for /v1/responses".into(),
        ));
    }

    let function_tools = req.normalize_function_tools()?;
    let tool_defs: Vec<gpt_oss_tokenizer::ToolDefinition> = function_tools
        .iter()
        .map(|tool| tool.to_tool_definition())
        .collect();

    let harmony_instructions = req.instructions.clone().filter(|value| !value.is_empty());
    let protocol = load_harmony_protocol()?;
    let prompt = protocol
        .render_prompt(
            &protocol_messages,
            harmony_instructions.as_deref(),
            if req.tools_enabled() { &tool_defs } else { &[] },
        )
        .map(|rendered| rendered.text)
        .map_err(|e| ApiError::Internal(format!("harmony render error: {}", e)))?;

    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let mut sampling_params = req.to_sampling_params();
    if is_gpt_oss_model(&state.model_name) {
        apply_gpt_oss_sampling_policy(&protocol, &mut sampling_params)?;
    }
    let tool_choice = req.effective_tool_choice();
    let response_tools = req.tools.clone().unwrap_or_default();

    info!(
        model = %req.model,
        stream = req.stream,
        store = req.store,
        tools = req.tools_enabled(),
        previous_response = req.previous_response_id.as_deref().unwrap_or("none"),
        runtime = %state.runtime_decision.summary(),
        "responses request"
    );

    if req.stream {
        if req.tools_enabled() {
            return stream_tool_response(
                state,
                req,
                prompt,
                sampling_params,
                response_id,
                input_items,
                conversation_items,
            )
            .await;
        }

        let model = state.model_name.clone();
        let input_items_clone = input_items.clone();
        let conversation_items_clone = conversation_items.clone();
        let req_clone = req.clone();
        let response_store = state.response_store.clone();
        let response_id_clone = response_id.clone();
        let tool_choice_clone = tool_choice.clone();
        let response_tools_clone = response_tools.clone();
        let is_gpt_oss = is_gpt_oss_model(&state.model_name);

        let (_request_id, mut output_stream) = state
            .engine
            .generate(prompt, sampling_params)
            .await
            .map_err(ApiError::from)?;

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::convert::Infallible>>(16);

        tokio::spawn(async move {
            let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
            let mut initial = ResponseObject::in_progress(
                response_id_clone.clone(),
                model.clone(),
                req_clone.instructions.clone(),
                req_clone.max_output_tokens,
                req_clone.previous_response_id.clone(),
                req_clone.store,
                req_clone.temperature,
                req_clone.top_p,
                req_clone.metadata.clone(),
                req_clone.parallel_tool_calls,
                tool_choice_clone.clone(),
                response_tools_clone.clone(),
            );

            if tx
                .send(Ok(format_sse_event(
                    "response.created",
                    &serde_json::json!({
                        "type": "response.created",
                        "response": initial,
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }

            initial.status = "in_progress".to_string();
            if tx
                .send(Ok(format_sse_event(
                    "response.in_progress",
                    &serde_json::json!({
                        "type": "response.in_progress",
                        "response": initial,
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }

            let mut content_open = false;
            let mut full_text = String::new();
            let mut final_output = None;
            let mut protocol_state = if is_gpt_oss {
                StreamedProtocolState::new().ok()
            } else {
                None
            };

            while let Some(output) = output_stream.next().await {
                if !content_open {
                    let item = ResponseOutputMessage::in_progress(message_id.clone());
                    if tx
                        .send(Ok(format_sse_event(
                            "response.output_item.added",
                            &serde_json::json!({
                                "type": "response.output_item.added",
                                "output_index": 0,
                                "item": ResponseOutputItem::Message(item),
                            }),
                        )))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    if tx
                        .send(Ok(format_sse_event(
                            "response.content_part.added",
                            &serde_json::json!({
                                "type": "response.content_part.added",
                                "item_id": message_id,
                                "output_index": 0,
                                "content_index": 0,
                                "part": {
                                    "type": "output_text",
                                    "text": "",
                                    "annotations": [],
                                },
                            }),
                        )))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    content_open = true;
                }

                if let Some(choice) = output.outputs.first() {
                    let next_text = if let Some(state) = protocol_state.as_mut() {
                        match state.ingest(&choice.token_ids, output.finished) {
                            Ok(messages) => {
                                visible_text_from_protocol_messages(&messages).unwrap_or_default()
                            }
                            Err(_) => return,
                        }
                    } else {
                        choice.text.clone()
                    };
                    let delta = diff_text(&full_text, &next_text);
                    full_text = next_text;
                    if !delta.is_empty()
                        && tx
                            .send(Ok(format_sse_event(
                                "response.output_text.delta",
                                &serde_json::json!({
                                    "type": "response.output_text.delta",
                                    "item_id": message_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": delta,
                                }),
                            )))
                            .await
                            .is_err()
                    {
                        return;
                    }
                }

                final_output = Some(output.clone());
                if output.finished {
                    break;
                }
            }

            let Some(output) = final_output else {
                let _ = tx
                    .send(Ok(format_sse_event(
                        "response.completed",
                        &serde_json::json!({
                            "type": "response.completed",
                            "response": ResponseObject::in_progress(
                                response_id_clone.clone(),
                                model,
                                req_clone.instructions.clone(),
                                req_clone.max_output_tokens,
                                req_clone.previous_response_id.clone(),
                                req_clone.store,
                                req_clone.temperature,
                                req_clone.top_p,
                                req_clone.metadata.clone(),
                                req_clone.parallel_tool_calls,
                                tool_choice_clone,
                                response_tools_clone,
                            ),
                        }),
                    )))
                    .await;
                return;
            };

            let output_items = vec![ResponseOutputItem::Message(
                ResponseOutputMessage::completed(message_id.clone(), full_text.clone()),
            )];
            let response = ResponseObject::completed(
                response_id_clone.clone(),
                model.clone(),
                req_clone.instructions.clone(),
                req_clone.max_output_tokens,
                req_clone.previous_response_id.clone(),
                req_clone.store,
                req_clone.temperature,
                req_clone.top_p,
                req_clone.metadata.clone(),
                output_items.clone(),
                ResponseUsage::from_request_output(&output),
                req_clone.parallel_tool_calls,
                req_clone.effective_tool_choice(),
                req_clone.tools.clone().unwrap_or_default(),
            );

            if tx
                .send(Ok(format_sse_event(
                    "response.output_text.done",
                    &serde_json::json!({
                        "type": "response.output_text.done",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "text": full_text,
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.content_part.done",
                    &serde_json::json!({
                        "type": "response.content_part.done",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": full_text,
                            "annotations": [],
                        },
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.output_item.done",
                    &serde_json::json!({
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": response.output[0],
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.completed",
                    &serde_json::json!({
                        "type": "response.completed",
                        "response": response,
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }

            if req_clone.store {
                let mut stored_items = conversation_items_clone;
                stored_items.extend(output_items.into_iter().map(StoredConversationItem::Output));

                let mut store = response_store.write().await;
                store.insert(
                    response_id_clone,
                    StoredResponse {
                        response,
                        input_items: input_items_clone,
                        conversation_items: stored_items,
                    },
                );
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
        let (_request_id, mut output_stream) = state
            .engine
            .generate(prompt, sampling_params)
            .await
            .map_err(ApiError::from)?;

        let mut last_output = None;
        while let Some(output) = output_stream.next().await {
            last_output = Some(output.clone());
            if output.finished {
                break;
            }
        }

        let output =
            last_output.ok_or_else(|| ApiError::Internal("engine produced no output".into()))?;

        let response = response_from_output(
            &response_id,
            &state.model_name,
            &req,
            &output,
            &function_tools,
            tool_choice,
            response_tools,
        )?;

        if req.store {
            let mut stored_items = conversation_items;
            if is_gpt_oss_model(&state.model_name) {
                if let Some(completion) = output.outputs.first() {
                    let protocol = load_harmony_protocol()?;
                    let parsed = parse_gpt_oss_completion(&protocol, &completion.token_ids)?;
                    stored_items.extend(stored_conversation_items_from_protocol_messages(
                        &parsed,
                        &response.output,
                    ));
                }
            } else {
                stored_items.extend(
                    response
                        .output
                        .iter()
                        .cloned()
                        .map(StoredConversationItem::Output),
                );
            }

            let mut store = state.response_store.write().await;
            store.insert(
                response_id,
                StoredResponse {
                    response: response.clone(),
                    input_items,
                    conversation_items: stored_items,
                },
            );
        }

        Ok(Json(response).into_response())
    }
}

#[derive(Debug, Clone)]
struct StreamedMessageState {
    id: String,
    text: String,
}

#[derive(Debug, Clone)]
struct StreamedFunctionCallState {
    id: String,
    call_id: String,
    name: String,
    arguments: String,
}

struct StreamedProtocolState {
    processed_tokens: usize,
    parser: gpt_oss_tokenizer::HarmonyStreamParser,
}

impl StreamedProtocolState {
    fn new() -> Result<Self, ApiError> {
        let protocol = load_harmony_protocol()?;
        let parser = new_gpt_oss_stream_parser(&protocol)?;
        Ok(Self {
            processed_tokens: 0,
            parser,
        })
    }

    fn ingest(
        &mut self,
        token_ids: &[u32],
        finished: bool,
    ) -> Result<Vec<gpt_oss_tokenizer::ParsedProtocolMessage>, ApiError> {
        for token in token_ids.iter().copied().skip(self.processed_tokens) {
            self.parser
                .push_token(token)
                .map_err(|e| ApiError::Internal(format!("harmony stream parse error: {}", e)))?;
        }
        self.processed_tokens = token_ids.len();
        if finished {
            self.parser
                .finish()
                .map_err(|e| ApiError::Internal(format!("harmony stream finalize error: {}", e)))?;
        }
        self.parser
            .messages()
            .map_err(|e| ApiError::Internal(format!("harmony stream read error: {}", e)))
    }
}

async fn stream_tool_response(
    state: Arc<AppState>,
    req: CreateResponseRequest,
    prompt: String,
    sampling_params: gpt_oss_core::prelude::SamplingParams,
    response_id: String,
    input_items: Vec<ResponseInputItem>,
    conversation_items: Vec<StoredConversationItem>,
) -> Result<Response, ApiError> {
    let model = state.model_name.clone();
    let response_store = state.response_store.clone();
    let tool_choice = req.effective_tool_choice();
    let response_tools = req.tools.clone().unwrap_or_default();
    let is_gpt_oss = is_gpt_oss_model(&state.model_name);

    let (_request_id, mut output_stream) = state
        .engine
        .generate(prompt, sampling_params)
        .await
        .map_err(ApiError::from)?;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::convert::Infallible>>(32);

    tokio::spawn(async move {
        let initial = ResponseObject::in_progress(
            response_id.clone(),
            model.clone(),
            req.instructions.clone(),
            req.max_output_tokens,
            req.previous_response_id.clone(),
            req.store,
            req.temperature,
            req.top_p,
            req.metadata.clone(),
            req.parallel_tool_calls,
            tool_choice.clone(),
            response_tools.clone(),
        );

        if tx
            .send(Ok(format_sse_event(
                "response.created",
                &serde_json::json!({
                    "type": "response.created",
                    "response": initial,
                }),
            )))
            .await
            .is_err()
        {
            return;
        }

        if tx
            .send(Ok(format_sse_event(
                "response.in_progress",
                &serde_json::json!({
                    "type": "response.in_progress",
                    "response": ResponseObject::in_progress(
                        response_id.clone(),
                        model.clone(),
                        req.instructions.clone(),
                        req.max_output_tokens,
                        req.previous_response_id.clone(),
                        req.store,
                        req.temperature,
                        req.top_p,
                        req.metadata.clone(),
                        req.parallel_tool_calls,
                        tool_choice.clone(),
                        response_tools.clone(),
                    ),
                }),
            )))
            .await
            .is_err()
        {
            return;
        }

        let mut full_text = String::new();
        let mut final_output = None;
        let mut prefix_state: Option<StreamedMessageState> = None;
        let mut tool_states: Vec<StreamedFunctionCallState> = Vec::new();
        let mut saw_tool_calls = false;
        let mut protocol_state = if is_gpt_oss {
            StreamedProtocolState::new().ok()
        } else {
            None
        };

        while let Some(output) = output_stream.next().await {
            if let Some(choice) = output.outputs.first() {
                let parse_result = if let Some(state) = protocol_state.as_mut() {
                    let messages = match state.ingest(&choice.token_ids, output.finished) {
                        Ok(messages) => messages,
                        Err(_) => return,
                    };
                    let prefix_text =
                        visible_text_from_protocol_messages(&messages).unwrap_or_default();
                    full_text = prefix_text.clone();
                    let calls = messages
                        .iter()
                        .filter(|message| message.role == "assistant")
                        .filter_map(|message| {
                            let recipient = message.recipient.as_ref()?;
                            Some(gpt_oss_tokenizer::ParsedToolCall {
                                id: format!(
                                    "{response_id}_0_{}",
                                    recipient
                                        .strip_prefix("functions.")
                                        .unwrap_or(recipient.as_str())
                                ),
                                name: recipient
                                    .strip_prefix("functions.")
                                    .unwrap_or(recipient.as_str())
                                    .to_string(),
                                arguments: message.content.clone(),
                            })
                        })
                        .collect::<Vec<_>>();
                    if calls.is_empty() {
                        gpt_oss_tokenizer::ToolParseResult::PlainText(prefix_text)
                    } else {
                        gpt_oss_tokenizer::ToolParseResult::ToolCalls { prefix_text, calls }
                    }
                } else {
                    full_text = choice.text.clone();
                    gpt_oss_tokenizer::parse_tool_calls(&full_text, &format!("{response_id}_0_"))
                };

                if let gpt_oss_tokenizer::ToolParseResult::ToolCalls { prefix_text, calls } =
                    parse_result
                {
                    saw_tool_calls = true;

                    if !prefix_text.is_empty() {
                        if prefix_state.is_none() {
                            let item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
                            if tx
                                .send(Ok(format_sse_event(
                                    "response.output_item.added",
                                    &serde_json::json!({
                                        "type": "response.output_item.added",
                                        "response_id": response_id,
                                        "output_index": 0,
                                        "item": ResponseOutputItem::Message(
                                            ResponseOutputMessage::in_progress(item_id.clone())
                                        ),
                                    }),
                                )))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            if tx
                                .send(Ok(format_sse_event(
                                    "response.content_part.added",
                                    &serde_json::json!({
                                        "type": "response.content_part.added",
                                        "item_id": item_id,
                                        "output_index": 0,
                                        "content_index": 0,
                                        "part": {
                                            "type": "output_text",
                                            "text": "",
                                            "annotations": [],
                                        },
                                    }),
                                )))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            prefix_state = Some(StreamedMessageState {
                                id: item_id,
                                text: String::new(),
                            });
                        }

                        if let Some(state) = prefix_state.as_mut() {
                            let delta = diff_text(&state.text, &prefix_text);
                            if !delta.is_empty()
                                && tx
                                    .send(Ok(format_sse_event(
                                        "response.output_text.delta",
                                        &serde_json::json!({
                                            "type": "response.output_text.delta",
                                            "item_id": state.id,
                                            "output_index": 0,
                                            "content_index": 0,
                                            "delta": delta,
                                        }),
                                    )))
                                    .await
                                    .is_err()
                            {
                                return;
                            }
                            state.text = prefix_text;
                        }
                    }

                    let tool_offset = usize::from(prefix_state.is_some());
                    for (index, call) in calls.into_iter().enumerate() {
                        if tool_states.len() <= index {
                            let item_id = format!("fc_{}", uuid::Uuid::new_v4().simple());
                            let output_index = tool_offset + index;
                            if tx
                                .send(Ok(format_sse_event(
                                    "response.output_item.added",
                                    &serde_json::json!({
                                        "type": "response.output_item.added",
                                        "response_id": response_id,
                                        "output_index": output_index,
                                        "item": ResponseOutputItem::FunctionCall(
                                            ResponseFunctionCallItem::in_progress(
                                                item_id.clone(),
                                                call.id.clone(),
                                                call.name.clone(),
                                            )
                                        ),
                                    }),
                                )))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            tool_states.push(StreamedFunctionCallState {
                                id: item_id,
                                call_id: call.id.clone(),
                                name: call.name.clone(),
                                arguments: String::new(),
                            });
                        }

                        let state = &mut tool_states[index];
                        let delta = diff_text(&state.arguments, &call.arguments);
                        if !delta.is_empty()
                            && tx
                                .send(Ok(format_sse_event(
                                    "response.function_call_arguments.delta",
                                    &serde_json::json!({
                                        "type": "response.function_call_arguments.delta",
                                        "response_id": response_id,
                                        "item_id": state.id,
                                        "output_index": tool_offset + index,
                                        "delta": delta,
                                    }),
                                )))
                                .await
                                .is_err()
                        {
                            return;
                        }
                        state.arguments = call.arguments;
                    }
                }
            }

            final_output = Some(output.clone());
            if output.finished {
                break;
            }
        }

        let Some(output) = final_output else {
            return;
        };

        let output_items = if saw_tool_calls {
            let mut items = Vec::new();

            if let Some(state) = prefix_state.as_ref() {
                if tx
                    .send(Ok(format_sse_event(
                        "response.output_text.done",
                        &serde_json::json!({
                            "type": "response.output_text.done",
                            "item_id": state.id,
                            "output_index": 0,
                            "content_index": 0,
                            "text": state.text,
                        }),
                    )))
                    .await
                    .is_err()
                {
                    return;
                }
                if tx
                    .send(Ok(format_sse_event(
                        "response.content_part.done",
                        &serde_json::json!({
                            "type": "response.content_part.done",
                            "item_id": state.id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": state.text,
                                "annotations": [],
                            },
                        }),
                    )))
                    .await
                    .is_err()
                {
                    return;
                }

                let item = ResponseOutputItem::Message(ResponseOutputMessage::completed(
                    state.id.clone(),
                    state.text.clone(),
                ));
                if tx
                    .send(Ok(format_sse_event(
                        "response.output_item.done",
                        &serde_json::json!({
                            "type": "response.output_item.done",
                            "response_id": response_id,
                            "output_index": 0,
                            "item": item,
                        }),
                    )))
                    .await
                    .is_err()
                {
                    return;
                }
                items.push(item);
            }

            let tool_offset = usize::from(prefix_state.is_some());
            for (index, state) in tool_states.iter().enumerate() {
                let item = ResponseOutputItem::FunctionCall(ResponseFunctionCallItem::completed(
                    state.id.clone(),
                    state.call_id.clone(),
                    state.name.clone(),
                    state.arguments.clone(),
                ));
                if tx
                    .send(Ok(format_sse_event(
                        "response.function_call_arguments.done",
                        &serde_json::json!({
                            "type": "response.function_call_arguments.done",
                            "response_id": response_id,
                            "output_index": tool_offset + index,
                            "item": item,
                        }),
                    )))
                    .await
                    .is_err()
                {
                    return;
                }
                if tx
                    .send(Ok(format_sse_event(
                        "response.output_item.done",
                        &serde_json::json!({
                            "type": "response.output_item.done",
                            "response_id": response_id,
                            "output_index": tool_offset + index,
                            "item": item,
                        }),
                    )))
                    .await
                    .is_err()
                {
                    return;
                }
                items.push(item);
            }

            items
        } else {
            let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
            let item = ResponseOutputItem::Message(ResponseOutputMessage::completed(
                message_id.clone(),
                full_text.clone(),
            ));

            if tx
                .send(Ok(format_sse_event(
                    "response.output_item.added",
                    &serde_json::json!({
                        "type": "response.output_item.added",
                        "response_id": response_id,
                        "output_index": 0,
                        "item": ResponseOutputItem::Message(ResponseOutputMessage::in_progress(message_id.clone())),
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.content_part.added",
                    &serde_json::json!({
                        "type": "response.content_part.added",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        },
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if !full_text.is_empty()
                && tx
                    .send(Ok(format_sse_event(
                        "response.output_text.delta",
                        &serde_json::json!({
                            "type": "response.output_text.delta",
                            "item_id": message_id,
                            "output_index": 0,
                            "content_index": 0,
                            "delta": full_text,
                        }),
                    )))
                    .await
                    .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.output_text.done",
                    &serde_json::json!({
                        "type": "response.output_text.done",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "text": full_text,
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.content_part.done",
                    &serde_json::json!({
                        "type": "response.content_part.done",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": full_text,
                            "annotations": [],
                        },
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }
            if tx
                .send(Ok(format_sse_event(
                    "response.output_item.done",
                    &serde_json::json!({
                        "type": "response.output_item.done",
                        "response_id": response_id,
                        "output_index": 0,
                        "item": item,
                    }),
                )))
                .await
                .is_err()
            {
                return;
            }

            vec![item]
        };

        let response = ResponseObject::completed(
            response_id.clone(),
            model.clone(),
            req.instructions.clone(),
            req.max_output_tokens,
            req.previous_response_id.clone(),
            req.store,
            req.temperature,
            req.top_p,
            req.metadata.clone(),
            output_items.clone(),
            ResponseUsage::from_request_output(&output),
            req.parallel_tool_calls,
            tool_choice,
            response_tools,
        );

        if tx
            .send(Ok(format_sse_event(
                "response.completed",
                &serde_json::json!({
                    "type": "response.completed",
                    "response": response,
                }),
            )))
            .await
            .is_err()
        {
            return;
        }

        if req.store {
            let mut stored_items = conversation_items;
            if is_gpt_oss {
                if let Some(completion) = output.outputs.first() {
                    let protocol = match load_harmony_protocol() {
                        Ok(protocol) => protocol,
                        Err(_) => return,
                    };
                    let parsed = match parse_gpt_oss_completion(&protocol, &completion.token_ids) {
                        Ok(parsed) => parsed,
                        Err(_) => return,
                    };
                    stored_items.extend(stored_conversation_items_from_protocol_messages(
                        &parsed,
                        &output_items,
                    ));
                }
            } else {
                stored_items.extend(output_items.into_iter().map(StoredConversationItem::Output));
            }

            let mut store = response_store.write().await;
            store.insert(
                response_id,
                StoredResponse {
                    response,
                    input_items,
                    conversation_items: stored_items,
                },
            );
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
}

/// GET /v1/responses/{response_id} -- retrieve a stored response object.
pub async fn get_response(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let store = state.response_store.read().await;
    let stored = store
        .get(&response_id)
        .ok_or_else(|| ApiError::NotFound(format!("response '{}' not found", response_id)))?;
    Ok(Json(stored.response.clone()))
}

/// GET /v1/responses/{response_id}/input_items -- list normalized input items.
pub async fn list_response_input_items(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let store = state.response_store.read().await;
    let stored = store
        .get(&response_id)
        .ok_or_else(|| ApiError::NotFound(format!("response '{}' not found", response_id)))?;
    Ok(Json(ResponseInputItemsList {
        object: "list".to_string(),
        data: stored.input_items.clone(),
        first_id: None,
        last_id: None,
        has_more: false,
    }))
}

fn response_from_output(
    response_id: &str,
    model: &str,
    req: &CreateResponseRequest,
    output: &gpt_oss_core::prelude::RequestOutput,
    function_tools: &[crate::types::responses::ResponseFunctionTool],
    tool_choice: ResponseToolChoice,
    response_tools: Vec<serde_json::Value>,
) -> Result<ResponseObject, ApiError> {
    let completion = output
        .outputs
        .first()
        .ok_or_else(|| ApiError::Internal("engine produced no completion output".into()))?;
    let output_items = response_output_items_from_completion(
        response_id,
        completion,
        req,
        function_tools,
        &tool_choice,
        model,
    )?;
    Ok(ResponseObject::completed(
        response_id.to_string(),
        model.to_string(),
        req.instructions.clone(),
        req.max_output_tokens,
        req.previous_response_id.clone(),
        req.store,
        req.temperature,
        req.top_p,
        req.metadata.clone(),
        output_items,
        ResponseUsage::from_request_output(output),
        req.parallel_tool_calls,
        tool_choice,
        response_tools,
    ))
}

fn response_output_items_from_text(
    response_id: &str,
    text: &str,
    req: &CreateResponseRequest,
    function_tools: &[crate::types::responses::ResponseFunctionTool],
    tool_choice: &ResponseToolChoice,
) -> Result<Vec<ResponseOutputItem>, ApiError> {
    if !req.tools_enabled() {
        return Ok(vec![ResponseOutputItem::Message(
            ResponseOutputMessage::completed(
                format!("msg_{}", uuid::Uuid::new_v4().simple()),
                text.to_string(),
            ),
        )]);
    }

    let allowed_tools: HashSet<&str> = function_tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    let call_prefix = format!("{response_id}_0_");
    match gpt_oss_tokenizer::parse_tool_calls(text, &call_prefix) {
        gpt_oss_tokenizer::ToolParseResult::ToolCalls { prefix_text, calls } => {
            let mut output_items = Vec::new();

            if !prefix_text.is_empty() {
                output_items.push(ResponseOutputItem::Message(
                    ResponseOutputMessage::completed(
                        format!("msg_{}", uuid::Uuid::new_v4().simple()),
                        prefix_text,
                    ),
                ));
            }

            let forced_tool_name = tool_choice.forced_tool_name();
            for call in calls {
                if !allowed_tools.contains(call.name.as_str()) {
                    return Err(ApiError::Internal(format!(
                        "model emitted undeclared function '{}'",
                        call.name
                    )));
                }
                if let Some(forced_name) = forced_tool_name {
                    if call.name != forced_name {
                        return Err(ApiError::Internal(format!(
                            "model emitted function '{}' but tool_choice requires '{}'",
                            call.name, forced_name
                        )));
                    }
                }
                output_items.push(ResponseOutputItem::FunctionCall(
                    ResponseFunctionCallItem::completed(
                        format!("fc_{}", uuid::Uuid::new_v4().simple()),
                        call.id,
                        call.name,
                        call.arguments,
                    ),
                ));
            }

            Ok(output_items)
        }
        gpt_oss_tokenizer::ToolParseResult::PlainText(text) => {
            if matches!(tool_choice, ResponseToolChoice::Mode(mode) if mode == "required")
                || tool_choice.forced_tool_name().is_some()
            {
                return Err(ApiError::Internal(
                    "model did not emit a required function call".into(),
                ));
            }
            Ok(vec![ResponseOutputItem::Message(
                ResponseOutputMessage::completed(
                    format!("msg_{}", uuid::Uuid::new_v4().simple()),
                    text,
                ),
            )])
        }
    }
}

fn response_output_items_from_completion(
    response_id: &str,
    output: &gpt_oss_core::prelude::CompletionOutput,
    req: &CreateResponseRequest,
    function_tools: &[crate::types::responses::ResponseFunctionTool],
    tool_choice: &ResponseToolChoice,
    model_name: &str,
) -> Result<Vec<ResponseOutputItem>, ApiError> {
    if is_gpt_oss_model(model_name) {
        let protocol = load_harmony_protocol()?;
        let parsed = parse_gpt_oss_completion(&protocol, &output.token_ids)?;
        return response_output_items_from_protocol_messages(
            response_id,
            &parsed,
            req,
            function_tools,
            tool_choice,
        );
    }

    response_output_items_from_text(response_id, &output.text, req, function_tools, tool_choice)
}

fn response_output_items_from_protocol_messages(
    response_id: &str,
    messages: &[gpt_oss_tokenizer::ParsedProtocolMessage],
    req: &CreateResponseRequest,
    function_tools: &[crate::types::responses::ResponseFunctionTool],
    tool_choice: &ResponseToolChoice,
) -> Result<Vec<ResponseOutputItem>, ApiError> {
    let allowed_tools: HashSet<&str> = function_tools.iter().map(|tool| tool.name.as_str()).collect();
    let forced_tool_name = tool_choice.forced_tool_name();
    let mut output_items = Vec::new();
    let mut saw_function_call = false;

    for message in messages {
        if message.role != "assistant" {
            continue;
        }

        if let Some(recipient) = &message.recipient {
            let name = recipient
                .strip_prefix("functions.")
                .unwrap_or(recipient.as_str())
                .to_string();
            if !allowed_tools.contains(name.as_str()) {
                return Err(ApiError::Internal(format!(
                    "model emitted undeclared function '{}'",
                    name
                )));
            }
            if let Some(forced_name) = forced_tool_name {
                if name != forced_name {
                    return Err(ApiError::Internal(format!(
                        "model emitted function '{}' but tool_choice requires '{}'",
                        name, forced_name
                    )));
                }
            }
            saw_function_call = true;
            output_items.push(ResponseOutputItem::FunctionCall(
                ResponseFunctionCallItem::completed(
                    format!("fc_{}", uuid::Uuid::new_v4().simple()),
                    format!("{response_id}_{}", output_items.len()),
                    name,
                    message.content.clone(),
                ),
            ));
            continue;
        }

        if message.channel.as_deref() == Some("analysis") {
            continue;
        }

        if !message.content.is_empty() {
            output_items.push(ResponseOutputItem::Message(ResponseOutputMessage::completed(
                format!("msg_{}", uuid::Uuid::new_v4().simple()),
                message.content.clone(),
            )));
        }
    }

    if !saw_function_call
        && (matches!(tool_choice, ResponseToolChoice::Mode(mode) if mode == "required")
            || tool_choice.forced_tool_name().is_some())
    {
        return Err(ApiError::Internal(
            "model did not emit a required function call".into(),
        ));
    }

    if output_items.is_empty() && !req.tools_enabled() {
        output_items.push(ResponseOutputItem::Message(ResponseOutputMessage::completed(
            format!("msg_{}", uuid::Uuid::new_v4().simple()),
            String::new(),
        )));
    }

    Ok(output_items)
}

fn render_conversation_items(
    items: &[StoredConversationItem],
    style: gpt_oss_tokenizer::ToolPromptStyle,
) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    let mut function_names = HashMap::new();

    for item in items {
        match item {
            StoredConversationItem::Input(ResponseInputItem::Message(message)) => {
                messages.push(message.to_chat_message());
            }
            StoredConversationItem::Input(ResponseInputItem::FunctionCallOutput(output)) => {
                let function_name = function_names.get(&output.call_id).map(String::as_str);
                messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: render_function_call_output(output, function_name, style),
                });
            }
            StoredConversationItem::Output(ResponseOutputItem::Message(message)) => {
                messages.push(message.to_chat_message());
            }
            StoredConversationItem::Output(ResponseOutputItem::FunctionCall(call)) => {
                function_names.insert(call.call_id.clone(), call.name.clone());
                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: render_function_call(call, style),
                });
            }
            StoredConversationItem::Protocol(_) => {}
        }
    }

    messages
}

fn render_conversation_protocol_items(
    items: &[StoredConversationItem],
) -> Vec<gpt_oss_tokenizer::ProtocolMessage> {
    let mut messages = Vec::new();
    let mut function_names = HashMap::new();

    for item in items {
        match item {
            StoredConversationItem::Input(ResponseInputItem::Message(message)) => {
                messages.push(gpt_oss_tokenizer::ProtocolMessage::new(
                    &message.role,
                    message
                        .content
                        .iter()
                        .map(|part| part.text.as_str())
                        .collect::<String>(),
                ));
            }
            StoredConversationItem::Input(ResponseInputItem::FunctionCallOutput(output)) => {
                if let Some(function_name) = function_names.get(&output.call_id) {
                    messages.push(
                        gpt_oss_tokenizer::ProtocolMessage::new("tool", output.output_text())
                            .with_author_name(format!("functions.{function_name}"))
                            .with_recipient("assistant")
                            .with_channel("commentary"),
                    );
                }
            }
            StoredConversationItem::Output(ResponseOutputItem::Message(message)) => {
                let content = message
                    .content
                    .iter()
                    .map(|part| part.text.as_str())
                    .collect::<String>();
                let mut protocol_message =
                    gpt_oss_tokenizer::ProtocolMessage::new(&message.role, content);
                if message.role == "assistant" {
                    protocol_message = protocol_message.with_channel("final");
                }
                messages.push(protocol_message);
            }
            StoredConversationItem::Output(ResponseOutputItem::FunctionCall(call)) => {
                function_names.insert(call.call_id.clone(), call.name.clone());
                messages.push(
                    gpt_oss_tokenizer::ProtocolMessage::new("assistant", call.arguments.clone())
                        .with_recipient(format!("functions.{}", call.name))
                        .with_channel("commentary"),
                );
            }
            StoredConversationItem::Protocol(message) => {
                messages.push(message.clone());
            }
        }
    }

    messages
}

fn protocol_message_from_parsed(
    message: &gpt_oss_tokenizer::ParsedProtocolMessage,
) -> gpt_oss_tokenizer::ProtocolMessage {
    let mut protocol = gpt_oss_tokenizer::ProtocolMessage::new(&message.role, message.content.clone());
    if let Some(author_name) = &message.author_name {
        protocol = protocol.with_author_name(author_name.clone());
    }
    if let Some(channel) = &message.channel {
        protocol = protocol.with_channel(channel.clone());
    }
    if let Some(recipient) = &message.recipient {
        protocol = protocol.with_recipient(recipient.clone());
    }
    if let Some(content_type) = &message.content_type {
        protocol = protocol.with_content_type(content_type.clone());
    }
    protocol
}

fn stored_conversation_items_from_protocol_messages(
    messages: &[gpt_oss_tokenizer::ParsedProtocolMessage],
    output_items: &[ResponseOutputItem],
) -> Vec<StoredConversationItem> {
    let mut stored_items = Vec::new();
    let mut visible_items = output_items.iter().cloned();

    for message in messages {
        if message.role != "assistant" {
            continue;
        }

        if message.channel.as_deref() == Some("analysis") {
            if !message.content.is_empty() {
                stored_items.push(StoredConversationItem::Protocol(
                    protocol_message_from_parsed(message),
                ));
            }
            continue;
        }

        if message.recipient.is_some() || !message.content.is_empty() {
            if let Some(item) = visible_items.next() {
                stored_items.push(StoredConversationItem::Output(item));
            }
        }
    }

    stored_items.extend(visible_items.map(StoredConversationItem::Output));
    stored_items
}

fn render_function_call(
    call: &ResponseFunctionCallItem,
    style: gpt_oss_tokenizer::ToolPromptStyle,
) -> String {
    let arguments = serde_json::from_str::<serde_json::Value>(&call.arguments)
        .unwrap_or_else(|_| serde_json::Value::String(call.arguments.clone()));
    let body = serde_json::json!({
        "type": "function_call",
        "call_id": call.call_id,
        "name": call.name,
        "arguments": arguments,
    });
    match style {
        gpt_oss_tokenizer::ToolPromptStyle::Harmony => body.to_string(),
        gpt_oss_tokenizer::ToolPromptStyle::Hermes => {
            format!("<tool_call>{}</tool_call>", body)
        }
    }
}

fn render_function_call_output(
    output: &ResponseFunctionCallOutputItem,
    function_name: Option<&str>,
    style: gpt_oss_tokenizer::ToolPromptStyle,
) -> String {
    let rendered_output = output.output_text();
    match style {
        gpt_oss_tokenizer::ToolPromptStyle::Harmony => {
            let output_value = serde_json::from_str::<serde_json::Value>(&rendered_output)
                .unwrap_or_else(|_| serde_json::Value::String(rendered_output.clone()));
            let mut body = serde_json::json!({
                "type": "function_call_output",
                "call_id": output.call_id,
                "output": output_value,
            });
            if let Some(name) = function_name {
                body["name"] = serde_json::Value::String(name.to_string());
            }
            body.to_string()
        }
        gpt_oss_tokenizer::ToolPromptStyle::Hermes => match function_name {
            Some(name) => format!(
                "Tool output for function {} (call_id {}):\n{}",
                name, output.call_id, rendered_output
            ),
            None => format!(
                "Tool output for call_id {}:\n{}",
                output.call_id, rendered_output
            ),
        },
    }
}

fn diff_text(previous: &str, current: &str) -> String {
    if let Some(suffix) = current.strip_prefix(previous) {
        suffix.to_string()
    } else {
        current.to_string()
    }
}

fn format_sse_event<T: Serialize>(event: &str, payload: &T) -> String {
    let json = serde_json::to_string(payload).unwrap_or_default();
    format!("event: {event}\ndata: {json}\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use crate::types::responses::{
        ResponseInputMessage, ResponseInputTextPart, ResponseSpecificToolChoice,
    };
    use crate::{build_router, AppState};
    use axum_test::TestServer;
    use gpt_oss_core::prelude::{
        CompletionOutput, FinishReason, RequestId, RequestOutput, SamplingParams,
    };
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::Tokenizer as HfTokenizer;
    use tokio::sync::Mutex as AsyncMutex;
    use tokio_stream::wrappers::ReceiverStream;

    struct FakeEngine {
        outputs: AsyncMutex<VecDeque<Vec<RequestOutput>>>,
        prompts: Mutex<Vec<String>>,
        sampling_params: Mutex<Vec<SamplingParams>>,
    }

    impl FakeEngine {
        fn new(outputs: Vec<Vec<RequestOutput>>) -> Self {
            Self {
                outputs: AsyncMutex::new(outputs.into()),
                prompts: Mutex::new(Vec::new()),
                sampling_params: Mutex::new(Vec::new()),
            }
        }

        fn prompts(&self) -> Vec<String> {
            self.prompts.lock().unwrap().clone()
        }

        fn sampling_params(&self) -> Vec<SamplingParams> {
            self.sampling_params.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl crate::server::InferenceEngine for FakeEngine {
        async fn generate(
            &self,
            prompt: String,
            params: SamplingParams,
        ) -> gpt_oss_core::prelude::Result<(RequestId, ReceiverStream<RequestOutput>)> {
            self.prompts.lock().unwrap().push(prompt);
            self.sampling_params.lock().unwrap().push(params);
            let maybe_outputs = self.outputs.lock().await.pop_front();
            let outputs = maybe_outputs.expect("fake engine ran out of queued outputs");
            let (tx, rx) = tokio::sync::mpsc::channel(outputs.len().max(1));
            for output in outputs {
                tx.send(output).await.unwrap();
            }
            drop(tx);
            Ok((RequestId(1), ReceiverStream::new(rx)))
        }
    }

    fn make_test_tokenizer() -> gpt_oss_tokenizer::Tokenizer {
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("tool".to_string(), 2);
        vocab.insert("call".to_string(), 3);
        vocab.insert("weather".to_string(), 4);
        vocab.insert("time".to_string(), 5);
        vocab.insert("[UNK]".to_string(), 6);

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, vec![])
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();

        let mut hf = HfTokenizer::new(bpe);
        hf.with_pre_tokenizer(Some(Whitespace {}));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        hf.save(&path, false).unwrap();
        gpt_oss_tokenizer::Tokenizer::from_file(&path).unwrap()
    }

    fn make_server(outputs: Vec<Vec<RequestOutput>>) -> (TestServer, Arc<FakeEngine>) {
        make_server_with_model("test", outputs)
    }

    fn make_server_with_model(
        model_name: &str,
        outputs: Vec<Vec<RequestOutput>>,
    ) -> (TestServer, Arc<FakeEngine>) {
        let engine = Arc::new(FakeEngine::new(outputs));
        let state = Arc::new(AppState::new(
            engine.clone(),
            model_name.to_string(),
            crate::runtime_policy::RuntimeDecision {
                runtime_mode: gpt_oss_engine::RuntimeMode::Experimental,
                backend_path: crate::runtime_policy::RuntimeBackendPath::Mock,
                reason: "test mock backend".into(),
            },
            make_test_tokenizer(),
        ));
        let server = TestServer::new(build_router(state)).unwrap();
        (server, engine)
    }

    fn request_output(text: &str, finished: bool) -> RequestOutput {
        RequestOutput {
            request_id: RequestId(1),
            prompt: "prompt".into(),
            prompt_token_ids: vec![1, 2, 3],
            prompt_logprobs: None,
            outputs: vec![CompletionOutput {
                index: 0,
                text: text.to_string(),
                token_ids: vec![10, 11],
                cumulative_logprob: -0.1,
                logprobs: None,
                finish_reason: finished.then_some(FinishReason::Stop),
            }],
            finished,
        }
    }

    fn gpt_oss_request_output(text: &str, visible_text: &str, finished: bool) -> RequestOutput {
        let token_ids = gpt_oss_tokenizer::HarmonyProtocol::gpt_oss()
            .unwrap()
            .encode_completion_text(text);
        RequestOutput {
            request_id: RequestId(1),
            prompt: "prompt".into(),
            prompt_token_ids: vec![1, 2, 3],
            prompt_logprobs: None,
            outputs: vec![CompletionOutput {
                index: 0,
                text: visible_text.to_string(),
                token_ids,
                cumulative_logprob: -0.1,
                logprobs: None,
                finish_reason: finished.then_some(FinishReason::Stop),
            }],
            finished,
        }
    }

    fn gpt_oss_stream_request_output_from_fragments(
        fragments: &[&str],
        visible_text: &str,
        finished: bool,
    ) -> RequestOutput {
        let protocol = gpt_oss_tokenizer::HarmonyProtocol::gpt_oss().unwrap();
        let mut token_ids = Vec::new();
        for fragment in fragments {
            token_ids.extend(protocol.encode_stream_fragment_text(fragment));
        }
        RequestOutput {
            request_id: RequestId(1),
            prompt: "prompt".into(),
            prompt_token_ids: vec![1, 2, 3],
            prompt_logprobs: None,
            outputs: vec![CompletionOutput {
                index: 0,
                text: visible_text.to_string(),
                token_ids,
                cumulative_logprob: -0.1,
                logprobs: None,
                finish_reason: finished.then_some(FinishReason::Stop),
            }],
            finished,
        }
    }

    fn parse_sse_events(body: &str) -> Vec<(String, serde_json::Value)> {
        body.split("\n\n")
            .filter_map(|chunk| {
                let mut event = None;
                let mut data = None;
                for line in chunk.lines() {
                    if let Some(value) = line.strip_prefix("event: ") {
                        event = Some(value.to_string());
                    } else if let Some(value) = line.strip_prefix("data: ") {
                        data = Some(serde_json::from_str(value).unwrap());
                    }
                }
                event.zip(data)
            })
            .collect()
    }

    fn make_tool_request(tool_choice: ResponseToolChoice) -> CreateResponseRequest {
        CreateResponseRequest {
            model: "test".into(),
            input: Some(crate::types::responses::ResponseInput::Text("hi".into())),
            instructions: None,
            max_output_tokens: Some(32),
            temperature: 1.0,
            top_p: 1.0,
            stream: false,
            store: true,
            previous_response_id: None,
            metadata: Default::default(),
            background: None,
            tools: Some(vec![serde_json::json!({
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"}
            })]),
            tool_choice: Some(tool_choice),
            parallel_tool_calls: true,
            text: None,
            reasoning: None,
            conversation: None,
            include: None,
            truncation: None,
        }
    }

    #[test]
    fn diff_text_returns_suffix_when_possible() {
        assert_eq!(diff_text("Hello", "Hello there"), " there");
    }

    #[test]
    fn diff_text_returns_current_when_prefix_does_not_match() {
        assert_eq!(diff_text("Hi", "Hello"), "Hello");
    }

    #[test]
    fn format_sse_event_includes_event_name() {
        let rendered = format_sse_event("response.created", &serde_json::json!({"ok": true}));
        assert!(rendered.starts_with("event: response.created\n"));
        assert!(rendered.contains("data: {\"ok\":true}\n\n"));
    }

    #[tokio::test]
    async fn create_response_route_adds_harmony_assistant_stop_strings_to_sampling_params() {
        let (server, engine) = make_server_with_model(
            "openai/gpt-oss-20b",
            vec![vec![gpt_oss_request_output(
                "<|channel|>final<|message|>Hello.<|return|>",
                "Hello.",
                true,
            )]],
        );

        let response = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "openai/gpt-oss-20b",
                "input": "Say hello.",
                "stream": false,
                "store": false,
            }))
            .await;

        response.assert_status_ok();
        let sampling_params = engine.sampling_params();
        assert_eq!(sampling_params.len(), 1);
        assert!(sampling_params[0]
            .stop_strings
            .contains(&"<|call|>".to_string()));
        assert!(sampling_params[0]
            .stop_strings
            .contains(&"<|return|>".to_string()));
    }

    #[test]
    fn response_output_items_parse_function_calls() {
        let req = make_tool_request(ResponseToolChoice::Mode("auto".into()));
        let tools = req.normalize_function_tools().unwrap();
        let items = response_output_items_from_text(
            "resp_test",
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Boston\"}}</tool_call>",
            &req,
            &tools,
            &req.effective_tool_choice(),
        )
        .unwrap();

        assert_eq!(items.len(), 1);
        match &items[0] {
            ResponseOutputItem::FunctionCall(call) => {
                assert_eq!(call.name, "get_weather");
                assert!(call.arguments.contains("Boston"));
            }
            _ => panic!("expected function_call item"),
        }
    }

    #[test]
    fn response_output_items_parse_harmony_function_calls() {
        let req = make_tool_request(ResponseToolChoice::Mode("auto".into()));
        let tools = req.normalize_function_tools().unwrap();
        let items = response_output_items_from_text(
            "resp_test",
            "{\"type\":\"function_call\",\"call_id\":\"abc123\",\"name\":\"get_weather\",\"arguments\":{\"location\":\"Boston\"}}",
            &req,
            &tools,
            &req.effective_tool_choice(),
        )
        .unwrap();

        assert_eq!(items.len(), 1);
        match &items[0] {
            ResponseOutputItem::FunctionCall(call) => {
                assert_eq!(call.name, "get_weather");
                assert!(call.arguments.contains("Boston"));
            }
            _ => panic!("expected function_call item"),
        }
    }

    #[test]
    fn response_output_items_reject_plain_text_when_required() {
        let req = make_tool_request(ResponseToolChoice::Mode("required".into()));
        let tools = req.normalize_function_tools().unwrap();
        let err = response_output_items_from_text(
            "resp_test",
            "plain text",
            &req,
            &tools,
            &req.effective_tool_choice(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("required function call"));
    }

    #[test]
    fn response_output_items_reject_wrong_forced_tool() {
        let mut req = make_tool_request(ResponseToolChoice::Specific(ResponseSpecificToolChoice {
            choice_type: "function".into(),
            name: "get_weather".into(),
        }));
        req.tools
            .as_mut()
            .unwrap()
            .push(serde_json::json!({"type": "function", "name": "get_time"}));
        let tools = req.normalize_function_tools().unwrap();
        let err = response_output_items_from_text(
            "resp_test",
            "<tool_call>{\"name\":\"get_time\",\"arguments\":{}}</tool_call>",
            &req,
            &tools,
            &req.effective_tool_choice(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("tool_choice requires"));
    }

    #[test]
    fn response_output_items_from_protocol_messages_reject_plain_text_when_required() {
        let req = make_tool_request(ResponseToolChoice::Mode("required".into()));
        let tools = req.normalize_function_tools().unwrap();
        let err = response_output_items_from_protocol_messages(
            "resp_test",
            &[gpt_oss_tokenizer::ParsedProtocolMessage {
                role: "assistant".into(),
                author_name: None,
                content: "plain text".into(),
                channel: Some("final".into()),
                recipient: None,
                content_type: None,
            }],
            &req,
            &tools,
            &req.effective_tool_choice(),
        )
        .unwrap_err();

        assert!(err.to_string().contains("required function call"));
    }

    #[test]
    fn response_output_items_from_protocol_messages_reject_wrong_forced_tool() {
        let mut req = make_tool_request(ResponseToolChoice::Specific(ResponseSpecificToolChoice {
            choice_type: "function".into(),
            name: "get_weather".into(),
        }));
        req.tools
            .as_mut()
            .unwrap()
            .push(serde_json::json!({"type": "function", "name": "get_time"}));
        let tools = req.normalize_function_tools().unwrap();
        let err = response_output_items_from_protocol_messages(
            "resp_test",
            &[gpt_oss_tokenizer::ParsedProtocolMessage {
                role: "assistant".into(),
                author_name: None,
                content: "{\"timezone\":\"UTC\"}".into(),
                channel: Some("commentary".into()),
                recipient: Some("functions.get_time".into()),
                content_type: Some("<|constrain|>json".into()),
            }],
            &req,
            &tools,
            &req.effective_tool_choice(),
        )
        .unwrap_err();

        assert!(err.to_string().contains("tool_choice requires"));
    }

    #[test]
    fn response_output_items_from_protocol_messages_ignore_analysis_and_keep_final_text() {
        let req = make_tool_request(ResponseToolChoice::Mode("auto".into()));
        let tools = req.normalize_function_tools().unwrap();
        let items = response_output_items_from_protocol_messages(
            "resp_test",
            &[
                gpt_oss_tokenizer::ParsedProtocolMessage {
                    role: "assistant".into(),
                    author_name: None,
                    content: "Need weather lookup.".into(),
                    channel: Some("analysis".into()),
                    recipient: None,
                    content_type: None,
                },
                gpt_oss_tokenizer::ParsedProtocolMessage {
                    role: "assistant".into(),
                    author_name: None,
                    content: "It is 18C and sunny.".into(),
                    channel: Some("final".into()),
                    recipient: None,
                    content_type: None,
                },
            ],
            &req,
            &tools,
            &req.effective_tool_choice(),
        )
        .unwrap();

        assert_eq!(items.len(), 1);
        match &items[0] {
            ResponseOutputItem::Message(message) => {
                assert_eq!(message.content[0].text, "It is 18C and sunny.");
            }
            _ => panic!("expected final message item"),
        }
    }

    #[test]
    fn stored_conversation_items_from_protocol_messages_preserve_hidden_analysis_order() {
        let output_items = vec![
            ResponseOutputItem::FunctionCall(ResponseFunctionCallItem::completed(
                "fc_1",
                "resp_test_0",
                "get_weather",
                "{\"location\":\"Boston\"}",
            )),
            ResponseOutputItem::Message(ResponseOutputMessage::completed(
                "msg_1",
                "It is 18C and sunny.",
            )),
        ];
        let stored_items = stored_conversation_items_from_protocol_messages(
            &[
                gpt_oss_tokenizer::ParsedProtocolMessage {
                    role: "assistant".into(),
                    author_name: None,
                    content: "Need weather lookup.".into(),
                    channel: Some("analysis".into()),
                    recipient: None,
                    content_type: None,
                },
                gpt_oss_tokenizer::ParsedProtocolMessage {
                    role: "assistant".into(),
                    author_name: None,
                    content: "{\"location\":\"Boston\"}".into(),
                    channel: Some("commentary".into()),
                    recipient: Some("functions.get_weather".into()),
                    content_type: Some("<|constrain|>json".into()),
                },
                gpt_oss_tokenizer::ParsedProtocolMessage {
                    role: "assistant".into(),
                    author_name: None,
                    content: "It is 18C and sunny.".into(),
                    channel: Some("final".into()),
                    recipient: None,
                    content_type: None,
                },
            ],
            &output_items,
        );

        assert_eq!(stored_items.len(), 3);
        match &stored_items[0] {
            StoredConversationItem::Protocol(message) => {
                assert_eq!(message.channel.as_deref(), Some("analysis"));
                assert_eq!(message.content, "Need weather lookup.");
            }
            _ => panic!("expected hidden protocol message"),
        }
        match &stored_items[1] {
            StoredConversationItem::Output(ResponseOutputItem::FunctionCall(call)) => {
                assert_eq!(call.call_id, "resp_test_0");
            }
            _ => panic!("expected function call output item"),
        }
        match &stored_items[2] {
            StoredConversationItem::Output(ResponseOutputItem::Message(message)) => {
                assert_eq!(message.content[0].text, "It is 18C and sunny.");
            }
            _ => panic!("expected visible message output item"),
        }
    }

    #[test]
    fn render_conversation_items_preserves_tool_history() {
        let items = vec![
            StoredConversationItem::Input(ResponseInputItem::Message(ResponseInputMessage::new(
                "user",
                vec![ResponseInputTextPart::new("Weather?")],
            ))),
            StoredConversationItem::Output(ResponseOutputItem::FunctionCall(
                ResponseFunctionCallItem::completed(
                    "fc_1",
                    "call_1",
                    "get_weather",
                    "{\"location\":\"Boston\"}",
                ),
            )),
            StoredConversationItem::Input(ResponseInputItem::FunctionCallOutput(
                ResponseFunctionCallOutputItem {
                    call_id: "call_1".into(),
                    output: serde_json::json!({"temp_c": 18}),
                },
            )),
        ];

        let messages =
            render_conversation_items(&items, gpt_oss_tokenizer::ToolPromptStyle::Hermes);
        assert_eq!(messages.len(), 3);
        assert!(messages[1].content.contains("<tool_call>"));
        assert!(messages[2].content.contains("get_weather"));
        assert!(messages[2].content.contains("temp_c"));
    }

    #[test]
    fn render_function_call_serializes_json_arguments() {
        let call = ResponseFunctionCallItem::completed(
            "fc_1",
            "call_1",
            "get_weather",
            "{\"location\":\"Boston\"}",
        );
        let rendered = render_function_call(&call, gpt_oss_tokenizer::ToolPromptStyle::Hermes);
        assert!(rendered.contains("<tool_call>"));
        assert!(rendered.contains("\"name\":\"get_weather\""));
        assert!(rendered.contains("\"location\":\"Boston\""));
    }

    #[test]
    fn render_function_call_output_includes_call_id() {
        let rendered = render_function_call_output(
            &ResponseFunctionCallOutputItem {
                call_id: "call_123".into(),
                output: serde_json::json!({"ok": true}),
            },
            None,
            gpt_oss_tokenizer::ToolPromptStyle::Hermes,
        );
        assert!(rendered.contains("call_123"));
        assert!(rendered.contains("\"ok\":true"));
    }

    #[test]
    fn render_conversation_items_uses_harmony_json_for_gpt_oss_history() {
        let items = vec![
            StoredConversationItem::Input(ResponseInputItem::Message(ResponseInputMessage::new(
                "user",
                vec![ResponseInputTextPart::new("Weather?")],
            ))),
            StoredConversationItem::Output(ResponseOutputItem::FunctionCall(
                ResponseFunctionCallItem::completed(
                    "fc_1",
                    "call_1",
                    "get_weather",
                    "{\"location\":\"Boston\"}",
                ),
            )),
            StoredConversationItem::Input(ResponseInputItem::FunctionCallOutput(
                ResponseFunctionCallOutputItem {
                    call_id: "call_1".into(),
                    output: serde_json::json!({"temp_c": 18}),
                },
            )),
        ];

        let messages =
            render_conversation_items(&items, gpt_oss_tokenizer::ToolPromptStyle::Harmony);
        assert_eq!(messages.len(), 3);
        assert!(messages[1].content.contains("\"type\":\"function_call\""));
        assert!(messages[1].content.contains("\"call_id\":\"call_1\""));
        assert!(messages[2]
            .content
            .contains("\"type\":\"function_call_output\""));
        assert!(messages[2].content.contains("\"temp_c\":18"));
        assert!(!messages[1].content.contains("<tool_call>"));
    }

    #[test]
    fn render_function_call_harmony_serializes_typed_json() {
        let call = ResponseFunctionCallItem::completed(
            "fc_1",
            "call_1",
            "get_weather",
            "{\"location\":\"Boston\"}",
        );
        let rendered = render_function_call(&call, gpt_oss_tokenizer::ToolPromptStyle::Harmony);
        assert!(rendered.contains("\"type\":\"function_call\""));
        assert!(rendered.contains("\"call_id\":\"call_1\""));
        assert!(rendered.contains("\"location\":\"Boston\""));
        assert!(!rendered.contains("<tool_call>"));
    }

    #[test]
    fn render_function_call_output_harmony_serializes_typed_json() {
        let rendered = render_function_call_output(
            &ResponseFunctionCallOutputItem {
                call_id: "call_123".into(),
                output: serde_json::json!({"ok": true}),
            },
            Some("lookup"),
            gpt_oss_tokenizer::ToolPromptStyle::Harmony,
        );
        assert!(rendered.contains("\"type\":\"function_call_output\""));
        assert!(rendered.contains("\"call_id\":\"call_123\""));
        assert!(rendered.contains("\"name\":\"lookup\""));
        assert!(rendered.contains("\"ok\":true"));
    }

    #[tokio::test]
    async fn create_response_route_returns_function_call_items() {
        let (server, _) = make_server(vec![vec![request_output(
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Boston\"}}</tool_call>",
            true,
        )]]);

        let response = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "test",
                "input": "What's the weather?",
                "store": true,
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }],
            }))
            .await;

        response.assert_status_ok();
        let body = response.json::<serde_json::Value>();
        assert_eq!(body["object"], "response");
        assert_eq!(body["status"], "completed");
        assert_eq!(body["output"][0]["type"], "function_call");
        assert_eq!(body["output"][0]["name"], "get_weather");
        assert_eq!(body["output"][0]["status"], "completed");
    }

    #[tokio::test]
    async fn previous_response_id_replays_function_calls_and_outputs() {
        let (server, engine) = make_server(vec![
            vec![request_output(
                "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Boston\"}}</tool_call>",
                true,
            )],
            vec![request_output("It is 18C and sunny.", true)],
        ]);

        let first = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "test",
                "input": "What's the weather?",
                "store": true,
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }],
            }))
            .await;
        first.assert_status_ok();
        let first_body = first.json::<serde_json::Value>();
        let first_id = first_body["id"].as_str().unwrap().to_string();
        let call_id = first_body["output"][0]["call_id"]
            .as_str()
            .unwrap()
            .to_string();

        let second = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "test",
                "previous_response_id": first_id,
                "store": true,
                "input": [{
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": {"temp_c": 18, "conditions": "sunny"},
                }],
            }))
            .await;
        second.assert_status_ok();
        let second_body = second.json::<serde_json::Value>();
        assert_eq!(second_body["output"][0]["type"], "message");
        assert_eq!(second_body["previous_response_id"], first_id);

        let prompts = engine.prompts();
        assert_eq!(prompts.len(), 2);
        assert!(prompts[1].contains("<tool_call>"));
        assert!(prompts[1].contains("Tool output for function get_weather"));
        assert!(prompts[1].contains("\"temp_c\":18"));

        let retrieved = server.get(&format!("/v1/responses/{}", first_id)).await;
        retrieved.assert_status_ok();
        let retrieved = retrieved.json::<serde_json::Value>();
        assert_eq!(retrieved["output"][0]["type"], "function_call");

        let input_items = server
            .get(&format!(
                "/v1/responses/{}/input_items",
                second_body["id"].as_str().unwrap()
            ))
            .await;
        input_items.assert_status_ok();
        let input_items = input_items.json::<serde_json::Value>();
        assert_eq!(input_items["data"][0]["type"], "function_call_output");
        assert_eq!(input_items["data"][0]["output"]["temp_c"], 18);
    }

    #[tokio::test]
    async fn previous_response_id_uses_harmony_json_for_gpt_oss_replay() {
        let (server, engine) = make_server_with_model(
            "openai/gpt-oss-20b",
            vec![
                vec![gpt_oss_request_output(
                    " to=functions.get_weather<|channel|>commentary<|constrain|>json<|message|>{\"location\":\"Boston\"}<|call|>",
                    "",
                    true,
                )],
                vec![gpt_oss_request_output(
                    "<|channel|>final<|message|>It is 18C and sunny.<|end|>",
                    "It is 18C and sunny.",
                    true,
                )],
            ],
        );

        let first = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "openai/gpt-oss-20b",
                "input": "What's the weather?",
                "store": true,
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }],
            }))
            .await;
        first.assert_status_ok();
        let first_body = first.json::<serde_json::Value>();
        let first_id = first_body["id"].as_str().unwrap().to_string();
        let call_id = first_body["output"][0]["call_id"]
            .as_str()
            .unwrap()
            .to_string();

        let second = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "openai/gpt-oss-20b",
                "previous_response_id": first_id,
                "store": true,
                "input": [{
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": {"temp_c": 18, "conditions": "sunny"},
                }],
            }))
            .await;
        second.assert_status_ok();

        let prompts = engine.prompts();
        assert_eq!(prompts.len(), 2);
        assert!(prompts[1].contains("to=functions.get_weather"), "{}", prompts[1]);
        assert!(prompts[1].contains("<|channel|>commentary"), "{}", prompts[1]);
        assert!(
            prompts[1].contains("<|start|>functions.get_weather"),
            "{}",
            prompts[1]
        );
        assert!(prompts[1].contains("to=assistant"), "{}", prompts[1]);
        assert!(prompts[1].contains("\"temp_c\":18"));
        assert!(!prompts[1].contains("<tool_call>"));
        assert!(!prompts[1].contains("Tool output for function"));
    }

    #[tokio::test]
    async fn previous_response_id_preserves_hidden_analysis_for_gpt_oss_replay() {
        let (server, engine) = make_server_with_model(
            "openai/gpt-oss-20b",
            vec![
                vec![gpt_oss_request_output(
                    "<|channel|>analysis<|message|>Need weather lookup.<|end|><|start|>assistant to=functions.get_weather<|channel|>commentary<|constrain|>json<|message|>{\"location\":\"Boston\"}<|call|>",
                    "",
                    true,
                )],
                vec![gpt_oss_request_output(
                    "<|channel|>final<|message|>It is 18C and sunny.<|end|>",
                    "It is 18C and sunny.",
                    true,
                )],
            ],
        );

        let first = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "openai/gpt-oss-20b",
                "input": "What's the weather?",
                "store": true,
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }],
            }))
            .await;
        first.assert_status_ok();
        let first_body = first.json::<serde_json::Value>();
        let first_id = first_body["id"].as_str().unwrap().to_string();
        let call_id = first_body["output"][0]["call_id"]
            .as_str()
            .unwrap()
            .to_string();

        let second = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "openai/gpt-oss-20b",
                "previous_response_id": first_id,
                "store": true,
                "input": [{
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": {"temp_c": 18, "conditions": "sunny"},
                }],
            }))
            .await;
        second.assert_status_ok();
        let second_body = second.json::<serde_json::Value>();

        assert_eq!(
            second_body["output"][0]["content"][0]["text"],
            "It is 18C and sunny."
        );
        assert_ne!(
            second_body["output"][0]["content"][0]["text"],
            "Need weather lookup."
        );

        let prompts = engine.prompts();
        assert_eq!(prompts.len(), 2);
        assert!(prompts[1].contains("<|channel|>analysis"), "{}", prompts[1]);
        assert!(prompts[1].contains("Need weather lookup."), "{}", prompts[1]);
        assert!(prompts[1].contains("to=functions.get_weather"), "{}", prompts[1]);
        assert!(prompts[1].contains("\"temp_c\":18"), "{}", prompts[1]);
    }

    #[tokio::test]
    async fn previous_response_id_replays_harmony_tool_chain_in_order() {
        let (server, engine) = make_server_with_model(
            "openai/gpt-oss-20b",
            vec![
                vec![gpt_oss_request_output(
                    " to=functions.get_weather<|channel|>commentary<|constrain|>json<|message|>{\"location\":\"Boston\"}<|call|>",
                    "",
                    true,
                )],
                vec![gpt_oss_request_output(
                    " to=functions.get_time<|channel|>commentary<|constrain|>json<|message|>{\"timezone\":\"America/New_York\"}<|call|>",
                    "",
                    true,
                )],
                vec![gpt_oss_request_output(
                    "<|channel|>final<|message|>It is 18C and 09:00.<|end|>",
                    "It is 18C and 09:00.",
                    true,
                )],
            ],
        );

        let first = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "openai/gpt-oss-20b",
                "input": "What's the weather and time?",
                "store": true,
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "get_time",
                        "description": "Get time",
                        "parameters": {"type": "object"},
                    }
                ],
            }))
            .await;
        first.assert_status_ok();
        let first_body = first.json::<serde_json::Value>();
        let first_id = first_body["id"].as_str().unwrap().to_string();
        let weather_call_id = first_body["output"][0]["call_id"]
            .as_str()
            .unwrap()
            .to_string();

        let second = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "openai/gpt-oss-20b",
                "previous_response_id": first_id,
                "store": true,
                "input": [{
                    "type": "function_call_output",
                    "call_id": weather_call_id,
                    "output": {"temp_c": 18, "conditions": "sunny"},
                }],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "get_time",
                        "description": "Get time",
                        "parameters": {"type": "object"},
                    }
                ],
            }))
            .await;
        second.assert_status_ok();
        let second_body = second.json::<serde_json::Value>();
        let second_id = second_body["id"].as_str().unwrap().to_string();
        let time_call_id = second_body["output"][0]["call_id"]
            .as_str()
            .unwrap()
            .to_string();

        let third = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "openai/gpt-oss-20b",
                "previous_response_id": second_id,
                "store": true,
                "input": [{
                    "type": "function_call_output",
                    "call_id": time_call_id,
                    "output": {"time": "09:00"},
                }],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "get_time",
                        "description": "Get time",
                        "parameters": {"type": "object"},
                    }
                ],
            }))
            .await;
        third.assert_status_ok();
        let third_body = third.json::<serde_json::Value>();
        assert_eq!(third_body["output"][0]["type"], "message");
        assert_eq!(
            third_body["output"][0]["content"][0]["text"],
            "It is 18C and 09:00."
        );

        let prompts = engine.prompts();
        assert_eq!(prompts.len(), 3);
        assert!(prompts[1].contains("to=functions.get_weather"), "{}", prompts[1]);
        assert!(prompts[1].contains("<|start|>functions.get_weather"), "{}", prompts[1]);
        assert!(prompts[1].contains("\"temp_c\":18"), "{}", prompts[1]);
        assert!(prompts[2].contains("to=functions.get_weather"), "{}", prompts[2]);
        assert!(prompts[2].contains("to=functions.get_time"), "{}", prompts[2]);
        assert!(prompts[2].contains("\"temp_c\":18"), "{}", prompts[2]);
        assert!(prompts[2].contains("\"time\":\"09:00\""), "{}", prompts[2]);

        let weather_call_idx = prompts[2].find("to=functions.get_weather").unwrap();
        let weather_output_idx = prompts[2].find("<|start|>functions.get_weather").unwrap();
        let time_call_idx = prompts[2].find("to=functions.get_time").unwrap();
        let time_output_idx = prompts[2]
            .rfind("<|start|>functions.get_time")
            .unwrap();
        assert!(weather_call_idx < weather_output_idx);
        assert!(weather_output_idx < time_call_idx);
        assert!(time_call_idx < time_output_idx);
        assert!(!prompts[2].contains("<tool_call>"));
    }

    #[tokio::test]
    async fn streaming_tool_responses_emit_function_call_events_and_store_output() {
        let (server, _) = make_server(vec![vec![
            request_output(
                "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Bos",
                false,
            ),
            request_output(
                "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Boston\"}}</tool_call><tool_call>{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"UT",
                false,
            ),
            request_output(
                "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"location\":\"Boston\"}}</tool_call><tool_call>{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"UTC\"}}</tool_call>",
                true,
            ),
        ]]);

        let response = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "test",
                "input": "Call the tools you need.",
                "stream": true,
                "store": true,
                "parallel_tool_calls": true,
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "get_time",
                        "description": "Get time",
                        "parameters": {"type": "object"},
                    }
                ],
            }))
            .await;

        response.assert_status_ok();
        let events = parse_sse_events(&response.text());
        let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert!(names.contains(&"response.created"));
        assert!(names.contains(&"response.in_progress"));
        assert!(names.contains(&"response.output_item.added"));
        assert!(names.contains(&"response.function_call_arguments.delta"));
        assert!(names.contains(&"response.function_call_arguments.done"));
        assert!(names.contains(&"response.output_item.done"));
        assert!(names.contains(&"response.completed"));

        let delta_count = names
            .iter()
            .filter(|name| **name == "response.function_call_arguments.delta")
            .count();
        assert!(delta_count >= 2);

        let completed = events
            .iter()
            .find(|(name, _)| name == "response.completed")
            .unwrap();
        let response_id = completed.1["response"]["id"].as_str().unwrap();
        assert_eq!(
            completed.1["response"]["output"].as_array().unwrap().len(),
            2
        );
        assert_eq!(
            completed.1["response"]["output"][0]["type"],
            "function_call"
        );
        assert_eq!(completed.1["response"]["output"][1]["name"], "get_time");

        let stored = server.get(&format!("/v1/responses/{response_id}")).await;
        stored.assert_status_ok();
        let stored = stored.json::<serde_json::Value>();
        assert_eq!(stored["output"].as_array().unwrap().len(), 2);
        assert_eq!(stored["output"][0]["name"], "get_weather");
        assert_eq!(stored["output"][1]["name"], "get_time");
    }

    #[tokio::test]
    async fn gpt_oss_streaming_tool_responses_use_incremental_protocol_parsing() {
        let (server, _) = make_server_with_model(
            "openai/gpt-oss-20b",
            vec![vec![gpt_oss_stream_request_output_from_fragments(
                &[
                    " to=functions.get_weather",
                    "<|channel|>commentary",
                    "<|constrain|>json",
                    "<|message|>",
                    "{\"location\":\"Boston\"}",
                    "<|call|>",
                ],
                "",
                true,
            )]],
        );

        let response = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "openai/gpt-oss-20b",
                "input": "Call the tools you need.",
                "stream": true,
                "store": true,
                "parallel_tool_calls": true,
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "get_time",
                        "description": "Get time",
                        "parameters": {"type": "object"},
                    }
                ],
            }))
            .await;

        response.assert_status_ok();
        let body = response.text();
        let events = parse_sse_events(&body);
        let names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();
        assert!(names.contains(&"response.function_call_arguments.delta"), "{body}");
        assert!(names.contains(&"response.function_call_arguments.done"));

        let completed = events
            .iter()
            .find(|(name, _)| name == "response.completed")
            .unwrap();
        assert_eq!(
            completed.1["response"]["output"][0]["name"],
            "get_weather"
        );
        assert_eq!(
            completed.1["response"]["output"][0]["arguments"],
            "{\"location\":\"Boston\"}"
        );
    }

    #[tokio::test]
    async fn gpt_oss_streaming_tool_response_events_are_ordered() {
        let (server, _) = make_server_with_model(
            "openai/gpt-oss-20b",
            vec![vec![gpt_oss_stream_request_output_from_fragments(
                &[
                    " to=functions.get_weather",
                    "<|channel|>commentary",
                    "<|constrain|>json",
                    "<|message|>",
                    "{\"location\":\"Boston\"}",
                    "<|call|>",
                ],
                "",
                true,
            )]],
        );

        let response = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "openai/gpt-oss-20b",
                "input": "Call the tools you need.",
                "stream": true,
                "store": true,
                "parallel_tool_calls": true,
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                }],
            }))
            .await;

        response.assert_status_ok();
        let events = parse_sse_events(&response.text());
        let event_names: Vec<&str> = events.iter().map(|(name, _)| name.as_str()).collect();

        let created_idx = event_names.iter().position(|name| *name == "response.created").unwrap();
        let in_progress_idx = event_names
            .iter()
            .position(|name| *name == "response.in_progress")
            .unwrap();
        let added_idx = event_names
            .iter()
            .position(|name| *name == "response.output_item.added")
            .unwrap();
        let delta_idx = event_names
            .iter()
            .position(|name| *name == "response.function_call_arguments.delta")
            .unwrap();
        let done_idx = event_names
            .iter()
            .position(|name| *name == "response.function_call_arguments.done")
            .unwrap();
        let output_done_idx = event_names
            .iter()
            .position(|name| *name == "response.output_item.done")
            .unwrap();
        let completed_idx = event_names
            .iter()
            .position(|name| *name == "response.completed")
            .unwrap();

        assert!(created_idx < in_progress_idx);
        assert!(in_progress_idx < added_idx);
        assert!(added_idx < delta_idx);
        assert!(delta_idx < done_idx);
        assert!(done_idx < output_done_idx);
        assert!(output_done_idx < completed_idx);
    }

    #[tokio::test]
    async fn gpt_oss_streaming_text_response_uses_incremental_protocol_parsing() {
        let (server, _) = make_server_with_model(
            "openai/gpt-oss-20b",
            vec![vec![
                gpt_oss_stream_request_output_from_fragments(
                    &["<|channel|>final", "<|message|>", "Hel"],
                    "",
                    false,
                ),
                gpt_oss_stream_request_output_from_fragments(
                    &["<|channel|>final", "<|message|>", "Hel", "lo world", "<|end|>"],
                    "",
                    true,
                ),
            ]],
        );

        let response = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "openai/gpt-oss-20b",
                "input": "Say hello.",
                "stream": true,
                "store": false,
            }))
            .await;

        response.assert_status_ok();
        let body = response.text();
        let events = parse_sse_events(&body);
        let text_deltas: Vec<String> = events
            .iter()
            .filter(|(name, _)| name == "response.output_text.delta")
            .map(|(_, payload)| payload["delta"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(text_deltas, vec!["Hello world".to_string()], "{body}");

        let completed = events
            .iter()
            .find(|(name, _)| name == "response.completed")
            .unwrap();
        assert_eq!(
            completed.1["response"]["output"][0]["content"][0]["text"],
            "Hello world"
        );
    }

    #[tokio::test]
    async fn gpt_oss_streaming_text_response_recovers_from_malformed_header() {
        let (server, _) = make_server_with_model(
            "openai/gpt-oss-20b",
            vec![vec![gpt_oss_stream_request_output_from_fragments(
                &["<|channel|>final Hello", "<|end|>"],
                "",
                true,
            )]],
        );

        let response = server
            .post("/v1/responses")
            .json(&serde_json::json!({
                "model": "openai/gpt-oss-20b",
                "input": "Say hello.",
                "stream": true,
                "store": false,
            }))
            .await;

        response.assert_status_ok();
        let body = response.text();
        let events = parse_sse_events(&body);
        let text_deltas: Vec<String> = events
            .iter()
            .filter(|(name, _)| name == "response.output_text.delta")
            .map(|(_, payload)| payload["delta"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(text_deltas, vec!["Hello".to_string()], "{body}");

        let completed = events
            .iter()
            .find(|(name, _)| name == "response.completed")
            .unwrap();
        assert_eq!(
            completed.1["response"]["output"][0]["content"][0]["text"],
            "Hello"
        );
    }
}
