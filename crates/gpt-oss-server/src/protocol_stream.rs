//! Shared Harmony protocol stream helpers for GPT-OSS routes.

use gpt_oss_server::error::ApiError;
use gpt_oss_server::types::streaming::{ChatFunctionCallDelta, ChatToolCallDelta};

pub(crate) fn visible_text_from_protocol_messages(
    messages: &[gpt_oss_tokenizer::ParsedProtocolMessage],
) -> Option<String> {
    let collected: Vec<&str> = messages
        .iter()
        .filter(|message| message.role == "assistant")
        .filter(|message| message.recipient.is_none())
        .filter(|message| message.channel.as_deref() != Some("analysis"))
        .map(|message| message.content.as_str())
        .filter(|content| !content.is_empty())
        .collect();

    if collected.is_empty() {
        None
    } else {
        Some(collected.join("\n"))
    }
}

pub(crate) fn diff_text(previous: &str, current: &str) -> String {
    if let Some(suffix) = current.strip_prefix(previous) {
        suffix.to_string()
    } else {
        current.to_string()
    }
}

#[derive(Debug)]
struct StreamedChatToolCallState {
    id: String,
    name: String,
    arguments: String,
}

pub(crate) struct StreamedChatChoiceState {
    processed_tokens: usize,
    parser: gpt_oss_tokenizer::HarmonyStreamParser,
    visible_text: String,
    tool_calls: Vec<StreamedChatToolCallState>,
}

impl StreamedChatChoiceState {
    pub(crate) fn new() -> Result<Self, ApiError> {
        let protocol = gpt_oss_tokenizer::HarmonyProtocol::gpt_oss()
            .map_err(|e| ApiError::Internal(format!("harmony init error: {}", e)))?;
        Ok(Self {
            processed_tokens: 0,
            parser: protocol
                .stream_parser()
                .map_err(|e| ApiError::Internal(format!("harmony stream init error: {}", e)))?,
            visible_text: String::new(),
            tool_calls: Vec::new(),
        })
    }

    pub(crate) fn ingest(
        &mut self,
        stream_id: &str,
        choice_index: usize,
        token_ids: &[u32],
        finished: bool,
    ) -> Result<(Option<String>, Vec<ChatToolCallDelta>), ApiError> {
        for token in token_ids.iter().copied().skip(self.processed_tokens) {
            self.parser
                .push_token(token)
                .map_err(|e| ApiError::Internal(format!("harmony stream parse error: {}", e)))?;
        }
        self.processed_tokens = token_ids.len();
        if finished {
            self.parser.finish().map_err(|e| {
                ApiError::Internal(format!("harmony stream finalize error: {}", e))
            })?;
        }

        let messages = self
            .parser
            .messages()
            .map_err(|e| ApiError::Internal(format!("harmony stream read error: {}", e)))?;
        let visible_text = visible_text_from_protocol_messages(&messages).unwrap_or_default();
        let content_delta = diff_text(&self.visible_text, &visible_text);
        self.visible_text = visible_text;

        let mut tool_call_deltas = Vec::new();
        let mut parsed_tool_messages: Vec<_> = messages
            .iter()
            .filter(|message| message.role == "assistant")
            .filter_map(|message| {
                message
                    .recipient
                    .as_ref()
                    .map(|recipient| (recipient.as_str(), message.content.as_str()))
            })
            .collect();

        for (tool_index, (recipient, arguments)) in parsed_tool_messages.drain(..).enumerate() {
            let name = recipient
                .strip_prefix("functions.")
                .unwrap_or(recipient)
                .to_string();
            if self.tool_calls.len() <= tool_index {
                self.tool_calls.push(StreamedChatToolCallState {
                    id: format!("{stream_id}_{choice_index}_{tool_index}"),
                    name: name.clone(),
                    arguments: String::new(),
                });
            }

            let state = &mut self.tool_calls[tool_index];
            let arguments_delta = diff_text(&state.arguments, arguments);
            if arguments_delta.is_empty() && state.name == name {
                continue;
            }
            let is_new_call = state.arguments.is_empty();
            state.name = name.clone();
            state.arguments = arguments.to_string();
            tool_call_deltas.push(ChatToolCallDelta {
                index: tool_index,
                id: is_new_call.then(|| state.id.clone()),
                call_type: is_new_call.then(|| "function".to_string()),
                function: Some(ChatFunctionCallDelta {
                    name: is_new_call.then_some(name),
                    arguments: (!arguments_delta.is_empty()).then_some(arguments_delta),
                }),
            });
        }

        Ok(((!content_delta.is_empty()).then_some(content_delta), tool_call_deltas))
    }

    pub(crate) fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}
