//! OpenAI-compatible request types.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::error::ApiError;

/// Function call details carried by an assistant chat message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ChatFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// A function tool call carried by an assistant chat message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ChatFunctionCall,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, ToSchema)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn content_text(&self) -> &str {
        self.content.as_deref().unwrap_or_default()
    }
}

/// POST /v1/completions request body.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_n")]
    pub n: usize,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub logprobs: Option<usize>,
    #[serde(default)]
    pub echo: bool,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
}

/// POST /v1/chat/completions request body.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_n")]
    pub n: usize,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub tools: Option<Vec<crate::routes::tools::RequestTool>>,
    #[serde(default)]
    pub tool_choice: Option<crate::routes::tools::ToolChoice>,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}
fn default_n() -> usize {
    1
}

impl CompletionRequest {
    pub fn validate(&self) -> Result<(), ApiError> {
        if self.model.is_empty() {
            return Err(ApiError::InvalidRequest("model is required".into()));
        }
        if self.prompt.is_empty() {
            return Err(ApiError::InvalidRequest("prompt is required".into()));
        }
        if self.max_tokens == 0 {
            return Err(ApiError::InvalidRequest(
                "max_tokens must be greater than 0".into(),
            ));
        }
        if self.temperature < 0.0 || self.temperature > 2.0 {
            return Err(ApiError::InvalidRequest(
                "temperature must be between 0.0 and 2.0".into(),
            ));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(ApiError::InvalidRequest(
                "top_p must be between 0.0 and 1.0".into(),
            ));
        }
        if self.n == 0 {
            return Err(ApiError::InvalidRequest("n must be greater than 0".into()));
        }
        Ok(())
    }

    pub fn to_sampling_params(&self) -> gpt_oss_core::prelude::SamplingParams {
        gpt_oss_core::prelude::SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            stop_strings: self.stop.clone().unwrap_or_default(),
            logprobs: self.logprobs,
            echo: self.echo,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            best_of: self.n,
            ..Default::default()
        }
    }
}

impl ChatCompletionRequest {
    pub fn validate(&self) -> Result<(), ApiError> {
        if self.model.is_empty() {
            return Err(ApiError::InvalidRequest("model is required".into()));
        }
        if self.messages.is_empty() {
            return Err(ApiError::InvalidRequest(
                "messages must not be empty".into(),
            ));
        }
        validate_chat_messages(&self.messages)?;
        if self.max_tokens == 0 {
            return Err(ApiError::InvalidRequest(
                "max_tokens must be greater than 0".into(),
            ));
        }
        if self.temperature < 0.0 || self.temperature > 2.0 {
            return Err(ApiError::InvalidRequest(
                "temperature must be between 0.0 and 2.0".into(),
            ));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(ApiError::InvalidRequest(
                "top_p must be between 0.0 and 1.0".into(),
            ));
        }
        if self.n == 0 {
            return Err(ApiError::InvalidRequest("n must be greater than 0".into()));
        }
        if let Some(tools) = &self.tools {
            let mut names = HashSet::new();
            for (index, tool) in tools.iter().enumerate() {
                if tool.tool_type != "function" {
                    return Err(ApiError::InvalidRequest(format!(
                        "tools[{index}].type must be 'function'"
                    )));
                }
                if tool.function.name.is_empty() {
                    return Err(ApiError::InvalidRequest(format!(
                        "tools[{index}].function.name is required"
                    )));
                }
                if !names.insert(tool.function.name.as_str()) {
                    return Err(ApiError::InvalidRequest(format!(
                        "duplicate tool function name '{}'",
                        tool.function.name
                    )));
                }
            }
        }
        if let Some(choice) = &self.tool_choice {
            match choice {
                crate::routes::tools::ToolChoice::Mode(mode) => {
                    if !["auto", "none", "required"].contains(&mode.as_str()) {
                        return Err(ApiError::InvalidRequest(format!(
                            "invalid tool_choice mode '{mode}', expected auto/none/required"
                        )));
                    }
                }
                crate::routes::tools::ToolChoice::Specific(choice) => {
                    if choice.choice_type != "function" || choice.function.name.is_empty() {
                        return Err(ApiError::InvalidRequest(
                            "specific tool_choice requires a named function".into(),
                        ));
                    }
                    if !self.tools.as_ref().is_some_and(|tools| {
                        tools
                            .iter()
                            .any(|tool| tool.function.name == choice.function.name)
                    }) {
                        return Err(ApiError::InvalidRequest(format!(
                            "tool_choice references unknown tool '{}'",
                            choice.function.name
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Convert Chat Completions history into the structured Harmony seam.
    ///
    /// Tool results carry only a call ID in the public schema. Preserve a map
    /// from assistant call IDs to function names so Harmony receives the
    /// required named tool author on the matching result message.
    pub fn to_protocol_messages(
        &self,
    ) -> Result<Vec<gpt_oss_tokenizer::ProtocolMessage>, ApiError> {
        validate_chat_messages(&self.messages)?;

        let mut call_names = HashMap::<String, String>::new();
        let mut converted = Vec::new();
        for message in &self.messages {
            match message.role.as_str() {
                "assistant" if message.tool_calls.is_some() => {
                    if !message.content_text().is_empty() {
                        converted.push(
                            gpt_oss_tokenizer::ProtocolMessage::new(
                                "assistant",
                                message.content_text(),
                            )
                            .with_channel("commentary"),
                        );
                    }
                    for call in message.tool_calls.as_deref().unwrap_or_default() {
                        call_names.insert(call.id.clone(), call.function.name.clone());
                        converted.push(
                            gpt_oss_tokenizer::ProtocolMessage::new(
                                "assistant",
                                call.function.arguments.clone(),
                            )
                            .with_channel("commentary")
                            .with_recipient(format!("functions.{}", call.function.name))
                            .with_content_type("<|constrain|>json"),
                        );
                    }
                }
                "tool" => {
                    let call_id = message.tool_call_id.as_deref().ok_or_else(|| {
                        ApiError::InvalidRequest("tool messages require tool_call_id".into())
                    })?;
                    let function_name = call_names.get(call_id).ok_or_else(|| {
                        ApiError::InvalidRequest(format!(
                            "tool message references unresolved tool_call_id '{call_id}'"
                        ))
                    })?;
                    converted.push(
                        gpt_oss_tokenizer::ProtocolMessage::new("tool", message.content_text())
                            .with_author_name(format!("functions.{function_name}"))
                            .with_channel("commentary")
                            .with_recipient("assistant"),
                    );
                }
                _ => {
                    let mut protocol = gpt_oss_tokenizer::ProtocolMessage::new(
                        &message.role,
                        message.content_text(),
                    );
                    if let Some(name) = &message.name {
                        protocol = protocol.with_author_name(name);
                    }
                    converted.push(protocol);
                }
            }
        }
        Ok(converted)
    }

    pub fn to_sampling_params(&self) -> gpt_oss_core::prelude::SamplingParams {
        gpt_oss_core::prelude::SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            stop_strings: self.stop.clone().unwrap_or_default(),
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            best_of: self.n,
            ..Default::default()
        }
    }
}

fn validate_chat_messages(messages: &[ChatMessage]) -> Result<(), ApiError> {
    let mut pending_calls = HashMap::<String, String>::new();
    let mut seen_call_ids = HashSet::<String>::new();
    let mut seen_results = HashSet::<String>::new();

    for (index, message) in messages.iter().enumerate() {
        if !["system", "developer", "user", "assistant", "tool"].contains(&message.role.as_str()) {
            return Err(ApiError::InvalidRequest(format!(
                "messages[{index}].role '{}' is not supported",
                message.role
            )));
        }

        if message.role != "tool" && !pending_calls.is_empty() {
            let mut unresolved: Vec<_> = pending_calls.keys().cloned().collect();
            unresolved.sort();
            return Err(ApiError::InvalidRequest(format!(
                "assistant tool calls are unresolved before messages[{index}]: {}",
                unresolved.join(", ")
            )));
        }

        match message.role.as_str() {
            "assistant" => {
                if message.tool_call_id.is_some() {
                    return Err(ApiError::InvalidRequest(format!(
                        "messages[{index}].tool_call_id is only valid for tool messages"
                    )));
                }
                match &message.tool_calls {
                    Some(calls) => {
                        if calls.is_empty() {
                            return Err(ApiError::InvalidRequest(format!(
                                "messages[{index}].tool_calls must not be empty"
                            )));
                        }
                        for (call_index, call) in calls.iter().enumerate() {
                            if call.id.is_empty() {
                                return Err(ApiError::InvalidRequest(format!(
                                    "messages[{index}].tool_calls[{call_index}].id is required"
                                )));
                            }
                            if call.call_type != "function" {
                                return Err(ApiError::InvalidRequest(format!(
                                    "messages[{index}].tool_calls[{call_index}].type must be 'function'"
                                )));
                            }
                            if call.function.name.is_empty() {
                                return Err(ApiError::InvalidRequest(format!(
                                    "messages[{index}].tool_calls[{call_index}].function.name is required"
                                )));
                            }
                            if !seen_call_ids.insert(call.id.clone()) {
                                return Err(ApiError::InvalidRequest(format!(
                                    "duplicate tool call id '{}'",
                                    call.id
                                )));
                            }
                            pending_calls.insert(call.id.clone(), call.function.name.clone());
                        }
                    }
                    None if message.content.is_none() => {
                        return Err(ApiError::InvalidRequest(format!(
                            "messages[{index}].content is required unless tool_calls are present"
                        )));
                    }
                    None => {}
                }
            }
            "tool" => {
                if message.content.is_none() {
                    return Err(ApiError::InvalidRequest(format!(
                        "messages[{index}].content is required for tool messages"
                    )));
                }
                if message.tool_calls.is_some() {
                    return Err(ApiError::InvalidRequest(format!(
                        "messages[{index}].tool_calls is only valid for assistant messages"
                    )));
                }
                let call_id = message.tool_call_id.as_deref().ok_or_else(|| {
                    ApiError::InvalidRequest(format!(
                        "messages[{index}].tool_call_id is required for tool messages"
                    ))
                })?;
                if call_id.is_empty() {
                    return Err(ApiError::InvalidRequest(format!(
                        "messages[{index}].tool_call_id must not be empty"
                    )));
                }
                let Some(function_name) = pending_calls.remove(call_id) else {
                    let detail = if seen_results.contains(call_id) {
                        "duplicate tool result"
                    } else {
                        "unresolved tool call"
                    };
                    return Err(ApiError::InvalidRequest(format!(
                        "messages[{index}] has {detail} id '{call_id}'"
                    )));
                };
                if !seen_results.insert(call_id.to_string()) {
                    return Err(ApiError::InvalidRequest(format!(
                        "duplicate tool result id '{call_id}'"
                    )));
                }
                if let Some(name) = &message.name {
                    let supplied = name.strip_prefix("functions.").unwrap_or(name);
                    if supplied != function_name {
                        return Err(ApiError::InvalidRequest(format!(
                            "messages[{index}].name '{name}' does not match tool call function '{function_name}'"
                        )));
                    }
                }
            }
            _ => {
                if message.content.is_none() {
                    return Err(ApiError::InvalidRequest(format!(
                        "messages[{index}].content is required for {} messages",
                        message.role
                    )));
                }
                if message.tool_call_id.is_some() || message.tool_calls.is_some() {
                    return Err(ApiError::InvalidRequest(format!(
                        "messages[{index}] contains tool metadata incompatible with role '{}'",
                        message.role
                    )));
                }
            }
        }
    }

    if !pending_calls.is_empty() {
        let mut unresolved: Vec<_> = pending_calls.keys().cloned().collect();
        unresolved.sort();
        return Err(ApiError::InvalidRequest(format!(
            "assistant tool calls are missing matching tool results: {}",
            unresolved.join(", ")
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_request_serde_roundtrip() {
        let req = CompletionRequest {
            model: "gpt-3.5-turbo".into(),
            prompt: "Hello".into(),
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            n: 1,
            stream: false,
            stop: Some(vec!["\n".into()]),
            logprobs: Some(5),
            echo: false,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: CompletionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "gpt-3.5-turbo");
        assert_eq!(back.max_tokens, 100);
    }

    #[test]
    fn chat_request_serde_roundtrip() {
        let req = ChatCompletionRequest {
            model: "gpt-4".into(),
            messages: vec![
                ChatMessage::new("system", "You are helpful."),
                ChatMessage::new("user", "Hello"),
            ],
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            n: 1,
            stream: true,
            stop: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
            tools: None,
            tool_choice: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "gpt-4");
        assert_eq!(back.messages.len(), 2);
        assert!(back.stream);
    }

    #[test]
    fn completion_request_defaults() {
        let json = r#"{"model":"m","prompt":"p"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.temperature, 1.0);
        assert_eq!(req.top_p, 1.0);
        assert_eq!(req.n, 1);
        assert!(!req.stream);
    }

    #[test]
    fn completion_validate_ok() {
        let req = CompletionRequest {
            model: "m".into(),
            prompt: "p".into(),
            max_tokens: 10,
            temperature: 0.5,
            top_p: 0.9,
            n: 1,
            stream: false,
            stop: None,
            logprobs: None,
            echo: false,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
        };
        assert!(req.validate().is_ok());
    }

    #[test]
    fn completion_validate_empty_model() {
        let req = CompletionRequest {
            model: "".into(),
            prompt: "p".into(),
            max_tokens: 10,
            temperature: 0.5,
            top_p: 0.9,
            n: 1,
            stream: false,
            stop: None,
            logprobs: None,
            echo: false,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn completion_validate_bad_temperature() {
        let req = CompletionRequest {
            model: "m".into(),
            prompt: "p".into(),
            max_tokens: 10,
            temperature: 3.0,
            top_p: 0.9,
            n: 1,
            stream: false,
            stop: None,
            logprobs: None,
            echo: false,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn chat_validate_empty_messages() {
        let req = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![],
            max_tokens: 10,
            temperature: 0.5,
            top_p: 0.9,
            n: 1,
            stream: false,
            stop: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
            tools: None,
            tool_choice: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn chat_validate_empty_role() {
        let req = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![ChatMessage::new("", "hi")],
            max_tokens: 10,
            temperature: 0.5,
            top_p: 0.9,
            n: 1,
            stream: false,
            stop: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            user: None,
            seed: None,
            tools: None,
            tool_choice: None,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn to_sampling_params_maps_fields() {
        let req = CompletionRequest {
            model: "m".into(),
            prompt: "p".into(),
            max_tokens: 42,
            temperature: 0.8,
            top_p: 0.95,
            n: 2,
            stream: false,
            stop: Some(vec!["END".into()]),
            logprobs: Some(3),
            echo: true,
            presence_penalty: 0.1,
            frequency_penalty: 0.2,
            user: None,
            seed: Some(123),
        };
        let sp = req.to_sampling_params();
        assert_eq!(sp.max_tokens, 42);
        assert_eq!(sp.temperature, 0.8);
        assert_eq!(sp.top_p, 0.95);
        assert_eq!(sp.stop_strings, vec!["END".to_string()]);
        assert_eq!(sp.logprobs, Some(3));
        assert!(sp.echo);
        assert_eq!(sp.seed, Some(123));
        assert_eq!(sp.best_of, 2);
    }

    #[test]
    fn chat_message_serde() {
        let msg = ChatMessage::new("user", "hello");
        let json = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back, msg);
    }

    fn chat_request_from_messages(messages: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": messages,
            "max_tokens": 16
        }))
        .unwrap()
    }

    #[test]
    fn chat_tool_history_maps_multiple_call_ids_to_harmony_authors() {
        let req = chat_request_from_messages(serde_json::json!([
            {"role": "user", "name": "alice", "content": "Weather and time?"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Boston\"}"}
                    },
                    {
                        "id": "call_time",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{\"timezone\":\"UTC\"}"}
                    }
                ]
            },
            {"role": "tool", "tool_call_id": "call_time", "content": "{\"time\":\"12:00\"}"},
            {"role": "tool", "tool_call_id": "call_weather", "content": "{\"temp_c\":18}"},
            {"role": "assistant", "content": "It is 18C and noon UTC."}
        ]));

        req.validate().unwrap();
        let converted = req.to_protocol_messages().unwrap();
        assert_eq!(converted.len(), 6);
        assert_eq!(converted[0].author_name.as_deref(), Some("alice"));
        assert_eq!(
            converted[1].recipient.as_deref(),
            Some("functions.get_weather")
        );
        assert_eq!(
            converted[2].recipient.as_deref(),
            Some("functions.get_time")
        );
        assert_eq!(
            converted[3].author_name.as_deref(),
            Some("functions.get_time")
        );
        assert_eq!(
            converted[4].author_name.as_deref(),
            Some("functions.get_weather")
        );

        let prompt = gpt_oss_tokenizer::HarmonyProtocol::gpt_oss()
            .unwrap()
            .render_prompt(&converted, None, &[])
            .unwrap();
        assert!(prompt.text.contains("to=functions.get_weather"));
        assert!(prompt.text.contains("<|start|>functions.get_weather"));
        assert!(prompt.text.contains("<|start|>functions.get_time"));
    }

    #[test]
    fn chat_tool_history_rejects_missing_duplicate_and_unresolved_metadata() {
        let missing_id = chat_request_from_messages(serde_json::json!([
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"}
            }]},
            {"role": "tool", "content": "ok"}
        ]));
        assert!(missing_id
            .validate()
            .unwrap_err()
            .to_string()
            .contains("tool_call_id is required"));

        let duplicate_id = chat_request_from_messages(serde_json::json!([
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "call_1", "type": "function", "function": {"name": "g", "arguments": "{}"}}
            ]}
        ]));
        assert!(duplicate_id
            .validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate tool call id"));

        let unresolved_id = chat_request_from_messages(serde_json::json!([
            {"role": "tool", "tool_call_id": "missing", "content": "ok"}
        ]));
        assert!(unresolved_id
            .validate()
            .unwrap_err()
            .to_string()
            .contains("unresolved tool call"));

        let missing_result = chat_request_from_messages(serde_json::json!([
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"}
            }]}
        ]));
        assert!(missing_result
            .validate()
            .unwrap_err()
            .to_string()
            .contains("missing matching tool results"));
    }
}
