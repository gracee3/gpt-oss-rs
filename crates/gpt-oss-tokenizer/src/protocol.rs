//! Harmony-native GPT-OSS protocol seam.
//!
//! This module owns the GPT-OSS-specific render/parse/stream-parse boundary.
//! The server should normalize API requests into `ProtocolMessage`s, render via
//! Harmony here, and treat the resulting structured parse state as the source
//! of truth for future SSE/event derivation.

use gpt_oss_core::prelude::{LLMError, Result, TokenId};
use openai_harmony::chat::{
    Author, Content, Conversation, DeveloperContent, Message, Role, SystemContent, ToolDescription,
};
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName, StreamableParser};

use crate::tool_parser::ToolDefinition;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtocolMessage {
    pub role: String,
    pub content: String,
    pub channel: Option<String>,
    pub recipient: Option<String>,
    pub content_type: Option<String>,
}

impl ProtocolMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            channel: None,
            recipient: None,
            content_type: None,
        }
    }

    pub fn with_channel(mut self, channel: impl Into<String>) -> Self {
        self.channel = Some(channel.into());
        self
    }

    pub fn with_recipient(mut self, recipient: impl Into<String>) -> Self {
        self.recipient = Some(recipient.into());
        self
    }

    pub fn with_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = Some(content_type.into());
        self
    }

    fn from_harmony_message(message: Message) -> Result<Self> {
        let content = flatten_message_content(&message.content)?;
        Ok(Self {
            role: message.author.role.as_str().to_string(),
            content,
            channel: message.channel,
            recipient: message.recipient,
            content_type: message.content_type,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedProtocolMessage {
    pub role: String,
    pub content: String,
    pub channel: Option<String>,
    pub recipient: Option<String>,
    pub content_type: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPrompt {
    pub text: String,
    pub token_ids: Vec<TokenId>,
}

pub struct HarmonyProtocol {
    encoding: openai_harmony::HarmonyEncoding,
}

impl HarmonyProtocol {
    pub fn gpt_oss() -> Result<Self> {
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            .map_err(map_harmony_error)?;
        Ok(Self { encoding })
    }

    pub fn normalize_conversation(
        &self,
        messages: &[ProtocolMessage],
        instructions: Option<&str>,
        tools: &[ToolDefinition],
    ) -> Result<Conversation> {
        let mut conversation = vec![Message::from_role_and_content(
            Role::System,
            SystemContent::new(),
        )];

        let mut developer_parts = Vec::new();
        if let Some(instructions) = instructions {
            if !instructions.is_empty() {
                developer_parts.push(instructions.to_string());
            }
        }

        for message in messages {
            if matches!(message.role.as_str(), "system" | "developer") && !message.content.is_empty()
            {
                developer_parts.push(message.content.clone());
            }
        }

        let mut developer = DeveloperContent::new();
        if !developer_parts.is_empty() {
            developer = developer.with_instructions(developer_parts.join("\n\n"));
        }
        if !tools.is_empty() {
            developer = developer.with_function_tools(
                tools
                    .iter()
                    .map(tool_definition_to_harmony)
                    .collect::<Vec<_>>(),
            );
        }
        if !developer_parts.is_empty() || !tools.is_empty() {
            conversation.push(Message::from_role_and_content(Role::Developer, developer));
        }

        for message in messages {
            if matches!(message.role.as_str(), "system" | "developer") {
                continue;
            }
            conversation.push(protocol_to_harmony_message(message)?);
        }

        Ok(Conversation::from_messages(conversation))
    }

    pub fn render_prompt(
        &self,
        messages: &[ProtocolMessage],
        instructions: Option<&str>,
        tools: &[ToolDefinition],
    ) -> Result<RenderedPrompt> {
        let conversation = self.normalize_conversation(messages, instructions, tools)?;
        let token_ids = self
            .encoding
            .render_conversation_for_completion(&conversation, Role::Assistant, None)
            .map_err(map_harmony_error)?;
        let text = self
            .encoding
            .tokenizer()
            .decode_utf8(&token_ids)
            .map_err(|error| LLMError::TokenizerError(format!("harmony decode error: {error}")))?;
        Ok(RenderedPrompt { text, token_ids })
    }

    pub fn parse_completion_tokens(&self, token_ids: &[TokenId]) -> Result<Vec<ParsedProtocolMessage>> {
        let parsed = self
            .encoding
            .parse_messages_from_completion_tokens(token_ids.iter().copied(), Some(Role::Assistant))
            .map_err(map_harmony_error)?;
        parsed
            .into_iter()
            .map(|message| {
                let protocol = ProtocolMessage::from_harmony_message(message)?;
                Ok(ParsedProtocolMessage {
                    role: protocol.role,
                    content: protocol.content,
                    channel: protocol.channel,
                    recipient: protocol.recipient,
                    content_type: protocol.content_type,
                })
            })
            .collect()
    }

    pub fn stream_parser(&self) -> Result<HarmonyStreamParser> {
        HarmonyStreamParser::new(self.encoding.clone())
    }
}

pub struct HarmonyStreamParser {
    parser: StreamableParser,
}

impl HarmonyStreamParser {
    pub fn new(encoding: openai_harmony::HarmonyEncoding) -> Result<Self> {
        let parser =
            StreamableParser::new(encoding, Some(Role::Assistant)).map_err(map_harmony_error)?;
        Ok(Self { parser })
    }

    pub fn push_token(&mut self, token: TokenId) -> Result<()> {
        self.parser.process(token).map_err(map_harmony_error)?;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        self.parser.process_eos().map_err(map_harmony_error)?;
        Ok(())
    }

    pub fn last_content_delta(&self) -> Result<Option<String>> {
        self.parser.last_content_delta().map_err(map_harmony_error)
    }

    pub fn current_content(&self) -> Result<String> {
        self.parser.current_content().map_err(map_harmony_error)
    }

    pub fn state_json(&self) -> Result<String> {
        self.parser.state_json().map_err(map_harmony_error)
    }

    pub fn messages(&self) -> Result<Vec<ParsedProtocolMessage>> {
        self.parser
            .messages()
            .iter()
            .cloned()
            .map(|message| {
                let protocol = ProtocolMessage::from_harmony_message(message)?;
                Ok(ParsedProtocolMessage {
                    role: protocol.role,
                    content: protocol.content,
                    channel: protocol.channel,
                    recipient: protocol.recipient,
                    content_type: protocol.content_type,
                })
            })
            .collect()
    }
}

fn protocol_to_harmony_message(message: &ProtocolMessage) -> Result<Message> {
    let role = Role::try_from(message.role.as_str()).map_err(|_| {
        LLMError::TokenizerError(format!("unsupported protocol role '{}'", message.role))
    })?;
    let mut converted = if role == Role::Tool {
        match &message.recipient {
            Some(name) => Message::from_author_and_content(
                Author::new(Role::Tool, name.clone()),
                message.content.clone(),
            ),
            None => Message::from_role_and_content(Role::Tool, message.content.clone()),
        }
    } else {
        Message::from_role_and_content(role, message.content.clone())
    };

    if let Some(channel) = &message.channel {
        converted = converted.with_channel(channel.clone());
    }
    if let Some(recipient) = &message.recipient {
        converted = converted.with_recipient(recipient.clone());
    }
    if let Some(content_type) = &message.content_type {
        converted = converted.with_content_type(content_type.clone());
    }
    Ok(converted)
}

fn tool_definition_to_harmony(tool: &ToolDefinition) -> ToolDescription {
    let parameters = tool
        .function
        .parameters
        .as_ref()
        .and_then(|params| serde_json::to_value(params).ok());
    ToolDescription::new(
        tool.function.name.clone(),
        tool.function.description.clone().unwrap_or_default(),
        parameters,
    )
}

fn flatten_message_content(content: &[Content]) -> Result<String> {
    let mut parts = Vec::new();
    for item in content {
        match item {
            Content::Text(text) => parts.push(text.text.clone()),
            Content::SystemContent(system) => {
                parts.push(serde_json::to_string(system).map_err(map_json_error)?)
            }
            Content::DeveloperContent(developer) => {
                parts.push(serde_json::to_string(developer).map_err(map_json_error)?)
            }
        }
    }
    Ok(parts.join("\n\n"))
}

fn map_harmony_error(error: anyhow::Error) -> LLMError {
    LLMError::TokenizerError(format!("harmony error: {error}"))
}

fn map_json_error(error: serde_json::Error) -> LLMError {
    LLMError::TokenizerError(format!("json error: {error}"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn render_prompt_normalizes_system_to_developer() {
        let protocol = HarmonyProtocol::gpt_oss().unwrap();
        let prompt = protocol
            .render_prompt(
                &[
                    ProtocolMessage::new("system", "Always answer tersely."),
                    ProtocolMessage::new("user", "Hello"),
                ],
                None,
                &[],
            )
            .unwrap();

        assert!(prompt.text.contains("<|start|>system"));
        assert!(prompt.text.contains("<|start|>developer"));
        assert!(prompt.text.contains("Always answer tersely."));
        assert!(prompt.text.contains("<|start|>user<|message|>Hello<|end|>"));
        assert!(prompt.text.ends_with("<|start|>assistant"));
    }

    #[test]
    fn parse_completion_tokens_preserves_recipient_channel_and_constrain() {
        let protocol = HarmonyProtocol::gpt_oss().unwrap();
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
        let text = " to=functions.get_weather<|channel|>commentary<|constrain|>json<|message|>{\"location\":\"Tokyo\"}<|call|>";
        let token_ids = encoding.tokenizer().encode_with_special_tokens(text);
        let parsed = protocol.parse_completion_tokens(&token_ids).unwrap();

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].role, "assistant");
        assert_eq!(parsed[0].channel.as_deref(), Some("commentary"));
        assert_eq!(parsed[0].recipient.as_deref(), Some("functions.get_weather"));
        assert_eq!(parsed[0].content_type.as_deref(), Some("<|constrain|>json"));
        assert_eq!(parsed[0].content, "{\"location\":\"Tokyo\"}");
    }

    #[test]
    fn stream_parser_handles_fragmented_tool_call() {
        let protocol = HarmonyProtocol::gpt_oss().unwrap();
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).unwrap();
        let text = " to=functions.get_weather<|channel|>commentary<|constrain|>json<|message|>{\"location\":\"Tokyo\"}<|call|>";
        let token_ids = encoding
            .tokenizer()
            .encode(
                text,
                &HashSet::from(["<|channel|>", "<|constrain|>", "<|message|>", "<|call|>"]),
            )
            .0;
        let mut parser = protocol.stream_parser().unwrap();

        for token in token_ids {
            parser.push_token(token).unwrap();
        }
        parser.finish().unwrap();

        let messages = parser.messages().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].recipient.as_deref(), Some("functions.get_weather"));
        assert_eq!(messages[0].content_type.as_deref(), Some("<|constrain|>json"));
        assert_eq!(messages[0].channel.as_deref(), Some("commentary"));
        assert_eq!(messages[0].content, "{\"location\":\"Tokyo\"}");
    }
}
