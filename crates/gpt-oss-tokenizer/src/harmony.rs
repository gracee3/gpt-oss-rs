//! Harmony-native prompt rendering and assistant-output parsing for GPT-OSS.

use std::sync::OnceLock;

use gpt_oss_core::prelude::{LLMError, Result, TokenId};
use openai_harmony::chat::{
    Content, Conversation, DeveloperContent, Message, Role, SystemContent, ToolDescription,
};
use openai_harmony::{load_harmony_encoding, HarmonyEncoding, HarmonyEncodingName, ParseOptions};

use crate::{ChatMessage, ToolDefinition};

static HARMONY_ENCODING: OnceLock<HarmonyEncoding> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarmonyConversationItem {
    Message(ChatMessage),
    FunctionCall { name: String, arguments: String },
    FunctionCallOutput { name: String, output: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarmonyPrompt {
    pub prompt: String,
    pub prompt_token_ids: Vec<TokenId>,
    pub stop_token_ids: Vec<TokenId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarmonyToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
    pub channel: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarmonyAssistantOutput {
    pub text: Option<String>,
    pub tool_calls: Vec<HarmonyToolCall>,
}

pub fn render_harmony_prompt(
    items: &[HarmonyConversationItem],
    developer_instructions: Option<&str>,
    tools: &[ToolDefinition],
) -> Result<HarmonyPrompt> {
    let encoding = harmony_encoding()?;
    let has_function_tools = !tools.is_empty()
        || items.iter().any(|item| {
            matches!(
                item,
                HarmonyConversationItem::FunctionCall { .. }
                    | HarmonyConversationItem::FunctionCallOutput { .. }
            )
        });

    let mut messages = vec![Message::from_role_and_content(
        Role::System,
        default_system_content(has_function_tools),
    )];

    if developer_instructions.is_some_and(|text| !text.trim().is_empty()) || !tools.is_empty() {
        let mut developer = DeveloperContent::new();
        if let Some(instructions) = developer_instructions.filter(|text| !text.trim().is_empty()) {
            developer = developer.with_instructions(instructions);
        }
        if !tools.is_empty() {
            developer = developer.with_function_tools(
                tools
                    .iter()
                    .map(|tool| {
                        ToolDescription::new(
                            tool.function.name.clone(),
                            tool.function.description.clone().unwrap_or_default(),
                            tool.parameters_json(),
                        )
                    })
                    .collect(),
            );
        }
        messages.push(Message::from_role_and_content(Role::Developer, developer));
    }

    for item in items {
        messages.push(item_to_harmony_message(item)?);
    }

    let conversation = Conversation::from_messages(messages);
    let prompt_ranks = encoding
        .render_conversation_for_completion(&conversation, Role::Assistant, None)
        .map_err(harmony_error)?;
    let prompt = encoding
        .tokenizer()
        .decode_utf8(prompt_ranks.iter().copied())
        .map_err(harmony_error)?;
    let prompt_token_ids = prompt_ranks
        .iter()
        .copied()
        .map(rank_to_token_id)
        .collect::<Result<Vec<_>>>()?;

    let mut stop_token_ids = encoding
        .stop_tokens_for_assistant_actions()
        .map_err(harmony_error)?
        .into_iter()
        .map(rank_to_token_id)
        .collect::<Result<Vec<_>>>()?;
    stop_token_ids.sort_unstable();

    Ok(HarmonyPrompt {
        prompt,
        prompt_token_ids,
        stop_token_ids,
    })
}

pub fn parse_harmony_assistant_output(
    tokens: &[TokenId],
    call_id_prefix: &str,
) -> Result<HarmonyAssistantOutput> {
    let encoding = harmony_encoding()?;
    let ranks = tokens
        .iter()
        .copied()
        .map(token_id_to_rank)
        .collect::<Result<Vec<_>>>()?;

    let messages = encoding
        .parse_messages_from_completion_tokens_with_options(
            ranks.iter().copied(),
            Some(Role::Assistant),
            ParseOptions { strict: false },
        )
        .map_err(harmony_error)?;

    let mut final_parts = Vec::new();
    let mut visible_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for message in messages {
        let text = message_text(&message);
        match message.author.role {
            Role::Assistant if message.recipient.is_some() => {
                let recipient = message.recipient.as_deref().unwrap_or_default();
                let name = normalize_tool_name(recipient);
                let id = format!("call_{}{}", call_id_prefix, tool_calls.len());
                tool_calls.push(HarmonyToolCall {
                    id,
                    name,
                    arguments: text,
                    channel: message.channel.clone(),
                });
            }
            Role::Assistant => match message.channel.as_deref() {
                Some("analysis") => {}
                Some("final") => {
                    if !text.is_empty() {
                        final_parts.push(text);
                    }
                }
                _ => {
                    if !text.is_empty() {
                        visible_parts.push(text);
                    }
                }
            },
            _ => {}
        }
    }

    let text = if !final_parts.is_empty() {
        Some(final_parts.join("\n\n"))
    } else if !visible_parts.is_empty() {
        Some(visible_parts.join("\n\n"))
    } else {
        None
    };

    Ok(HarmonyAssistantOutput { text, tool_calls })
}

fn harmony_encoding() -> Result<&'static HarmonyEncoding> {
    if let Some(encoding) = HARMONY_ENCODING.get() {
        return Ok(encoding);
    }

    let encoding =
        load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss).map_err(harmony_error)?;
    let _ = HARMONY_ENCODING.set(encoding);
    HARMONY_ENCODING
        .get()
        .ok_or_else(|| harmony_error("Harmony encoding was not initialized"))
}

fn harmony_error(error: impl std::fmt::Display) -> LLMError {
    LLMError::TokenizerError(format!("harmony error: {error}"))
}

fn default_system_content(has_function_tools: bool) -> SystemContent {
    if has_function_tools {
        SystemContent::new().with_required_channels(["commentary", "final"])
    } else {
        SystemContent::new().with_required_channels(["final"])
    }
}

fn item_to_harmony_message(item: &HarmonyConversationItem) -> Result<Message> {
    match item {
        HarmonyConversationItem::Message(message) => {
            let role = parse_role(&message.role)?;
            Ok(Message::from_role_and_content(role, message.content.clone()))
        }
        HarmonyConversationItem::FunctionCall { name, arguments } => Ok(
            Message::from_role_and_content(Role::Assistant, arguments.clone())
                .with_channel("commentary")
                .with_recipient(format!("functions.{name}"))
                .with_content_type("<|constrain|>json"),
        ),
        HarmonyConversationItem::FunctionCallOutput { name, output } => Ok(
            Message::from_author_and_content(
                openai_harmony::chat::Author::new(Role::Tool, format!("functions.{name}")),
                output.clone(),
            )
            .with_channel("commentary")
            .with_recipient("assistant"),
        ),
    }
}

fn parse_role(role: &str) -> Result<Role> {
    Role::try_from(role).map_err(|_| {
        LLMError::TokenizerError(format!("unsupported Harmony role '{role}' for GPT-OSS prompt"))
    })
}

fn rank_to_token_id(rank: u32) -> Result<TokenId> {
    Ok(rank)
}

fn token_id_to_rank(token_id: TokenId) -> Result<u32> {
    Ok(token_id)
}

fn message_text(message: &Message) -> String {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            Content::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn normalize_tool_name(recipient: &str) -> String {
    recipient
        .strip_prefix("functions.")
        .unwrap_or(recipient)
        .to_string()
}

trait ToolDefinitionExt {
    fn parameters_json(&self) -> Option<serde_json::Value>;
}

impl ToolDefinitionExt for ToolDefinition {
    fn parameters_json(&self) -> Option<serde_json::Value> {
        self.function
            .parameters
            .as_ref()
            .and_then(|params| serde_json::to_value(params).ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_harmony_prompt_with_tools_includes_stop_tokens() {
        let prompt = render_harmony_prompt(
            &[HarmonyConversationItem::Message(ChatMessage::user("Hello"))],
            Some("Be terse."),
            &[ToolDefinition {
                tool_type: "function".to_string(),
                function: crate::FunctionDefinition {
                    name: "lookup_weather".to_string(),
                    description: Some("Look up weather.".to_string()),
                    parameters: Some(crate::ToolParameters {
                        schema_type: "object".to_string(),
                        properties: Default::default(),
                        required: Vec::new(),
                    }),
                },
            }],
        )
        .unwrap();

        assert!(!prompt.prompt_token_ids.is_empty());
        assert!(!prompt.stop_token_ids.is_empty());
        assert!(prompt.prompt.contains("<|start|>system"));
        assert!(prompt.prompt.contains("namespace functions"));
    }

    #[test]
    fn parse_harmony_assistant_tool_call_strips_functions_namespace() {
        let prompt = render_harmony_prompt(
            &[HarmonyConversationItem::Message(ChatMessage::user(
                "What is the weather in Tokyo?",
            ))],
            None,
            &[ToolDefinition {
                tool_type: "function".to_string(),
                function: crate::FunctionDefinition {
                    name: "lookup_weather".to_string(),
                    description: Some("Look up weather.".to_string()),
                    parameters: Some(crate::ToolParameters {
                        schema_type: "object".to_string(),
                        properties: Default::default(),
                        required: Vec::new(),
                    }),
                },
            }],
        )
        .unwrap();

        let encoding = harmony_encoding().unwrap();
        let generated = encoding
            .tokenizer()
            .encode(
                "<|channel|>commentary to=functions.lookup_weather<|constrain|>json<|message|>{\"location\":\"Tokyo\"}<|call|>",
                &encoding.tokenizer().special_tokens(),
            )
            .0;
        let generated = generated
            .into_iter()
            .map(|rank| rank_to_token_id(rank).unwrap())
            .collect::<Vec<_>>();

        let parsed = parse_harmony_assistant_output(&generated, "resp_").unwrap();
        assert_eq!(parsed.text, None);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "lookup_weather");
        assert!(parsed.tool_calls[0].arguments.contains("Tokyo"));
        assert!(!prompt.prompt_token_ids.is_empty());
    }
}
