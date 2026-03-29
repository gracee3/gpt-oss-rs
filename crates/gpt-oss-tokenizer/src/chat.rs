//! Chat message types and simple template rendering.

use gpt_oss_core::prelude::{LLMError, Result};

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

/// Apply a ChatML-style template to messages.
/// This is the default fallback when no Jinja2 template is available.
pub(crate) fn apply_chatml(
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    if messages.is_empty() {
        return Err(LLMError::TokenizerError("empty message list".into()));
    }

    let mut out = String::new();
    for msg in messages {
        out.push_str("<|im_start|>");
        out.push_str(&msg.role);
        out.push('\n');
        out.push_str(&msg.content);
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_basic() {
        let msgs = vec![
            ChatMessage::new("system", "You are helpful."),
            ChatMessage::new("user", "Hello"),
        ];
        let result = apply_chatml(&msgs, true).unwrap();
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn chatml_no_generation_prompt() {
        let msgs = vec![ChatMessage::new("user", "Hi")];
        let result = apply_chatml(&msgs, false).unwrap();
        assert!(!result.contains("assistant"));
    }

    #[test]
    fn chatml_empty_errors() {
        assert!(apply_chatml(&[], true).is_err());
    }

    #[test]
    fn chat_message_constructor() {
        let m = ChatMessage::new("system", "test");
        assert_eq!(m.role, "system");
        assert_eq!(m.content, "test");
    }
}
