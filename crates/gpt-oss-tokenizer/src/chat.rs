//! Minimal chat message types and fallback ChatML rendering.

use gpt_oss_core::prelude::{LLMError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

impl std::fmt::Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(ChatRole::System.as_str(), content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(ChatRole::User.as_str(), content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(ChatRole::Assistant.as_str(), content)
    }
}

pub(crate) fn apply_chatml(
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String> {
    if messages.is_empty() {
        return Err(LLMError::TokenizerError("empty message list".into()));
    }

    let mut output = String::new();
    for message in messages {
        output.push_str("<|im_start|>");
        output.push_str(&message.role);
        output.push('\n');
        output.push_str(&message.content);
        output.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        output.push_str("<|im_start|>assistant\n");
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatml_renders_generation_prompt() {
        let rendered = apply_chatml(&[ChatMessage::user("hello")], true).unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn empty_conversation_is_rejected() {
        assert!(apply_chatml(&[], true).is_err());
    }
}
