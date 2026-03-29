#![forbid(unsafe_code)]
//! Tokenization and detokenization for gpt-oss-rs.
//!
//! Wraps the HuggingFace `tokenizers` crate with streaming decode,
//! chat template support, and special token detection.

pub mod chat;
pub mod harmony;
pub mod incremental;
pub mod tokenizer;
pub mod tool_parser;

pub use chat::{ChatMessage, ChatRole};
pub use harmony::{
    parse_harmony_assistant_output, render_harmony_prompt, HarmonyAssistantOutput,
    HarmonyConversationItem, HarmonyPrompt, HarmonyToolCall,
};
pub use incremental::IncrementalDecoder;
pub use tokenizer::Tokenizer;
pub use tool_parser::{
    format_tool_definitions, parse_tool_calls, FunctionDefinition, ParsedToolCall, ToolDefinition,
    ToolParameterProperty, ToolParameters, ToolParseResult, ToolPromptStyle,
};
