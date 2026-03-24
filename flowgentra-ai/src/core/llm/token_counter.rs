//! Token counting and context window management.
//!
//! Provides estimation-based token counting (no external tokenizer dependency)
//! and utilities for truncating message history to fit within model context windows.

use super::Message;

/// Known model context window sizes.
pub fn context_window(model: &str) -> Option<usize> {
    // Normalize model name for lookup
    let m = model.to_lowercase();
    match m.as_str() {
        // OpenAI
        s if s.starts_with("gpt-4o") => Some(128_000),
        s if s.starts_with("gpt-4-turbo") => Some(128_000),
        s if s.starts_with("gpt-4-32k") => Some(32_768),
        "gpt-4" => Some(8_192),
        s if s.starts_with("gpt-3.5-turbo-16k") => Some(16_384),
        s if s.starts_with("gpt-3.5-turbo") => Some(4_096),
        s if s.starts_with("o1") || s.starts_with("o3") || s.starts_with("o4") => Some(200_000),
        // Anthropic
        s if s.contains("claude-3")
            || s.contains("claude-opus")
            || s.contains("claude-sonnet")
            || s.contains("claude-haiku") =>
        {
            Some(200_000)
        }
        s if s.contains("claude-2") => Some(100_000),
        s if s.contains("claude") => Some(200_000),
        // Mistral
        s if s.contains("mistral-large") => Some(128_000),
        s if s.contains("mistral") => Some(32_768),
        // Groq
        s if s.contains("llama-3") => Some(8_192),
        s if s.contains("mixtral") => Some(32_768),
        _ => None,
    }
}

/// Estimate the number of tokens in a string.
///
/// Uses the common heuristic of ~4 characters per token for English text.
/// This is intentionally conservative (overestimates slightly) to avoid
/// exceeding limits.
pub fn estimate_tokens(text: &str) -> usize {
    // ~4 chars per token for English, ~3 for code/mixed
    // We use 3.5 (round up) as a conservative estimate
    let chars = text.len();
    (chars as f64 / 3.5).ceil() as usize
}

/// Estimate token count for a list of messages.
///
/// Includes overhead for message formatting (~4 tokens per message for
/// role/formatting tokens).
pub fn estimate_messages_tokens(messages: &[Message]) -> usize {
    let mut total = 0;
    for msg in messages {
        total += 4; // role + formatting overhead
        total += estimate_tokens(&msg.content);
    }
    total += 2; // conversation priming
    total
}

/// Context window manager that truncates message history to fit.
pub struct ContextWindow {
    /// Maximum tokens for the model
    pub max_tokens: usize,
    /// Reserve this many tokens for the completion
    pub reserve_for_completion: usize,
}

impl ContextWindow {
    /// Create a new context window manager.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            reserve_for_completion: 1024,
        }
    }

    /// Create from a model name (auto-detect context window).
    pub fn for_model(model: &str) -> Option<Self> {
        context_window(model).map(Self::new)
    }

    /// Set the number of tokens to reserve for the completion.
    pub fn with_reserve(mut self, reserve: usize) -> Self {
        self.reserve_for_completion = reserve;
        self
    }

    /// Available tokens for the prompt (max - reserve).
    pub fn available_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.reserve_for_completion)
    }

    /// Check if messages fit within the context window.
    pub fn fits(&self, messages: &[Message]) -> bool {
        estimate_messages_tokens(messages) <= self.available_tokens()
    }

    /// Truncate messages to fit within the context window.
    ///
    /// Strategy: always keep the system message (first) and the most recent
    /// user message (last). Remove oldest non-system messages first.
    pub fn truncate(&self, messages: &[Message]) -> Vec<Message> {
        let available = self.available_tokens();

        if estimate_messages_tokens(messages) <= available {
            return messages.to_vec();
        }

        if messages.is_empty() {
            return vec![];
        }

        // Always keep system messages and the last message
        let mut system_msgs: Vec<&Message> = vec![];
        let mut other_msgs: Vec<&Message> = vec![];

        for msg in messages {
            if msg.role == super::MessageRole::System {
                system_msgs.push(msg);
            } else {
                other_msgs.push(msg);
            }
        }

        // Start with system messages
        let mut result: Vec<Message> = system_msgs.iter().map(|m| (*m).clone()).collect();
        let mut used = estimate_messages_tokens(&result);

        // Add messages from the end (most recent first) until we run out of space
        let mut from_end: Vec<Message> = Vec::new();
        for msg in other_msgs.iter().rev() {
            let msg_tokens = 4 + estimate_tokens(&msg.content);
            if used + msg_tokens <= available {
                from_end.push((*msg).clone());
                used += msg_tokens;
            } else {
                break;
            }
        }

        // Reverse to maintain chronological order
        from_end.reverse();
        result.extend(from_end);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::llm::{Message, MessageRole};

    #[test]
    fn test_estimate_tokens() {
        // "hello world" = 11 chars → ~4 tokens
        let count = estimate_tokens("hello world");
        assert!(count >= 3 && count <= 5);
    }

    #[test]
    fn test_context_window_lookup() {
        assert_eq!(context_window("gpt-4o"), Some(128_000));
        assert_eq!(context_window("claude-3-opus-20240229"), Some(200_000));
        assert_eq!(context_window("unknown-model-xyz"), None);
    }

    #[test]
    fn test_truncation() {
        let cw = ContextWindow::new(100).with_reserve(20);
        let messages: Vec<Message> = (0..50)
            .map(|i| Message {
                role: MessageRole::User,
                content: format!("Message number {} with some content", i),
                tool_calls: None,
                tool_call_id: None,
            })
            .collect();

        let truncated = cw.truncate(&messages);
        assert!(estimate_messages_tokens(&truncated) <= cw.available_tokens());
        // Last message should be preserved
        assert_eq!(
            truncated.last().unwrap().content,
            messages.last().unwrap().content
        );
    }

    #[test]
    fn test_system_message_preserved() {
        let cw = ContextWindow::new(80).with_reserve(20);
        let messages = vec![
            Message::system("You are helpful."),
            Message::user(&"Very long message that takes many tokens and should probably be dropped if needed ".repeat(5)),
            Message::assistant("Short reply."),
            Message::user("Latest question."),
        ];

        let truncated = cw.truncate(&messages);
        // System message should always be first
        assert_eq!(truncated[0].role, MessageRole::System);
    }
}
