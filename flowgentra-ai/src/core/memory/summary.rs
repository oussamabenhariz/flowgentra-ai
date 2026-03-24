//! Summary and Token Buffer conversation memory types
//!
//! - **SummaryMemory** — summarizes old messages to stay within a token budget
//! - **TokenBufferMemory** — keeps recent messages up to a token limit

use crate::core::error::Result;
use crate::core::llm::Message;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

use super::conversation::ConversationMemory;

// =============================================================================
// TokenBufferMemory
// =============================================================================

/// Conversation memory that keeps messages up to a token budget.
///
/// When the total token count exceeds `max_tokens`, the oldest non-system
/// messages are dropped. System messages are always preserved.
///
/// Token estimation uses ~4 chars per token.
pub struct TokenBufferMemory {
    store: RwLock<HashMap<String, Vec<Message>>>,
    max_tokens: usize,
}

impl TokenBufferMemory {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
            max_tokens,
        }
    }

    fn estimate_tokens(msg: &Message) -> usize {
        // ~4 chars per token, plus overhead for role
        (msg.content.len() + 10) / 4
    }

    fn trim_to_budget(messages: &mut Vec<Message>, max_tokens: usize) {
        loop {
            let total: usize = messages.iter().map(Self::estimate_tokens).sum();
            if total <= max_tokens || messages.len() <= 1 {
                break;
            }

            // Find the first non-system message to remove
            if let Some(idx) = messages.iter().position(|m| !m.is_system()) {
                messages.remove(idx);
            } else {
                break;
            }
        }
    }
}

impl ConversationMemory for TokenBufferMemory {
    fn add_message(&self, thread_id: &str, message: Message) -> Result<()> {
        let mut guard = self
            .store
            .write()
            .map_err(|e| crate::core::error::FlowgentraError::StateError(e.to_string()))?;
        let list = guard.entry(thread_id.to_string()).or_default();
        list.push(message);
        Self::trim_to_budget(list, self.max_tokens);
        Ok(())
    }

    fn messages(&self, thread_id: &str, limit: Option<usize>) -> Result<Vec<Message>> {
        let guard = self
            .store
            .read()
            .map_err(|e| crate::core::error::FlowgentraError::StateError(e.to_string()))?;
        let list = guard.get(thread_id).cloned().unwrap_or_default();
        let out = match limit {
            Some(n) if list.len() > n => list[list.len().saturating_sub(n)..].to_vec(),
            _ => list,
        };
        Ok(out)
    }

    fn clear(&self, thread_id: &str) -> Result<()> {
        let mut guard = self
            .store
            .write()
            .map_err(|e| crate::core::error::FlowgentraError::StateError(e.to_string()))?;
        guard.remove(thread_id);
        Ok(())
    }
}

// =============================================================================
// SummaryMemory
// =============================================================================

/// Configuration for summary memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryConfig {
    /// Maximum number of messages to keep in full before summarizing
    pub buffer_size: usize,
    /// Maximum tokens for the summary itself
    pub max_summary_tokens: usize,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10,
            max_summary_tokens: 200,
        }
    }
}

/// Conversation memory that summarizes old messages to stay within context limits.
///
/// Keeps the most recent `buffer_size` messages in full, and stores a running
/// summary of older messages. The summary is prepended as a system message
/// when retrieving the conversation history.
///
/// The caller provides a summary function (typically wrapping an LLM call).
pub struct SummaryMemory<F>
where
    F: Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
{
    store: RwLock<HashMap<String, ThreadState>>,
    config: SummaryConfig,
    summarize_fn: F,
}

#[derive(Debug, Clone, Default)]
struct ThreadState {
    messages: Vec<Message>,
    summary: Option<String>,
}

impl<F> SummaryMemory<F>
where
    F: Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
{
    /// Create a new summary memory.
    ///
    /// `summarize_fn` should take conversation text and return a summary.
    ///
    /// # Example
    /// ```ignore
    /// let memory = SummaryMemory::new(SummaryConfig::default(), |text| {
    ///     Box::pin(async move {
    ///         let msg = Message::user(format!("Summarize this conversation:\n{}", text));
    ///         let resp = llm.chat(vec![msg]).await?;
    ///         Ok(resp.content)
    ///     })
    /// });
    /// ```
    pub fn new(config: SummaryConfig, summarize_fn: F) -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
            config,
            summarize_fn,
        }
    }

    /// Get the current summary for a thread.
    pub fn get_summary(&self, thread_id: &str) -> Option<String> {
        self.store
            .read()
            .ok()
            .and_then(|guard| guard.get(thread_id).and_then(|ts| ts.summary.clone()))
    }

    /// Trigger summarization of old messages for a thread.
    ///
    /// This should be called periodically (e.g., after each turn) to keep
    /// the buffer trimmed.
    pub async fn summarize_if_needed(&self, thread_id: &str) -> Result<()> {
        let to_summarize = {
            let guard = self
                .store
                .read()
                .map_err(|e| crate::core::error::FlowgentraError::StateError(e.to_string()))?;
            let state = match guard.get(thread_id) {
                Some(s) => s,
                None => return Ok(()),
            };

            if state.messages.len() <= self.config.buffer_size {
                return Ok(());
            }

            // Collect messages to summarize (everything except the last buffer_size)
            let cut = state.messages.len() - self.config.buffer_size;
            let old_messages: Vec<_> = state.messages[..cut].to_vec();

            // Build text to summarize
            let mut text = String::new();
            if let Some(ref existing) = state.summary {
                text.push_str("Previous summary: ");
                text.push_str(existing);
                text.push_str("\n\nNew messages:\n");
            }
            for msg in &old_messages {
                text.push_str(&format!("[{}] {}\n", msg.role_str(), msg.content));
            }
            text
        };

        // Call summarize function
        let new_summary = (self.summarize_fn)(to_summarize).await?;

        // Update state
        let mut guard = self
            .store
            .write()
            .map_err(|e| crate::core::error::FlowgentraError::StateError(e.to_string()))?;
        if let Some(state) = guard.get_mut(thread_id) {
            let cut = state.messages.len().saturating_sub(self.config.buffer_size);
            state.messages.drain(..cut);
            state.summary = Some(new_summary);
        }

        Ok(())
    }
}

impl<F> ConversationMemory for SummaryMemory<F>
where
    F: Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
{
    fn add_message(&self, thread_id: &str, message: Message) -> Result<()> {
        let mut guard = self
            .store
            .write()
            .map_err(|e| crate::core::error::FlowgentraError::StateError(e.to_string()))?;
        let state = guard.entry(thread_id.to_string()).or_default();
        state.messages.push(message);
        Ok(())
    }

    fn messages(&self, thread_id: &str, limit: Option<usize>) -> Result<Vec<Message>> {
        let guard = self
            .store
            .read()
            .map_err(|e| crate::core::error::FlowgentraError::StateError(e.to_string()))?;
        let state = guard.get(thread_id);

        let mut result = Vec::new();

        if let Some(state) = state {
            // Prepend summary as a system message if available
            if let Some(ref summary) = state.summary {
                result.push(Message::system(format!(
                    "Summary of earlier conversation: {}",
                    summary
                )));
            }

            let msgs = &state.messages;
            let out = match limit {
                Some(n) if msgs.len() > n => msgs[msgs.len().saturating_sub(n)..].to_vec(),
                _ => msgs.clone(),
            };
            result.extend(out);
        }

        Ok(result)
    }

    fn clear(&self, thread_id: &str) -> Result<()> {
        let mut guard = self
            .store
            .write()
            .map_err(|e| crate::core::error::FlowgentraError::StateError(e.to_string()))?;
        guard.remove(thread_id);
        Ok(())
    }
}

/// Helper extension for Message to get role as string
trait MessageRoleStr {
    fn role_str(&self) -> &str;
}

impl MessageRoleStr for Message {
    fn role_str(&self) -> &str {
        if self.is_system() {
            "system"
        } else if self.is_user() {
            "user"
        } else if self.is_assistant() {
            "assistant"
        } else {
            "tool"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_buffer_memory() {
        let memory = TokenBufferMemory::new(100); // ~100 tokens

        // Add messages
        memory
            .add_message("t1", Message::system("You are helpful"))
            .unwrap();
        memory.add_message("t1", Message::user("Hello")).unwrap();
        memory
            .add_message("t1", Message::assistant("Hi there!"))
            .unwrap();

        let msgs = memory.messages("t1", None).unwrap();
        assert!(msgs.len() >= 2);
    }

    #[test]
    fn test_token_buffer_trims() {
        let memory = TokenBufferMemory::new(20); // Very small budget

        memory.add_message("t1", Message::system("sys")).unwrap();
        memory
            .add_message(
                "t1",
                Message::user("A long message that exceeds the budget"),
            )
            .unwrap();
        memory
            .add_message("t1", Message::assistant("Another long response here"))
            .unwrap();

        let msgs = memory.messages("t1", None).unwrap();
        // Should have trimmed some messages
        assert!(msgs.len() <= 3);
    }

    #[test]
    fn test_token_buffer_preserves_system() {
        let memory = TokenBufferMemory::new(30);

        memory.add_message("t1", Message::system("sys")).unwrap();
        for i in 0..10 {
            memory
                .add_message("t1", Message::user(format!("msg {}", i)))
                .unwrap();
        }

        let msgs = memory.messages("t1", None).unwrap();
        // System message should still be there
        assert!(msgs.iter().any(|m| m.is_system()));
    }

    #[test]
    fn test_token_estimate() {
        let msg = Message::user("Hello world"); // 11 + 10 = 21 chars / 4 ≈ 5 tokens
        let tokens = TokenBufferMemory::estimate_tokens(&msg);
        assert!(tokens > 0);
        assert!(tokens < 20);
    }
}
