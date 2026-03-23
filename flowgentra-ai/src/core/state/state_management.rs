//! # State Management for LangGraph-style Memory
//!
//! Provides easy-to-use memory patterns inspired by LangGraph:
//! 1. **Persistent State** - Basic state that flows through graph
//! 2. **Message History** - Automatic message list management
//! 3. **Summary/Compression** - Summarize old messages to manage tokens
//! 4. **Custom State Fields** - Any additional state data
//! 5. **Thread-Scoped Memory** - Multi-tenant conversations

use crate::core::error::Result;
use crate::core::llm::{Message, MessageRole};
use crate::core::state::State;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Helper to work with message history in state
///
/// # Example
/// ```ignore
/// let mut history = MessageHistory::from_state(&state);
/// history.add_user_message("What is Rust?");
/// history.save_to_state(&state);
/// ```
pub struct MessageHistory {
    messages: Vec<MessageHistoryEntry>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MessageHistoryEntry {
    pub role: String,
    pub content: String,
}

impl MessageHistory {
    /// Create empty message history
    pub fn new() -> Self {
        MessageHistory {
            messages: Vec::new(),
        }
    }

    /// Load from state field `messages` (LangGraph-style)
    pub fn from_state<T: State>(state: &T) -> Result<Self> {
        if let Some(messages_value) = state.get("messages") {
            if let Ok(messages) = serde_json::from_value::<Vec<MessageHistoryEntry>>(messages_value)
            {
                return Ok(MessageHistory { messages });
            }
        }
        Ok(MessageHistory::new())
    }

    /// Add user message
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.messages.push(MessageHistoryEntry {
            role: "user".to_string(),
            content: content.into(),
        });
    }

    /// Add assistant message
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.messages.push(MessageHistoryEntry {
            role: "assistant".to_string(),
            content: content.into(),
        });
    }

    /// Add system message
    pub fn add_system_message(&mut self, content: impl Into<String>) {
        self.messages.push(MessageHistoryEntry {
            role: "system".to_string(),
            content: content.into(),
        });
    }

    /// Save back to state
    pub fn save_to_state<T: State>(&self, state: &T) -> Result<()> {
        let messages_json = serde_json::to_value(&self.messages)?;
        state.set("messages", messages_json);
        Ok(())
    }

    /// Get all messages
    pub fn messages(&self) -> &[MessageHistoryEntry] {
        &self.messages
    }

    /// Get count
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Clear messages
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Convert to Message objects for LLM
    pub fn to_llm_messages(&self) -> Vec<Message> {
        self.messages
            .iter()
            .map(|entry| {
                let role = match entry.role.as_str() {
                    "user" => MessageRole::User,
                    "assistant" => MessageRole::Assistant,
                    "system" => MessageRole::System,
                    _ => MessageRole::User,
                };
                Message {
                    role,
                    content: entry.content.clone(),
                    ..Default::default()
                }
            })
            .collect()
    }
}

impl Default for MessageHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Compression manager - summarize old messages
pub struct CompressionManager {
    max_recent_messages: usize,
}

impl CompressionManager {
    /// Create new compression manager
    ///
    /// # Arguments
    /// * `max_recent_messages` - Keep this many recent messages, summarize older ones
    pub fn new(max_recent_messages: usize) -> Self {
        CompressionManager {
            max_recent_messages,
        }
    }

    /// Apply compression to message history
    /// Keeps recent messages, summarizes older ones
    pub fn compress_history(&self, history: &mut MessageHistory) -> Result<()> {
        let total_messages = history.len();

        if total_messages <= self.max_recent_messages {
            return Ok(()); // No compression needed
        }

        let messages = history.messages.clone();
        let summary_count = total_messages - self.max_recent_messages;

        // Create summary of old messages
        let old_messages = &messages[0..summary_count];
        let summary = self.create_summary(old_messages);

        // Keep summary + recent messages
        let mut compressed = vec![MessageHistoryEntry {
            role: "system".to_string(),
            content: format!("Previous conversation summary:\n{}", summary),
        }];

        compressed.extend_from_slice(&messages[summary_count..]);
        history.messages = compressed;

        Ok(())
    }

    fn create_summary(&self, messages: &[MessageHistoryEntry]) -> String {
        let user_count = messages.iter().filter(|m| m.role == "user").count();
        let assistant_count = messages.iter().filter(|m| m.role == "assistant").count();

        format!(
            "Previous conversation ({} user messages, {} assistant messages)",
            user_count, assistant_count
        )
    }
}

/// Thread-scoped memory manager for multi-tenant support
#[allow(dead_code)]
pub struct ThreadManager {
    threads: HashMap<String, Value>,
}

impl ThreadManager {
    #[allow(dead_code)]
    /// Create new thread manager
    pub fn new() -> Self {
        ThreadManager {
            threads: HashMap::new(),
        }
    }

    /// Get thread state
    #[allow(dead_code)]
    pub fn get_thread_state(&self, thread_id: &str) -> Option<&Value> {
        self.threads.get(thread_id)
    }

    /// Set thread state
    #[allow(dead_code)]
    pub fn set_thread_state(&mut self, thread_id: impl Into<String>, state: Value) {
        self.threads.insert(thread_id.into(), state);
    }

    /// Create thread if doesn't exist
    #[allow(dead_code)]
    pub fn create_thread(&mut self, thread_id: impl Into<String>) {
        let id = thread_id.into();
        self.threads.entry(id).or_insert_with(|| json!({}));
    }

    /// Get all thread IDs
    #[allow(dead_code)]
    pub fn thread_ids(&self) -> Vec<&String> {
        self.threads.keys().collect()
    }

    /// Clear specific thread
    #[allow(dead_code)]
    pub fn clear_thread(&mut self, thread_id: &str) {
        self.threads.remove(thread_id);
    }
}

impl Default for ThreadManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Custom state field helper  
pub struct CustomState {
    fields: HashMap<String, Value>,
}

impl CustomState {
    /// Create new custom state
    pub fn new() -> Self {
        CustomState {
            fields: HashMap::new(),
        }
    }

    /// Set a custom field
    pub fn set(&mut self, key: impl Into<String>, value: Value) {
        self.fields.insert(key.into(), value);
    }

    /// Get a custom field
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.fields.get(key)
    }

    /// Save all fields to state
    pub fn save_to_state<T: State>(&self, state: &T) -> Result<()> {
        state.set("_custom_state", serde_json::to_value(&self.fields)?);
        Ok(())
    }

    /// Load from state
    pub fn from_state<T: State>(state: &T) -> Result<Self> {
        if let Some(custom_value) = state.get("_custom_state") {
            if let Ok(fields) = serde_json::from_value(custom_value) {
                return Ok(CustomState { fields });
            }
        }
        Ok(CustomState::new())
    }
}

impl Default for CustomState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_history() {
        let mut history = MessageHistory::new();
        history.add_user_message("Hello");
        history.add_assistant_message("Hi!");

        assert_eq!(history.len(), 2);
        assert!(!history.is_empty());
    }

    #[test]
    fn test_compression() {
        let manager = CompressionManager::new(2);
        let mut history = MessageHistory::new();
        for i in 0..5 {
            history.add_user_message(format!("Message {}", i));
        }

        assert_eq!(history.len(), 5);
        manager.compress_history(&mut history).ok();
        // Should have summary + 2 recent = 3 entries
        assert!(history.len() <= 3);
    }

    #[test]
    fn test_thread_manager() {
        let mut tm = ThreadManager::new();
        tm.create_thread("user_1");
        tm.create_thread("user_2");

        assert_eq!(tm.thread_ids().len(), 2);
    }
}
