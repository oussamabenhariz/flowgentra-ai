//! # Conversation memory and buffer/window
//!
//! Stores message history per thread with optional last-N (buffer/window) limiting.

use crate::core::error::Result;
use crate::core::llm::Message;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

/// Configuration for buffer/window: limit how many messages are kept or returned.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BufferWindowConfig {
    /// Maximum number of messages to keep per thread (sliding window). None = no limit.
    #[serde(default)]
    pub max_messages: Option<usize>,
}

/// Conversation memory: message history per thread with optional buffer/window.
pub trait ConversationMemory: Send + Sync {
    /// Add a message for the given thread.
    fn add_message(&self, thread_id: &str, message: Message) -> Result<()>;

    /// Get messages for the thread in chronological order (oldest first).
    /// When `limit` is set, returns the last N messages in chronological order.
    fn messages(&self, thread_id: &str, limit: Option<usize>) -> Result<Vec<Message>>;

    /// Clear all messages for the thread.
    fn clear(&self, thread_id: &str) -> Result<()>;
}

/// In-memory conversation memory with optional max-messages window per thread.
pub struct InMemoryConversationMemory {
    /// thread_id -> list of messages (oldest first)
    store: RwLock<HashMap<String, Vec<Message>>>,
    /// If set, only the last N messages are kept when adding (sliding window).
    max_messages: Option<usize>,
}

impl InMemoryConversationMemory {
    pub fn new() -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
            max_messages: None,
        }
    }

    /// Set a sliding window: only keep the last N messages per thread.
    pub fn with_max_messages(mut self, max: usize) -> Self {
        self.max_messages = Some(max);
        self
    }

    /// Create from config (e.g. from YAML).
    pub fn with_config(config: &BufferWindowConfig) -> Self {
        let mut m = Self::new();
        if let Some(max) = config.max_messages {
            m.max_messages = Some(max);
        }
        m
    }
}

impl Default for InMemoryConversationMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversationMemory for InMemoryConversationMemory {
    fn add_message(&self, thread_id: &str, message: Message) -> Result<()> {
        let mut guard = self
            .store
            .write()
            .map_err(|e| crate::core::error::FlowgentraError::StateError(e.to_string()))?;
        let list = guard.entry(thread_id.to_string()).or_default();
        list.push(message);
        if let Some(max) = self.max_messages {
            if list.len() > max {
                list.drain(0..(list.len() - max));
            }
        }
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
