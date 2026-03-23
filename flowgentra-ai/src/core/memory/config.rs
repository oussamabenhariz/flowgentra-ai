//! # Memory configuration (YAML and programmatic)
//!
//! Configures checkpointer and conversation memory from config.yaml or code.

use serde::{Deserialize, Serialize};

use super::conversation::BufferWindowConfig;

/// Memory section in config.yaml.
///
/// Example:
/// ```yaml
/// memory:
///   checkpointer:
///     type: memory
///   conversation:
///     enabled: true
///     state_key: messages
///   buffer:
///     max_messages: 20
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryConfig {
    #[serde(default)]
    pub checkpointer: CheckpointerConfig,

    #[serde(default)]
    pub conversation: ConversationMemoryConfig,

    #[serde(default)]
    pub buffer: BufferWindowConfig,
}

/// Checkpointer backend selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointerConfig {
    /// Backend type: "memory" (in-memory), "none" or omitted to disable.
    #[serde(rename = "type", default = "default_checkpointer_type")]
    pub type_: String,
}

fn default_checkpointer_type() -> String {
    "none".to_string()
}

impl Default for CheckpointerConfig {
    fn default() -> Self {
        Self {
            type_: "none".to_string(),
        }
    }
}

/// Conversation memory config.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConversationMemoryConfig {
    /// Enable conversation memory (message history per thread).
    #[serde(default)]
    pub enabled: bool,

    /// Optional state key to sync messages into state (e.g. "messages").
    #[serde(default)]
    pub state_key: Option<String>,
}
