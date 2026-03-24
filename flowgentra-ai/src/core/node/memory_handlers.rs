//! # Memory Handlers
//!
//! Built-in handlers for managing message history, compression, and custom state.
//! These can be added to your graph via config (YAML) or programmatically.
//!
//! # Example (Config)
//! ```yaml
//! nodes:
//!   - name: my_appender
//!     handler: memory::append_message
//!     config:
//!       role: user
//! ```
//!
//! # Example (Code)
//! ```ignore
//! agent.add_node("append", handlers::memory::append_message_handler)?;
//! ```

use crate::core::error::Result;
use crate::core::state::State;
use crate::core::state::{CompressionManager, MessageHistory};
use serde_json::json;

/// Append a message to the message history
///
/// Expects state fields:
/// - `input` or `user_input` - the message content
/// - `role` (from config) - one of: "user", "assistant", "system"
///
/// # Config Example
/// ```yaml
/// nodes:
///   - name: append_user_message
///     handler: memory::append_message
///     config:
///       role: user
/// ```
pub async fn append_message_handler(
    state: crate::core::state::SharedState,
) -> Result<crate::core::state::SharedState> {
    let role = state
        .get_typed::<String>("role")
        .unwrap_or_else(|_| "user".to_string());

    let content = state
        .get("input")
        .or_else(|| state.get("user_input"))
        .or_else(|| state.get("message"))
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .ok_or_else(|| {
            crate::core::error::FlowgentraError::ValidationError(
                "No input/user_input/message field found for append_message".to_string(),
            )
        })?;

    let mut history = MessageHistory::from_state(&state)?;

    match role.as_str() {
        "user" => history.add_user_message(content),
        "assistant" => history.add_assistant_message(content),
        "system" => history.add_system_message(content),
        _ => {
            return Err(crate::core::error::FlowgentraError::ValidationError(
                format!("Invalid role: {}", role),
            ))
        }
    }

    history.save_to_state(&state)?;
    Ok(state)
}

/// Compress message history to manage token usage
///
/// Keeps recent messages, summarizes older ones.
///
/// # Config Example
/// ```yaml
/// nodes:
///   - name: compress_history
///     handler: memory::compress_history
///     config:
///       max_recent_messages: 10
/// ```
pub async fn compress_history_handler(
    state: crate::core::state::SharedState,
) -> Result<crate::core::state::SharedState> {
    let max_recent = state
        .get_typed::<usize>("max_recent_messages")
        .unwrap_or(10);

    let manager = CompressionManager::new(max_recent);
    let mut history = MessageHistory::from_state(&state)?;
    manager.compress_history(&mut history)?;
    history.save_to_state(&state)?;

    Ok(state)
}

/// Clear all messages from history
///
/// # Config Example
/// ```yaml
/// nodes:
///   - name: clear_messages
///     handler: memory::clear_history
/// ```
pub async fn clear_history_handler<T: State>(state: T) -> Result<T> {
    let mut history = MessageHistory::from_state(&state)?;
    history.clear();
    history.save_to_state(&state)?;

    state.set("cleared_at", json!(chrono::Utc::now().to_rfc3339()));
    Ok(state)
}

/// Get message count for logging/debugging
///
/// Sets `message_count` in state
pub async fn get_message_count_handler<T: State>(state: T) -> Result<T> {
    let history = MessageHistory::from_state(&state)?;
    state.set("message_count", json!(history.len()));
    Ok(state)
}

/// Format message history for LLM context
///
/// Converts message history to formatted text for use in prompts
pub async fn format_history_for_context_handler<T: State>(state: T) -> Result<T> {
    let history = MessageHistory::from_state(&state)?;

    let formatted = history
        .messages()
        .iter()
        .map(|msg| format!("{}: {}", msg.role, msg.content))
        .collect::<Vec<_>>()
        .join("\n");

    state.set("formatted_history", json!(formatted));
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::SharedState;

    #[tokio::test]
    async fn test_append_message() {
        let state = SharedState::new(Default::default());
        state.set("input", json!("Hello"));
        state.set("role", json!("user"));

        let result = append_message_handler(state.clone()).await;
        assert!(result.is_ok());

        let history = MessageHistory::from_state(&state).unwrap();
        assert_eq!(history.len(), 1);
    }

    #[tokio::test]
    async fn test_clear_history() {
        let state = SharedState::new(Default::default());
        let mut history = MessageHistory::new();
        history.add_user_message("Test");
        history.save_to_state(&state).unwrap();

        let result = clear_history_handler(state.clone()).await;
        assert!(result.is_ok());

        let history = MessageHistory::from_state(&state).unwrap();
        assert!(history.is_empty());
    }
}
