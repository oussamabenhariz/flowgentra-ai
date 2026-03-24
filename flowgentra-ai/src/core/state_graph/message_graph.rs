//! MessageGraph — a convenience wrapper for chat-focused workflows
//!
//! Pre-configures a StateGraph with message accumulation using the Append reducer.
//! Messages flow through the graph and accumulate automatically.

use serde_json::json;

use crate::core::llm::Message;
use crate::core::state::PlainState;
use crate::core::state_graph::error::Result;
use crate::core::state_graph::executor::{StateGraph, StateGraphBuilder};
use crate::core::state_graph::RouterFn;

/// Builder for message-centric graphs.
///
/// Automatically manages a "messages" array in state with append semantics.
/// Each node receives the full message history and can add new messages.
///
/// # Example
/// ```ignore
/// use flowgentra_ai::core::state_graph::message_graph::MessageGraphBuilder;
/// use flowgentra_ai::core::llm::Message;
///
/// let graph = MessageGraphBuilder::new()
///     .add_fn("echo", |state: &PlainState| {
///         let messages = MessageGraphBuilder::get_messages(state);
///         let last = messages.last().map(|m| m.content.clone()).unwrap_or_default();
///         let mut s = state.clone();
///         s = MessageGraphBuilder::add_message(s, Message::assistant(format!("Echo: {}", last)));
///         Box::pin(async move { Ok(s) })
///     })
///     .set_entry_point("echo")
///     .add_edge("echo", "__end__")
///     .compile()?;
/// ```
pub struct MessageGraphBuilder {
    inner: StateGraphBuilder<PlainState>,
}

impl MessageGraphBuilder {
    pub fn new() -> Self {
        Self {
            inner: StateGraphBuilder::new(),
        }
    }

    /// Add a node function. The function receives `&PlainState` and returns a pinned future.
    pub fn add_fn<F>(mut self, name: &str, f: F) -> Self
    where
        F: Fn(
                &PlainState,
            )
                -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<PlainState>> + Send>>
            + Send
            + Sync
            + 'static,
    {
        self.inner = self.inner.add_fn(name, f);
        self
    }

    /// Set the entry point.
    pub fn set_entry_point(mut self, name: &str) -> Self {
        self.inner = self.inner.set_entry_point(name);
        self
    }

    /// Add a fixed edge.
    pub fn add_edge(mut self, from: &str, to: &str) -> Self {
        self.inner = self.inner.add_edge(from, to);
        self
    }

    /// Add a conditional edge with a sync router.
    pub fn add_conditional_edge(mut self, from: &str, router: RouterFn<PlainState>) -> Self {
        self.inner = self.inner.add_conditional_edge(from, router);
        self
    }

    /// Compile the graph.
    pub fn compile(self) -> Result<StateGraph<PlainState>> {
        self.inner.compile()
    }

    // ── Helper functions for working with messages in state ──

    /// Create initial state with messages.
    pub fn initial_state(messages: Vec<Message>) -> PlainState {
        let mut state = PlainState::new();
        let msg_values: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|m| {
                json!({
                    "role": format!("{:?}", m.role).to_lowercase(),
                    "content": m.content
                })
            })
            .collect();
        state.set("messages", json!(msg_values));
        state
    }

    /// Get messages from state.
    pub fn get_messages(state: &PlainState) -> Vec<Message> {
        state
            .get("messages")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        let role = v.get("role")?.as_str()?;
                        let content = v.get("content")?.as_str()?.to_string();
                        Some(match role {
                            "system" => Message::system(content),
                            "user" => Message::user(content),
                            "assistant" => Message::assistant(content),
                            "tool" => Message::tool(content),
                            _ => Message::user(content),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Add a message to state (append semantics).
    pub fn add_message(mut state: PlainState, message: Message) -> PlainState {
        let msg_value = json!({
            "role": format!("{:?}", message.role).to_lowercase(),
            "content": message.content
        });

        let mut messages = state
            .get("messages")
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        messages.push(msg_value);
        state.set("messages", json!(messages));
        state
    }

    /// Get the last message from state.
    pub fn last_message(state: &PlainState) -> Option<Message> {
        Self::get_messages(state).into_iter().last()
    }
}

impl Default for MessageGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let state = MessageGraphBuilder::initial_state(vec![
            Message::user("Hello"),
            Message::system("sys"),
        ]);

        let messages = MessageGraphBuilder::get_messages(&state);
        assert_eq!(messages.len(), 2);
        assert!(messages[0].is_user());
        assert_eq!(messages[0].content, "Hello");
    }

    #[test]
    fn test_add_message() {
        let state = MessageGraphBuilder::initial_state(vec![Message::user("Hi")]);
        let state = MessageGraphBuilder::add_message(state, Message::assistant("Hello!"));

        let messages = MessageGraphBuilder::get_messages(&state);
        assert_eq!(messages.len(), 2);
        assert!(messages[1].is_assistant());
    }

    #[test]
    fn test_last_message() {
        let state = MessageGraphBuilder::initial_state(vec![
            Message::user("Hi"),
            Message::assistant("Hello!"),
        ]);

        let last = MessageGraphBuilder::last_message(&state).unwrap();
        assert!(last.is_assistant());
        assert_eq!(last.content, "Hello!");
    }

    #[tokio::test]
    async fn test_message_graph_compile() {
        let graph = MessageGraphBuilder::new()
            .add_fn("echo", |state: &PlainState| {
                let messages = MessageGraphBuilder::get_messages(state);
                let last = messages
                    .last()
                    .map(|m| m.content.clone())
                    .unwrap_or_default();
                let mut s = state.clone();
                s = MessageGraphBuilder::add_message(
                    s,
                    Message::assistant(format!("Echo: {}", last)),
                );
                Box::pin(async move { Ok(s) })
            })
            .set_entry_point("echo")
            .add_edge("echo", "__end__")
            .compile()
            .unwrap();

        let state = MessageGraphBuilder::initial_state(vec![Message::user("Hello")]);
        let result = graph.invoke(state).await.unwrap();

        let messages = MessageGraphBuilder::get_messages(&result);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content, "Echo: Hello");
    }
}
