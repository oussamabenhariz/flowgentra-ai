//! MessageGraph — a convenience wrapper for chat-focused workflows
//!
//! Provides `MessageState` — a pre-built typed state with message accumulation.

use std::sync::Arc;

use crate::core::llm::Message;
use crate::core::reducer::{Append, Reducer};
use crate::core::state::{Context, State};
use crate::core::state_graph::error::Result;
use crate::core::state_graph::executor::{StateGraph, StateGraphBuilder};
use crate::core::state_graph::node::Node;
use crate::core::state_graph::RouterFn;

/// Pre-built typed state for chat workflows.
///
/// Messages automatically accumulate via the `Append` reducer.
///
/// # Example
/// ```ignore
/// use std::sync::Arc;
/// use flowgentra_ai::core::state_graph::node::FunctionNode;
///
/// let graph = MessageGraphBuilder::new()
///     .add_node("echo", Arc::new(FunctionNode::new("echo", |state: &MessageState, _ctx: &Context| {
///         let last = state.messages.last().map(|m| m.content.clone()).unwrap_or_default();
///         Box::pin(async move {
///             Ok(MessageStateUpdate::new()
///                 .messages(vec![Message::assistant(format!("Echo: {}", last))]))
///         })
///     })))
///     .set_entry_point("echo")
///     .add_edge("echo", "__end__")
///     .compile()?;
/// ```
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MessageState {
    /// Chat messages — uses Append reducer to accumulate.
    pub messages: Vec<Message>,
}

/// Partial update for MessageState.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct MessageStateUpdate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<Message>>,
}

impl MessageStateUpdate {
    pub fn new() -> Self {
        Self { messages: None }
    }

    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = Some(messages);
        self
    }
}

impl State for MessageState {
    type Update = MessageStateUpdate;

    fn apply_update(&mut self, update: Self::Update) {
        if let Some(messages) = update.messages {
            <Append as Reducer<Vec<Message>>>::merge(&mut self.messages, messages);
        }
    }
}

impl MessageState {
    /// Create a new MessageState with initial messages.
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }

    /// Create an empty MessageState.
    pub fn empty() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Get the last message.
    pub fn last_message(&self) -> Option<&Message> {
        self.messages.last()
    }
}

/// Builder for message-centric graphs using `MessageState`.
pub struct MessageGraphBuilder {
    inner: StateGraphBuilder<MessageState>,
}

impl MessageGraphBuilder {
    pub fn new() -> Self {
        Self {
            inner: StateGraphBuilder::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(mut self, name: &str, node: Arc<dyn Node<MessageState>>) -> Self {
        self.inner = self.inner.add_node(name, node);
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
    pub fn add_conditional_edge(mut self, from: &str, router: RouterFn<MessageState>) -> Self {
        self.inner = self.inner.add_conditional_edge(from, router);
        self
    }

    /// Set the framework context.
    pub fn set_context(mut self, ctx: Context) -> Self {
        self.inner = self.inner.set_context(ctx);
        self
    }

    /// Compile the graph.
    pub fn compile(self) -> Result<StateGraph<MessageState>> {
        self.inner.compile()
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
    fn test_message_state_append() {
        let mut state = MessageState::new(vec![Message::user("Hello")]);

        let update = MessageStateUpdate::new().messages(vec![Message::assistant("Hi!")]);
        state.apply_update(update);

        assert_eq!(state.messages.len(), 2);
        assert!(state.messages[1].is_assistant());
    }

    #[test]
    fn test_last_message() {
        let state = MessageState::new(vec![
            Message::user("Hi"),
            Message::assistant("Hello!"),
        ]);

        let last = state.last_message().unwrap();
        assert!(last.is_assistant());
        assert_eq!(last.content, "Hello!");
    }

    #[tokio::test]
    async fn test_message_graph_compile() {
        use crate::core::state_graph::node::FunctionNode;
        let graph = MessageGraphBuilder::new()
            .add_node("echo", Arc::new(FunctionNode::new("echo", |state: &MessageState, _ctx: &Context| {
                let last = state
                    .messages
                    .last()
                    .map(|m| m.content.clone())
                    .unwrap_or_default();
                Box::pin(async move {
                    Ok(MessageStateUpdate::new()
                        .messages(vec![Message::assistant(format!("Echo: {}", last))]))
                })
            })))
            .set_entry_point("echo")
            .add_edge("echo", "__end__")
            .compile()
            .unwrap();

        let state = MessageState::new(vec![Message::user("Hello")]);
        let result = graph.invoke(state).await.unwrap();

        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[1].content, "Echo: Hello");
    }
}
