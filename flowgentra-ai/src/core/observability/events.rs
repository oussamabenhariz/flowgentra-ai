//! Real-time execution event broadcast.
//!
//! Provides a `tokio::sync::broadcast` channel for subscribing to graph
//! execution events from outside the graph (e.g., a UI dashboard, logger,
//! or external monitoring system).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::broadcast;

/// An execution event emitted during graph execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionEvent {
    /// Graph execution started.
    GraphStarted { graph_id: String },
    /// A node is about to execute.
    NodeStarted { node_name: String, step: usize },
    /// A node finished execution.
    NodeCompleted {
        node_name: String,
        step: usize,
        duration_ms: u64,
        /// Optionally includes the state snapshot after this node
        state_snapshot: Option<Value>,
    },
    /// A node failed.
    NodeFailed {
        node_name: String,
        step: usize,
        error: String,
    },
    /// An edge was traversed (routing decision).
    EdgeTraversed {
        from: String,
        to: String,
        condition: Option<String>,
    },
    /// Graph execution completed.
    GraphCompleted {
        total_steps: usize,
        total_duration_ms: u64,
    },
    /// Graph execution failed.
    GraphFailed {
        error: String,
        last_node: Option<String>,
    },
    /// A streaming chunk from an LLM call inside a node.
    ///
    /// Emitted incrementally as the LLM produces tokens. Use
    /// `EventBroadcaster::emit_llm_chunk()` from your handler.
    LLMStreaming {
        node_name: String,
        /// The text chunk produced by the LLM (may be a single token or a few words).
        chunk: String,
        /// Running total of chunks emitted so far for this node.
        chunk_index: usize,
    },
    /// An LLM streaming call finished for a node.
    LLMStreamingCompleted {
        node_name: String,
        total_chunks: usize,
    },
    /// A tool was invoked by a node.
    ToolCalled {
        node_name: String,
        tool_name: String,
        args: Value,
    },
    /// A tool returned a result.
    ToolResult {
        node_name: String,
        tool_name: String,
        result: Value,
        success: bool,
    },
}

/// Event broadcaster for real-time execution monitoring.
///
/// Create one per graph and share with `Arc`. Subscribers call `subscribe()`
/// to get a `broadcast::Receiver<ExecutionEvent>`.
///
/// # Example
/// ```ignore
/// let broadcaster = EventBroadcaster::new(256);
/// let mut rx = broadcaster.subscribe();
///
/// // In another task:
/// tokio::spawn(async move {
///     while let Ok(event) = rx.recv().await {
///         println!("Event: {:?}", event);
///     }
/// });
///
/// // During graph execution:
/// broadcaster.emit(ExecutionEvent::NodeStarted { node_name: "foo".into(), step: 0 });
/// ```
#[derive(Clone)]
pub struct EventBroadcaster {
    tx: broadcast::Sender<ExecutionEvent>,
}

impl std::fmt::Debug for EventBroadcaster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventBroadcaster")
            .field("subscribers", &self.tx.receiver_count())
            .finish()
    }
}

impl EventBroadcaster {
    /// Create a new broadcaster with the given channel capacity.
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Subscribe to execution events.
    pub fn subscribe(&self) -> broadcast::Receiver<ExecutionEvent> {
        self.tx.subscribe()
    }

    /// Emit an event to all subscribers.
    /// Silently drops the event if there are no active subscribers.
    pub fn emit(&self, event: ExecutionEvent) {
        let _ = self.tx.send(event);
    }

    /// Get the number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }

    /// Convenience: emit a `NodeStarted` event.
    pub fn node_started(&self, node_name: impl Into<String>, step: usize) {
        self.emit(ExecutionEvent::NodeStarted {
            node_name: node_name.into(),
            step,
        });
    }

    /// Convenience: emit a `NodeCompleted` event.
    pub fn node_completed(
        &self,
        node_name: impl Into<String>,
        step: usize,
        duration_ms: u64,
        state_snapshot: Option<Value>,
    ) {
        self.emit(ExecutionEvent::NodeCompleted {
            node_name: node_name.into(),
            step,
            duration_ms,
            state_snapshot,
        });
    }

    /// Convenience: emit a `NodeFailed` event.
    pub fn node_failed(&self, node_name: impl Into<String>, step: usize, error: impl Into<String>) {
        self.emit(ExecutionEvent::NodeFailed {
            node_name: node_name.into(),
            step,
            error: error.into(),
        });
    }

    /// Convenience: emit an `EdgeTraversed` event.
    pub fn edge_traversed(
        &self,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: Option<String>,
    ) {
        self.emit(ExecutionEvent::EdgeTraversed {
            from: from.into(),
            to: to.into(),
            condition,
        });
    }

    /// Emit an LLM token chunk for a node. Call this from inside handlers that
    /// stream LLM output token-by-token.
    ///
    /// ```ignore
    /// // Inside your handler:
    /// if let Some(broadcaster) = ctx.event_broadcaster() {
    ///     broadcaster.emit_llm_chunk("my_node", token, chunk_index);
    /// }
    /// ```
    pub fn emit_llm_chunk(
        &self,
        node_name: impl Into<String>,
        chunk: impl Into<String>,
        chunk_index: usize,
    ) {
        self.emit(ExecutionEvent::LLMStreaming {
            node_name: node_name.into(),
            chunk: chunk.into(),
            chunk_index,
        });
    }

    /// Emit a `ToolCalled` event. Call this before invoking a tool in a handler.
    pub fn tool_called(
        &self,
        node_name: impl Into<String>,
        tool_name: impl Into<String>,
        args: Value,
    ) {
        self.emit(ExecutionEvent::ToolCalled {
            node_name: node_name.into(),
            tool_name: tool_name.into(),
            args,
        });
    }

    /// Emit a `ToolResult` event. Call this after a tool returns in a handler.
    pub fn tool_result(
        &self,
        node_name: impl Into<String>,
        tool_name: impl Into<String>,
        result: Value,
        success: bool,
    ) {
        self.emit(ExecutionEvent::ToolResult {
            node_name: node_name.into(),
            tool_name: tool_name.into(),
            result,
            success,
        });
    }
}

impl Default for EventBroadcaster {
    fn default() -> Self {
        Self::new(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_event_broadcast() {
        let broadcaster = EventBroadcaster::new(16);
        let mut rx = broadcaster.subscribe();

        broadcaster.emit(ExecutionEvent::GraphStarted {
            graph_id: "test".to_string(),
        });
        broadcaster.emit(ExecutionEvent::NodeStarted {
            node_name: "node1".to_string(),
            step: 0,
        });

        let e1 = rx.recv().await.unwrap();
        assert!(matches!(e1, ExecutionEvent::GraphStarted { .. }));

        let e2 = rx.recv().await.unwrap();
        assert!(matches!(e2, ExecutionEvent::NodeStarted { .. }));
    }

    #[test]
    fn test_no_subscribers_doesnt_panic() {
        let broadcaster = EventBroadcaster::new(16);
        broadcaster.emit(ExecutionEvent::GraphStarted {
            graph_id: "test".to_string(),
        });
        // No panic, event just dropped
    }
}
