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
