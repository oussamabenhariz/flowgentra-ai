//! Error types for state graph execution

use thiserror::Error;

/// Errors that can occur during graph execution
#[derive(Error, Debug)]
pub enum StateGraphError {
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Edge not found from {from} to {to}")]
    EdgeNotFound { from: String, to: String },

    #[error("Execution error in node '{node}': {reason}")]
    ExecutionError { node: String, reason: String },

    #[error("State serialization error: {0}")]
    SerializationError(String),

    #[error("Checkpointing error: {0}")]
    CheckpointError(String),

    #[error("Invalid graph: {0}")]
    InvalidGraph(String),

    #[error("Cycle detected with no exit condition — add a conditional edge to END (e.g. add_conditional_edges(\"node\", router)) or the graph will loop forever")]
    UnterminatedCycle,

    #[error("Router error in node '{0}': the routing function returned an unknown node name. Ensure all returned names are valid node names or the END constant.")]
    RouterError(String),

    #[error("Node execution timed out: {0}. Increase the timeout in config.yaml for this node, or move blocking work into tokio::task::spawn_blocking.")]
    Timeout(String),

    #[error("Interrupted by breakpoint at node '{node}'")]
    InterruptedAtBreakpoint { node: String },

    #[error("Resume failed: {0}")]
    ResumeFailed(String),

    #[error("Type error: {0}")]
    TypeError(String),

    #[error(
        "Subgraph/supervisor nesting limit exceeded at depth {depth} (max {limit}). \
         Restructure the graph to reduce nesting depth, or increase the limit by \
         setting the FLOWGENTRA_MAX_NESTING environment variable."
    )]
    RecursionLimitExceeded { depth: usize, limit: usize },

    #[error(
        "Graph execution exceeded the wall-clock budget of {budget_secs:.1}s after {elapsed_secs:.1}s \
         at node '{node}'. Raise the budget with set_max_duration(), or check for a node that \
         hangs or loops."
    )]
    WallClockExceeded {
        budget_secs: f64,
        elapsed_secs: f64,
        node: String,
    },

    #[error("Graph execution was cancelled at node '{node}' (step {step}). State up to the last completed node is checkpointed under the thread id.")]
    Cancelled { node: String, step: usize },

    #[error(
        "Node '{node}' requested an interrupt (human input needed): {payload}. \
         Inspect the payload, then call resume_with_state(thread_id, {{...}}) to \
         inject the answer — the node re-runs with the injected state."
    )]
    InterruptedByNode {
        node: String,
        payload: serde_json::Value,
    },
}

/// Request a human-in-the-loop interrupt from inside a node.
///
/// Return this as the node's error to pause the run at this node. The state
/// at node entry is checkpointed; resuming re-runs this node, so read the
/// injected answer from state:
///
/// ```ignore
/// async fn approve(state: &MyState, _ctx: &Context) -> Result<MyStateUpdate> {
///     match &state.approval {
///         Some(answer) => Ok(update! { approved: answer == "yes" }),
///         None => Err(interrupt(serde_json::json!({
///             "question": "Approve the draft?", "options": ["yes", "no"],
///         }))),
///     }
/// }
/// ```
pub fn interrupt(payload: serde_json::Value) -> StateGraphError {
    StateGraphError::InterruptedByNode {
        // Filled in by the executor with the actual node name.
        node: String::new(),
        payload,
    }
}

/// Result type for state graph operations
pub type Result<T> = std::result::Result<T, StateGraphError>;
