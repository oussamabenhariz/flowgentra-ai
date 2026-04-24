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
}

/// Result type for state graph operations
pub type Result<T> = std::result::Result<T, StateGraphError>;
