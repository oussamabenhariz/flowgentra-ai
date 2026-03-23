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

    #[error("Cycle detected with no exit condition")]
    UnterminatedCycle,

    #[error("Router error: {0}")]
    RouterError(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Interrupted by breakpoint at node '{node}'")]
    InterruptedAtBreakpoint { node: String },

    #[error("Resume failed: {0}")]
    ResumeFailed(String),

    #[error("Type error: {0}")]
    TypeError(String),
}

/// Result type for state graph operations
pub type Result<T> = std::result::Result<T, StateGraphError>;
