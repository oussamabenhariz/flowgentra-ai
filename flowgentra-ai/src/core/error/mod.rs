//! # Error Handling
//!
//! Comprehensive error types for all FlowgentraAI operations.
//!
//! This module provides a unified error type `FlowgentraError` that covers all possible
//! failure modes in the framework, along with convenient conversion implementations
//! for common error sources.
//!
//! ## Usage
//!
//! ```no_run
//! use flowgentra_ai::core::error::{Result, FlowgentraError};
//!
//! // Most functions return Result<T>
//! fn my_operation() -> Result<String> {
//!     Err(FlowgentraError::ConfigError("Invalid configuration".into()))
//! }
//! ```

use thiserror::Error;

/// Convenient type alias for Results in FlowgentraAI operations
pub type Result<T> = std::result::Result<T, FlowgentraError>;

/// Comprehensive error enum covering all possible FlowgentraAI failures
///
/// Each variant is designed to provide context about what went wrong,
/// making it easy to handle different error scenarios.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum FlowgentraError {
    // Configuration Errors
    /// Invalid or malformed configuration
    #[error("Configuration error: {0}")]
    ConfigError(String),

    // Graph Errors
    /// Issues with graph structure or operations
    #[error("Graph error: {0}")]
    GraphError(String),

    /// Node not found in the graph
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Invalid edge definition
    #[error("Invalid edge: {0}")]
    InvalidEdge(String),

    /// Routing condition failed
    #[error("Routing error: {0}")]
    RoutingError(String),

    /// Cycle detected in acyclic graph
    #[error("Cycle detected in graph")]
    CycleDetected,

    /// Recursion limit exceeded during graph execution
    #[error("Recursion limit of {limit} steps exceeded. Ensure your graph has a termination condition that routes to END. You can raise the limit via `recursion_limit` in the graph config.")]
    RecursionLimitExceeded { limit: usize },

    /// Cyclic graph has nodes with no path to END (infinite loop risk)
    #[error("Graph validation warning: node(s) [{nodes}] have cycles but no path to END. Add a conditional edge to END or the graph will loop forever.")]
    NoTerminationPath { nodes: String },

    // Node/Runtime Errors
    /// Error during node execution
    #[error("Node execution error: {0}")]
    NodeExecutionError(String),

    /// Runtime orchestration error
    #[error("Runtime error: {0}")]
    RuntimeError(String),

    /// Invalid state transition
    #[error("Invalid state transition: {0}")]
    InvalidStateTransition(String),

    /// Execution failed
    #[error("Execution error: {0}")]
    ExecutionError(String),

    /// Execution aborted by middleware
    #[error("Execution aborted: {0}")]
    ExecutionAborted(String),

    /// Operation timed out
    #[error("Timeout error")]
    TimeoutError,

    /// Execution timeout
    #[error("Execution timeout: {0}")]
    ExecutionTimeout(String),

    // Parallel Execution Errors
    /// Error during parallel execution
    #[error("Parallel execution error: {0}")]
    ParallelExecutionError(String),

    // State Errors
    /// Error with state management
    #[error("State error: {0}")]
    StateError(String),

    /// Error context wrapper preserving the original error trace
    #[error("{0}")]
    Context(String, #[source] Box<FlowgentraError>),

    // Service Integration Errors
    /// LLM operation failed
    #[error("LLM error: {0}")]
    LLMError(String),

    /// MCP (Model Context Protocol) operation failed
    #[error("MCP error: {0}")]
    MCPError(String),

    /// MCP transport-level error (connection refused, timeout, DNS failure).
    /// Safe to retry because the request never reached the server.
    #[error("MCP transport error: {0}")]
    MCPTransportError(String),

    /// MCP server-side error (HTTP 5xx, tool execution failure).
    /// NOT safe to retry blindly — the tool may have executed.
    #[error("MCP server error: {0}")]
    MCPServerError(String),

    /// Tool operation failed
    #[error("Tool error: {0}")]
    ToolError(String),

    /// Validation error (e.g., schema validation)
    #[error("Validation error: {0}")]
    ValidationError(String),

    // Serialization Errors
    /// JSON serialization/deserialization failed
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// YAML parsing failed
    #[error("YAML parse error: {0}")]
    YamlError(String),

    // System Errors
    /// File I/O operation failed
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl FlowgentraError {
    /// Returns true if this is a transport-level error safe to retry.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            FlowgentraError::MCPTransportError(_) | FlowgentraError::TimeoutError
        )
    }
}

// NOTE: `From<String>` and `From<&str>` impls were intentionally removed.
// They silently converted any string into `ExecutionError`, making it easy
// to accidentally swallow structured errors via the `?` operator.
// Use explicit error variants instead: `FlowgentraError::ExecutionError(msg)`.

// =============================================================================
// Error Context & Helper Methods
// =============================================================================

impl FlowgentraError {
    /// Add context to an error message
    ///
    /// Useful for providing debugging information without re-wrapping the error.
    ///
    /// # Example
    /// ```no_run
    /// use flowgentra_ai::core::error::FlowgentraError;
    ///
    /// let err = FlowgentraError::ConfigError("invalid value".to_string());
    /// let contextualized = err.context("while loading agent config from 'config.yaml'");
    /// ```
    pub fn context(self, msg: &str) -> Self {
        match self {
            FlowgentraError::Context(existing_msg, inner) => {
                FlowgentraError::Context(format!("{}\nContext: {}", existing_msg, msg), inner)
            }
            _ => FlowgentraError::Context(msg.to_string(), Box::new(self)),
        }
    }

    /// Check if error is a timeout
    pub fn is_timeout(&self) -> bool {
        match self {
            FlowgentraError::TimeoutError | FlowgentraError::ExecutionTimeout(_) => true,
            FlowgentraError::Context(_, inner) => inner.is_timeout(),
            _ => false,
        }
    }

    /// Check if error is a validation error
    pub fn is_validation_error(&self) -> bool {
        match self {
            FlowgentraError::ValidationError(_) => true,
            FlowgentraError::Context(_, inner) => inner.is_validation_error(),
            _ => false,
        }
    }

    /// Check if error is an LLM error
    pub fn is_llm_error(&self) -> bool {
        match self {
            FlowgentraError::LLMError(_) => true,
            FlowgentraError::Context(_, inner) => inner.is_llm_error(),
            _ => false,
        }
    }

    /// Check if error is a state-related error
    pub fn is_state_error(&self) -> bool {
        match self {
            FlowgentraError::StateError(_) => true,
            FlowgentraError::Context(_, inner) => inner.is_state_error(),
            _ => false,
        }
    }

    /// Get a hint for common error scenarios
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            FlowgentraError::NodeNotFound(_) => {
                Some("Make sure the handler name in config.yaml matches a #[register_handler] function name")
            }
            FlowgentraError::StateError(msg) if msg.contains("not found") => {
                Some("Check that previous nodes set this state field, or set it in main before agent.run()")
            }
            FlowgentraError::LLMError(msg) if msg.contains("401") || msg.contains("Unauthorized") => {
                Some("Check your API key is valid and set correctly (e.g., $MISTRAL_API_KEY environment variable)")
            }
            FlowgentraError::TimeoutError | FlowgentraError::ExecutionTimeout(_) => {
                Some("Increase the timeout value in config.yaml for this node, or optimize handler performance")
            }
            FlowgentraError::ConfigError(_) => {
                Some("Ensure config.yaml is valid YAML and all required fields are present")
            }
            FlowgentraError::Context(_, inner) => inner.hint(),
            _ => None,
        }
    }
}
