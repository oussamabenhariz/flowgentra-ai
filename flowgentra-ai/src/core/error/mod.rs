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
    #[error("Cycle detected in graph involving node(s): [{nodes}]. Set `allow_cycles: true` in config.yaml if cycles are intentional, or remove the back-edge between these nodes.")]
    CycleDetected { nodes: String },

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
    #[error("Operation timed out. Increase the timeout in config.yaml, or check for blocking calls inside async handlers (use tokio::task::spawn_blocking for CPU-heavy work).")]
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

    /// Get a hint for common error scenarios.
    ///
    /// Returns a human-readable suggestion that helps developers resolve the error
    /// without needing to search documentation.
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            // ── Graph / Node errors ──
            FlowgentraError::NodeNotFound(_) => {
                Some("Make sure the handler name in config.yaml matches a #[register_handler] \
                      function name. Run the binary with RUST_LOG=debug to see registered handlers.")
            }
            FlowgentraError::InvalidEdge(_) => {
                Some("Edge references a node that was not added to the graph. \
                      Call builder.add_node(\"name\", ...) before adding edges to it.")
            }
            FlowgentraError::CycleDetected { .. } => {
                Some("Your graph has a cycle but no conditional edge routing to END. \
                      Add .add_conditional_edge(\"node\", router) where the router returns END \
                      when the termination condition is met.")
            }
            FlowgentraError::RecursionLimitExceeded { .. } => {
                Some("The graph ran more steps than allowed. Either your cycle never terminates \
                      (add a conditional edge to END) or raise the limit with \
                      builder.set_max_steps(n).")
            }
            FlowgentraError::NoTerminationPath { .. } => {
                Some("One or more nodes loop back without a path to END. \
                      Add a conditional edge that routes to END when done.")
            }
            FlowgentraError::RoutingError(_) => {
                Some("The routing function returned an unknown node name. \
                      Make sure all strings returned by your router are valid node names \
                      or the END constant.")
            }
            // ── State errors ──
            FlowgentraError::StateError(msg) if msg.contains("not found") => {
                Some("Check that previous nodes set this state field before reading it. \
                      Consider initialising the field in the initial state passed to invoke().")
            }
            FlowgentraError::StateError(msg) if msg.contains("type mismatch") => {
                Some("A state field was written with a different type than expected. \
                      Check that all nodes agree on the field's JSON type.")
            }
            FlowgentraError::InvalidStateTransition(_) => {
                Some("A node attempted a state transition that violates a reducer constraint. \
                      Review the field's #[reducer(...)] annotation and the values being written.")
            }
            // ── LLM errors ──
            FlowgentraError::LLMError(msg)
                if msg.contains("401") || msg.contains("Unauthorized") =>
            {
                Some("Your API key is invalid or expired. \
                      Check that the correct environment variable is set \
                      (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY, MISTRAL_API_KEY).")
            }
            FlowgentraError::LLMError(msg)
                if msg.contains("429") || msg.contains("rate limit") =>
            {
                Some("You have exceeded the LLM provider's rate limit. \
                      Add exponential back-off or reduce request frequency. \
                      Consider enabling the built-in LLM retry middleware.")
            }
            FlowgentraError::LLMError(msg)
                if msg.contains("context_length") || msg.contains("max_tokens") =>
            {
                Some("The prompt exceeds the model's context window. \
                      Truncate conversation history, use message summarisation \
                      (ConversationMemory::with_summary), or switch to a model with \
                      a larger context window.")
            }
            FlowgentraError::LLMError(msg) if msg.contains("model") => {
                Some("The specified model name may be incorrect or unavailable for your account. \
                      Double-check the model field in config.yaml.")
            }
            FlowgentraError::LLMError(_) => {
                Some("An LLM call failed. Enable RUST_LOG=debug to see the full request/response. \
                      Check network connectivity and provider status page.")
            }
            // ── Tool errors ──
            FlowgentraError::ToolError(msg) if msg.contains("not found") => {
                Some("The tool was not registered. \
                      Call registry.register(\"name\", Arc::new(MyTool)) before passing it \
                      to the agent, or add it via context.with_tool_registry(registry).")
            }
            FlowgentraError::ToolError(msg) if msg.contains("already registered") => {
                Some("A tool with this name is already in the registry. \
                      Use a unique name or call registry.unregister(\"name\") first.")
            }
            FlowgentraError::ValidationError(_) => {
                Some("A value did not match the expected JSON schema. \
                      Check the tool's input_schema or the state field's type annotation.")
            }
            // ── Config errors ──
            FlowgentraError::ConfigError(msg) if msg.contains("LLM") => {
                Some("LLM is not configured in this context. \
                      Add an `llm:` section to config.yaml or call \
                      builder.set_context(Context::new().with_llm_config(...)).")
            }
            FlowgentraError::ConfigError(msg) if msg.contains("MCP") => {
                Some("The named MCP server is not configured. \
                      Add the server under `mcp_servers:` in config.yaml and make sure \
                      the name matches the one used in the node's `mcps:` list.")
            }
            FlowgentraError::ConfigError(_) => {
                Some("Ensure config.yaml is valid YAML and all required fields are present. \
                      Run with RUST_LOG=debug for the full parse error.")
            }
            // ── Runtime / execution ──
            FlowgentraError::TimeoutError | FlowgentraError::ExecutionTimeout(_) => {
                Some("A node exceeded its time budget. \
                      Increase the timeout in config.yaml for this node, \
                      or optimise the handler (e.g., avoid blocking calls inside async code).")
            }
            FlowgentraError::ParallelExecutionError(_) => {
                Some("One branch of a parallel execution failed. \
                      Check the inner error message for which branch failed and why.")
            }
            FlowgentraError::ExecutionAborted(_) => {
                Some("Execution was aborted by middleware (e.g., evaluation threshold not met). \
                      Check evaluation configuration or disable the evaluation middleware.")
            }
            // ── MCP errors ──
            FlowgentraError::MCPTransportError(_) => {
                Some("Could not connect to the MCP server. \
                      Check that the server process is running and the endpoint URL is correct. \
                      This error is safe to retry.")
            }
            FlowgentraError::MCPServerError(_) => {
                Some("The MCP server returned an error response. \
                      Check the server logs. Do NOT blindly retry — the tool may have executed.")
            }
            // ── IO / serialization ──
            FlowgentraError::IoError(_) => {
                Some("A file or network I/O operation failed. \
                      Check file paths, permissions, and network connectivity.")
            }
            FlowgentraError::YamlError(_) => {
                Some("Failed to parse YAML. Validate config.yaml with a YAML linter \
                      (e.g., yamllint) and check for tabs vs. spaces, missing quotes, etc.")
            }
            // ── Context wrapper (unwrap inner hint) ──
            FlowgentraError::Context(_, inner) => inner.hint(),
            _ => None,
        }
    }

    /// Format the error with its hint, suitable for displaying to developers.
    ///
    /// ```no_run
    /// use flowgentra_ai::core::error::FlowgentraError;
    ///
    /// let err = FlowgentraError::NodeNotFound("my_handler".into());
    /// eprintln!("{}", err.display_with_hint());
    /// ```
    pub fn display_with_hint(&self) -> String {
        match self.hint() {
            Some(hint) => format!("{}\n  Hint: {}", self, hint),
            None => self.to_string(),
        }
    }
}
