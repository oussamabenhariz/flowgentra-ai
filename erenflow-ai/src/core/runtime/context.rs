//! # Unified Execution Context
//!
//! Comprehensive context passed through graph execution, containing all information
//! nodes, conditions, and middleware need to make decisions.
//!
//! ## Features
//!
//! - **State Management** - Current execution state
//! - **Service Access** - LLM and MCP clients
//! - **Timing & Metrics** - Execution timing and performance data
//! - **Metadata & Labels** - Arbitrary key-value pairs
//! - **Execution Tracking** - Path, depth, and iteration info
//! - **Error Context** - Error information when applicable
//!
//! ## Example
//!
//! ```no_run
//! use erenflow_ai::core::context::ExecutionContext;
//! use erenflow_ai::core::state::State;
//! use std::sync::Arc;
//!
//! let ctx = ExecutionContext::new("my_node", State::new());
//! println!("Node: {}", ctx.node_name());
//! println!("Depth: {}", ctx.depth());
//! ```

use crate::core::error::ErenFlowError;
use crate::core::llm::LLMClient;
use crate::core::mcp::MCPClient;
use crate::core::state::State;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Comprehensive execution context for the entire graph execution
///
/// Contains all information needed for:
/// - Node handler execution
/// - Condition evaluation
/// - Middleware processing
/// - Debugging and monitoring
#[derive(Clone)]
pub struct ExecutionContext {
    // === Core Execution Info ===
    /// Name of the current node being executed
    node_name: String,

    /// Current execution state (shared across all nodes)
    state: State,

    /// Execution path: list of nodes executed so far
    execution_path: Vec<String>,

    /// Current depth in execution tree
    depth: usize,

    /// Iteration count (for loops)
    iteration: usize,

    /// Branch identifier (for parallel execution)
    branch_id: Option<String>,

    // === Service Clients ===
    /// LLM client for AI operations
    llm_client: Option<Arc<dyn LLMClient>>,

    /// MCP clients available to this node
    mcp_clients: HashMap<String, Arc<dyn MCPClient>>,

    // === Timing & Performance ===
    /// When execution started (overall)
    start_time: Instant,

    /// When this node started
    node_start_time: Option<Instant>,

    /// Cumulative time spent in nodes
    accumulated_duration: Duration,

    // === Execution Tracking ===
    /// Attempt number (for retries)
    attempt: usize,

    /// Maximum allowed attempts
    max_attempts: usize,

    /// Whether execution should stop after this node
    should_stop: bool,

    /// Whether this node should be skipped
    should_skip: bool,

    // === Metadata & Context ===
    /// Arbitrary metadata key-value pairs
    metadata: HashMap<String, serde_json::Value>,

    /// Custom labels for grouping/filtering
    labels: HashMap<String, String>,

    /// Error information (if execution failed at this point)
    last_error: Option<Box<ErenFlowError>>,

    // === Loop Context ===
    /// Maximum iterations for loops
    max_iterations: Option<usize>,

    /// Loop break condition (if set, loop exits when true)
    loop_break_condition: Option<String>,
}

// =============================================================================
// Constructor & Basic Methods
// =============================================================================

impl ExecutionContext {
    /// Create a new execution context for a node
    pub fn new(node_name: impl Into<String>, state: State) -> Self {
        let now = Instant::now();
        Self {
            node_name: node_name.into(),
            state,
            execution_path: Vec::new(),
            depth: 0,
            iteration: 0,
            branch_id: None,
            llm_client: None,
            mcp_clients: HashMap::new(),
            start_time: now,
            node_start_time: None,
            accumulated_duration: Duration::ZERO,
            attempt: 1,
            max_attempts: 1,
            should_stop: false,
            should_skip: false,
            metadata: HashMap::new(),
            labels: HashMap::new(),
            last_error: None,
            max_iterations: None,
            loop_break_condition: None,
        }
    }

    /// Create a child context for nested execution
    pub fn child(&self, node_name: impl Into<String>) -> Self {
        let mut child = Self::new(node_name, self.state.clone());
        child.execution_path = self.execution_path.clone();
        child.execution_path.push(self.node_name.clone());
        child.depth = self.depth + 1;
        child.start_time = self.start_time;
        child.llm_client = self.llm_client.clone();
        child.mcp_clients = self.mcp_clients.clone();
        child.labels = self.labels.clone();
        child
    }

    // === Core Accessors ===

    /// Get the node name
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// Get the current state
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Get mutable access to state
    pub fn state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    /// Set the entire state
    pub fn set_state(&mut self, state: State) {
        self.state = state;
    }

    // === Execution Path & Depth ===

    /// Get the execution path (list of executed nodes)
    pub fn execution_path(&self) -> &[String] {
        &self.execution_path
    }

    /// Add a node to the execution path
    pub fn push_path(&mut self, node: String) {
        self.execution_path.push(node);
    }

    /// Get current depth
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Get iteration number (for loops)
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Set iteration number
    pub fn set_iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    /// Get branch ID (for parallel execution)
    pub fn branch_id(&self) -> Option<&str> {
        self.branch_id.as_deref()
    }

    /// Set branch ID
    pub fn set_branch_id(&mut self, branch_id: impl Into<String>) {
        self.branch_id = Some(branch_id.into());
    }

    // === Service Clients ===

    /// Set the LLM client
    pub fn with_llm_client(mut self, client: Arc<dyn LLMClient>) -> Self {
        self.llm_client = Some(client);
        self
    }

    /// Get the LLM client
    pub fn llm_client(&self) -> Option<Arc<dyn LLMClient>> {
        self.llm_client.clone()
    }

    /// Add an MCP client
    pub fn add_mcp_client(&mut self, name: impl Into<String>, client: Arc<dyn MCPClient>) {
        self.mcp_clients.insert(name.into(), client);
    }

    /// Get an MCP client by name
    pub fn mcp_client(&self, name: &str) -> Option<Arc<dyn MCPClient>> {
        self.mcp_clients.get(name).cloned()
    }

    /// Get all MCP clients
    pub fn mcp_clients(&self) -> &HashMap<String, Arc<dyn MCPClient>> {
        &self.mcp_clients
    }

    // === Timing & Performance ===

    /// Get elapsed time since overall execution started
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get elapsed time for just this node
    pub fn node_elapsed(&self) -> Duration {
        if let Some(start) = self.node_start_time {
            start.elapsed()
        } else {
            Duration::ZERO
        }
    }

    /// Mark the start of node execution
    pub fn mark_node_start(&mut self) {
        self.node_start_time = Some(Instant::now());
    }

    /// Mark the end of node execution and accumulate time
    pub fn mark_node_end(&mut self) {
        if let Some(start) = self.node_start_time {
            self.accumulated_duration += start.elapsed();
            self.node_start_time = None;
        }
    }

    /// Get total accumulated time in all nodes
    pub fn accumulated_duration(&self) -> Duration {
        self.accumulated_duration
    }

    // === Execution Control ===

    /// Get current attempt number
    pub fn attempt(&self) -> usize {
        self.attempt
    }

    /// Set attempt number
    pub fn set_attempt(&mut self, attempt: usize) {
        self.attempt = attempt;
    }

    /// Get maximum allowed attempts
    pub fn max_attempts(&self) -> usize {
        self.max_attempts
    }

    /// Set maximum allowed attempts
    pub fn set_max_attempts(&mut self, max: usize) {
        self.max_attempts = max;
    }

    /// Check if should stop execution
    pub fn should_stop(&self) -> bool {
        self.should_stop
    }

    /// Set stop flag
    pub fn set_should_stop(&mut self, value: bool) {
        self.should_stop = value;
    }

    /// Check if should skip this node
    pub fn should_skip(&self) -> bool {
        self.should_skip
    }

    /// Set skip flag
    pub fn set_should_skip(&mut self, value: bool) {
        self.should_skip = value;
    }

    // === Metadata & Labels ===

    /// Set metadata value
    pub fn set_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.metadata.insert(key.into(), value);
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }

    /// Get all metadata
    pub fn metadata(&self) -> &HashMap<String, serde_json::Value> {
        &self.metadata
    }

    /// Set a label
    pub fn set_label(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.labels.insert(key.into(), value.into());
    }

    /// Get a label
    pub fn get_label(&self, key: &str) -> Option<&str> {
        self.labels.get(key).map(|s| s.as_str())
    }

    /// Get all labels
    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }

    // === Error Context ===

    /// Set error information
    pub fn set_error(&mut self, error: ErenFlowError) {
        self.last_error = Some(Box::new(error));
    }

    /// Get last error
    pub fn last_error(&self) -> Option<&ErenFlowError> {
        self.last_error.as_ref().map(|e| e.as_ref())
    }

    /// Clear error
    pub fn clear_error(&mut self) {
        self.last_error = None;
    }

    // === Loop Context ===

    /// Set maximum iterations for loops
    pub fn set_max_iterations(&mut self, max: usize) {
        self.max_iterations = Some(max);
    }

    /// Get maximum iterations
    pub fn max_iterations(&self) -> Option<usize> {
        self.max_iterations
    }

    /// Check if iteration limit reached
    pub fn is_iteration_limit_reached(&self) -> bool {
        if let Some(max) = self.max_iterations {
            self.iteration >= max
        } else {
            false
        }
    }

    /// Set loop break condition expression
    pub fn set_loop_break_condition(&mut self, condition: impl Into<String>) {
        self.loop_break_condition = Some(condition.into());
    }

    /// Get loop break condition
    pub fn loop_break_condition(&self) -> Option<&str> {
        self.loop_break_condition.as_deref()
    }

    // === Utility Methods ===

    /// Create a summary of the execution context for logging
    pub fn summary(&self) -> String {
        format!(
            "ExecutionContext {{ node: '{}', depth: {}, iteration: {}, attempt: {}/{}, elapsed: {:?}, path: {:?} }}",
            self.node_name,
            self.depth,
            self.iteration,
            self.attempt,
            self.max_attempts,
            self.elapsed(),
            self.execution_path,
        )
    }
}

impl std::fmt::Debug for ExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("node_name", &self.node_name)
            .field(
                "state_keys",
                &self.state.keys().cloned().collect::<Vec<_>>(),
            )
            .field("execution_path", &self.execution_path)
            .field("depth", &self.depth)
            .field("iteration", &self.iteration)
            .field("branch_id", &self.branch_id)
            .field("attempt", &self.attempt)
            .field("max_attempts", &self.max_attempts)
            .field("elapsed", &self.elapsed())
            .field("should_stop", &self.should_stop)
            .field("should_skip", &self.should_skip)
            .field(
                "metadata_keys",
                &self.metadata.keys().cloned().collect::<Vec<_>>(),
            )
            .field("labels", &self.labels)
            .field("has_error", &self.last_error.is_some())
            .finish()
    }
}

// =============================================================================
// Builder Pattern for Convenience
// =============================================================================

impl ExecutionContext {
    /// Builder-style method to set LLM client
    pub fn with_llm(mut self, client: Arc<dyn LLMClient>) -> Self {
        self.llm_client = Some(client);
        self
    }

    /// Builder-style method to add MCP client
    pub fn with_mcp(mut self, name: impl Into<String>, client: Arc<dyn MCPClient>) -> Self {
        self.mcp_clients.insert(name.into(), client);
        self
    }

    /// Builder-style method to set metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Builder-style method to set label
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Builder-style method to set max attempts
    pub fn with_max_attempts(mut self, max: usize) -> Self {
        self.max_attempts = max;
        self
    }

    /// Builder-style method to set max iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = Some(max);
        self
    }

    /// Builder-style method to set loop break condition
    pub fn with_loop_break(mut self, condition: impl Into<String>) -> Self {
        self.loop_break_condition = Some(condition.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_context_creation() {
        let state = State::new();
        let ctx = ExecutionContext::new("my_node", state);
        assert_eq!(ctx.node_name(), "my_node");
        assert_eq!(ctx.depth(), 0);
        assert_eq!(ctx.iteration(), 0);
    }

    #[test]
    fn test_context_child() {
        let state = State::new();
        let ctx = ExecutionContext::new("parent", state);
        let child = ctx.child("child");

        assert_eq!(child.node_name(), "child");
        assert_eq!(child.depth(), 1);
        assert_eq!(child.execution_path(), &["parent"]);
    }

    #[test]
    fn test_metadata() {
        let state = State::new();
        let mut ctx = ExecutionContext::new("test", state);

        ctx.set_metadata("key1", json!("value1"));
        assert_eq!(ctx.get_metadata("key1"), Some(&json!("value1")));
    }

    #[test]
    fn test_labels() {
        let state = State::new();
        let mut ctx = ExecutionContext::new("test", state);

        ctx.set_label("env", "prod");
        assert_eq!(ctx.get_label("env"), Some("prod"));
    }

    #[test]
    fn test_builder_pattern() {
        let state = State::new();
        let ctx = ExecutionContext::new("test", state)
            .with_max_attempts(3)
            .with_max_iterations(5)
            .with_label("type", "critical");

        assert_eq!(ctx.max_attempts(), 3);
        assert_eq!(ctx.max_iterations(), Some(5));
        assert_eq!(ctx.get_label("type"), Some("critical"));
    }

    #[test]
    fn test_execution_path() {
        let state = State::new();
        let mut ctx = ExecutionContext::new("node1", state);

        ctx.push_path("node1".to_string());
        ctx.push_path("node2".to_string());

        assert_eq!(ctx.execution_path(), &["node1", "node2"]);
    }

    #[test]
    fn test_iteration_limit() {
        let state = State::new();
        let mut ctx = ExecutionContext::new("test", state);

        ctx.set_max_iterations(3);
        ctx.set_iteration(2);
        assert!(!ctx.is_iteration_limit_reached());

        ctx.set_iteration(3);
        assert!(ctx.is_iteration_limit_reached());
    }
}
