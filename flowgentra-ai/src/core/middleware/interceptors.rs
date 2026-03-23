//! # Middleware & Interceptor System
//!
//! Provides extensible middleware hooks for cross-cutting concerns like
//! logging, metrics, authentication, and state validation.
//!
//! ## Features
//!
//! - **Pre/Post Node Hooks** - Execute before and after each node
//! - **Error Handling** - Custom error processing
//! - **State Inspection** - View and validate state at any point
//! - **Composable** - Stack multiple middleware together
//! - **Type-Safe** - Compile-time checks where possible
//!
//! ## Example
//!
//! ```ignore
//! use flowgentra_ai::core::middleware::{Middleware, ExecutionContext};
//! use std::sync::Arc;
//!
//! struct LoggingMiddleware;
//!
//! #[async_trait::async_trait]
//! impl Middleware for LoggingMiddleware {
//!     async fn before_node(&self, ctx: &mut ExecutionContext) -> MiddlewareResult {
//!         println!("Entering node: {}", ctx.node_name);
//!         Ok(())
//!     }
//!
//!     async fn after_node(&self, ctx: &mut ExecutionContext) -> MiddlewareResult {
//!         println!("Exiting node: {}", ctx.node_name);
//!         Ok(())
//!     }
//! }
//! ```

use crate::core::error::FlowgentraError;
use crate::core::state::State;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

/// Context passed to middleware hooks
#[derive(Debug, Clone)]
pub struct ExecutionContext<T: State> {
    /// Name of the node being executed
    pub node_name: String,

    /// Current execution state
    pub state: T,

    /// Execution metadata
    pub metadata: HashMap<String, String>,

    /// Whether this is a retry attempt
    pub attempt: usize,

    /// Timestamp of execution start (ms since UNIX_EPOCH)
    pub start_time_ms: u128,
}

impl<T: State> ExecutionContext<T> {
    /// Create new execution context
    pub fn new(node_name: impl Into<String>, state: T) -> Self {
        Self {
            node_name: node_name.into(),
            state,
            metadata: HashMap::new(),
            attempt: 1,
            start_time_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
        }
    }

    /// Get elapsed time since execution started (ms)
    pub fn elapsed_ms(&self) -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            - self.start_time_ms
    }

    /// Add metadata to context
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }
}

/// Result of middleware execution
#[derive(Debug, Clone)]
pub enum MiddlewareResult<T: State> {
    /// Continue execution normally
    Continue,
    /// Skip this node and continue to next
    Skip,
    /// Abort execution and return error
    Abort(String),
    /// Modify state and continue
    ModifyState(T),
}

/// Core middleware trait
#[async_trait]
pub trait Middleware<T: State>: Send + Sync {
    /// Called before node execution starts
    ///
    /// Use for: logging, metrics setup, state validation
    async fn before_node(&self, ctx: &mut ExecutionContext<T>) -> MiddlewareResult<T> {
        let _ctx = ctx;
        MiddlewareResult::Continue
    }

    /// Called after node execution completes successfully
    ///
    /// Use for: metrics collection, state inspection, side effects
    async fn after_node(&self, ctx: &mut ExecutionContext<T>) -> MiddlewareResult<T> {
        let _ctx = ctx;
        MiddlewareResult::Continue
    }

    /// Called when node execution fails
    ///
    /// Use for: error logging, retry decisions, error transformation
    async fn on_error(
        &self,
        node_name: &str,
        error: &FlowgentraError,
        ctx: &ExecutionContext<T>,
    ) -> MiddlewareResult<T> {
        let _ = (node_name, error, ctx);
        MiddlewareResult::Continue
    }

    /// Called after the entire graph execution completes
    ///
    /// Use for: cleanup, final logging, report generation
    async fn on_complete(&self, final_state: &T) {
        let _final_state = final_state;
    }

    /// Name of this middleware for logging/debugging
    fn name(&self) -> &str {
        "unknown_middleware"
    }
}

/// Middleware pipeline that manages multiple middleware
pub struct MiddlewarePipeline<T: State> {
    middleware: Vec<Arc<dyn Middleware<T>>>,
}

impl<T: State> Default for MiddlewarePipeline<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: State> MiddlewarePipeline<T> {
    /// Create new empty pipeline
    pub fn new() -> Self {
        Self {
            middleware: Vec::new(),
        }
    }

    /// Add middleware to pipeline
    pub fn use_middleware(&mut self, mw: Arc<dyn Middleware<T>>) -> &mut Self {
        tracing::debug!("Registering middleware: {}", mw.name());
        self.middleware.push(mw);
        self
    }

    /// Execute before_node hooks for all middleware
    pub async fn execute_before_node(
        &self,
        ctx: &mut ExecutionContext<T>,
    ) -> Result<(), FlowgentraError> {
        for mw in &self.middleware {
            match mw.before_node(ctx).await {
                MiddlewareResult::Continue => {}
                MiddlewareResult::Skip => {
                    tracing::warn!(
                        "Middleware {} requested skip for node {}",
                        mw.name(),
                        ctx.node_name
                    );
                    // Note: Actual skip handled by runtime
                }
                MiddlewareResult::Abort(reason) => {
                    return Err(FlowgentraError::ExecutionError(format!(
                        "Middleware {} aborted execution: {}",
                        mw.name(),
                        reason
                    )));
                }
                MiddlewareResult::ModifyState(state) => {
                    ctx.state = state;
                }
            }
        }
        Ok(())
    }

    /// Execute after_node hooks for all middleware
    pub async fn execute_after_node(
        &self,
        ctx: &mut ExecutionContext<T>,
    ) -> Result<(), FlowgentraError> {
        for mw in &self.middleware {
            match mw.after_node(ctx).await {
                MiddlewareResult::Continue => {}
                MiddlewareResult::Skip => {}
                MiddlewareResult::Abort(reason) => {
                    return Err(FlowgentraError::ExecutionError(format!(
                        "Middleware {} aborted execution: {}",
                        mw.name(),
                        reason
                    )));
                }
                MiddlewareResult::ModifyState(state) => {
                    ctx.state = state;
                }
            }
        }
        Ok(())
    }

    /// Execute on_error hooks for all middleware
    pub async fn execute_on_error(
        &self,
        node_name: &str,
        error: &FlowgentraError,
        ctx: &ExecutionContext<T>,
    ) {
        for mw in &self.middleware {
            let _ = mw.on_error(node_name, error, ctx).await;
        }
    }

    /// Execute on_complete hooks for all middleware
    pub async fn execute_on_complete(&self, final_state: &T) {
        for mw in &self.middleware {
            mw.on_complete(final_state).await;
        }
    }

    /// Get number of middleware in pipeline
    pub fn len(&self) -> usize {
        self.middleware.len()
    }

    /// Check if pipeline is empty
    pub fn is_empty(&self) -> bool {
        self.middleware.is_empty()
    }
}

// =============================================================================
// Built-in Middleware
// =============================================================================

/// Logging middleware for execution tracing
pub struct LoggingMiddleware {
    verbose: bool,
}

impl LoggingMiddleware {
    /// Create new logging middleware
    pub fn new() -> Self {
        Self { verbose: false }
    }

    /// Enable verbose logging
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

impl Default for LoggingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<T: State> Middleware<T> for LoggingMiddleware {
    async fn before_node(&self, ctx: &mut ExecutionContext<T>) -> MiddlewareResult<T> {
        if self.verbose {
            tracing::debug!(
                node = %ctx.node_name,
                attempt = ctx.attempt,
                "Node execution started"
            );
        } else {
            tracing::trace!(node = %ctx.node_name, "Node execution started");
        }
        MiddlewareResult::Continue
    }

    async fn after_node(&self, ctx: &mut ExecutionContext<T>) -> MiddlewareResult<T> {
        let elapsed = ctx.elapsed_ms();
        if self.verbose {
            tracing::debug!(
                node = %ctx.node_name,
                elapsed_ms = elapsed,
                "Node execution completed"
            );
        } else {
            tracing::trace!(node = %ctx.node_name, elapsed_ms = elapsed, "Node completed");
        }
        MiddlewareResult::Continue
    }

    async fn on_error(
        &self,
        node_name: &str,
        error: &FlowgentraError,
        _ctx: &ExecutionContext<T>,
    ) -> MiddlewareResult<T> {
        tracing::error!(node = %node_name, error = %error, "Node execution failed");
        MiddlewareResult::Continue
    }

    fn name(&self) -> &str {
        "LoggingMiddleware"
    }
}

/// Metrics collection middleware
pub struct MetricsMiddleware {
    metrics: std::sync::Arc<tokio::sync::Mutex<ExecutionMetrics>>,
}

/// Collected metrics from execution
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    /// Total nodes executed
    pub nodes_executed: usize,
    /// Total errors encountered
    pub errors: usize,
    /// Per-node timing data
    pub node_timings: HashMap<String, Vec<u128>>,
}

impl ExecutionMetrics {
    /// Get average timing for a node
    pub fn avg_timing(&self, node: &str) -> Option<u128> {
        self.node_timings.get(node).and_then(|timings| {
            if timings.is_empty() {
                None
            } else {
                Some(timings.iter().sum::<u128>() / timings.len() as u128)
            }
        })
    }
}

impl MetricsMiddleware {
    /// Create new metrics middleware
    pub fn new() -> Self {
        Self {
            metrics: std::sync::Arc::new(tokio::sync::Mutex::new(ExecutionMetrics::default())),
        }
    }

    /// Get collected metrics
    pub async fn metrics(&self) -> ExecutionMetrics {
        self.metrics.lock().await.clone()
    }
}

impl Default for MetricsMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<T: State> Middleware<T> for MetricsMiddleware {
    async fn before_node(&self, ctx: &mut ExecutionContext<T>) -> MiddlewareResult<T> {
        let mut metrics = self.metrics.lock().await;
        metrics.nodes_executed += 1;
        let elapsed = ctx.elapsed_ms();
        metrics
            .node_timings
            .entry(ctx.node_name.clone())
            .or_insert_with(Vec::new)
            .push(elapsed);
        MiddlewareResult::Continue
    }

    async fn on_error(
        &self,
        _node_name: &str,
        _error: &FlowgentraError,
        _ctx: &ExecutionContext<T>,
    ) -> MiddlewareResult<T> {
        let mut metrics = self.metrics.lock().await;
        metrics.errors += 1;
        MiddlewareResult::Continue
    }

    fn name(&self) -> &str {
        "MetricsMiddleware"
    }
}

/// State validation middleware
pub struct ValidationMiddleware {
    required_fields: Vec<String>,
}

impl ValidationMiddleware {
    /// Create new validation middleware
    pub fn new() -> Self {
        Self {
            required_fields: Vec::new(),
        }
    }

    /// Require a field to exist in state
    pub fn require_field(mut self, field: impl Into<String>) -> Self {
        self.required_fields.push(field.into());
        self
    }

    /// Require multiple fields
    pub fn require_fields(mut self, fields: Vec<String>) -> Self {
        self.required_fields.extend(fields);
        self
    }
}

impl Default for ValidationMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<T: State> Middleware<T> for ValidationMiddleware {
    async fn before_node(&self, ctx: &mut ExecutionContext<T>) -> MiddlewareResult<T> {
        for field in &self.required_fields {
            if !ctx.state.contains_key(field) {
                return MiddlewareResult::Abort(format!("Required field missing: {}", field));
            }
        }
        MiddlewareResult::Continue
    }

    fn name(&self) -> &str {
        "ValidationMiddleware"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::SharedState;

    #[test]
    fn pipeline_creation() {
        let pipeline: MiddlewarePipeline<SharedState> = MiddlewarePipeline::new();
        assert!(pipeline.is_empty());
    }

    #[test]
    fn add_middleware() {
        let mut pipeline: MiddlewarePipeline<SharedState> = MiddlewarePipeline::new();
        pipeline.use_middleware(Arc::new(LoggingMiddleware::new()));
        assert_eq!(pipeline.len(), 1);
    }

    #[tokio::test]
    async fn execution_context_creation() {
        let state = SharedState::new(Default::default());
        let ctx = ExecutionContext::new("test_node", state);
        assert_eq!(ctx.node_name, "test_node");
        assert_eq!(ctx.attempt, 1);
    }

    #[tokio::test]
    async fn logging_middleware() {
        let mw = LoggingMiddleware::new();
        let mut ctx = ExecutionContext::new("test", SharedState::new(Default::default()));
        let result = mw.before_node(&mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Continue));
    }

    #[tokio::test]
    async fn metrics_collection() {
        let mw = MetricsMiddleware::new();
        let mut ctx = ExecutionContext::new("node1", SharedState::new(Default::default()));
        let _ = mw.before_node(&mut ctx).await;

        let metrics = mw.metrics().await;
        assert_eq!(metrics.nodes_executed, 1);
    }

    #[tokio::test]
    async fn validation_middleware() {
        let mw = ValidationMiddleware::new().require_field("required_field");
        let state = SharedState::new(Default::default());
        state.set("required_field", serde_json::json!("value"));
        let mut ctx = ExecutionContext::new("test", state);

        let result = mw.before_node(&mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Continue));
    }

    #[tokio::test]
    async fn validation_middleware_fails_missing_field() {
        let mw = ValidationMiddleware::new().require_field("required_field");
        let ctx = ExecutionContext::new("test", SharedState::new(Default::default()));
        let mut ctx = ctx;

        let result = mw.before_node(&mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Abort(_)));
    }
}
