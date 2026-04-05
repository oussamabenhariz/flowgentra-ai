//! # Middleware & Interceptor System
//!
//! Provides extensible middleware hooks for cross-cutting concerns like
//! logging, metrics, authentication, and state validation.

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

    pub fn elapsed_ms(&self) -> u128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            - self.start_time_ms
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }
}

/// Result of middleware execution
#[derive(Debug, Clone)]
pub enum MiddlewareResult<T: State> {
    Continue,
    Skip,
    Abort(String),
    ModifyState(T),
}

/// Core middleware trait
#[async_trait]
pub trait Middleware<T: State>: Send + Sync {
    async fn before_node(&self, ctx: &mut ExecutionContext<T>) -> MiddlewareResult<T> {
        let _ctx = ctx;
        MiddlewareResult::Continue
    }

    async fn after_node(&self, ctx: &mut ExecutionContext<T>) -> MiddlewareResult<T> {
        let _ctx = ctx;
        MiddlewareResult::Continue
    }

    async fn on_error(
        &self,
        node_name: &str,
        error: &FlowgentraError,
        ctx: &ExecutionContext<T>,
    ) -> MiddlewareResult<T> {
        let _ = (node_name, error, ctx);
        MiddlewareResult::Continue
    }

    async fn on_complete(&self, final_state: &T) {
        let _final_state = final_state;
    }

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
    pub fn new() -> Self {
        Self {
            middleware: Vec::new(),
        }
    }

    pub fn use_middleware(&mut self, mw: Arc<dyn Middleware<T>>) -> &mut Self {
        tracing::debug!("Registering middleware: {}", mw.name());
        self.middleware.push(mw);
        self
    }

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

    pub async fn execute_on_complete(&self, final_state: &T) {
        for mw in &self.middleware {
            mw.on_complete(final_state).await;
        }
    }

    pub fn len(&self) -> usize {
        self.middleware.len()
    }

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
    pub fn new() -> Self {
        Self { verbose: false }
    }

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
            tracing::debug!(node = %ctx.node_name, attempt = ctx.attempt, "Node execution started");
        } else {
            tracing::trace!(node = %ctx.node_name, "Node execution started");
        }
        MiddlewareResult::Continue
    }

    async fn after_node(&self, ctx: &mut ExecutionContext<T>) -> MiddlewareResult<T> {
        let elapsed = ctx.elapsed_ms();
        if self.verbose {
            tracing::debug!(node = %ctx.node_name, elapsed_ms = elapsed, "Node execution completed");
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

#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    pub nodes_executed: usize,
    pub errors: usize,
    pub node_timings: HashMap<String, Vec<u128>>,
}

impl ExecutionMetrics {
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
    pub fn new() -> Self {
        Self {
            metrics: std::sync::Arc::new(tokio::sync::Mutex::new(ExecutionMetrics::default())),
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state_graph::message_graph::MessageState;

    #[test]
    fn pipeline_creation() {
        let pipeline: MiddlewarePipeline<MessageState> = MiddlewarePipeline::new();
        assert!(pipeline.is_empty());
    }

    #[test]
    fn add_middleware() {
        let mut pipeline: MiddlewarePipeline<MessageState> = MiddlewarePipeline::new();
        pipeline.use_middleware(Arc::new(LoggingMiddleware::new()));
        assert_eq!(pipeline.len(), 1);
    }

    #[tokio::test]
    async fn execution_context_creation() {
        let state = MessageState::empty();
        let ctx = ExecutionContext::new("test_node", state);
        assert_eq!(ctx.node_name, "test_node");
        assert_eq!(ctx.attempt, 1);
    }

    #[tokio::test]
    async fn logging_middleware() {
        let mw = LoggingMiddleware::new();
        let mut ctx = ExecutionContext::new("test", MessageState::empty());
        let result = mw.before_node(&mut ctx).await;
        assert!(matches!(result, MiddlewareResult::Continue));
    }

    #[tokio::test]
    async fn metrics_collection() {
        let mw = MetricsMiddleware::new();
        let mut ctx = ExecutionContext::new("node1", MessageState::empty());
        let _ = mw.before_node(&mut ctx).await;

        let metrics = mw.metrics().await;
        assert_eq!(metrics.nodes_executed, 1);
    }
}
