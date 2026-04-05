//! # Observability Middleware
//!
//! Collects execution traces, node timings, token usage, and failure snapshots.

use crate::core::error::FlowgentraError;
use crate::core::middleware::{ExecutionContext, Middleware, MiddlewareResult};
use crate::core::state::DynState;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::trace::{ExecutionTrace, TraceStatus};
use super::types::{FailureSnapshot, TokenUsage};

/// Key in state where handlers can store token usage: `_token_usage`
pub const TOKEN_USAGE_STATE_KEY: &str = "_token_usage";

/// Observability middleware - execution tracing, node timing, token usage, failure snapshots
pub struct ObservabilityMiddleware {
    trace: Arc<RwLock<ExecutionTrace>>,
    agent_name: Option<String>,
    capture_failure_snapshots: bool,
    max_snapshot_size: usize,
}

impl ObservabilityMiddleware {
    pub fn new() -> Self {
        Self {
            trace: Arc::new(RwLock::new(ExecutionTrace::new(None))),
            agent_name: None,
            capture_failure_snapshots: true,
            max_snapshot_size: 100_000,
        }
    }

    pub fn with_agent_name(mut self, name: impl Into<String>) -> Self {
        self.agent_name = Some(name.into());
        self
    }

    pub fn with_failure_snapshots(mut self, enable: bool) -> Self {
        self.capture_failure_snapshots = enable;
        self
    }

    pub fn max_snapshot_size(mut self, bytes: usize) -> Self {
        self.max_snapshot_size = bytes;
        self
    }

    /// Get a clone of the current trace (for export after execution)
    pub async fn trace(&self) -> ExecutionTrace {
        self.trace.read().await.clone()
    }

    fn extract_token_usage(state: &DynState) -> Option<TokenUsage> {
        state
            .get(TOKEN_USAGE_STATE_KEY)
            .and_then(|v: serde_json::Value| {
                let prompt = v.get("prompt_tokens")?.as_u64()?;
                let completion = v.get("completion_tokens")?.as_u64()?;
                Some(TokenUsage::new(prompt, completion))
            })
    }

    fn state_to_snapshot(state: &DynState, max_size: usize) -> Option<String> {
        let json = state.to_json_string().ok()?;
        if json.len() <= max_size {
            Some(json)
        } else {
            Some(format!(
                "{}... (truncated, {} bytes)",
                &json[..max_size.min(json.len())],
                json.len()
            ))
        }
    }
}

impl Default for ObservabilityMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware<DynState> for ObservabilityMiddleware {
    async fn before_node(&self, _ctx: &mut ExecutionContext<DynState>) -> MiddlewareResult<DynState> {
        if let Some(ref name) = &self.agent_name {
            let mut trace = self.trace.write().await;
            if trace.agent_name.is_none() {
                trace.agent_name = Some(name.clone());
            }
        }
        MiddlewareResult::Continue
    }

    async fn after_node(&self, ctx: &mut ExecutionContext<DynState>) -> MiddlewareResult<DynState> {
        let elapsed = ctx.elapsed_ms();
        let start_time = chrono::Utc::now() - chrono::Duration::milliseconds(elapsed as i64);

        let mut trace = self.trace.write().await;
        trace.record_node(&ctx.node_name, elapsed as u64, start_time);

        // Aggregate token usage if handler set it
        if let Some(usage) = Self::extract_token_usage(&ctx.state) {
            trace.add_token_usage(&usage);
        }

        MiddlewareResult::Continue
    }

    async fn on_error(
        &self,
        node_name: &str,
        error: &FlowgentraError,
        ctx: &ExecutionContext<DynState>,
    ) -> MiddlewareResult<DynState> {
        let mut trace = self.trace.write().await;
        trace.mark_failed(node_name, error.to_string());
        trace.complete();

        if self.capture_failure_snapshots {
            let snapshot = FailureSnapshot {
                trace_id: trace.trace_id.clone(),
                failed_node: node_name.to_string(),
                error_message: error.to_string(),
                state_snapshot: Self::state_to_snapshot(&ctx.state, self.max_snapshot_size),
                execution_path: trace.execution_path(),
                node_timings: trace
                    .node_timings
                    .iter()
                    .map(|t| (t.node_name.clone(), t.duration_ms))
                    .collect::<HashMap<_, _>>(),
                token_usage: Some(trace.token_usage.clone()),
                timestamp: chrono::Utc::now(),
            };
            trace.metadata.insert(
                "failure_snapshot".to_string(),
                serde_json::to_value(snapshot).unwrap_or_default(),
            );
        }

        MiddlewareResult::Continue
    }

    async fn on_complete(&self, final_state: &DynState) {
        let mut trace = self.trace.write().await;
        if matches!(trace.status, TraceStatus::Completed) {
            trace.complete();
        }
        if let Some(usage) = Self::extract_token_usage(final_state) {
            trace.add_token_usage(&usage);
        }
    }

    fn name(&self) -> &str {
        "ObservabilityMiddleware"
    }
}
