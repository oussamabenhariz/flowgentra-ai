//! # Observability Module
//!
//!
//! - **Execution tracing** - Hierarchical spans, trace IDs
//! - **Node timing metrics** - Per-node latency
//! - **Token usage tracking** - LLM cost visibility
//! - **DAG visualization** - Graph with execution overlay
//! - **Replay mode** - Load traces and step through
//! - **Failure snapshots** - State + context on error
//! - **Tracing UI** - Built-in web UI (optional feature)
//!
//! ## Quick Start
//!
//! ```ignore
//! use std::sync::Arc;
//!
//! let mut runtime = AgentRuntime::from_config(config)?;
//! runtime.add_middleware(Arc::new(
//!     ObservabilityMiddleware::new().with_agent_name("my_agent")
//! ));
//!
//! let result = runtime.execute(state).await?;
//!
//! // Get trace for export/analysis
//! let mw = runtime.middleware_pipeline()...; // get ObservabilityMiddleware
//! let trace = mw.trace().await;
//! ```
//!
//! ## Token Usage
//!
//! Handlers using LLM should call `chat_with_usage()` and record usage:
//!
//! ```ignore
//! let (message, usage) = llm_client.chat_with_usage(messages).await?;
//! if let Some(u) = usage {
//!     record_token_usage(&mut state, &u);
//! }
//! ```

pub mod events;
pub mod graph_visualizer;
mod middleware;
pub mod otel;
pub mod prometheus;
mod replay;
mod trace;
mod types;
pub mod ui;
pub mod visualization;

pub use events::{EventBroadcaster, ExecutionEvent};
pub use prometheus::{MetricsCollector, PrometheusExporter, record_llm_tokens};
pub use graph_visualizer::{
    ExecutionStatistics, NodeExecutionStatus, NodeStatistics, NodeType, StateGraphEdge,
    StateGraphNode, StateGraphVisualization, StateGraphVisualizer,
};
pub use middleware::{ObservabilityMiddleware, TOKEN_USAGE_STATE_KEY};
pub use replay::ReplayMode;
pub use trace::ExecutionTrace;
pub use types::{FailureSnapshot, NodeTiming, PathSegment};
pub use visualization::{ExecutionTracer, GraphVisualizer, LayoutAlgorithm};

use crate::core::llm::TokenUsage;
use crate::core::state::DynState;
use serde_json::json;

/// Record token usage in state for ObservabilityMiddleware to aggregate.
/// Call this from handlers after LLM calls when using `chat_with_usage()`.
///
/// Token counts are **accumulated** across multiple calls, not overwritten.
/// This means graphs with multiple LLM-calling nodes will report the total
/// token usage across all calls.
pub fn record_token_usage(state: &mut DynState, usage: &TokenUsage) {
    // Read existing usage (if any) and accumulate
    let existing: TokenUsage = state
        .get(TOKEN_USAGE_STATE_KEY)
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default();
    let accumulated = existing.add(usage);

    state.set(
        TOKEN_USAGE_STATE_KEY,
        json!({
            "prompt_tokens": accumulated.prompt_tokens,
            "completion_tokens": accumulated.completion_tokens,
            "total_tokens": accumulated.total_tokens,
        }),
    );
}
