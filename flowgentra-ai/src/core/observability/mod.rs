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
//! let (message, usage) = llm.chat_with_usage(messages).await?;
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
pub use graph_visualizer::{
    ExecutionStatistics, NodeExecutionStatus, NodeStatistics, NodeType, StateGraphEdge,
    StateGraphNode, StateGraphVisualization, StateGraphVisualizer,
};
pub use middleware::{ObservabilityMiddleware, TOKEN_USAGE_STATE_KEY};
pub use prometheus::{record_llm_tokens, MetricsCollector, PrometheusExporter};
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

/// State field holding cumulative estimated cost in USD across a run.
pub const COST_USAGE_STATE_KEY: &str = "_cost_usd";

/// Record token usage **and** its estimated USD cost for the given model.
///
/// Accumulates tokens exactly like [`record_token_usage`], and additionally
/// adds the per-call cost (from the model's price, including any override set
/// via `llm::set_model_price`) into the `_cost_usd` field that the cost budget
/// checks. A run that mixes models sums each call at its own price.
///
/// When the model is not priced, the cost contribution is `0.0` and a warning
/// is logged once per model — the budget under-counts rather than blocking the
/// run (audit F-10 decision).
pub fn record_usage_with_cost(state: &mut DynState, usage: &TokenUsage, model: &str) {
    record_token_usage(state, usage);

    let call_cost = match usage.estimated_cost(model) {
        Some(c) => c,
        None => {
            warn_unpriced_model(model);
            0.0
        }
    };

    let existing = state
        .get(COST_USAGE_STATE_KEY)
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    state.set(COST_USAGE_STATE_KEY, json!(existing + call_cost));
}

/// Models already warned about, so the "no price" log fires once each.
static WARNED_MODELS: once_cell::sync::Lazy<std::sync::Mutex<std::collections::HashSet<String>>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(std::collections::HashSet::new()));

fn warn_unpriced_model(model: &str) {
    if let Ok(mut seen) = WARNED_MODELS.lock() {
        if seen.insert(model.to_string()) {
            tracing::warn!(
                model,
                "no price found for model; its calls count as $0 toward the cost budget. \
                 Add one with llm::set_model_price(\"{model}\", input, output).",
            );
        }
    }
}

#[cfg(test)]
mod cost_tests {
    use super::*;

    fn cost_of(state: &DynState) -> f64 {
        state
            .get(COST_USAGE_STATE_KEY)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
    }

    #[test]
    fn cost_accumulates_across_models() {
        let mut state = DynState::new();

        // 1M input + 1M output on gpt-4o = $2.50 + $10.00 = $12.50.
        record_usage_with_cost(&mut state, &TokenUsage::new(1_000_000, 1_000_000), "gpt-4o");
        assert!(
            (cost_of(&state) - 12.50).abs() < 1e-9,
            "{}",
            cost_of(&state)
        );

        // gpt-4o-mini: $0.15 + $0.60 = $0.75, summed on top.
        record_usage_with_cost(
            &mut state,
            &TokenUsage::new(1_000_000, 1_000_000),
            "gpt-4o-mini",
        );
        assert!(
            (cost_of(&state) - 13.25).abs() < 1e-9,
            "{}",
            cost_of(&state)
        );

        // Tokens accumulate too.
        let tokens = state
            .get(TOKEN_USAGE_STATE_KEY)
            .and_then(|v| v.get("total_tokens").and_then(|t| t.as_u64()))
            .unwrap_or(0);
        assert_eq!(tokens, 4_000_000);
    }

    #[test]
    fn unpriced_model_counts_as_zero() {
        let mut state = DynState::new();
        record_usage_with_cost(
            &mut state,
            &TokenUsage::new(1000, 1000),
            "no-such-model-abc",
        );
        assert_eq!(cost_of(&state), 0.0);
    }
}
