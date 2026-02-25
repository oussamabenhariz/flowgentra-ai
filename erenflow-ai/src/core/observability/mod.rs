//! # Observability Module
//!
//! Production-grade observability for agent execution:
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
//! use erenflow_ai::core::observability::{ObservabilityMiddleware, record_token_usage};
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

mod middleware;
mod replay;
mod trace;
mod types;

pub use middleware::{ObservabilityMiddleware, TOKEN_USAGE_STATE_KEY};
pub use replay::ReplayMode;
pub use trace::ExecutionTrace;
pub use types::{FailureSnapshot, NodeTiming, PathSegment};

use crate::core::llm::TokenUsage;
use crate::core::state::State;
use serde_json::json;

/// Record token usage in state for ObservabilityMiddleware to aggregate.
/// Call this from handlers after LLM calls when using `chat_with_usage()`.
pub fn record_token_usage(state: &mut State, usage: &TokenUsage) {
    state.set(
        TOKEN_USAGE_STATE_KEY,
        json!({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }),
    );
}

#[cfg(feature = "observability-ui")]
mod ui;
#[cfg(feature = "observability-ui")]
pub use ui::TracingUIServer;
