//! # Prometheus Metrics Exporter
//!
//! Provides a Prometheus-compatible `/metrics` HTTP endpoint that exports
//! per-node execution counters, latency histograms, LLM token usage, and
//! error rates.
//!
//! ## Quick Start
//!
//! ```ignore
//! use flowgentra_ai::core::observability::prometheus::PrometheusExporter;
//!
//! // Start the /metrics endpoint on port 9090
//! let exporter = PrometheusExporter::new("0.0.0.0:9090");
//! exporter.install().expect("failed to install Prometheus exporter");
//!
//! // From now on, all graph executions automatically emit metrics.
//! // Scrape http://localhost:9090/metrics with Prometheus or Grafana.
//! ```
//!
//! ## Metrics emitted
//!
//! | Metric | Type | Labels | Description |
//! |--------|------|--------|-------------|
//! | `flowgentra_node_executions_total` | counter | `node`, `status` | Nodes executed (status: success/failure) |
//! | `flowgentra_node_duration_seconds` | histogram | `node` | Per-node wall-clock latency |
//! | `flowgentra_graph_executions_total` | counter | `status` | Graph-level completions/failures |
//! | `flowgentra_graph_duration_seconds` | histogram | — | Full graph wall-clock latency |
//! | `flowgentra_llm_tokens_total` | counter | `type` | Prompt/completion tokens (type: prompt/completion) |
//! | `flowgentra_llm_streaming_chunks_total` | counter | `node` | LLM streaming chunks emitted per node |

use crate::core::observability::events::{EventBroadcaster, ExecutionEvent};
use metrics::{counter, histogram};
use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;
use std::sync::Arc;

// ─── Metric names ──────────────────────────────────────────────────────────

pub const NODE_EXECUTIONS_TOTAL: &str = "flowgentra_node_executions_total";
pub const NODE_DURATION_SECONDS: &str = "flowgentra_node_duration_seconds";
pub const GRAPH_EXECUTIONS_TOTAL: &str = "flowgentra_graph_executions_total";
pub const GRAPH_DURATION_SECONDS: &str = "flowgentra_graph_duration_seconds";
pub const LLM_TOKENS_TOTAL: &str = "flowgentra_llm_tokens_total";
pub const LLM_STREAMING_CHUNKS_TOTAL: &str = "flowgentra_llm_streaming_chunks_total";
pub const TOOL_CALLS_TOTAL: &str = "flowgentra_tool_calls_total";

// ─── PrometheusExporter ────────────────────────────────────────────────────

/// Installs a Prometheus metrics exporter listening on the given address.
///
/// After calling `install()`, the `metrics` crate macros (`counter!`, `histogram!`)
/// are backed by the Prometheus registry. Prometheus can then scrape the HTTP
/// endpoint to collect all recorded metrics.
pub struct PrometheusExporter {
    addr: SocketAddr,
}

impl PrometheusExporter {
    /// Create a new exporter bound to the given address (e.g. `"0.0.0.0:9090"`).
    pub fn new(addr: impl std::fmt::Display) -> Self {
        let addr: SocketAddr = addr
            .to_string()
            .parse()
            .unwrap_or_else(|_| "0.0.0.0:9090".parse().unwrap());
        Self { addr }
    }

    /// Install the Prometheus recorder and start the HTTP listener.
    ///
    /// This is a one-time call — calling it twice will return an error.
    pub fn install(self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        PrometheusBuilder::new()
            .with_http_listener(self.addr)
            .install()?;
        tracing::info!(addr = %self.addr, "Prometheus /metrics endpoint started");
        Ok(())
    }
}

// ─── MetricsCollector ──────────────────────────────────────────────────────

/// Subscribes to an `EventBroadcaster` and records metrics for every event.
///
/// Spawn this in a background task alongside your graph execution:
///
/// ```ignore
/// let collector = MetricsCollector::new(graph.event_broadcaster());
/// tokio::spawn(collector.run());
/// let result = graph.invoke(state).await?;
/// ```
pub struct MetricsCollector {
    broadcaster: Arc<EventBroadcaster>,
}

impl MetricsCollector {
    pub fn new(broadcaster: &Arc<EventBroadcaster>) -> Self {
        Self {
            broadcaster: Arc::clone(broadcaster),
        }
    }

    /// Subscribe to events and record metrics until the graph completes.
    pub async fn run(self) {
        let mut rx = self.broadcaster.subscribe();
        // Track graph start time for overall duration
        let mut graph_start: Option<std::time::Instant> = None;

        while let Ok(event) = rx.recv().await {
            self.record_event(&event, &mut graph_start);
            match event {
                ExecutionEvent::GraphCompleted { .. } | ExecutionEvent::GraphFailed { .. } => break,
                _ => {}
            }
        }
    }

    fn record_event(&self, event: &ExecutionEvent, graph_start: &mut Option<std::time::Instant>) {
        match event {
            ExecutionEvent::GraphStarted { .. } => {
                *graph_start = Some(std::time::Instant::now());
            }
            ExecutionEvent::NodeCompleted {
                node_name,
                duration_ms,
                ..
            } => {
                counter!(NODE_EXECUTIONS_TOTAL, "node" => node_name.clone(), "status" => "success")
                    .increment(1);
                histogram!(NODE_DURATION_SECONDS, "node" => node_name.clone())
                    .record(*duration_ms as f64 / 1_000.0);
            }
            ExecutionEvent::NodeFailed { node_name, .. } => {
                counter!(NODE_EXECUTIONS_TOTAL, "node" => node_name.clone(), "status" => "failure")
                    .increment(1);
            }
            ExecutionEvent::GraphCompleted {
                total_duration_ms, ..
            } => {
                counter!(GRAPH_EXECUTIONS_TOTAL, "status" => "success").increment(1);
                histogram!(GRAPH_DURATION_SECONDS).record(*total_duration_ms as f64 / 1_000.0);
            }
            ExecutionEvent::GraphFailed { .. } => {
                counter!(GRAPH_EXECUTIONS_TOTAL, "status" => "failure").increment(1);
                if let Some(start) = graph_start.take() {
                    histogram!(GRAPH_DURATION_SECONDS).record(start.elapsed().as_secs_f64());
                }
            }
            ExecutionEvent::LLMStreaming { node_name, .. } => {
                counter!(LLM_STREAMING_CHUNKS_TOTAL, "node" => node_name.clone()).increment(1);
            }
            ExecutionEvent::ToolCalled { tool_name, .. } => {
                counter!(TOOL_CALLS_TOTAL, "tool" => tool_name.clone(), "status" => "called")
                    .increment(1);
            }
            ExecutionEvent::ToolResult {
                tool_name, success, ..
            } => {
                let status = if *success { "success" } else { "failure" };
                counter!(TOOL_CALLS_TOTAL, "tool" => tool_name.clone(), "status" => status)
                    .increment(1);
            }
            _ => {}
        }
    }
}

// ─── Convenience: record LLM token usage ──────────────────────────────────

/// Record prompt and completion token counts from an LLM response.
///
/// Call this from your handler after each LLM call to feed token metrics.
///
/// ```ignore
/// let (msg, usage) = llm.chat_with_usage(messages).await?;
/// if let Some(u) = usage {
///     flowgentra_ai::core::observability::prometheus::record_llm_tokens(u.prompt_tokens, u.completion_tokens);
/// }
/// ```
pub fn record_llm_tokens(prompt_tokens: u64, completion_tokens: u64) {
    counter!(LLM_TOKENS_TOTAL, "type" => "prompt").increment(prompt_tokens);
    counter!(LLM_TOKENS_TOTAL, "type" => "completion").increment(completion_tokens);
}
