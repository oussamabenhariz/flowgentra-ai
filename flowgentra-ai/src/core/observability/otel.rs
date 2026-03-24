//! OpenTelemetry-compatible trace export.
//!
//! Converts internal `ExecutionTrace` to OTLP-compatible JSON spans
//! for export to Jaeger, Datadog, Honeycomb, or any OTLP collector.

use super::trace::ExecutionTrace;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An OTLP-compatible span representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelSpan {
    #[serde(rename = "traceId")]
    pub trace_id: String,
    #[serde(rename = "spanId")]
    pub span_id: String,
    #[serde(rename = "parentSpanId", skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    #[serde(rename = "operationName")]
    pub operation_name: String,
    #[serde(rename = "startTimeUnixNano")]
    pub start_time_unix_nano: u64,
    #[serde(rename = "endTimeUnixNano")]
    pub end_time_unix_nano: u64,
    pub attributes: Vec<OtelAttribute>,
    pub status: OtelStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelAttribute {
    pub key: String,
    pub value: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelStatus {
    pub code: u32,  // 0=Unset, 1=Ok, 2=Error
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// Convert an `ExecutionTrace` to a list of OTLP-compatible spans.
///
/// Creates a root span for the entire graph execution, with child spans
/// for each node execution.
pub fn trace_to_otel_spans(trace: &ExecutionTrace) -> Vec<OtelSpan> {
    let trace_id = trace.trace_id.clone();
    let root_span_id = format!("{:016x}", rand_id());

    let mut spans = Vec::new();

    // Root span
    let first_start = trace.node_timings.first()
        .map(|t| t.start_time.timestamp_nanos_opt().unwrap_or(0) as u64)
        .unwrap_or(0);
    let total_duration_ns: u64 = trace.node_timings.iter()
        .map(|t| t.duration_ms * 1_000_000)
        .sum();

    spans.push(OtelSpan {
        trace_id: trace_id.clone(),
        span_id: root_span_id.clone(),
        parent_span_id: None,
        operation_name: "graph_execution".to_string(),
        start_time_unix_nano: first_start,
        end_time_unix_nano: first_start + total_duration_ns,
        attributes: vec![
            OtelAttribute {
                key: "agent.name".to_string(),
                value: Value::String(trace.agent_name.clone().unwrap_or_default()),
            },
            OtelAttribute {
                key: "graph.total_nodes".to_string(),
                value: serde_json::json!(trace.node_timings.len()),
            },
        ],
        status: match &trace.status {
            super::trace::TraceStatus::Completed => OtelStatus { code: 1, message: None },
            super::trace::TraceStatus::Failed { error, .. } => OtelStatus {
                code: 2,
                message: Some(error.clone()),
            },
        },
    });

    // Child spans for each node
    for timing in &trace.node_timings {
        let span_id = format!("{:016x}", rand_id());
        let start_ns = timing.start_time.timestamp_nanos_opt().unwrap_or(0) as u64;
        let duration_ns = timing.duration_ms * 1_000_000;

        spans.push(OtelSpan {
            trace_id: trace_id.clone(),
            span_id,
            parent_span_id: Some(root_span_id.clone()),
            operation_name: format!("node.{}", timing.node_name),
            start_time_unix_nano: start_ns,
            end_time_unix_nano: start_ns + duration_ns,
            attributes: vec![
                OtelAttribute {
                    key: "node.name".to_string(),
                    value: Value::String(timing.node_name.clone()),
                },
                OtelAttribute {
                    key: "node.duration_ms".to_string(),
                    value: serde_json::json!(timing.duration_ms),
                },
            ],
            status: OtelStatus { code: 1, message: None },
        });
    }

    spans
}

/// Export spans as OTLP JSON (for sending to a collector).
pub fn spans_to_otlp_json(spans: &[OtelSpan]) -> Value {
    serde_json::json!({
        "resourceSpans": [{
            "resource": {
                "attributes": [{
                    "key": "service.name",
                    "value": { "stringValue": "flowgentra-ai" }
                }]
            },
            "scopeSpans": [{
                "scope": {
                    "name": "flowgentra-ai",
                    "version": env!("CARGO_PKG_VERSION"),
                },
                "spans": spans,
            }]
        }]
    })
}

/// Send spans to an OTLP HTTP collector endpoint.
pub async fn export_to_otlp(
    endpoint: &str,
    spans: &[OtelSpan],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let payload = spans_to_otlp_json(spans);
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/v1/traces", endpoint))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await?;

    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("OTLP export failed: {}", body).into());
    }
    Ok(())
}

fn rand_id() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    hasher.finish()
}
