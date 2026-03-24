# Observability Guide

Monitor your agent in real time, trace execution, and export to external monitoring systems.

## Event Broadcaster

Stream real-time execution events using a tokio broadcast channel:

```rust
use flowgentra_ai::core::observability::{EventBroadcaster, ExecutionEvent};

let broadcaster = EventBroadcaster::new(100); // buffer capacity

// Subscribe to events
let mut rx = broadcaster.subscribe();

// In a separate task, listen for events
tokio::spawn(async move {
    while let Ok(event) = rx.recv().await {
        match event {
            ExecutionEvent::GraphStarted { graph_id } => {
                println!("Graph {} started", graph_id);
            }
            ExecutionEvent::NodeStarted { name, .. } => {
                println!("  Node '{}' started", name);
            }
            ExecutionEvent::NodeCompleted { name, duration, .. } => {
                println!("  Node '{}' completed in {:?}", name, duration);
            }
            ExecutionEvent::NodeFailed { name, error, .. } => {
                eprintln!("  Node '{}' FAILED: {}", name, error);
            }
            ExecutionEvent::EdgeTraversed { from, to, .. } => {
                println!("  Edge: {} -> {}", from, to);
            }
            _ => {}
        }
    }
});
```

### Event Types

| Event | When It Fires |
|-------|---------------|
| `GraphStarted` | Graph execution begins |
| `NodeStarted` | A node begins executing |
| `NodeCompleted` | A node finishes successfully |
| `NodeFailed` | A node throws an error |
| `EdgeTraversed` | Execution moves from one node to another |

---

## Execution Tracing

Record the full execution path with timing and state snapshots:

```rust
use flowgentra_ai::core::observability::ExecutionTrace;

let trace = ExecutionTrace::new("my-graph");

// Record nodes as they execute
trace.record_node("validate", duration);

// Record with state snapshot
trace.record_node_with_state("process", duration, &state);
```

### Execution Replay

Step through a past execution:

```rust
use flowgentra_ai::core::observability::ReplayMode;

let replay = ReplayMode::new(trace);

// Get state at any step
let state_at_step_3 = replay.state_at(3);

// Get current (latest) state
let current = replay.current_state();

// See what changed between steps
let diff = replay.diff_states(2, 3);
```

---

## OpenTelemetry Export

Export execution traces in OTLP format for Jaeger, Datadog, Honeycomb, Grafana Tempo, or any OpenTelemetry-compatible backend:

```rust
use flowgentra_ai::core::observability::otel::{
    trace_to_otel_spans,
    spans_to_otlp_json,
    export_to_otlp,
};

// Convert execution trace to OTLP spans
let spans = trace_to_otel_spans(&execution_trace);

// Serialize to OTLP JSON
let json = spans_to_otlp_json(&spans);

// Export to an OTLP collector
export_to_otlp("http://localhost:4318/v1/traces", &spans).await?;
```

### Config-Based Setup

```yaml
observability:
  tracing_enabled: true

  exporters:
    - type: otlp_http
      endpoint: "http://localhost:4318/v1/traces"

    - type: jaeger
      endpoint: "http://localhost:14268/api/traces"

    - type: datadog
      api_key: ${DATADOG_API_KEY}
      service: "my-agent"
```

---

## Logging

### Configuration

```yaml
middleware:
  - name: logging
    enabled: true
    level: info           # debug, info, warn, error
    include_state: false  # Set true to log full state (verbose)
```

### Log Levels

| Level | What Gets Logged |
|-------|-----------------|
| `debug` | Everything: state, payloads, internal decisions |
| `info` | Progress: node start/end, graph lifecycle |
| `warn` | Potential issues: slow nodes, retries |
| `error` | Failures only |

Use `debug` during development, `info` or `warn` in production.

---

## Health Monitoring

```yaml
health:
  enabled: true
  check_interval: 30         # Check every 30 seconds

  checks:
    - llm_connectivity       # Can reach the LLM provider
    - rag_availability       # Vector store is accessible
    - mcp_tools              # External tools respond
    - memory_usage           # Not running out of memory
    - response_time          # Response time is acceptable

  memory_threshold: 80       # Alert if memory > 80%
  response_time_threshold: 5000  # Alert if response > 5s
```

---

## Performance Monitoring

### Caching

```yaml
middleware:
  - name: cache
    enabled: true
    ttl: 3600              # Cache responses for 1 hour
    strategy: content_hash # Cache key based on input content
```

### Rate Limiting

```yaml
middleware:
  - name: rate_limiting
    enabled: true
    rpm: 60                # 60 requests per minute
    burst_size: 10         # Allow 10 rapid requests
```

---

## Best Practices

1. **Use EventBroadcaster for live dashboards** -- subscribe from a websocket handler or logging task
2. **Enable tracing in production** -- the overhead is minimal and debugging without traces is painful
3. **Export to OTLP** -- standard format, works with any observability platform
4. **Use state snapshots sparingly** -- they increase memory usage; enable only for debugging
5. **Set health check thresholds** -- catch degradation before users notice
6. **Use `info` level in production** -- `debug` generates too much output

---

See [FEATURES.md](../FEATURES.md) for the complete feature list.
