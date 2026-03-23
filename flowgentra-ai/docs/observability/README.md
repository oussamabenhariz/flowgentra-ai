# Observability & Monitoring Guide

Watch your agent run - see what it's doing, how fast it's working, and catch problems early.

## Logging

### Enable Logging

```yaml
middleware:
  - name: logging
    enabled: true
    level: debug          # debug, info, warn, error
    include_state: true   # Log full state
```

## Log Levels

- **debug**: Detailed execution info (useful while developing)
- **info**: General progress (default for production)
- **warn**: Warnings and potential issues
- **error**: Errors only (minimal logging)

## Distributed Tracing

### Enable Tracing

```yaml
observability:
  tracing_enabled: true
  trace_level: debug
  
  spans:
    - name: handler_execution
      sample_rate: 1.0       # Always trace
    
    - name: llm_calls
      sample_rate: 0.5       # Trace 50%
    
    - name: rag_queries
      sample_rate: 0.5       # Trace 50%
    
    - name: mcp_tool_execution
      sample_rate: 1.0       # Always trace
```

### Export Tracing

```yaml
observability:
  tracing_enabled: true
  
  exporters:
    # Jaeger
    - type: jaeger
      endpoint: "http://localhost:14268/api/traces"
    
    # OTLP
    - type: otlp_http
      endpoint: "http://localhost:4318/v1/traces"
    
    # Datadog
    - type: datadog
      api_key: ${DATADOG_API_KEY}
      service: "flowgentra-agent"
```

## Health Monitoring

### Enable Health Checks

```yaml
health:
  enabled: true
  check_interval: 30        # Check every 30 seconds
  
  checks:
    - llm_connectivity      # Can reach LLM
    - rag_availability      # Vector store accessible
    - mcp_tools             # External tools respond
    - memory_usage          # Memory not exhausted
    - response_time         # Response time acceptable
  
  memory_threshold: 80      # % memory usage alert
  response_time_threshold: 5000  # ms max response time
  
  on_failure: "alert"       # alert, fallback, stop
```

## Caching & Performance

### Response Caching

```yaml
middleware:
  - name: cache
    enabled: true
    ttl: 3600               # 1 hour
    strategy: content_hash  # Cache based on input
```

### Rate Limiting

```yaml
middleware:
  - name: rate_limiting
    enabled: true
    rpm: 60                 # 60 requests per minute
    burst_size: 10          # Allow 10 rapid requests
```

## Debugging

### Enable Debug Mode

```yaml
environment:
  debug_mode: true
  
  log_file:
    enabled: true
    path: "./logs/agent.log"
    max_size: "100Mi"
    max_backups: 5
    max_age_days: 7
```

### Inspect Runtime

```rust
// Check agent metrics
let metrics = agent.get_metrics()?;
println!("Total requests: {}", metrics.total_requests);
println!("Avg response time: {}ms", metrics.avg_response_time);
println!("Cache hit rate: {}%", metrics.cache_hit_rate);

// Check health status
let health = agent.check_health().await?;
println!("LLM status: {}", health.llm_status);
println!("Memory: {}%", health.memory_percent);
```

## Monitoring Traces

### What to Monitor

1. **Handler Execution** - How long each handler takes
2. **LLM Calls** - Token usage, latency
3. **RAG Queries** - Retrieval latency, doc count
4. **Tool Execution** - Tool response times
5. **State Size** - Memory usage

### Common Metrics

```
Handler Performance:
├─ validate_input: 10ms
├─ retrieve_context: 250ms
├─ generate_response: 1200ms
└─ Total: 1460ms

LLM Usage:
├─ Calls: 3
├─ Input tokens: 1250
├─ Output tokens: 450
└─ Cost: $0.042

Cache:
├─ Hits: 5
├─ Misses: 2
└─ Hit rate: 71%
```

## Alerting

### Configure Alerts

```yaml
health:
  on_failure: "alert"
  
  # Alert if LLM is down
  # Alert if memory > 80%
  # Alert if response time > 5s
```

### Example: Alert Handler

```rust
pub async fn alert_on_failure(mut state: State) -> Result<State> {
    let health = agent.check_health().await?;
    
    if !health.is_healthy {
        // Send alert
        send_alert(format!("Agent error: {}", health.error))?;
        state.set("alert_sent", json!(true));
    }
    
    Ok(state)
}
```

## Best Practices

1. Always log errors
2. Trace the critical paths (LLM calls, RAG queries)
3. Set realistic latency thresholds
4. Use caching to reduce repeated work
5. Set rate limits to prevent abuse
6. Clean up old logs regularly

## Tools

### View Logs

```bash
# Follow logs
tail -f ./logs/agent.log

# Search logs
grep "ERROR" ./logs/agent.log
```

### Analyze Traces

- **Jaeger UI** - http://localhost:16686
- **Datadog APM** - https://app.datadoghq.com
- **Custom dashboards** - Build in Grafana

---

See [configuration/CONFIG_GUIDE.md](../configuration/CONFIG_GUIDE.md) for complete reference.
