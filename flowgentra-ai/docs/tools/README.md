# Tools and MCP Guide

Connect your agent to external tools, APIs, and services.

## Two Types of Tools

### Local Tools

Simple function specifications your agent can call:

```rust
use flowgentra_ai::core::agents::ToolSpec;

let calculator = ToolSpec::new("calculator", "Perform math calculations")
    .with_parameter("expression", "string")
    .required("expression");

let search = ToolSpec::new("search", "Search the web")
    .with_parameter("query", "string")
    .with_parameter("max_results", "number")
    .required("query");

let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_tool(calculator)
    .with_tool(search)
    .build()?;
```

### MCP Tools (Remote Services)

Connect to external services via the Model Context Protocol.

---

## MCP Transports

### SSE (Server-Sent Events)

```yaml
mcp:
  tools:
    - name: web_search
      type: sse
      url: "http://localhost:8000"
      timeout: 30
```

### Stdio

```yaml
mcp:
  tools:
    - name: python_tool
      type: stdio
      command: "python /path/to/tool.py"
      timeout: 15
```

### Docker

```yaml
mcp:
  tools:
    - name: nlp_service
      type: docker
      image: "nlp-tool:latest"
      timeout: 60
```

---

## Reconnecting MCP Client

For long-running agents, connections can drop. The `ReconnectingMCPClient` automatically reconnects:

```rust
use flowgentra_ai::core::mcp::ReconnectingMCPClient;

let client = ReconnectingMCPClient::new(|| async {
    // Factory function that creates a new connection
    DefaultMCPClient::connect_sse("http://localhost:8000").await
})
.with_max_reconnects(5);

// If the connection drops, it reconnects automatically
let tools = client.list_tools().await?;
let result = client.call_tool("search", args).await?;
```

The factory pattern means each reconnection creates a fresh connection. The client detects connection errors and retries transparently.

---

## MCP Resources

Read external resources (files, databases, APIs) through MCP:

```rust
// List available resources
let resources = client.list_resources().await?;
for r in &resources {
    println!("{}: {} ({})", r.uri, r.name, r.mime_type.as_deref().unwrap_or("unknown"));
}

// Read a specific resource
let content = client.read_resource("file:///docs/readme.md").await?;
println!("{}", content.text.unwrap_or_default());
```

### Resource Types

Resources can be files, database records, API responses, or any external data:

```rust
pub struct MCPResource {
    pub uri: String,            // e.g. "file:///path" or "db://table/id"
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
}
```

---

## MCP Prompts

Use prompt templates managed by MCP servers:

```rust
// List available prompts
let prompts = client.list_prompts().await?;
for p in &prompts {
    println!("{}: {}", p.name, p.description.as_deref().unwrap_or(""));
    for arg in &p.arguments {
        println!("  arg: {} (required: {})", arg.name, arg.required);
    }
}

// Get a rendered prompt
let result = client.get_prompt("summarize", json!({
    "text": "Long document content here..."
})).await?;

for msg in &result.messages {
    println!("[{}] {}", msg.role, msg.content);
}
```

---

## Using Tools in Handlers

```rust
#[register_handler]
pub async fn research_handler(mut state: State) -> Result<State> {
    let mcp = state.get_mcp_client()?;
    let query = state.get_str("user_query")?;

    // Execute a tool
    let results = mcp.execute_tool(
        "web_search",
        json!({"query": query, "max_results": 5})
    ).await?;

    state.set("search_results", results);
    Ok(state)
}
```

### Running Multiple Tools Concurrently

```rust
let (search_result, calc_result) = tokio::join!(
    mcp.execute_tool("search", json!({"query": "..."})),
    mcp.execute_tool("calculator", json!({"expr": "2+2"}))
);

state.set("search", search_result?);
state.set("calc", calc_result?);
```

---

## MCP Configuration in YAML

```yaml
mcp:
  enabled: true
  execution:
    parallel: true          # Run tools concurrently
    max_parallel: 3         # Max concurrent calls
    timeout: 30             # Default timeout
    error_handling: continue  # continue or fail

  tools:
    - name: web_search
      type: sse
      url: "http://search-api:8000"
      timeout: 30

    - name: email
      type: stdio
      command: "/usr/bin/email-tool"
      timeout: 15

    - name: database
      type: docker
      image: "db-tool:latest"
      timeout: 60
```

---

## Best Practices

1. **Use clear tool names and descriptions** -- the LLM uses these to decide when to call a tool
2. **Mark required parameters** -- prevents the LLM from omitting critical inputs
3. **Set timeouts** -- always set timeouts on external tool calls
4. **Use ReconnectingMCPClient for long-running agents** -- connections will drop eventually
5. **Run independent tools in parallel** -- use `tokio::join!` for concurrent execution
6. **Handle errors gracefully** -- tools can fail; decide whether to retry or skip

---

See [FEATURES.md](../FEATURES.md) for the complete feature list.
