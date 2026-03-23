# Tools & MCP Guide

Connect your agent to external tools, APIs, and services so it can do more than just chat.

## Two Types of Tools

### 1. Local Tools

Simple functions your agent can call directly:

```rust
let calculator = ToolSpec::new("calculator", "Do math")
    .with_parameter("expression", "string")
    .required("expression");

let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_tool(calculator)
    .build()?;
```

### 2. MCP Tools (Remote Services)

Connect to external services via HTTP, stdio, or Docker:

```yaml
mcp:
  enabled: true
  tools:
    - name: web_search
      description: "Search the web"
      type: external
      endpoint: "http://localhost:3000/search"
      method: POST
      auth:
        type: bearer
        token: ${SEARCH_TOKEN}
      timeout: 30
    
    - name: calculator
      description: "Math operations"
      type: builtin
      capabilities: [basic_math, trigonometry]
```

## Creating Local Tools

```rust
let web_search = ToolSpec::new("search", "Search the web")
    .with_parameter("query", "string")
    .with_parameter("max_results", "number")
    .required("query");

let translator = ToolSpec::new("translate", "Translate text")
    .with_parameter("text", "string")
    .with_parameter("language", "string")
    .required("text")
    .required("language");

let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_tool(web_search)
    .with_tool(translator)
    .build()?;
```

## MCP Configuration

### HTTP/SSE Tools

```yaml
mcp:
  tools:
    - name: api_service
      type: external
      endpoint: "http://api.example.com/execute"
      method: POST
      auth:
        type: bearer
        token: ${API_TOKEN}
      timeout: 30
```

### Stdio Tools

```yaml
mcp:
  tools:
    - name: python_script
      type: external
      command: "python /path/to/script.py"
      timeout: 15
```

### Execution Options

```yaml
mcp:
  enabled: true
  execution:
    parallel: true         # Run multiple tools concurrently
    max_parallel: 3        # Max concurrent executions
    timeout: 30            # Tool execution timeout
    error_handling: "continue"  # Continue on tool error
```

## Tool Best Practices

1. Use clear, specific names (not just "search")
2. Write good descriptions
3. Always mark required fields
4. Provide sensible defaults
5. Each tool should do one thing well
6. Handle errors gracefully

## Using Tools in Handlers

```rust
pub async fn search_web(mut state: State) -> Result<State> {
    let query = state.get_str("user_query")?;
    
    // Execute MCP tool
    let results = mcp_client.execute_tool(
        "web_search",
        json!({"query": query, "max_results": 5})
    ).await?;
    
    state.set("search_results", results);
    Ok(state)
}
```

## Common Tools

### Search
- Web search (Google, Bing)
- Document search
- Database queries

### Calculation
- Math operations
- Unit conversion
- Statistics

### Communication
- Email sending
- SMS/messaging
- Notifications

### Data Processing
- CSV/Excel parsing
- JSON transformation
- File operations

### Integration
- API calls
- Database access
- Third-party services

## Tool Configuration in Graph

```yaml
graph:
  nodes:
    - name: search
      handler: handlers::search
      uses_mcp: true
      mcp_tools:
        - web_search
        - calculator
      timeout: 30
    
    - name: process
      handler: handlers::process
      uses_mcp: true
      mcp_tools:
        - file_processor
```

## Fallback for Tools

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_mcp(primary_search_service)
    .with_mcp(fallback_search_service)
    .build()?;
```

---

See [configuration/CONFIG_GUIDE.md](../configuration/CONFIG_GUIDE.md) for complete reference.
