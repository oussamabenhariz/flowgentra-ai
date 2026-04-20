# Handler Development Guide

Write the functions that power each node in your workflow.

## The Pattern

Every handler follows the same shape: take state in, do work, return updated state.

```rust
pub async fn my_handler(mut state: State) -> Result<State> {
    let input = state.get("input").and_then(|v| v.as_str()).unwrap_or("");
    let result = process(input);
    state.set("output", json!(result));
    Ok(state)
}
```

That's it. Every handler is an async function with this signature.

---

## Two Ways to Use Handlers

### With StateGraph (Programmatic)

Pass functions directly -- no macro needed:

```rust
let graph = StateGraphBuilder::new()
    .add_fn("validate", validate_input)
    .add_fn("process", process_data)
    .set_entry_point("validate")
    .add_edge("validate", "process")
    .add_edge("process", "__end__")
    .compile()?;
```

### With Config Files (YAML)

Use `#[register_handler]` for auto-discovery:

```rust
use flowgentra_ai::prelude::*;

#[register_handler]
pub async fn validate_input(mut state: State) -> Result<State> {
    let input = state.get_str("user_input")?;
    state.set("valid", json!(!input.is_empty()));
    Ok(state)
}
```

```yaml
graph:
  nodes:
    - name: step1
      handler: "validate_input"  # Matches function name
```

```rust
let agent = from_config_path("config.yaml")?;  // Handlers auto-discovered
```

Requirements for `#[register_handler]`: `pub`, `async fn`, one `State` param, returns `Result<State>`.

See [macros/README.md](../macros/README.md) for full details.

---

## Common Patterns

### Using LLM

```rust
pub async fn generate_response(mut state: State) -> Result<State> {
    let llm = state.get_llm()?;
    let input = state.get_str("input")?;

    let response = llm.chat(vec![
        Message::system("You are helpful."),
        Message::user(&input),
    ]).await?;

    state.set("response", json!(response.content));
    Ok(state)
}
```

### Using RAG

```rust
pub async fn retrieve_context(mut state: State) -> Result<State> {
    let rag = state.get_rag_client()?;
    let query = state.get_str("query")?;

    let docs = rag.retrieve(query, 5).await?;
    let context = docs.iter().map(|d| d.content.clone()).collect::<Vec<_>>().join("\n---\n");

    state.set("context", json!(context));
    Ok(state)
}
```

### Using MCP Tools

```rust
pub async fn search_web(mut state: State) -> Result<State> {
    let mcp = state.get_mcp_client()?;
    let query = state.get_str("query")?;

    let results = mcp.execute_tool(
        "web_search",
        json!({"query": query, "max_results": 5})
    ).await?;

    state.set("search_results", results);
    Ok(state)
}
```

### Running Concurrent Operations

```rust
pub async fn parallel_work(mut state: State) -> Result<State> {
    let mcp = state.get_mcp_client()?;

    let (search, calc) = tokio::join!(
        mcp.execute_tool("search", json!({"query": "rust"})),
        mcp.execute_tool("calculator", json!({"expr": "2+2"}))
    );

    state.set("search", search?);
    state.set("calc", calc?);
    Ok(state)
}
```

### Branching Logic

Set a field that conditional edges can route on:

```rust
pub async fn classify(mut state: State) -> Result<State> {
    let score = state.get("score").and_then(|v| v.as_i64()).unwrap_or(0);

    if score > 70 {
        state.set("path", json!("complex"));
    } else {
        state.set("path", json!("simple"));
    }

    Ok(state)
}
```

### Passthrough (Logging / Debug)

```rust
pub async fn debug_state(state: State) -> Result<State> {
    println!("State: {:#?}", state);
    Ok(state)  // No changes
}
```

---

## Error Handling

```rust
pub async fn safe_handler(mut state: State) -> Result<State> {
    // Propagate errors with ?
    let required = state.get_str("required_field")?;

    // Provide defaults
    let optional = state.get("optional")
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    // Custom errors
    let data = state.get("data")
        .ok_or_else(|| FlowgentraError::StateError("data field required".into()))?;

    Ok(state)
}
```

Always return `Result<State>`. Never panic in handlers.

---

## Testing Handlers

Handlers are plain async functions, so they're straightforward to test:

```rust
#[tokio::test]
async fn test_validate_input() {
    let mut state = State::new();
    state.set("user_input", json!("hello"));

    let result = validate_input(state).await.unwrap();
    assert_eq!(result.get_bool("valid").unwrap(), true);
}

#[tokio::test]
async fn test_empty_input() {
    let mut state = State::new();
    state.set("user_input", json!(""));

    let result = validate_input(state).await.unwrap();
    assert_eq!(result.get_bool("valid").unwrap(), false);
}
```

---

## Organizing Handlers

For larger projects, split handlers across files:

```
src/
  main.rs              -- mod handlers;
  handlers/
    mod.rs             -- pub mod validation; pub mod processing;
    validation.rs      -- validate_input, validate_schema
    processing.rs      -- process_data, enrich_data
    output.rs          -- format_output, send_response
```

All handlers are discovered through module imports.

---

## Best Practices

1. **One handler, one job** -- keep handlers focused
2. **Clear names** -- `validate_email` not `step3`
3. **Return errors, don't panic** -- use `?` and `Result`
4. **Document state reads/writes** -- comment which keys a handler expects and produces
5. **Test independently** -- each handler should be testable without running the full graph

---

See [macros/README.md](../macros/README.md) for macro details.
See [graph/README.md](../graph/README.md) for wiring handlers into graphs.
