# Handler Development Guide

Write the functions that power each node in your workflow.

## The Pattern

Every handler follows the same shape: take state by mutable reference, do work, return `Ok(())`.

```rust
#[node]
pub async fn my_handler(state: &mut MyState) -> Result<()> {
    let result = process(&state.input);
    state.output = result;
    Ok(())
}
```

That's it. Every handler is an async function decorated with `#[node]`.

---

## Two Ways to Use Handlers

### With StateGraph (Programmatic)

Pass functions directly -- no macro needed:

```rust
// #[derive(State, Default, Clone)] struct PipelineState { ... }
let graph = StateGraph::<PipelineState>::builder()
    .add_fn("validate", validate_input)
    .add_fn("process", process_data)
    .set_entry("validate")
    .add_edge("validate", "process")
    .set_finish("process")
    .build()?;
```

### With Config Files (YAML)

Use `#[register_handler]` for auto-discovery:

```rust
use flowgentra_ai::prelude::*;

#[register_handler]
pub async fn validate_input(state: &mut MyState) -> Result<()> {
    state.valid = !state.user_input.is_empty();
    Ok(())
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

Requirements for `#[register_handler]`: `pub`, `async fn`, one `&mut MyState` param, returns `Result<()`.

See [macros/README.md](../macros/README.md) for full details.

---

## Common Patterns

### Using LLM

```rust
#[node]
pub async fn generate_response(state: &mut MyState) -> Result<()> {
    let llm = state.get_llm()?;

    let response = llm.chat(vec![
        Message::system("You are helpful."),
        Message::user(&state.input),
    ]).await?;

    state.response = response.content;
    Ok(())
}
```

### Using RAG

```rust
#[node]
pub async fn retrieve_context(state: &mut MyState) -> Result<()> {
    let rag = state.get_rag_client()?;

    let docs = rag.retrieve(&state.query, 5).await?;
    state.context = docs.iter().map(|d| d.content.clone()).collect::<Vec<_>>().join("\n---\n");

    Ok(())
}
```

### Using MCP Tools

```rust
#[node]
pub async fn search_web(state: &mut MyState) -> Result<()> {
    let mcp = state.get_mcp_client()?;

    let results = mcp.execute_tool(
        "web_search",
        json!({"query": state.query, "max_results": 5})
    ).await?;

    state.search_results = results;
    Ok(())
}
```

### Running Concurrent Operations

```rust
#[node]
pub async fn parallel_work(state: &mut MyState) -> Result<()> {
    let mcp = state.get_mcp_client()?;

    let (search, calc) = tokio::join!(
        mcp.execute_tool("search", json!({"query": "rust"})),
        mcp.execute_tool("calculator", json!({"expr": "2+2"}))
    );

    state.search = search?;
    state.calc = calc?;
    Ok(())
}
```

### Branching Logic

Set a field that conditional edges can route on:

```rust
#[node]
pub async fn classify(state: &mut MyState) -> Result<()> {
    if state.score > 70 {
        state.path = "complex".to_string();
    } else {
        state.path = "simple".to_string();
    }

    Ok(())
}
```

### Passthrough (Logging / Debug)

```rust
#[node]
pub async fn debug_state(state: &mut MyState) -> Result<()> {
    println!("State: {:#?}", state);
    Ok(())  // No changes
}
```

---

## Error Handling

```rust
#[node]
pub async fn safe_handler(state: &mut MyState) -> Result<()> {
    // Propagate errors with ?
    let required = state.required_field.as_str();

    // Provide defaults
    let optional = state.optional.as_deref().unwrap_or("default");

    // Custom errors
    if state.data.is_empty() {
        return Err(FlowgentraError::StateError("data field required".into()));
    }

    Ok(())
}
```

Always return `Result<()>`. Never panic in handlers.

---

## Testing Handlers

Handlers are plain async functions, so they're straightforward to test:

```rust
#[tokio::test]
async fn test_validate_input() {
    let mut state = MyState { user_input: "hello".into(), ..Default::default() };

    validate_input(&mut state).await.unwrap();
    assert_eq!(state.valid, true);
}

#[tokio::test]
async fn test_empty_input() {
    let mut state = MyState { user_input: "".into(), ..Default::default() };

    validate_input(&mut state).await.unwrap();
    assert_eq!(state.valid, false);
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
