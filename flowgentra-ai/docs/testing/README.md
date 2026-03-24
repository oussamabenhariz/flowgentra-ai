# Testing Guide

Strategies for testing handlers, graphs, and full workflows.

## Testing Handlers

Handlers are plain async functions -- test them directly:

```rust
#[tokio::test]
async fn test_validate_input() {
    let mut state = PlainState::new();
    state.set("input", json!("test data"));

    let result = validate_input(state).await;

    assert!(result.is_ok());
    let final_state = result.unwrap();
    assert_eq!(final_state.get("valid").unwrap(), &json!(true));
}
```

### Test Error Cases

```rust
#[tokio::test]
async fn test_missing_required_field() {
    let state = PlainState::new();
    // Missing "input" field

    let result = validate_input(state).await;
    assert!(result.is_err());
}
```

---

## Testing State Management

### State Operations

```rust
#[test]
fn test_state_operations() {
    let mut state = PlainState::new();

    state.set("key", json!("value"));
    assert_eq!(state.get("key").unwrap(), &json!("value"));

    state.remove("key");
    assert!(state.get("key").is_none());
}
```

### Reducers

```rust
#[test]
fn test_reducer_append() {
    let reducer = JsonReducer::Append;
    let current = json!([1, 2]);
    let update = json!([3]);
    let result = reducer.apply(&current, &update);
    assert_eq!(result, json!([1, 2, 3]));
}

#[test]
fn test_reducer_config() {
    let config = ReducerConfig::default()
        .field("items", JsonReducer::Append)
        .field("total", JsonReducer::Sum);

    let current = json!({"items": [1], "total": 10});
    let update = json!({"items": [2], "total": 5});
    let merged = config.merge_values(&current, &update);

    assert_eq!(merged["items"], json!([1, 2]));
    assert_eq!(merged["total"], json!(15));
}
```

### ScopedState

```rust
#[test]
fn test_scoped_state_isolation() {
    let shared = SharedState::new();

    let scope_a = ScopedState::new(shared.clone(), "a");
    let scope_b = ScopedState::new(shared.clone(), "b");

    scope_a.set("value", json!(1));
    scope_b.set("value", json!(2));

    assert_eq!(scope_a.get("value").unwrap(), json!(1));
    assert_eq!(scope_b.get("value").unwrap(), json!(2));
}
```

---

## Testing StateGraph Workflows

### Full Graph Execution

```rust
#[tokio::test]
async fn test_graph_workflow() {
    let graph = StateGraphBuilder::new()
        .add_fn("step1", |mut s: PlainState| async move {
            s.set("step1", json!(true));
            Ok(s)
        })
        .add_fn("step2", |mut s: PlainState| async move {
            assert!(s.get("step1").is_some()); // step1 ran first
            s.set("step2", json!(true));
            Ok(s)
        })
        .set_entry_point("step1")
        .add_edge("step1", "step2")
        .add_edge("step2", "__end__")
        .compile()
        .unwrap();

    let result = graph.run(PlainState::new()).await.unwrap();
    assert_eq!(result.get("step1").unwrap(), &json!(true));
    assert_eq!(result.get("step2").unwrap(), &json!(true));
}
```

### Test Conditional Routing

```rust
#[tokio::test]
async fn test_conditional_takes_simple_path() {
    let mut state = PlainState::new();
    state.set("score", json!(30)); // Low score -> simple path

    let result = graph.run(state).await.unwrap();
    assert_eq!(result.get("path_taken").unwrap(), &json!("simple"));
}

#[tokio::test]
async fn test_conditional_takes_complex_path() {
    let mut state = PlainState::new();
    state.set("score", json!(90)); // High score -> complex path

    let result = graph.run(state).await.unwrap();
    assert_eq!(result.get("path_taken").unwrap(), &json!("complex"));
}
```

### Test Graph Export

```rust
#[test]
fn test_graph_exports() {
    let graph = build_test_graph().unwrap();

    let dot = graph.to_dot();
    assert!(dot.contains("step1"));
    assert!(dot.contains("step2"));

    let mermaid = graph.to_mermaid();
    assert!(mermaid.contains("-->"));

    let json = graph.to_json();
    assert!(json["nodes"].is_array());
}
```

---

## Testing Multi-Handler Pipelines

Simulate a full pipeline without the graph runtime:

```rust
#[tokio::test]
async fn test_pipeline() {
    let mut state = PlainState::new();
    state.set("input", json!("test data"));

    // Run handlers in sequence
    state = validate(state).await.unwrap();
    assert_eq!(state.get("valid").unwrap(), &json!(true));

    state = process(state).await.unwrap();
    assert!(state.get("processed").is_some());

    state = format_output(state).await.unwrap();
    assert!(state.get("output").is_some());
}
```

---

## Mocking External Services

### Mock LLM Responses

```rust
fn mock_state_with_llm_response() -> PlainState {
    let mut state = PlainState::new();
    state.set("llm_response", json!("Mocked answer"));
    state
}

#[tokio::test]
async fn test_with_mocked_llm() {
    let state = mock_state_with_llm_response();
    let result = format_response(state).await.unwrap();
    assert!(result.get("output").is_some());
}
```

### Mock MCP Tool Results

```rust
#[tokio::test]
async fn test_with_mocked_tools() {
    let mut state = PlainState::new();
    state.set("search_results", json!([
        {"title": "Result 1", "url": "https://example.com"}
    ]));

    let result = summarize_results(state).await.unwrap();
    assert!(result.get("summary").is_some());
}
```

---

## Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_validate_input

# With output visible
cargo test -- --nocapture

# Only unit tests
cargo test --lib

# Only integration tests
cargo test --test '*'
```

---

## Project Structure

```
tests/
  unit/
    handlers_test.rs      -- Individual handler tests
    state_test.rs         -- State operations and reducers
    graph_test.rs         -- Graph compilation and export
  integration/
    workflow_test.rs      -- Full pipeline tests
    agent_test.rs         -- Agent integration tests
```

---

## Best Practices

1. **Test one thing per test** -- clear names, focused assertions
2. **Arrange-Act-Assert** -- set up state, run handler, check results
3. **Test both paths** -- success AND failure cases
4. **Mock external services** -- don't hit real APIs in tests
5. **Test each conditional branch** -- create state that triggers each path
6. **Keep tests independent** -- no shared mutable state between tests

---

See [handlers/README.md](../handlers/README.md) for handler development.
See [state/README.md](../state/README.md) for state management patterns.
