# Testing Guide

Strategies for testing handlers, graphs, and full workflows.

## Testing Handlers

Handlers are plain async functions -- test them directly:

```rust
#[tokio::test]
async fn test_validate_input() {
    let mut state = MyState { input: "test data".into(), ..Default::default() };

    let result = validate_input(&mut state).await;

    assert!(result.is_ok());
    assert_eq!(state.valid, true);
}
```

### Test Error Cases

```rust
#[tokio::test]
async fn test_missing_required_field() {
    let mut state = MyState::default();
    // Missing input field

    let result = validate_input(&mut state).await;
    assert!(result.is_err());
}
```

---

## Testing State Management

### State Operations

```rust
#[derive(State, Default, Clone)]
struct TestState {
    key: Option<String>,
}

#[test]
fn test_state_operations() {
    let mut state = TestState::default();

    state.key = Some("value".into());
    assert_eq!(state.key.as_deref(), Some("value"));

    state.key = None;
    assert!(state.key.is_none());
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
#[derive(State, Default, Clone)]
struct WorkflowState {
    step1: bool,
    step2: bool,
}

#[tokio::test]
async fn test_graph_workflow() {
    let graph = StateGraph::<WorkflowState>::builder()
        .add_fn("step1", |s: &mut WorkflowState| async move {
            s.step1 = true;
            Ok(())
        })
        .add_fn("step2", |s: &mut WorkflowState| async move {
            assert!(s.step1); // step1 ran first
            s.step2 = true;
            Ok(())
        })
        .set_entry("step1")
        .add_edge("step1", "step2")
        .set_finish("step2")
        .build()
        .unwrap();

    let result = graph.invoke(WorkflowState::default()).await.unwrap();
    assert!(result.step1);
    assert!(result.step2);
}
```

### Test Conditional Routing

```rust
#[tokio::test]
async fn test_conditional_takes_simple_path() {
    let state = RoutingState { score: 30, ..Default::default() }; // Low score -> simple path

    let result = graph.invoke(state).await.unwrap();
    assert_eq!(result.path_taken.as_str(), "simple");
}

#[tokio::test]
async fn test_conditional_takes_complex_path() {
    let state = RoutingState { score: 90, ..Default::default() }; // High score -> complex path

    let result = graph.invoke(state).await.unwrap();
    assert_eq!(result.path_taken.as_str(), "complex");
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
    let mut state = PipelineState { input: "test data".into(), ..Default::default() };

    // Run handlers in sequence
    validate(&mut state).await.unwrap();
    assert!(state.valid);

    process(&mut state).await.unwrap();
    assert!(state.processed);

    format_output(&mut state).await.unwrap();
    assert!(!state.output.is_empty());
}
```

---

## Mocking External Services

### Mock LLM Responses

```rust
fn mock_state_with_llm_response() -> MyState {
    MyState { llm_response: "Mocked answer".into(), ..Default::default() }
}

#[tokio::test]
async fn test_with_mocked_llm() {
    let mut state = mock_state_with_llm_response();
    format_response(&mut state).await.unwrap();
    assert!(!state.output.is_empty());
}
```

### Mock MCP Tool Results

```rust
#[tokio::test]
async fn test_with_mocked_tools() {
    let mut state = MyState {
        search_results: vec![SearchResult { title: "Result 1".into(), url: "https://example.com".into() }],
        ..Default::default()
    };

    summarize_results(&mut state).await.unwrap();
    assert!(!state.summary.is_empty());
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
