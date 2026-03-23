# Testing Guide

Write tests to catch bugs before they hit production.

## Unit Testing Handlers

### Basic Handler Test

```rust
#[tokio::test]
async fn test_handler() {
    // Setup
    let mut state = State::new();
    state.set("input", json!("test"));
    
    // Execute
    let result = my_handler(state).await;
    
    // Assert
    assert!(result.is_ok());
    let final_state = result.unwrap();
    assert_eq!(
        final_state.get_str("output").unwrap(),
        "expected"
    );
}
```

### Testing with Mocks

```rust
#[tokio::test]
async fn test_handler_with_mock() {
    // Create mock state with defaults
    let mut state = State::new();
    state.set("input", json!("test"));
    state.set("count", json!(0));
    
    // Execute handler
    let result = my_handler(state).await;
    
    // Verify behavior
    assert!(result.is_ok());
    let final_state = result.unwrap();
    assert!(final_state.get("output").is_some());
}
```

## Testing Agent Components

### Test State Flow

```rust
#[tokio::test]
async fn test_state_flow() {
    let mut state = State::new();
    
    // Simulate handler 1
    state.set("step1", json!("done"));
    assert_eq!(state.get_str("step1").unwrap(), "done");
    
    // Simulate handler 2
    state.set("step2", json!("done"));
    assert_eq!(state.get_str("step2").unwrap(), "done");
}
```

### Test Validation

```rust
#[tokio::test]
async fn test_validation() {
    let mut state = State::new();
    state.set("input", json!(""));
    
    // Should handle empty input
    let result = validate_input(state).await;
    
    assert!(result.is_ok());
    let final_state = result.unwrap();
    assert_eq!(final_state.get_bool("valid").unwrap(), false);
}
```

## Integration Testing

### Test Multiple Handlers

```rust
#[tokio::test]
async fn test_workflow() {
    let mut state = State::new();
    state.set("input", json!("test data"));
    
    // Handler 1: Validate
    state = validate_input(state).await.unwrap();
    assert_eq!(state.get_bool("valid").unwrap(), true);
    
    // Handler 2: Process
    state = process_input(state).await.unwrap();
    assert!(state.get("processed").is_some());
    
    // Handler 3: Generate
    state = generate_response(state).await.unwrap();
    assert!(state.get("response").is_some());
}
```

### Test Error Cases

```rust
#[tokio::test]
async fn test_error_handling() {
    let mut state = State::new();
    // Missing required field
    
    let result = my_handler(state).await;
    assert!(result.is_err());
}
```

## Testing with Async Operations

```rust
#[tokio::test]
async fn test_concurrent_operations() {
    let mut state = State::new();
    
    // Test concurrent handler execution
    let result = concurrent_handler(state).await;
    
    assert!(result.is_ok());
}
```

## Mocking External Services

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock LLM response
    fn create_mock_state() -> State {
        let mut state = State::new();
        state.set("llm_response", json!("mocked response"));
        state
    }
    
    #[tokio::test]
    async fn test_with_mock() {
        let state = create_mock_state();
        let result = my_handler(state).await;
        assert!(result.is_ok());
    }
}
```

## Testing Best Practices

1. Test one thing per test
2. Use clear test names
3. Follow Arrange-Act-Assert: Setup, execute, verify
4. Keep tests independent
5. Mock external services
6. Test both success and failure cases

## Test Structure

```
tests/
├── unit/
│   ├── handlers_test.rs
│   ├── state_test.rs
│   └── validation_test.rs
└── integration/
    ├── workflow_test.rs
    └── end_to_end_test.rs
```

## Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_handler

# Run with output
cargo test -- --nocapture

# Run integration tests
cargo test --test integration_tests
```

---

See [handlers/README.md](../handlers/README.md) for handler development.
