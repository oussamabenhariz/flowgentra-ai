# State Management Guide

Understand how data flows through your agent workflow.

## How State Works

State is a JSON object that flows through your workflow, getting updated by each handler.

## State Schema

Document what data flows through your agent:

```yaml
state_schema:
  # Input
  user_input:
    type: string
    description: "User's question or command"
  
  # Processing
  analysis:
    type: object
    description: "Analysis results"
  
  context:
    type: array
    description: "Retrieved context documents"
  
  # Output
  response:
    type: string
    description: "Final response to user"
```

## Reading State

```rust
pub async fn my_handler(mut state: State) -> Result<State> {
    // Get string value
    let input = state.get_str("user_input")?;
    
    // Get JSON value
    let value = state.get("analysis")?;
    
    // Get with default
    let count = state.get_i64("count").unwrap_or(0);
    
    Ok(state)
}
```

## Writing State

```rust
pub async fn my_handler(mut state: State) -> Result<State> {
    // Set simple value
    state.set("processed", json!(true));
    
    // Set complex object
    state.set("analysis", json!({
        "score": 0.95,
        "tags": ["important", "urgent"]
    }));
    
    // Merge with existing
    state.set("metadata", json!({"timestamp": "2024-03-02"}));
    
    Ok(state)
}
```

## State Validation

```yaml
state_schema:
  required_field:
    type: string
    required: true
  
  optional_field:
    type: string
    required: false
```

```rust
// Validate state conforms to schema
state.validate()?;  // Returns error if invalid
```

## Common Patterns

### Accumulation

```rust
// Add to a list
let mut results = state.get_vec("results")
    .unwrap_or_default();
results.push(json!("new_item"));
state.set("results", json!(results));
```

### Conditional Processing

```rust
if state.get_bool("is_complex").unwrap_or(false) {
    // Complex processing
} else {
    // Simple processing
}
```

### Chaining Results

```rust
// Handler 1 output → Handler 2 input
state.set("handler1_result", json!(result));
// ...
let input = state.get("handler1_result")?;
```

## State Flow Example

```yaml
graph:
  nodes:
    - name: validate
      handler: handlers::validate
    
    - name: search
      handler: handlers::search
      uses_rag: true
    
    - name: analyze
      handler: handlers::analyze
    
    - name: respond
      handler: handlers::generate_response
```

**State progression**:
1. validate: `{}` → `{input: "...", valid: true}`
2. search: `{..., valid: true}` → `{..., context: [...]}`
3. analyze: `{..., context: [...]}` → `{..., analysis: {...}}`
4. respond: `{..., analysis: {...}}` → `{..., response: "..."}`

## Debugging State

```rust
// Print full state
println!("State: {:#?}", state);

// Print specific field
if let Some(value) = state.get("field_name") {
    println!("Value: {}", value);
}

// List all keys
for key in state.keys() {
    println!("Key: {}", key);
}
```

## Best Practices

1. Document your schema upfront
2. Use consistent naming (snake_case for keys)
3. Validate state early (in first handler)
4. Only modify what you need
5. Keep the state clean - don't accumulate trash data
6. Keep types consistent across handlers

## Large State Handling

```rust
// Check size
let size = state.size_bytes();
if size > 10_000_000 {
    // Consider checkpoint
}

// Clear old data
state.remove("old_field");
```

---

See [handlers/README.md](../handlers/README.md) for handler patterns.
