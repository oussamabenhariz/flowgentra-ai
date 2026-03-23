# Handler Development Guide

Write the custom functions that power your agent's workflow.

## Must Know

Handlers are async functions that take state in, do something, and return updated state.

```rust
pub async fn my_handler(mut state: State) -> Result<State> {
    // Read from state
    let input = state.get("input")?;
    
    // Do something
    let result = process(input);
    
    // Update state
    state.set("output", json!(result));
    
    Ok(state)
}
```

That's it - every handler follows this pattern.

## Registering Handlers (Required!)

For your handlers to be discoverable by the agent, you must use the `#[register_handler]` macro:

```rust
use flowgentra_ai::prelude::*;

#[register_handler]
pub async fn my_handler(mut state: State) -> Result<State> {
    // Read from state
    let input = state.get("input")?;
    
    // Do something
    let result = process(input);
    
    // Update state
    state.set("output", json!(result));
    
    Ok(state)
}
```

### How It Works

1. **Decorate with `#[register_handler]`** - This registers your function globally
2. **Reference in config.yaml** - Use the function name in your agent config
3. **Auto-discovery** - When you call `from_config_path("config.yaml")`, all registered handlers are automatically available

**Example config.yaml:**
```yaml
graph:
  nodes:
    - name: "step1"
      handler: "my_handler"  # Matches function name
```

**Loading the agent:**
```rust
#[tokio::main]
async fn main() -> Result<()> {
    let mut agent = from_config_path("config.yaml")?;  // Handlers auto-discovered!
    Ok(())
}
```

### Requirements for `#[register_handler]`

- ✅ Must be `pub` (public)
- ✅ Must be `async fn`
- ✅ Must take exactly one `State` parameter
- ✅ Must return `Result<State>`

**For complete details and advanced patterns, see [macros/README.md](../macros/README.md)**

---

## Simple Handler

```rust
pub async fn validate_input(mut state: State) -> Result<State> {
    let input = state.get_str("user_input")?;
    
    if input.is_empty() {
        state.set("valid", json!(false));
        state.set("error", json!("Input cannot be empty"));
    } else {
        state.set("valid", json!(true));
        state.set("cleaned_input", json!(input.trim()));
    }
    
    Ok(state)
}
```

To use this handler:

```rust
#[register_handler]
pub async fn validate_input(mut state: State) -> Result<State> {
    // ... implementation
}
```

Then in config.yaml:

```yaml
nodes:
  - name: "validation_step"
    handler: "validate_input"
```

```rust
pub async fn validate_input(mut state: State) -> Result<State> {
    let input = state.get_str("user_input")?;
    
    if input.is_empty() {
        state.set("valid", json!(false));
        state.set("error", json!("Input cannot be empty"));
    } else {
        state.set("valid", json!(true));
        state.set("cleaned_input", json!(input.trim()));
    }
    
    Ok(state)
}
```

## Using LLM in Handlers

```rust
#[register_handler]
pub async fn generate_response(mut state: State) -> Result<State> {
    let llm = state.get_llm_client()?;
    let input = state.get_str("input")?;
    
    // Call LLM
    let messages = vec![Message::new("user", input)];
    let response = llm.chat(messages).await?;
    
    state.set("response", json!(response));
    Ok(state)
}
```

## Using RAG in Handlers

```rust
#[register_handler]
pub async fn retrieve_context(mut state: State) -> Result<State> {
    let rag = state.get_rag_client()?;
    let query = state.get_str("user_query")?;
    
    // Retrieve documents
    let docs = rag.retrieve(query, 5).await?;
    
    // Format as context
    let context = docs.iter()
        .map(|d| d.content.clone())
        .collect::<Vec<_>>()
        .join("\n---\n");
    
    state.set("context", json!(context));
    state.set("docs", json!(docs));
    
    Ok(state)
}
```

## Using MCP Tools in Handlers

```rust
#[register_handler]
pub async fn search_web(mut state: State) -> Result<State> {
    let mcp = state.get_mcp_client()?;
    let query = state.get_str("query")?;
    
    // Execute tool
    let results = mcp.execute_tool(
        "web_search",
        json!({"query": query, "max_results": 5})
    ).await?;
    
    state.set("search_results", results);
    Ok(state)
}
```

## Error Handling

```rust
#[register_handler]
pub async fn safe_handler(mut state: State) -> Result<State> {
    // Option 1: Handle and continue
    let value = state.get("field").unwrap_or(json!(null));
    
    // Option 2: Propagate error
    let required = state.get_str("required_field")?;
    
    // Option 3: Recover with default
    let count = state.get_i64("count")
        .map_err(|_| anyhow!("Invalid count"))?;
    
    Ok(state)
}
```

## Complex Logic

```rust
#[register_handler]
pub async fn multi_step_handler(mut state: State) -> Result<State> {
    // Step 1: Validate
    let input = state.get_str("input")?;
    if input.is_empty() {
        return Err(anyhow!("Empty input"));
    }
    
    // Step 2: Process
    let processed = input.to_uppercase();
    state.set("step1_result", json!(processed));
    
    // Step 3: Enrich
    let llm = state.get_llm_client()?;
    let analysis = llm.chat(vec![
        Message::new("user", format!("Analyze: {}", processed))
    ]).await?;
    
    state.set("step2_result", json!(analysis));
    
    Ok(state)
}
```

## Async Operations

```rust
#[register_handler]
pub async fn concurrent_handler(mut state: State) -> Result<State> {
    let mcp = state.get_mcp_client()?;
    
    // Run multiple tools concurrently
    let (search_result, calc_result) = tokio::join!(
        mcp.execute_tool("web_search", json!({"query": "..."})),
        mcp.execute_tool("calculator", json!({"expr": "..."}))
    );
    
    state.set("search", search_result?);
    state.set("calc", calc_result?);
    
    Ok(state)
}
```

## Testing Handlers

```rust
#[register_handler]
pub async fn my_handler(mut state: State) -> Result<State> {
    let input = state.get_str("input")?;
    state.set("output", json!(input.to_uppercase()));
    Ok(state)
}

#[tokio::test]
async fn test_my_handler() {
    let mut state = State::new();
    state.set("input", json!("test input"));
    
    let result = my_handler(state).await;
    
    assert!(result.is_ok());
    let final_state = result.unwrap();
    assert_eq!(
        final_state.get_str("output").unwrap(),
        "TEST INPUT"
    );
}
```

## Common Patterns

### Passthrough
```rust
#[register_handler]
pub async fn logging_handler(mut state: State) -> Result<State> {
    println!("Current state: {:#?}", state);
    Ok(state)  // No changes
}
```

### Transformation
```rust
#[register_handler]
pub async fn transform_handler(mut state: State) -> Result<State> {
    let input = state.get("data")?;
    let transformed = transform(input);
    state.set("data", transformed);
    Ok(state)
}
```

### Branching
```rust
#[register_handler]
pub async fn branch_handler(mut state: State) -> Result<State> {
    let value = state.get_i64("score")?;
    
    if value > 70 {
        state.set("path", json!("complex"));
    } else {
        state.set("path", json!("simple"));
    }
    
    Ok(state)
}
```

## Best Practices

1. **Always use `#[register_handler]`** - Required for auto-discovery from config files
2. One handler, one job
3. Use clear names (not cryptic abbreviations)
4. Document what state it reads and writes
5. Return errors, don't panic
6. Keep handlers lightweight and fast
7. Write unit tests

---

## Organizing Handlers in Modules

For larger projects, organize handlers into separate files:

```
src/
  main.rs
  handlers/
    mod.rs
    validation.rs
    processing.rs
    output.rs
```

**src/handlers/validation.rs:**
```rust
use flowgentra_ai::prelude::*;

#[register_handler]
pub async fn validate_input(state: State) -> Result<State> {
    // implementation
    Ok(state)
}
```

**src/handlers/mod.rs:**
```rust
pub mod validation;
pub mod processing;
pub mod output;
```

**src/main.rs:**
```rust
use flowgentra_ai::prelude::*;

mod handlers;  // Important: must import the handlers module!

#[tokio::main]
async fn main() -> Result<()> {
    let mut agent = from_config_path("config.yaml")?;
    let result = agent.run().await?;
    Ok(())
}
```

All handlers are automatically discovered through the module imports!

---

## Detailed Macro Documentation

For complete details on the `#[register_handler]` macro, including:
- Advanced patterns
- Error handling
- Conditional registration
- Module organization
- Troubleshooting

See [../macros/README.md](../macros/README.md)
