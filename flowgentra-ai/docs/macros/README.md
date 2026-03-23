# Handler Registration Macro Guide

Learn how to use the `#[register_handler]` macro to make your handlers discoverable by FlowgentraAI agents.

## The Problem It Solves

Normally, if you write handler functions and want to use them in an agent, you need to manually register them. The `#[register_handler]` macro **automatically handles this registration** for you.

Instead of:
```rust
// ❌ Manual registration (tedious)
let mut registry = HandlerRegistry::new();
registry.register("handler1", Arc::new(|state| Box::pin(handler1(state))));
registry.register("handler2", Arc::new(|state| Box::pin(handler2(state))));
registry.register("handler3", Arc::new(|state| Box::pin(handler3(state))));
// ... repeat for every handler
```

You can write:
```rust
// ✅ Automatic registration (simple!)
#[register_handler]
pub async fn handler1(state: State) -> Result<State> { /* ... */ }

#[register_handler]
pub async fn handler2(state: State) -> Result<State> { /* ... */ }

#[register_handler]
pub async fn handler3(state: State) -> Result<State> { /* ... */ }
```

Then when you load your agent from a config file, all handlers are automatically discovered!

---

## How to Use It

### Step 1: Define Your Handler with the Macro

```rust
use flowgentra_ai::prelude::*;
use serde_json::json;

#[register_handler]
pub async fn validate_input(state: State) -> Result<State> {
    let input = state.get("input")
        .ok_or_else(|| FlowgentraError::StateError("input required".to_string()))?
        .as_str()
        .ok_or_else(|| FlowgentraError::StateError("input must be string".to_string()))?
        .to_string();
    
    state.set("validated_input", json!(input));
    Ok(state)
}
```

### Step 2: Reference the Handler in Your Config

In `config.yaml`, reference the handler by its **function name**:

```yaml
graph:
  nodes:
    - name: "step1"
      handler: "validate_input"  # Matches the function name
```

### Step 3: Load the Agent

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // Handlers are auto-discovered! ✨
    let mut agent = from_config_path("config.yaml")?;
    
    let result = agent.run().await?;
    Ok(())
}
```

That's it! The macro handles the registration automatically through the `inventory` crate.

---

## Requirements

Functions decorated with `#[register_handler]` must follow these rules:

| Requirement | Details |
|------------|---------|
| **Visibility** | Must be `pub` (public) |
| **Async** | Must be `async fn` |
| **Parameter** | Takes exactly one `State` parameter (can be `mut` or immutable) |
| **Return Type** | Must return `Result<State>` |
| **Module** | Can be in any module, including separate files |

### ✅ Valid

```rust
#[register_handler]
pub async fn simple_handler(state: State) -> Result<State> {
    Ok(state)
}

#[register_handler]
pub async fn mutable_handler(mut state: State) -> Result<State> {
    state.set("key", json!("value"));
    Ok(state)
}
```

### ❌ Invalid

```rust
// ❌ Not public
#[register_handler]
async fn private_handler(state: State) -> Result<State> {
    Ok(state)
}

// ❌ Not async
#[register_handler]
pub fn sync_handler(state: State) -> Result<State> {
    Ok(state)
}

// ❌ Wrong return type
#[register_handler]
pub async fn wrong_return(state: State) -> State {
    state
}

// ❌ Too many parameters
#[register_handler]
pub async fn too_many_params(state: State, extra: String) -> Result<State> {
    Ok(state)
}
```

---

## How It Works Behind the Scenes

The `#[register_handler]` macro uses Rust's `inventory` crate to create a global registry:

```
1. Your code:
   #[register_handler]
   pub async fn my_handler(state: State) -> Result<State> { ... }

2. Macro expansion:
   pub async fn my_handler(state: State) -> Result<State> { ... }
   
   inventory::submit! {
       flowgentra_ai::core::agent::HandlerEntry::new(
           "my_handler",  // Auto-derived from function name
           Arc::new(|state| Box::pin(my_handler(state)))
       )
   }

3. At runtime:
   When from_config_path() is called, it collects all registered
   handlers and makes them available to the agent.
```

This is a **compile-time macro**, so there's zero runtime overhead!

---

## Common Patterns

### Pattern 1: Handlers in Separate Files

You can organize handlers in separate modules:

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
pub async fn validate_input(mut state: State) -> Result<State> {
    // handler logic
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

mod handlers;

#[tokio::main]
async fn main() -> Result<()> {
    let mut agent = from_config_path("config.yaml")?;
    let result = agent.run().await?;
    Ok(())
}
```

All handlers are automatically discovered!

### Pattern 2: Conditional Handler Registration

You can use Rust's feature flags to conditionally register handlers:

**Cargo.toml:**
```toml
[features]
default = []
advanced-features = []
```

**handlers.rs:**
```rust
#[register_handler]
pub async fn basic_handler(state: State) -> Result<State> {
    Ok(state)
}

#[cfg(feature = "advanced-features")]
#[register_handler]
pub async fn advanced_handler(state: State) -> Result<State> {
    Ok(state)
}
```

Build with features:
```bash
cargo run --features advanced-features
```

### Pattern 3: Error Handling in Handlers

Always return `Result<State>`:

```rust
#[register_handler]
pub async fn process_data(state: State) -> Result<State> {
    // Option 1: Return error if required field missing
    let input = state.get("input")
        .ok_or_else(|| FlowgentraError::StateError(
            "input field required".to_string()
        ))?;
    
    // Option 2: Use ? operator for automatic error propagation
    let value = state.get_str("value")?;
    
    // Option 3: Handle error and continue
    let optional_field = state.get("optional")
        .ok_or_else(|| FlowgentraError::StateError("not found".to_string()))
        .unwrap_or_else(|_| json!(null));
    
    Ok(state)
}
```

### Pattern 4: Using Other Clients in Handlers

Get LLM, RAG, and MCP clients from state:

```rust
#[register_handler]
pub async fn smart_handler(state: State) -> Result<State> {
    // Get LLM client
    let llm = state.get_llm_client()?;
    
    // Get RAG client
    let rag = state.get_rag_client()?;
    
    // Get MCP client
    let mcp = state.get_mcp_client()?;
    
    // Use them
    let llm_response = llm.chat(messages).await?;
    let docs = rag.retrieve(query, 5).await?;
    let tool_result = mcp.execute_tool(tool_name, args).await?;
    
    Ok(state)
}
```

### Pattern 5: Multiple Handlers in One File

You can register multiple handlers in the same file:

```rust
use flowgentra_ai::prelude::*;

#[register_handler]
pub async fn step1(mut state: State) -> Result<State> {
    state.set("step1_complete", json!(true));
    Ok(state)
}

#[register_handler]
pub async fn step2(mut state: State) -> Result<State> {
    state.set("step2_complete", json!(true));
    Ok(state)
}

#[register_handler]
pub async fn step3(mut state: State) -> Result<State> {
    state.set("step3_complete", json!(true));
    Ok(state)
}
```

---

## Troubleshooting

### "Handler not found" Error

**Problem:** You reference a handler in config.yaml that doesn't exist.

**Solution:** Make sure:
1. The handler function is decorated with `#[register_handler]`
2. The handler name in config matches the **function name** exactly (case-sensitive)
3. The module containing the handler is imported in main.rs or a parent module

```rust
// ✅ This works
mod handlers;  // Module is imported

#[tokio::main]
async fn main() -> Result<()> {
    let agent = from_config_path("config.yaml")?;  // Can find handlers
    Ok(())
}
```

```rust
// ❌ This doesn't work - module not imported
// Missing: mod handlers;

#[tokio::main]
async fn main() -> Result<()> {
    let agent = from_config_path("config.yaml")?;  // Can't find handlers!
    Ok(())
}
```

### Compiler Error: "expected async fn"

**Problem:** Your handler is not async.

**Solution:** Add `async` keyword:

```rust
// ❌ Wrong
#[register_handler]
pub fn my_handler(state: State) -> Result<State> {
    Ok(state)
}

// ✅ Correct
#[register_handler]
pub async fn my_handler(state: State) -> Result<State> {
    Ok(state)
}
```

### Compiler Error: "cannot find type in this scope"

**Problem:** You're missing the prelude import.

**Solution:** Import the prelude at the top of your handler file:

```rust
// ✅ Always include this
use flowgentra_ai::prelude::*;
```

### Multiple Handlers with Same Name

**Problem:** Two handlers with the same function name (causes conflicts).

**Solution:** Give handlers unique names:

```rust
// ❌ Conflict - both named "validate"
#[register_handler]
pub async fn validate(state: State) -> Result<State> { /* ... */ }

#[register_handler]
pub async fn validate(state: State) -> Result<State> { /* ... */ }

// ✅ Unique names
#[register_handler]
pub async fn validate_input(state: State) -> Result<State> { /* ... */ }

#[register_handler]
pub async fn validate_output(state: State) -> Result<State> { /* ... */ }
```

---

## Complete Example

Here's a complete working example:

**handlers.rs:**
```rust
use flowgentra_ai::prelude::*;
use serde_json::json;

/// Validates user input
#[register_handler]
pub async fn validate_input(state: State) -> Result<State> {
    println!("🔍 Validating input...");
    
    let input = state
        .get("input")
        .ok_or_else(|| FlowgentraError::StateError(
            "input required".to_string()
        ))?
        .as_str()
        .ok_or_else(|| FlowgentraError::StateError(
            "input must be string".to_string()
        ))?
        .to_string();
    
    if input.is_empty() {
        return Err(FlowgentraError::ValidationError(
            "input cannot be empty".to_string(),
        ));
    }
    
    state.set("validated_input", json!(input));
    println!("✅ Input validated");
    Ok(state)
}

/// Processes with LLM
#[register_handler]
pub async fn process_with_llm(state: State) -> Result<State> {
    println!("🤖 Processing with LLM...");
    
    let input = state.get_str("validated_input")?;
    let llm = state.get_llm_client()?;
    
    let response = llm.chat(vec![
        Message::system("You are helpful."),
        Message::user(&input),
    ]).await?;
    
    state.set("llm_response", json!(response.content));
    println!("✅ LLM processing complete");
    Ok(state)
}

/// Formats output
#[register_handler]
pub async fn format_output(state: State) -> Result<State> {
    println!("📝 Formatting output...");
    
    let response = state.get_str("llm_response")?;
    let output = format!("Final Answer: {}", response);
    
    state.set("final_output", json!(output));
    println!("✅ Output formatted");
    Ok(state)
}
```

**config.yaml:**
```yaml
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}

state_schema:
  input: { type: string }
  validated_input: { type: string }
  llm_response: { type: string }
  final_output: { type: string }

graph:
  nodes:
    - name: "step1"
      handler: "validate_input"
    - name: "step2"
      handler: "process_with_llm"
    - name: "step3"
      handler: "format_output"

  edges:
    - from: START
      to: step1
    - from: step1
      to: step2
    - from: step2
      to: step3
    - from: step3
      to: END
```

**main.rs:**
```rust
use flowgentra_ai::prelude::*;
use serde_json::json;

mod handlers;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Starting agent...\n");
    
    let mut agent = from_config_path("config.yaml")?;
    agent.state.set("input", json!("What is Rust?"));
    
    let result = agent.run().await?;
    
    if let Some(output) = result.get("final_output") {
        println!("\n{}", output);
    }
    
    Ok(())
}
```

Run it:
```bash
export OPENAI_API_KEY="sk-..."
cargo run
```

---

## When NOT to Use the Macro

You don't need `#[register_handler]` if:

1. **You're building the graph manually** (without config file):
   ```rust
   let mut graph = GraphBuilder::new()
       .add_node("step1", my_handler) // Direct reference
       .add_edge("START", "step1")
       .build()?;
   ```

2. **You're using predefined agents** (no custom handlers):
   ```rust
   let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
       .build()?;
   ```

In these cases, the macro is optional. But it's always fine to use it!

---

## Summary

| Concept | Details |
|---------|---------|
| **Purpose** | Auto-register handlers for discovery from config files |
| **Syntax** | `#[register_handler]` above your handler function |
| **Requirements** | `pub`, `async`, `State` parameter, `Result<State>` return |
| **Config Reference** | Use function name in `config.yaml` |
| **Discovery** | Via `from_config_path("config.yaml")` |
| **Module Structure** | Works with any module structure |

Start using `#[register_handler]` to make handler registration automatic and painless!
