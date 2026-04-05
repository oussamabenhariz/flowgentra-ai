# FlowgentraAI Quick Start

Get a working agent running in 5 minutes. No theory, just results.

## What You'll Build

A simple agent that answers questions, then a graph-based workflow with state management.

## Step 1: Create a Project

```bash
cargo new my_agent
cd my_agent
```

## Step 2: Add Dependencies

```toml
[package]
name = "my_agent"
version = "0.1.0"
edition = "2021"

[dependencies]
flowgentra-ai = "0.1"
tokio = { version = "1", features = ["full"] }
serde_json = "1.0"
```

Run `cargo build` (takes a minute the first time).

## Step 3: Create Your Agent

### Option A: Simple Agent from Config (Recommended Start)

Use a configuration file for flexibility:

```rust
use flowgentra_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Load agent from config.yaml
    let agent = from_config_path("config.yaml")?;
    
    // Create dynamic state
    let state = DynState::new();
    state.set("input", serde_json::json!("What is Rust?"));
    
    // Run the agent
    let output = agent.run(&state).await?;
    println!("Agent response: {}", output);
    
    Ok(())
}
```

### Option B: StateGraph (Full Control)

For custom workflows with typed state and conditional routing:

```rust
use flowgentra_ai::prelude::*;
use serde_json::json;

// Define node functions
async fn greet(state: &DynState) -> Result<DynState> {
    let name = state.get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("World");
    
    let greeting = format!("Hello, {}!", name);
    state.set("greeting", json!(greeting));
    Ok(state.clone())
}

async fn format_output(state: &DynState) -> Result<DynState> {
    let greeting = state.get("greeting")
        .and_then(|v| v.as_str())
        .unwrap_or("???");
    
    state.set("output", json!(format!("[{}]", greeting)));
    Ok(state.clone())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Build the graph
    let graph = StateGraphBuilder::new()
        .add_node("greet", Box::new(FunctionNode::new(greet)))
        .add_node("format", Box::new(FunctionNode::new(format_output)))
        .set_entry_point("greet")
        .add_edge("greet", "format")
        .add_edge("format", "__end__")
        .compile()?;

    // Create state and run
    let state = DynState::new();
    state.set("name", json!("Alice"));

    let result = graph.invoke(state).await?;
    println!("{}", result.get("output").unwrap());
    // Prints: ["Hello, Alice!"]

    Ok(())
}
```

## Step 4: Create config.yaml (Optional, for Option A)

```yaml
name: MyAgent
handlers: []
```

See [configuration/CONFIG_GUIDE.md](./configuration/CONFIG_GUIDE.md) for full configuration options.

## Step 5: Set Your API Key

```bash
export OPENAI_API_KEY="sk-your-actual-key-here"
```

## Step 6: Run It

```bash
cargo run
```

## Core Concepts

### StateGraph

The fundamental building block. A directed acyclic graph where:
- **Nodes** are async functions that process state updates
- **Edges** define transitions (fixed or conditional)
- **State** flows through the graph as updates

### DynState

A flexible, dynamic key-value store for state:
```rust
use flowgentra_ai::prelude::*;
use serde_json::json;

let state = DynState::new();
state.set("user_input", json!("hello"));
state.set("count", json!(42));

assert_eq!(state.get("user_input"), Some(json!("hello")));
```

### Conditional Routing

Route execution based on state values at runtime:

```rust
let graph = StateGraphBuilder::new()
    .add_node("classify", Box::new(FunctionNode::new(classify)))
    .add_node("simple_path", Box::new(FunctionNode::new(simple)))
    .add_node("complex_path", Box::new(FunctionNode::new(complex)))
    .set_entry_point("classify")
    .add_conditional_edge("classify", |state| {
        let complexity = state.get("score")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        
        if complexity > 5 {
            Ok("complex_path".to_string())
        } else {
            Ok("simple_path".to_string())
        }
    })
    .add_edge("simple_path", "__end__")
    .add_edge("complex_path", "__end__")
    .compile()?;
```

---

## Next Steps by Complexity

---

## Next Steps by Complexity

### Add Conditional Routing (Shown Above)

Route to different branches based on state:
```rust
.add_conditional_edge("classify", |state| {
    // Return next node name
    let score = state.get("score").and_then(|v| v.as_i64()).unwrap_or(0);
    if score > 5 { Ok("complex".into()) } else { Ok("simple".into()) }
})
```

### Add Tools/LLM Integration

```rust
use flowgentra_ai::prelude::*;

// Define a tool the agent can use
let tool = ToolDefinition {
    name: "calculator".to_string(),
    description: "Perform math operations".to_string(),
    input_schema: serde_json::json!({
        "type": "object",
        "properties": {
            "expression": {"type": "string"}
        }
    }),
};

// Create an LLM client
let llm_config = LLMConfig {
    provider: LLMProvider::OpenAI,
    model: "gpt-4".to_string(),
    ..Default::default()
};
```

### Add Checkpointing

Save and resume state at key points:

```rust
use flowgentra_ai::prelude::*;

// File-based checkpoints for durable recovery
path: "./checkpoints",
create_missing: true,
```

### Add Message Graph (For Chat)

For multi-turn conversations with message history:

```rust
use flowgentra_ai::prelude::*;

let graph = MessageGraphBuilder::new()
    .add_node("llm", Box::new(FunctionNode::new(llm_node)))
    .add_node("tools", Box::new(FunctionNode::new(tool_node)))
    .set_entry_point("llm")
    .add_edge("llm", "tools")
    .add_edge("tools", "__end__")
    .compile()?;
```

---

## I Want to...

### Get Structured JSON Output

```rust
use flowgentra_ai::core::llm::{LLMConfig, ResponseFormat};

let config = LLMConfig::new(provider, model, api_key)
    .with_response_format(ResponseFormat::Json);
```

---

## Common Issues

**"Invalid API key"**
```bash
echo $OPENAI_API_KEY    # Check it's set
export OPENAI_API_KEY="sk-..."  # Set it
```

**"Handler not found"**
- Make sure handlers are decorated with `#[register_handler]`
- Ensure the module is imported in `main.rs`

**Slow first build?**
- Normal. First compile takes 1-2 min. Subsequent runs are fast.

---

## Where to Go Next

| Goal | Guide |
|------|-------|
| See every feature | [FEATURES.md](FEATURES.md) |
| Build graph workflows | [graph/README.md](graph/README.md) |
| Set up LLM providers | [llm/README.md](llm/README.md) |
| Manage state with reducers | [state/README.md](state/README.md) |
| Add vector search (RAG) | [rag/README.md](rag/README.md) |
| Advanced patterns | [development/DEVELOPER_GUIDE.md](development/DEVELOPER_GUIDE.md) |
