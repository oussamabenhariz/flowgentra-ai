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
flowgentra-ai = "0.2"
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

// 1. Define typed state
#[derive(State, Default, Clone)]
struct GreetState {
    name:     String,
    greeting: String,
    output:   String,
}

// 2. Define nodes with the #[node] macro
#[node]
async fn greet(state: &mut GreetState) -> Result<()> {
    state.greeting = format!("Hello, {}!", state.name);
    Ok(())
}

#[node]
async fn format_output(state: &mut GreetState) -> Result<()> {
    state.output = format!("[{}]", state.greeting);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // 3. Build the graph
    let graph = StateGraph::<GreetState>::builder()
        .add_node("greet",  greet)
        .add_node("format", format_output)
        .set_entry("greet")
        .add_edge("greet", "format")
        .set_finish("format")
        .build()?;

    // 4. Run with initial state
    let result = graph.invoke(GreetState {
        name: "Alice".into(),
        ..Default::default()
    }).await?;

    println!("{}", result.output);
    // Prints: [Hello, Alice!]
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
#[derive(State, Default, Clone)]
struct RouterState { score: i64, result: String }

#[node] async fn classify(s: &mut RouterState)    -> Result<()> { Ok(()) }
#[node] async fn simple_path(s: &mut RouterState) -> Result<()> { s.result = "simple".into(); Ok(()) }
#[node] async fn complex_path(s: &mut RouterState)-> Result<()> { s.result = "complex".into(); Ok(()) }

let graph = StateGraph::<RouterState>::builder()
    .add_node("classify",     classify)
    .add_node("simple_path",  simple_path)
    .add_node("complex_path", complex_path)
    .set_entry("classify")
    .conditional_edge("classify", |state| {
        if state.score > 5 { "complex_path" } else { "simple_path" }
    })
    .set_finish("simple_path")
    .set_finish("complex_path")
    .build()?;
```

---

## Next Steps by Complexity

### Add Conditional Routing (Shown Above)

Route to different branches based on state:
```rust
.conditional_edge("classify", |state: &RouterState| {
    if state.score > 5 { "complex" } else { "simple" }
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

// Create an LLM
let client = LLM::from_config(LLMConfig::openai("gpt-4o"))?;
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

let graph = StateGraph::<MessageState>::builder()
    .add_node("llm",   llm_node)
    .add_node("tools", tool_node)
    .set_entry("llm")
    .add_edge("llm", "tools")
    .set_finish("tools")
    .build()?;
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
