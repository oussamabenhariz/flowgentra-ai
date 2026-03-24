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

### Option A: Predefined Agent (Simplest)

```rust
use flowgentra_ai::core::agents::{AgentBuilder, AgentType};
use flowgentra_ai::core::state::SharedState;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
        .with_name("my_assistant")
        .with_llm_config("gpt-4")
        .build()?;

    let state = SharedState::new();
    state.set("input", json!("What is Rust?"));

    let mut agent = agent;
    agent.initialize(&mut state.clone())?;

    let response = agent.process("What is Rust?", &state)?;
    println!("Agent says: {}", response);

    Ok(())
}
```

### Option B: StateGraph (Recommended for Custom Workflows)

The `StateGraph` API gives you full control over nodes, edges, and state flow:

```rust
use flowgentra_ai::core::state_graph::StateGraphBuilder;
use flowgentra_ai::core::state::PlainState;
use flowgentra_ai::core::error::Result;

// Define node functions
async fn greet(mut state: PlainState) -> Result<PlainState> {
    let name = state.get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("World");
    state.set("greeting", serde_json::json!(format!("Hello, {}!", name)));
    Ok(state)
}

async fn format_output(mut state: PlainState) -> Result<PlainState> {
    let greeting = state.get("greeting")
        .and_then(|v| v.as_str())
        .unwrap_or("???");
    state.set("output", serde_json::json!(format!("[{}]", greeting)));
    Ok(state)
}

#[tokio::main]
async fn main() -> Result<()> {
    let graph = StateGraphBuilder::new()
        .add_fn("greet", greet)
        .add_fn("format", format_output)
        .set_entry_point("greet")
        .add_edge("greet", "format")
        .add_edge("format", "__end__")
        .compile()?;

    let mut state = PlainState::new();
    state.set("name", serde_json::json!("Alice"));

    let result = graph.run(state).await?;
    println!("{}", result.get("output").unwrap());
    // Prints: ["Hello, Alice!"]

    Ok(())
}
```

## Step 4: Set Your API Key

```bash
export OPENAI_API_KEY="sk-your-actual-key-here"
```

## Step 5: Run It

```bash
cargo run
```

---

## Three Agent Types

| Type | Best For | Key Feature |
|------|----------|-------------|
| **ZeroShotReAct** | General reasoning, open-ended questions | Thinks + acts without examples |
| **FewShotReAct** | Classification, pattern-based tasks | Learns from examples you provide |
| **Conversational** | Chatbots, multi-turn dialogue | Remembers conversation history |

```rust
// ZeroShotReAct
AgentBuilder::new(AgentType::ZeroShotReAct)

// FewShotReAct
AgentBuilder::new(AgentType::FewShotReAct)

// Conversational (with memory)
AgentBuilder::new(AgentType::Conversational)
    .with_memory_steps(20)
```

---

## Next Steps by Complexity

### Add Conditional Routing

```rust
let graph = StateGraphBuilder::new()
    .add_fn("classify", classify_input)
    .add_fn("simple", handle_simple)
    .add_fn("complex", handle_complex)
    .set_entry_point("classify")
    .add_conditional_edge("classify", |state| {
        let score = state.get("complexity")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if score > 5 { Ok("complex".into()) } else { Ok("simple".into()) }
    })
    .add_edge("simple", "__end__")
    .add_edge("complex", "__end__")
    .compile()?;
```

### Add State Reducers

Control how state fields merge when updated by multiple nodes:

```rust
use flowgentra_ai::core::reducer::{ReducerConfig, JsonReducer};

let reducers = ReducerConfig::default()
    .field("messages", JsonReducer::Append)      // Append to list
    .field("score", JsonReducer::Sum)             // Sum values
    .field("config", JsonReducer::DeepMerge);     // Deep merge objects
```

### Add Tools

```rust
use flowgentra_ai::core::agents::ToolSpec;

let calculator = ToolSpec::new("calculator", "Do math")
    .with_parameter("expression", "string")
    .required("expression");

let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_tool(calculator)
    .build()?;
```

### Add Checkpointing

```rust
use flowgentra_ai::core::state_graph::FileCheckpointer;

// File-based checkpoints for durable recovery
let checkpointer = FileCheckpointer::new("./checkpoints");
```

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
