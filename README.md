# FlowgentraAI: Build Intelligent Agents in Rust

<div align="center">
  <img src="logo/logo.png" alt="FlowgentraAI Logo" width="200">
</div>

**Making AI agents simple, powerful, and fun to build.**

FlowgentraAI is your toolkit for creating sophisticated AI agents that think, plan, and solve problems.

## 💡 What Can You Build?

- **Research agents** that gather and synthesize information
- **Support bots** that remember conversations and help customers  
- **Auto-healing systems** that diagnose and fix themselves
- **Planning engines** that decide what to do next (dynamically!)
- **Analysis pipelines** that process documents and data
- **Hybrid workflows** mixing human decisions with AI reasoning

## ✨ Key Features

### 🤖 Predefined Agents
No complex graphs needed - just pick your agent type:
- **ZeroShotReAct** - General reasoning agent
- **FewShotReAct** - Agent that learns from examples
- **Conversational** - Chat agent with memory

### 💾 Memory & Checkpointing
Your agents remember:
- Conversation history (for multi-turn chats)
- Execution state (resume interrupted workflows)

### 🎓 Auto-Evaluation
Agents grade their own work and improve:
- Automatic confidence scoring
- Self-correction on low quality  
- Intelligent retries

### 🧠 Dynamic Planning
LLM decides what to do next (no hardcoding):
- Flexible, adaptive workflows
- Responds to changing conditions
- Complex multi-step reasoning

### 🔌 Tools & Integrations
Connect to anything:
- **Local Tools** - Run functions directly
- **MCP Services** - HTTP, stdio, or Docker services
- **Vector Stores** - Pinecone, Weaviate, Chroma
- **LLMs** - OpenAI, Anthropic, Mistral, Groq, Ollama, Azure

### 📊 Multi-LLM Support
- Fallback chains (try OpenAI, fall back to Anthropic)
- Provider-agnostic handlers
- Easy switching

## 🚀 Quick Start

### 1. Add to Cargo.toml

```toml
[dependencies]
flowgentra-ai = "0.2.5"
tokio = { version = "1", features = ["full"] }
serde_json = "1.0"
```

### 2. Build a Graph

The core API is `StateGraph` - build workflows as directed graphs:

```rust
use flowgentra_ai::prelude::*;

// 1. Define your state
#[derive(State, Default, Clone)]
struct GreetState {
    name:     String,
    greeting: String,
}

// 2. Write a node
#[node]
async fn greet(state: &mut GreetState) -> Result<()> {
    state.greeting = format!("Hello, {}!", state.name);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // 3. Build and run
    let graph = StateGraph::<GreetState>::builder()
        .add_node("greet", greet)
        .set_entry("greet")
        .build()?;

    let result = graph.invoke(GreetState {
        name: "Alice".into(),
        ..Default::default()
    }).await?;

    println!("{}", result.greeting);
    // Prints: "Hello, Alice!"
    Ok(())
}
```

### 3. Load from Config (Optional)

For more complex agents, use a config file:

```rust
let agent = from_config_path("config.yaml")?;
let state = DynState::new();
let output = agent.run(&state).await?;
```

**→ [See Full Documentation](./flowgentra-ai/docs/README.md)**

## 📚 Documentation  

**Hosted docs:** [oussamabenhariz.github.io/flowgentra-ai-docs](https://oussamabenhariz.github.io/flowgentra-ai-docs/) — guides, API reference, and examples for both Rust and Python.

| What | Where | Time |
|------|-------|------|
| **Getting Started** | [flowgentra-ai/docs/QUICKSTART.md](flowgentra-ai/docs/QUICKSTART.md) | 5 min |
| **Feature Guide** | [flowgentra-ai/docs/FEATURES.md](flowgentra-ai/docs/FEATURES.md) | 15 min |
| **State Management** | [flowgentra-ai/docs/state/README.md](flowgentra-ai/docs/state/README.md) | 15 min |
| **Graph Engine** | [flowgentra-ai/docs/graph/README.md](flowgentra-ai/docs/graph/README.md) | 20 min |
| **Configuration Guide** | [flowgentra-ai/docs/configuration/CONFIG_GUIDE.md](flowgentra-ai/docs/configuration/CONFIG_GUIDE.md) | 20 min |
| **Advanced Patterns** | [flowgentra-ai/docs/DEVELOPER_GUIDE.md](flowgentra-ai/docs/DEVELOPER_GUIDE.md) | 30 min |
| **Full Documentation Hub** | [flowgentra-ai/docs/README.md](flowgentra-ai/docs/README.md) | Browse all topics |

## 📖 Core Concepts

### StateGraph

The core of FlowgentraAI is a **directed acyclic graph (DAG)** where:
- **Nodes** = Async functions that process state
- **Edges** = Connections defining execution flow  
- **State** = Key-value data flowing through the graph
- **Conditional Edges** = Runtime routing based on state

### State

Use `#[derive(State)]` for typed state. For config-driven agents, `DynState` (JSON key-value) is also available:

```rust
// Typed (recommended for code-driven graphs)
#[derive(State, Default, Clone)]
struct MyState {
    user_input: String,
    score: u32,
}

// Dynamic (for config-driven / YAML agents)
let state = DynState::new();
state.set("user_input", json!("Hello"));
state.set("score", json!(42));
```

### Minimal Example

```rust
use flowgentra_ai::prelude::*;

#[derive(State, Default, Clone)]
struct MyState { output: String }

#[node]
async fn transform(state: &mut MyState) -> Result<()> {
    state.output = "transformed".into();
    Ok(())
}

let graph = StateGraph::<MyState>::builder()
    .add_node("transform", transform)
    .set_entry("transform")
    .build()?;

let result = graph.invoke(MyState::default()).await?;
println!("{}", result.output); // "transformed"
```

## 🔗 Integration

FlowgentraAI supports:
- **LLM Providers**: OpenAI, Anthropic, Mistral, Groq, Azure, HuggingFace, Ollama
- **Vector Stores**: Pinecone, Qdrant, Weaviate, Chroma, In-Memory
- **Tools**: MCP Protocol (stdio, HTTP, Docker)
- **Memory**: Conversation history, Checkpointing

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Ready to build? Start with [flowgentra-ai/docs/QUICKSTART.md](flowgentra-ai/docs/QUICKSTART.md)** 🚀
