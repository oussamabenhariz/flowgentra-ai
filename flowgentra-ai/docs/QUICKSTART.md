````markdown
# FlowgentraAI Quick Start - Get Running in 5 Minutes

Let's build a working agent **right now**. No theory, just results.

## What You'll Build

A simple agent that can answer questions. No complex setup needed.

## Step 1: Create a Project

```bash
cargo new my_agent
cd my_agent
```

## Step 2: Add Dependencies

Edit `Cargo.toml`:

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

Run: `cargo build` (this might take a minute the first time)

## Step 3: Create Your Agent

Edit `src/main.rs`:

```rust
use flowgentra_ai::core::agents::{AgentBuilder, AgentType};
use flowgentra_ai::core::state::State;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple reasoning agent
    let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
        .with_name("my_assistant")
        .with_llm_config("gpt-4")
        .build()?;

    // Prepare state
    let mut state = State::new();
    state.set("input", json!("What is Rust?"));

    // Initialize agent
    let mut agent = agent;
    agent.initialize(&mut state)?;

    // Run it!
    let response = agent.process("What is Rust?", &state)?;
    println!("Agent says: {}", response);

    Ok(())
}
```

## Step 4: Set Your API Key

```bash
export OPENAI_API_KEY="sk-your-actual-key-here"
```

## Step 5: Run It!

```bash
cargo run
```

**That's it!** 🎉 You now have a working agent.

---

## What Just Happened?

1. ✅ Created a `ZeroShotReAct` agent (a reasoning agent template)
2. ✅ Configured it to use GPT-4
3. ✅ Passed it a question
4. ✅ It generated a thoughtful answer

## Next Steps

### Add Memory (Conversations)

```rust
let agent = AgentBuilder::new(AgentType::Conversational)
    .with_name("chat_bot")
    .with_llm_config("gpt-4")
    .with_memory_steps(20)  // Remember last 20 messages!
    .build()?;

// First turn
agent.process("Hi, I'm Alice", &state)?;

// Second turn - agent remembers you're Alice!
agent.process("What's my name?", &state)?;
```

### Add Tools

```rust
let calculator = ToolSpec::new("calculator", "Do math")
    .with_parameter("expression", "string")
    .required("expression");

let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_name("math_tutor")
    .with_llm_config("gpt-4")
    .with_tool(calculator)
    .build()?;

// Now agent can use the calculator!
agent.process("What is 123 + 456?", &state)?;
```

### Add External Services

```rust
let web_search = MCPConfig::builder()
    .name("web_search")
    .sse("http://localhost:8000")
    .timeout_secs(30)
    .build()?;

let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_llm_config("gpt-4")
    .with_mcp(web_search)
    .build()?;

// Agent can now search the web!
agent.process("What's trending on Twitter?", &state)?;
```

### Add Auto-Evaluation

Self-correcting agents:

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_llm_config("gpt-4")
    // ... other config ...
    .build()?;

// With evaluation enabled in config:
// - Agent grades its answer automatically
// - If quality is low, it retries with "improve your answer"
// - Returns only when confident
```

### Enable Checkpointing

Resume workflows:

```yaml
# In your config.yaml
memory:
  checkpointer:
    enabled: true
    checkpoint_dir: "./checkpoints"
```

Then in code:

```rust
// Save progress
let checkpoint = agent.save_checkpoint("workflow_123")?;

// Later, resume
agent.load_checkpoint("workflow_123")?;
agent.run(state).await?;
```

---

## Three Agent Types Available

### ZeroShotReAct (Simplest)
- General reasoning and problem-solving
- Works without examples
- Best for: Open-ended questions

```rust
AgentBuilder::new(AgentType::ZeroShotReAct)
```

### FewShotReAct (Smarter)
- Works better with examples
- Shows the LLM how you want it to behave
- Best for: Tasks where examples help

```rust
AgentBuilder::new(AgentType::FewShotReAct)
 // .add_example(...)
```

### Conversational (Most Human)
- Remembers chat history
- Multi-turn conversations
- Best for: Chatbots and assistants

```rust
AgentBuilder::new(AgentType::Conversational)
    .with_memory_steps(10)
```

---

## Common Issues

**Error: "Invalid API key"**
```bash
# Check it's set
echo $OPENAI_API_KEY

# Set it if needed
export OPENAI_API_KEY="sk-..."
```

**Error: "Handler not found"**
- Check you're using a predefined agent (no handlers needed for quick start)

**Slow on first run?**
- First compilation takes time (1-2 min)
- Subsequent runs are fast

---

## Learn More

- **Understanding all features**: [FEATURES.md](FEATURES.md)
- **Advanced: Memory, Evaluation, Planner**: [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- **Building custom logic**: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- **Adding tools and integrations**: [examples/MCP_AGENTS_GUIDE.md](flowgentra-ai/examples/MCP_AGENTS_GUIDE.md)

---

## You're Ready! 🚀

You've got a working agent. Now:
1. Try different agent types
2. Add tools
3. Export your results
4. Build something awesome!

Happy coding!


---

## Next Steps

### Add More Nodes

Extend your workflow by adding more nodes to `config.yaml`:

```yaml
graph:
  nodes:
    - name: validate
      handler: handlers::validate_input
    
    - name: search
      handler: handlers::search_documents
      uses_rag: true
    
    - name: analyze
      handler: handlers::analyze
    
    - name: respond
      handler: handlers::generate_response

  edges:
    - from: START
      to: validate
    - from: validate
      to: search
    - from: search
      to: analyze
    - from: analyze
      to: respond
    - from: respond
      to: END
```

### Add Conditions

Create branching logic:

```yaml
edges:
  - from: validate
    to: complex_handler
    condition: is_complex

  - from: validate
    to: simple_handler
    condition: is_simple

  - from: complex_handler
    to: respond

  - from: simple_handler
    to: respond
```

Add condition functions in handlers:

```rust
pub fn is_complex(state: &State) -> bool {
    state.get("complexity_score")
        .and_then(|v| v.as_i64())
        .map(|s| s > 70)
        .unwrap_or(false)
}

pub fn is_simple(state: &State) -> bool {
    !is_complex(state)
}
```

### Add Retrieval-Augmented Generation (RAG)

Enable semantic search:

```yaml
rag:
  enabled: true
  vector_store:
    type: pinecone
    index_name: "my-docs"
    environment: "us-west-2-aws"
    api_key: ${PINECONE_API_KEY}
  embedding_model: openai
  embedding_dimension: 1536
  top_k: 5
```

Use in handler:

```rust
pub async fn search_documents(mut state: State) -> Result<State> {
    let query = state.get("input")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    
    // Retrieve from vector store
    let docs = rag_client.retrieve(query, 5).await?;
    
    state.set("retrieved_docs", json!(docs));
    Ok(state)
}
```

### Add External Tools (MCP)

```yaml
mcp:
  enabled: true
  tools:
    - name: web_search
      description: "Search the web"
      type: external
      endpoint: "http://localhost:3000/search"

    - name: calculator
      description: "Do math"
      type: builtin
```

Use in handler:

```rust
pub async fn search_info(mut state: State) -> Result<State> {
    let query = state.get("input")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    
    let results = mcp_client.execute_tool(
        "web_search",
        json!({"query": query})
    ).await?;
    
    state.set("search_results", results);
    Ok(state)
}
```

---

## Common Tasks

### Change LLM Provider

**To Anthropic:**

```yaml
llm:
  provider: anthropic
  model: claude-3-opus-20240229
  api_key: ${ANTHROPIC_API_KEY}
```

**To Mistral:**

```yaml
llm:
  provider: mistral
  model: mistral-large
  api_key: ${MISTRAL_API_KEY}
```

**To Local Ollama:**

```yaml
llm:
  provider: ollama
  model: mistral
  base_url: "http://localhost:11434"
```

### Add Logging

```yaml
middleware:
  - name: logging
    enabled: true
    level: debug
```

### Cache Responses

```yaml
middleware:
  - name: cache
    enabled: true
    ttl: 3600  # 1 hour
```

### Limit Request Rate

```yaml
middleware:
  - name: rate_limiting
    enabled: true
    rpm: 60  # 60 requests per minute
```

### Monitor Health

```yaml
health:
  enabled: true
  check_interval: 30
  checks:
    - llm_connectivity
    - memory_usage
```

### Enable Debug Traces

```yaml
observability:
  tracing_enabled: true
  trace_level: debug
```

---

## Troubleshooting

**Error: `Failed to load config`**
- Ensure `config.yaml` exists in project root
- Check YAML syntax (no hard tabs, proper indentation)

**Error: `Handler 'handlers::process_input' not found`**
- Verify function name matches exactly
- Ensure function is `pub async fn`
- Make sure handler is in `src/handlers.rs`

**Error: `Invalid API key`**
- Check: `echo $OPENAI_API_KEY`
- Set key: `export OPENAI_API_KEY="sk-..."`
- Or create `.env` file with `OPENAI_API_KEY=sk-...`

**Error: `Timeout` or "Request failed"**
- Check internet connection
- Verify API key is valid
- Check provider status page

---

````
