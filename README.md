# FlowgentraAI: Build Intelligent Agents in Rust

<div align="center">
  <img src="logo/logo.png" alt="FlowgentraAI Logo" width="200">
</div>

**Making AI agents simple, powerful, and fun to build.**

FlowgentraAI is your toolkit for creating sophisticated AI agents that think, plan, and solve problems. Inspired by LangGraph but built for Rust's performance and safety.

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
flowgentra-ai = "0.1"
tokio = { version = "1", features = ["full"] }
```

### 2. Pick Your Agent Type

```rust
// Option 1: Predefined agent (simplest)
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_llm_config("gpt-4")
    .build()?;

// Option 2: With memory for conversations
let agent = AgentBuilder::new(AgentType::Conversational)
    .with_llm_config("gpt-4")
    .with_memory_steps(10)
    .build()?;

// Option 3: With tools
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_llm_config("gpt-4")
    .with_tool(calculator_tool)
    .with_mcp(web_search_mcp)
    .build()?;
```

### 3. Run It

```rust
let mut state = State::new();
state.set("input", json!("What is Rust?"));

let response = agent.process("What is Rust?", &state)?;
println!("{}", response);
```

**→ [See Full Quickstart](QUICKSTART.md)**

## 📚 Documentation  

| What | Where | Time |
|------|-------|------|
| **Getting Started** | [QUICKSTART.md](QUICKSTART.md) | 5 min |
| **Feature Guide** | [FEATURES.md](FEATURES.md) | 15 min |
| **Understanding Memory, Planner, Evaluation** | [FEATURES.md](FEATURES.md) | 20 min |
| **Config Setup** | [CONFIG_GUIDE.md](CONFIG_GUIDE.md) | 20 min |
| **Advanced Patterns** | [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | 30 min |
| **Using Tools (MCP)** | [examples/MCP_AGENTS_GUIDE.md](flowgentra-ai/examples/MCP_AGENTS_GUIDE.md) | 15 min |

  edges:
    - from: START
      to: process_input

    - from: process_input
      to: generate_response

    - from: generate_response
      to: END

# Define what data flows through your agent
state_schema:
  input:
    type: string
    description: "User input"
  response:
    type: string
    description: "Generated response"
```

### 3. Implement Your Handlers

Create `handlers.rs`:

```rust
use flowgentra_ai::prelude::*;
use serde_json::json;

pub async fn process_input(mut state: State) -> Result<State> {
    // Extract and validate user input
    if let Some(input) = state.get("input") {
        if let Some(text) = input.as_str() {
            println!("Processing input: {}", text);
            state.set("input_validated", json!(true));
        }
    }
    Ok(state)
}

pub async fn generate_response(mut state: State) -> Result<State> {
    // Use LLM to generate response
    let input = state.get("input")
        .and_then(|v| v.as_str())
        .unwrap_or("Help me!");
    
    // Your LLM call here
    let response = format!("Response to: {}", input);
    state.set("response", json!(response));
    
    Ok(state)
}

// Important: Create a handler module reference for the agent
pub mod handlers {
    pub use super::{process_input, generate_response};
}
```

### 4. Run Your Agent

In `main.rs`:

```rust
use flowgentra_ai::prelude::*;
use serde_json::json;

mod handlers;

#[tokio::main]
async fn main() -> Result<()> {
    // Load agent from config.yaml
    let mut agent = Agent::from_config_with_handlers(
        "config.yaml",
        &handlers,
    )?;

    // Prepare initial state
    let mut state = State::new();
    state.set("input", json!("What is Rust?"));

    // Run the agent
    let result = agent.run(state).await?;

    // Access results
    if let Some(response) = result.get("response") {
        println!("Agent responded: {}", response);
    }

    Ok(())
}
```

## 📖 Core Concepts

### Graph Structure

An FlowgentraAI agent is fundamentally a **directed graph** where:

- **Nodes** = Computational steps (handlers)
- **Edges** = Connections between nodes with optional conditions
- **State** = Data passed between nodes
- **START/END** = Special nodes marking workflow boundaries

### Simple Flow Example

```
START 
  ↓
process_input (validate user input)
  ↓
check_complexity (is this a complex query?)
  ├─ YES → gather_info → analyze
  └─ NO  → analyze
  ↓
generate_response (create response)
  ↓
END
```

### Handlers

Handlers are async functions that receive state and return updated state:

```rust
pub async fn my_handler(mut state: State) -> Result<State> {
    // 1. Read from state
    let input = state.get("input");
    
    // 2. Do some work
    let result = process_data(input);
    
    // 3. Update state
    state.set("output", json!(result));
    
    Ok(state)
}
```

### Conditional Routing

Control workflow flow with conditions:

```yaml
edges:
  - from: plan_query
    to: complex_query_handler
    condition: is_complex_query
    
  - from: plan_query
    to: simple_query_handler  
    condition: is_simple_query
```

Implement conditions in your handlers module:

```rust
pub fn is_complex_query(state: &State) -> bool {
    state.get("complexity_score")
        .and_then(|v| v.as_i64())
        .map(|score| score > 70)
        .unwrap_or(false)
}
```

### Parallel Execution (DAG-Based)

Multiple nodes in the same frontier run **concurrently** (Rust/tokio). Use list syntax for parallel targets:

```yaml
edges:
  - from: START
    to: [analyze_logs, analyze_pcap]   # Run in parallel
  - from: analyze_logs
    to: merge
  - from: analyze_pcap
    to: merge
  - from: merge
    to: END
```

Or multiple edges from one node (same effect):

```yaml
  - from: parse_code
    to: check_style
  - from: parse_code
    to: check_security
  - from: parse_code
    to: analyze_complexity
```

## 🔌 LLM Integration

### Setting Up LLMs

Configure any supported LLM provider in your `config.yaml`:

```yaml
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7
  api_key: ${OPENAI_API_KEY}
  timeout: 30
```

**Supported Providers:**
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude)
- Mistral
- Groq
- Azure OpenAI
- Ollama (local)

### Using LLMs in Handlers

To use the LLM in a handler, access it through the state:

```rust
pub async fn generate_response(mut state: State) -> Result<State> {
    let input = state.get("input").and_then(|v| v.as_str()).unwrap_or("");
    
    // Get the LLM client from state
    let llm = state.get_llm_client()?;
    
    // Create messages and call LLM
    let response = llm.chat(vec![
        Message {
            role: "user".to_string(),
            content: input.to_string(),
        }
    ]).await?;
    
    state.set("response", json!(response.content));
    Ok(state)
}
```

### Provider Fallbacks

Set up automatic fallback chains:

```yaml
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
  fallbacks:
    - provider: anthropic
      model: claude-3-opus
      api_key: ${ANTHROPIC_API_KEY}
    - provider: ollama
      model: mistral
      base_url: "http://localhost:11434"
```

## 🔍 RAG (Retrieval-Augmented Generation)

### Configuration

Enable RAG in your config:

```yaml
rag:
  enabled: true
  vector_store:
    type: pinecone
    index_name: "my-index"
    environment: "us-west-2-aws"
    api_key: ${PINECONE_API_KEY}
  embedding_model: openai
  embedding_dimension: 1536
  retrieval_strategy: semantic
  chunk_size: 1024
  top_k: 5
```

### Using RAG in Handlers

```yaml
- name: retrieve_context
  handler: handlers::retrieve_context
  uses_rag: true  # ← Enable RAG for this handler
  timeout: 15
```

In your handler:

```rust
pub async fn retrieve_context(mut state: State) -> Result<State> {
    let query = state.get("input").and_then(|v| v.as_str()).unwrap_or("");
    
    // Retrieve similar documents from vector store
    let docs = rag_client.retrieve(query, 5).await?;
    
    state.set("retrieved_docs", json!(docs));
    Ok(state)
}
```

## 🛠️ MCP (Model Context Protocol)

### Defining Tools

Configure external tools in `config.yaml`:

```yaml
mcp:
  enabled: true
  tools:
    - name: web_search
      description: "Search the web for information"
      type: external
      endpoint: "http://api.example.com/search"
      
    - name: calculator
      description: "Perform calculations"
      type: builtin
```

### Using Tools in Handlers

```yaml
- name: gather_info
  handler: handlers::gather_info
  uses_mcp: true  # ← Enable MCP tools
  mcp_tools:
    - web_search
    - calculator
```

In your handler:

```rust
pub async fn gather_info(mut state: State) -> Result<State> {
    // Use MCP tools to gather information
    let search_query = "latest Rust news";
    let results = mcp_client.execute_tool("web_search", 
        json!({"query": search_query})
    ).await?;
    
    state.set("search_results", results);
    Ok(state)
}
```

## 🔧 State Management

### Defining State Schema

Document your state structure in `config.yaml`:

```yaml
state_schema:
  input:
    type: string
    description: "User query"
  input_validated:
    type: boolean
    description: "Whether input passed validation"
  retrieved_docs:
    type: Array<Document>
    description: "Retrieved context"
  response:
    type: string
    description: "Final response"
  metadata:
    type: object
    description: "Additional metadata"
```

### Working with State

In your handlers:

```rust
pub async fn my_handler(mut state: State) -> Result<State> {
    // Get values
    let input = state.get("input");
    
    // Set values
    state.set("key", json!(value));
    
    // Access nested values
    if let Some(obj) = state.get_mut("metadata") {
        if let Some(map) = obj.as_object_mut() {
            map.insert("processed".to_string(), json!(true));
        }
    }
    
    Ok(state)
}
```

## 📊 Middleware Pipeline

Add preprocessing/postprocessing middleware:

```yaml
middleware:
  - name: logging
    enabled: true
    
  - name: rate_limiting
    enabled: true
    rpm: 60  # 60 requests per minute
    
  - name: cache
    enabled: true
    ttl: 3600  # 1 hour cache
```

## 🏥 Health Monitoring

Monitor your agent's health:

```yaml
health:
  enabled: true
  check_interval: 30
  checks:
    - llm_connectivity
    - rag_availability
    - mcp_tools
    - memory_usage
```

Query health status:

```rust
let health = agent.health().await?;
if !health.is_healthy {
    eprintln!("Agent is unhealthy: {:?}", health.failures);
}
```

## 📡 Observability & Tracing

Enable distributed tracing:

```yaml
observability:
  tracing_enabled: true
  trace_level: debug
  spans:
    - name: handler_execution
      sample_rate: 1.0
    - name: llm_calls
      sample_rate: 0.5
  exporters:
    - type: jaeger
      endpoint: "http://localhost:14268/api/traces"
```

## 📁 Creating a config.yaml File

The `config.yaml` file is the heart of your FlowgentraAI agent. Here's a complete guide:

### Required Fields

Every config must have:

```yaml
name: "agent_name"              # Unique identifier
description: "What it does"     # Human-readable description

llm:                            # LLM provider config
  provider: openai              # or: anthropic, mistral, groq, ollama, azure
  model: gpt-4                  # Model identifier
  api_key: ${OPENAI_API_KEY}    # Environment variable substitution

graph:                          # Workflow definition
  nodes:                        # List of computational steps
    - name: node_name
      handler: module::function # Path to handler function
  
  edges:                        # Connections between nodes
    - from: START
      to: first_node
    - from: first_node
      to: END
```

### Optional Fields

```yaml
# Optionally declare state schema (for documentation)
state_schema:
  field_name:
    type: string
    description: "Field description"
  another_field: "type - description"

# Add middleware
middleware:
  - name: logging
    enabled: true

# Enable RAG
rag:
  enabled: true
  vector_store:
    type: pinecone
    # ...

# Enable MCP tools
mcp:
  enabled: true
  tools:
    - name: tool_name
      # ...

# Health checks
health:
  enabled: true
  check_interval: 30
  checks:
    - llm_connectivity
    - rag_availability
```

### Environment Variables

Use `${VAR_NAME}` for sensitive information:

```yaml
llm:
  api_key: ${OPENAI_API_KEY}        # Reads OPENAI_API_KEY environment variable

rag:
  vector_store:
    api_key: ${PINECONE_API_KEY}    # Reads PINECONE_API_KEY environment variable
```

Before running your agent, set environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export PINECONE_API_KEY="your-key"
cargo run
```

## 🏗️ Project Structure

Recommended layout for an FlowgentraAI project:

```
my_agent_project/
├── Cargo.toml
├── config.yaml              # ← Agent configuration (REQUIRED)
├── src/
│   ├── main.rs              # Entry point
│   ├── handlers.rs          # Handler implementations
│   └── lib.rs               # Library exports
├── examples/
│   └── basic_example.rs     # Example usage
└── README.md
```

## 📚 Examples

See the [examples/](examples/) directory for complete working examples:

- **comprehensive_agent** - Full-featured agent with RAG, MCP, LLM
- More examples coming soon!

Run examples with:

```bash
cargo run --example comprehensive_agent
```

## 🔗 API Documentation

Full API documentation is available via rustdoc. Generate and open it locally with:

```bash
cargo doc --open
```

When published to crates.io, docs are also available at [docs.rs/flowgentra-ai](https://docs.rs/flowgentra-ai).

See [CHANGELOG.md](CHANGELOG.md) for version history and migration notes.

Key types to reference:

- `Agent` - Main API for creating and running agents
- `State` - Shared state container (`get_str`, `require_str`, `get_i64`, `get_bool`)
- `Message`, `MessageRole` - LLM conversation messages (user, assistant, system, tool)
- `Graph` - Workflow structure
- `AgentConfig` - Configuration parsing
- `Handler` - Handler function type

## ⚙️ Configuration Reference

### LLM Providers

#### OpenAI

```yaml
llm:
  provider: openai
  model: gpt-4          # or gpt-3.5-turbo, gpt-4-turbo, etc.
  temperature: 0.7
  max_tokens: 2000
  api_key: ${OPENAI_API_KEY}
  timeout: 30
```

#### Anthropic

```yaml
llm:
  provider: anthropic
  model: claude-3-opus-20240229
  temperature: 0.7
  api_key: ${ANTHROPIC_API_KEY}
```

#### Mistral

```yaml
llm:
  provider: mistral
  model: mistral-large
  api_key: ${MISTRAL_API_KEY}
```

#### Ollama (Local)

```yaml
llm:
  provider: ollama
  model: mistral
  base_url: "http://localhost:11434"
```

### Vector Stores (RAG)

#### Pinecone

```yaml
rag:
  vector_store:
    type: pinecone
    index_name: "my-index"
    environment: "us-west-2-aws"
    api_key: ${PINECONE_API_KEY}
```

#### Weaviate

```yaml
rag:
  vector_store:
    type: weaviate
    url: "http://localhost:8080"
    api_key: ${WEAVIATE_API_KEY}
```

## 🐛 Troubleshooting

### Agent fails to load config

**Error:** `Failed to load config.yaml`

**Solution:** 
- Ensure `config.yaml` exists in the path provided
- Check YAML syntax (indentation, colons, etc.)
- Verify environment variables are set: `echo $OPENAI_API_KEY`

### Handlers not found

**Error:** `Handler 'my_handler' not found`

**Solution:**
- Ensure handler function is public: `pub async fn my_handler(...)`
- Check handler path in config matches rust module path: `handlers::my_handler`
- Verify handler signature matches: `async fn(State) -> Result<State>`

### LLM API key errors

**Error:** `Invalid API key` 

**Solution:**
- Verify API key is valid: `echo $OPENAI_API_KEY`
- Ensure env variable name matches config: `api_key: ${OPENAI_API_KEY}`
- Load `.env` file: Create `.env` file with `OPENAI_API_KEY=sk-...`

### State validation errors

**Error:** `State validation failed`

**Solution:**
- Check that handlers set all required state fields
- Review `state_schema` in config for required fields
- Use `state.set()` to add fields in handlers

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review the examples directory

---

**Happy building! 🚀**
