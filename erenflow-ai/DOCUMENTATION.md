# ErenFlowAI Documentation Index

Welcome to ErenFlowAI! This is a LangGraph-inspired Rust library for building AI agents with graph-based workflows.

## 📚 Documentation Guide

### For New Users (Start Here!)

**[QUICKSTART.md](QUICKSTART.md)** ⚡ *5 minutes*
- Get a working agent running in 5 minutes
- Perfect if you're impatient and want to see results fast
- Includes complete working example

**[README.md](README.md)** 📖 *20 minutes*
- Overview of ErenFlowAI concepts
- Features and capabilities
- Architecture explanation
- Common patterns and examples

### For Config File Creation (REQUIRED!)

**[CONFIG_GUIDE.md](CONFIG_GUIDE.md)** ⚙️ *Complete Reference*
- **Why:** You MUST create a `config.yaml` file to use ErenFlowAI
- **What:** Complete guide to every config option
- **How:** Step-by-step walkthrough for creating your first config
- **Reference:** Provider configurations, vector stores, tools

**Key Concepts:**
- Minimal config example (get started with 20 lines)
- Full config example (with all features)
- Environment variable substitution (for API keys)
- Provider-specific configurations (OpenAI, Anthropic, Mistral, etc.)
- Vector store configs (Pinecone, Weaviate, Chroma)

### For Advanced Development

**[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** 👨‍💻 *Advanced Patterns*
- Writing custom handlers
- State management deep dive
- Conditional routing implementation
- Custom middleware
- Error handling patterns
- Testing and debugging
- Performance optimization

## 🎯 Quick Navigation

### I want to...

#### Get Started Immediately
→ [QUICKSTART.md](QUICKSTART.md)

#### Learn How to Create config.yaml
→ [CONFIG_GUIDE.md](CONFIG_GUIDE.md)

#### Understand the Whole Picture
→ [README.md](README.md)

#### Build Custom Handlers
→ [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)

#### Configure LLM Providers
→ [CONFIG_GUIDE.md#provider-configuration-examples](CONFIG_GUIDE.md)

#### Set Up RAG/Vector Stores
→ [CONFIG_GUIDE.md#vector-store-configuration-examples](CONFIG_GUIDE.md)

#### Add External Tools (MCP)
→ [CONFIG_GUIDE.md#7-mcp-configuration-optional](CONFIG_GUIDE.md)

#### Debug My Agent
→ [DEVELOPER_GUIDE.md#debugging](DEVELOPER_GUIDE.md)

#### Write Tests
→ [DEVELOPER_GUIDE.md#testing-agents](DEVELOPER_GUIDE.md)

## 📋 Key Concepts

### What is config.yaml?

The `config.yaml` file is **REQUIRED**. It defines:
- Your LLM provider and model
- Your workflow (nodes and edges)
- State schema
- RAG, MCP, middleware, health checks

**Without config.yaml, your agent cannot run.**

Create one with:
```bash
# Minimal example
cat > config.yaml << 'EOF'
name: "my_agent"
description: "My first agent"

llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}

graph:
  nodes:
    - name: process
      handler: handlers::process_input

  edges:
    - from: START
      to: process
    - from: process
      to: END
EOF
```

Then see [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for complete options.

### What is a Handler?

A handler is an async Rust function that processes state:

```rust
pub async fn my_handler(mut state: State) -> Result<State> {
    // Read from state
    let input = state.get("input");
    
    // Do some work
    let result = process(input);
    
    // Update state
    state.set("output", json!(result));
    
    Ok(state)
}
```

See [DEVELOPER_GUIDE.md#handler-development](DEVELOPER_GUIDE.md) for patterns.

### What is State?

State is a JSON object that flows through your agent:

```
START → Handler1 → Handler2 → Handler3 → END
  ↓         ↓         ↓         ↓       ↓
State0 → State1 → State2 → State3 → State4
```

Each handler receives the previous output state and returns the next state.

### How Do I Use LLMs?

In your config.yaml:
```yaml
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
```

Then mark handlers that use LLMs:
```yaml
- name: my_handler
  handler: handlers::generate_response
  uses_llm: true
```

See [CONFIG_GUIDE.md#llm-configuration-required](CONFIG_GUIDE.md) for all providers.

### How Do I Add RAG?

Enable in config.yaml:
```yaml
rag:
  enabled: true
  vector_store:
    type: pinecone
    index_name: "my-documents"
    api_key: ${PINECONE_API_KEY}
  embedding_model: openai
```

Mark handlers that use RAG:
```yaml
- name: retrieve_docs
  handler: handlers::retrieve_context
  uses_rag: true
```

See [CONFIG_GUIDE.md#6-rag-configuration-optional](CONFIG_GUIDE.md).

### How Do I Connect External Tools?

Use MCP in config.yaml:
```yaml
mcp:
  enabled: true
  tools:
    - name: web_search
      type: external
      endpoint: "http://api.example.com/search"
```

Then use in handlers:
```yaml
- name: search_handler
  handler: handlers::search
  uses_mcp: true
  mcp_tools:
    - web_search
```

See [CONFIG_GUIDE.md#7-mcp-configuration-optional](CONFIG_GUIDE.md).

## 🚀 Getting Started Path

1. **Read:** [QUICKSTART.md](QUICKSTART.md) (5 min)
2. **Create:** `config.yaml` following [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
3. **Write:** Handler functions in Rust
4. **Run:** Your first agent!
5. **Explore:** [README.md](README.md) for features
6. **Advance:** [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for patterns

## 📝 Documentation Files

| File | Purpose | Audience | Time |
|------|---------|----------|------|
| [QUICKSTART.md](QUICKSTART.md) | Get working in 5 minutes | New users | 5 min |
| [README.md](README.md) | Overview & features | Everyone | 20 min |
| [CONFIG_GUIDE.md](CONFIG_GUIDE.md) | Configuration reference | Config creators | 30 min |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Handler patterns & advanced | Developers | 1 hour |
| This file | Navigation & concepts | Everyone | 5 min |

## ❓ FAQ

### Do I need to create config.yaml?

**Yes.** It's required. Without it, your agent cannot run. See [CONFIG_GUIDE.md](CONFIG_GUIDE.md).

### What LLM providers are supported?

- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude)
- Mistral
- Groq
- Azure OpenAI
- Ollama (local)

See [CONFIG_GUIDE.md#provider-configuration-examples](CONFIG_GUIDE.md).

### Can I use RAG?

Yes! ErenFlowAI supports:
- Pinecone
- Weaviate
- Chroma

See [CONFIG_GUIDE.md#vector-store-configuration-examples](CONFIG_GUIDE.md).

### How do I handle secrets like API keys?

Use environment variables in config.yaml:
```yaml
llm:
  api_key: ${OPENAI_API_KEY}
```

Set before running:
```bash
export OPENAI_API_KEY="sk-..."
cargo run
```

Or create `.env` file. See [CONFIG_GUIDE.md#environment-variables](CONFIG_GUIDE.md).

### Can I have conditional routing?

Yes! Use conditions in edges:
```yaml
edges:
  - from: node_a
    to: complex_handler
    condition: is_complex_query
  
  - from: node_a
    to: simple_handler
    condition: is_simple_query
```

See [DEVELOPER_GUIDE.md#conditions--routing](DEVELOPER_GUIDE.md).

### How do I test my handlers?

Write async unit tests:
```rust
#[tokio::test]
async fn test_my_handler() {
    let mut state = State::new();
    state.set("input", json!("test"));
    
    let result = my_handler(state).await;
    assert!(result.is_ok());
}
```

See [DEVELOPER_GUIDE.md#testing-agents](DEVELOPER_GUIDE.md).

### How do I debug?

Enable debug logging in config.yaml:
```yaml
observability:
  tracing_enabled: true
  trace_level: debug
```

See [DEVELOPER_GUIDE.md#debugging](DEVELOPER_GUIDE.md).

## 📚 Examples

See the [examples/](examples/) directory for working code:
- `comprehensive_agent` - Full-featured example with RAG, MCP, LLM

Run with:
```bash
cargo run --example comprehensive_agent
```

## 🔗 API Documentation

Full API documentation available:
```bash
cargo doc --open
```

Key types:
- `Agent` - Main API
- `State` - State container
- `Graph` - Workflow structure
- `Handler` - Handler function type
- `AgentConfig` - Configuration

## 💡 Tips for Success

1. **Start Simple** - Create a config with one node
2. **Test Locally** - Run your agent before deploying
3. **Check Env Vars** - Verify API keys are set
4. **Read Errors** - Error messages are helpful
5. **Look at Examples** - Check `examples/` for patterns
6. **Enable Logging** - Turn on debug logs to see what's happening
7. **Validate YAML** - Use YAML linters to check syntax

## 🆘 Troubleshooting

### Issue: "Failed to load config"
**Solution:** Ensure config.yaml exists and has valid YAML syntax. See [CONFIG_GUIDE.md](CONFIG_GUIDE.md).

### Issue: "Handler not found"
**Solution:** Check handler name matches exactly and function is `pub async fn`. See [DEVELOPER_GUIDE.md#handler-development](DEVELOPER_GUIDE.md).

### Issue: "Invalid API key"
**Solution:** Verify environment variable is set: `echo $OPENAI_API_KEY`. See [CONFIG_GUIDE.md#environment-variables](CONFIG_GUIDE.md).

### Issue: State validation failed
**Solution:** Check handlers set required state fields. See [DEVELOPER_GUIDE.md#state-management](DEVELOPER_GUIDE.md).

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📞 Support

- **Bug Reports:** Open an issue on GitHub
- **Questions:** Check documentation first
- **Feature Requests:** GitHub discussions

## 📄 License

MIT License - See LICENSE file

---

## Quick Links Summary

**Just starting?**
→ [QUICKSTART.md](QUICKSTART.md)

**Need config help?**
→ [CONFIG_GUIDE.md](CONFIG_GUIDE.md)

**Want to understand it all?**
→ [README.md](README.md)

**Building something advanced?**
→ [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)

---

**Happy building! 🚀**
