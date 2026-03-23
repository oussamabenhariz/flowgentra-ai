# Flowgentra AI Documentation

Welcome to the FlowgentraAI documentation! Everything you need to build AI agents is organized here.

## Documentation Structure

```
docs/
├── README.md (this file - main navigation)
├── QUICKSTART.md (get running in 5 minutes)
├── FEATURES.md (feature overview)
├── config.yaml.template (config template to copy)
│
├── Feature Guides (organized by capability)
│   ├── agents/              → Predefined agent types (ZeroShotReAct, FewShot, Conversational)
│   ├── llm/              → Setup OpenAI, Anthropic, Mistral, Groq, HuggingFace, etc.
│   ├── rag/              → Semantic search with vector stores
│   ├── memory/           → Conversation history & checkpointing
│   ├── evaluation/       → Auto-evaluation & self-correction
│   ├── tools/            → Custom tools & MCP tools
│   ├── state/            → State management & data flow
│   ├── planning/         → Dynamic planning & adaptive workflows
│   ├── handlers/         → Custom handler development
│   ├── testing/          → Testing strategies & unit tests
│   └── observability/    → Logging, tracing, monitoring
│
├── configuration/
│   ├── CONFIG_GUIDE.md (complete config reference)
│   └── CONFIG_FEATURES.md (config examples & patterns)
│
└── development/
    └── DEVELOPER_GUIDE.md (advanced patterns & development)
```

## Quick Navigation

### Get Started in 5 Minutes
**[QUICKSTART.md](./QUICKSTART.md)** - Minimal example to get your first agent working

### Understand All Features
**[FEATURES.md](./FEATURES.md)** - Overview of core features (memory, evaluation, RAG, MCP, etc.)

### Configure Your Agent
**[configuration/CONFIG_GUIDE.md](./configuration/CONFIG_GUIDE.md)** - Complete config.yaml reference  
**[configuration/CONFIG_FEATURES.md](./configuration/CONFIG_FEATURES.md)** - Memory, evaluation, planner configs

### Feature-Specific Guides
| What I Want to Do | Guide |
|---|---|
| Use ZeroShotReAct, FewShot, or Conversational | [agents/README.md](./agents/README.md) |
| Set up OpenAI, Claude, Mistral, Groq, HuggingFace... | [llm/README.md](./llm/README.md) |
| Enable semantic search with RAG | [rag/README.md](./rag/README.md) |
| Add conversation memory/checkpoints | [memory/README.md](./memory/README.md) |
| Auto-grade & retry on low quality | [evaluation/README.md](./evaluation/README.md) |
| Connect external tools & APIs | [tools/README.md](./tools/README.md) |
| Understand data flow in workflows | [state/README.md](./state/README.md) |
| Let LLM decide next step | [planning/README.md](./planning/README.md) |
| Write custom handler functions | [handlers/README.md](./handlers/README.md) |
| Write unit & integration tests | [testing/README.md](./testing/README.md) |
| Monitor & debug execution | [observability/README.md](./observability/README.md) |

### Build Custom Code
**[development/DEVELOPER_GUIDE.md](./development/DEVELOPER_GUIDE.md)** - Advanced patterns and development  
**[handlers/README.md](./handlers/README.md)** - Build handler functions

### Template to Start
**[config.yaml.template](./config.yaml.template)** - Copy this to create your config.yaml

---

## Learning Path

### Beginner (30 minutes)
1. [QUICKSTART.md](./QUICKSTART.md) - 5 minute quickstart
2. [config.yaml.template](./config.yaml.template) - Copy and customize
3. [configuration/CONFIG_GUIDE.md](./configuration/CONFIG_GUIDE.md) - Understand options

### Intermediate (2 hours)
1. [FEATURES.md](./FEATURES.md) - Learn all capabilities
2. Pick a feature: [llm/](./llm/), [rag/](./rag/), [memory/](./memory/), [evaluation/](./evaluation/)
3. [handlers/README.md](./handlers/README.md) - Build custom handlers
4. Check examples in `../examples/`

### Advanced (1+ hours)
1. [planning/README.md](./planning/README.md) - Dynamic workflows
2. [state/README.md](./state/README.md) - State management
3. [testing/README.md](./testing/README.md) - Testing strategies
4. [development/DEVELOPER_GUIDE.md](./development/DEVELOPER_GUIDE.md) - Advanced patterns
5. Run `cargo doc --open` - Full API docs

---

## What You'll Learn

- **5 minutes**: Create and run your first agent
- **30 minutes**: Configure memory, evaluation, and tools
- **2 hours**: Build custom handlers and advanced workflows
- **Advanced**: Dynamic planning, complex state management, testing

---

## Key Concepts

### config.yaml
The required configuration file that defines:
- Your LLM provider (OpenAI, Anthropic, etc.)
- Your workflow (nodes and edges)
- State schema
- Tools, RAG, middleware, monitoring

**Rule**: You MUST create `config.yaml` in your project root.

### Handlers
Async Rust functions that process your agent's state:
```rust
pub async fn my_handler(mut state: State) -> Result<State> {
    // Read, process, update state
    Ok(state)
}
```

### State
A JSON object that flows through your agent's workflow:
```
START → Handler1 → Handler2 → Handler3 → END
  ↓       ↓        ↓        ↓       ↓
State0 → State1 → State2 → State3 → State4
```

---

## Common Need → Documentation Map

| I want to | File | Time |
|-----------|------|------|
| Get started quickly | [QUICKSTART.md](./QUICKSTART.md) | 5 min |
| Learn about features | [FEATURES.md](./FEATURES.md) | 20 min |
| Create config.yaml | [configuration/CONFIG_GUIDE.md](./configuration/CONFIG_GUIDE.md) | 20 min |
| **Use predefined agents** | [agents/README.md](./agents/README.md) | 15 min |
| **Set up LLM providers** | [llm/README.md](./llm/README.md) | 15 min |
| **Enable RAG/vector store** | [rag/README.md](./rag/README.md) | 20 min |
| **Add conversation memory** | [memory/README.md](./memory/README.md) | 10 min |
| **Auto-evaluate quality** | [evaluation/README.md](./evaluation/README.md) | 15 min |
| **Connect tools & APIs** | [tools/README.md](./tools/README.md) | 20 min |
| **Understand state flow** | [state/README.md](./state/README.md) | 15 min |
| **Dynamic planning** | [planning/README.md](./planning/README.md) | 15 min |
| Build custom handlers | [handlers/README.md](./handlers/README.md) | 30 min |
| Write tests | [testing/README.md](./testing/README.md) | 30 min |
| Monitor & trace | [observability/README.md](./observability/README.md) | 20 min |
| Advanced patterns | [development/DEVELOPER_GUIDE.md](./development/DEVELOPER_GUIDE.md) | 1 hour |
| Copy a template | [config.yaml.template](./config.yaml.template) | 1 min |

---

## Cleanup Note

The following files in the root of `flowgentra-ai/` should be removed (they've been moved to `docs/`):
- `CONFIG_FEATURES.md` → moved to `docs/configuration/CONFIG_FEATURES.md`
- `CONFIG_GUIDE.md` → moved to `docs/configuration/CONFIG_GUIDE.md`
- `DEVELOPER_GUIDE.md` → moved to `docs/development/DEVELOPER_GUIDE.md`
- `DOCUMENTATION.md` → superseded by this `README.md`
- `DOCS_SUMMARY.md` → superseded by this `README.md`
- `FEATURES.md` → moved to `docs/FEATURES.md`
- `QUICKSTART.md` → moved to `docs/QUICKSTART.md`
- `config.yaml.template` → moved to `docs/config.yaml.template`

**To clean up**, run from the `flowgentra-ai` directory:
```bash
rm CONFIG_FEATURES.md CONFIG_GUIDE.md DEVELOPER_GUIDE.md \
   DOCUMENTATION.md DOCS_SUMMARY.md FEATURES.md QUICKSTART.md \
   config.yaml.template
```

Or delete them manually from your file explorer.

---

## Feature Highlights

Memory & Checkpointing - Save conversation history and resume workflows  
Auto-Evaluation - LLM grades output quality and retries automatically  
Dynamic Planning - LLM decides next step based on state (no hardcoding)  
RAG - Semantic search over your documents (Pinecone, Weaviate, Chroma)  
MCP Tools - Connect external services and APIs  
Multiple LLMs - OpenAI, Anthropic, Mistral, Groq, Azure, HuggingFace, Ollama  
State Management - Type-safe data flow through your workflow  
Middleware - Logging, caching, rate limiting, tracing

---

## Need Help?

Quick question? Check the file that matches your topic above.  
Not sure where to start? Begin with [QUICKSTART.md](./QUICKSTART.md).  
Want to learn a specific feature? Browse the feature-specific guides.  
Building something advanced? Read [development/DEVELOPER_GUIDE.md](./development/DEVELOPER_GUIDE.md).

---

## File Reference

**Core Guides**
- **README.md** - Main navigation (you are here)
- **QUICKSTART.md** - 5-minute quickstart
- **FEATURES.md** - Feature overview
- **config.yaml.template** - Template configuration file

**Feature Guides** (pick what you need)
- **agents/** - Predefined agent types (ZeroShotReAct, FewShot, Conversational)
- **llm/** - LLM provider setup & configuration
- **rag/** - RAG & vector store setup
- **memory/** - Conversation memory & checkpointing
- **evaluation/** - Auto-evaluation & self-correction
- **tools/** - Custom tools & MCP integration
- **state/** - State management & data flow
- **planning/** - Dynamic planning & routing
- **handlers/** - Custom handler development
- **testing/** - Testing strategies & examples
- **observability/** - Monitoring & tracing

**Configuration & Development**
- **configuration/** - Complete config reference
- **development/** - Advanced development guide

---

Happy building!
