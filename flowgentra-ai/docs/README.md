# Flowgentra AI Documentation

Welcome to the Flowgentra AI documentation. This is your starting point for building intelligent agents, multi-agent systems, and stateful workflows in Rust.

Flowgentra AI is an agent framework, designed to give you fine-grained control over graph-based execution, LLM orchestration, tool use, and state management -- all with the performance and safety guarantees of Rust.

Whether you are wiring up your first chatbot or orchestrating a fleet of cooperating agents, this hub will point you to the right guide.

---

## Documentation Map

```
docs/
|-- README.md                    You are here
|-- QUICKSTART.md                Get running in 5 minutes
|-- FEATURES.md                  Full feature overview
|-- config.yaml.template         Starter config to copy
|
|-- agents/                      Predefined agent types
|   ZeroShotReAct, FewShotReAct, Conversational, Supervisor
|
|-- graph/                       Graph engine & compiler
|   SubgraphNode, ParallelExecutor, async conditional edges,
|   graph export (DOT / Mermaid / JSON), human-in-the-loop,
|   MessageGraphBuilder, ToolNode, tools_condition
|
|-- llm/                         LLM providers & features
|   OpenAI, Anthropic, Mistral, Groq, HuggingFace, Ollama,
|   RetryLLM, CachedLLM, FallbackLLM,
|   token counting, cost tracking, structured output,
|   PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate,
|   OutputParser (JSON, List, Structured)
|
|-- rag/                         Retrieval-Augmented Generation
|   PineconeStore, QdrantStore, ChromaStore, InMemoryStore,
|   Text splitters, Retriever, HuggingFace embeddings,
|   Cross-encoder reranker, Document loaders, Ingestion pipeline
|
|-- memory/                      Conversation history & checkpointing
|   InMemoryCheckpointer, FileCheckpointer,
|   TokenBufferMemory, SummaryMemory
|
|-- state/                       State management & data flow
|   PlainState, SharedState, ScopedState,
|   JsonReducer, ReducerConfig, #[derive(State)]
|
|-- tools/                       Custom tools & MCP integration
|   MCP SSE, Stdio, Docker, reconnecting client,
|   MCP resources & prompts protocol
|
|-- evaluation/                  Auto-evaluation & self-correction
|-- planning/                    Dynamic planning & adaptive workflows
|-- handlers/                    Custom handler development
|-- macros/                      Proc macros: #[node], #[register_handler], #[derive(State)]
|-- observability/               Logging, tracing, monitoring
|   Event broadcaster, OpenTelemetry (OTLP) export
|
|-- testing/                     Testing strategies & examples
|-- configuration/               Complete config reference
|   CONFIG_GUIDE.md, CONFIG_FEATURES.md
|
|-- development/                 Advanced patterns & internals
    DEVELOPER_GUIDE.md
```

---

## Quick Navigation

### "I want to..." Table

| I want to...                                      | Start here                                                         | Est. time |
| ------------------------------------------------- | ------------------------------------------------------------------ | --------- |
| Get started quickly                                | [QUICKSTART.md](./QUICKSTART.md)                                   | 5 min     |
| See every feature at a glance                      | [FEATURES.md](./FEATURES.md)                                       | 15 min    |
| Create or customize config.yaml                    | [configuration/CONFIG_GUIDE.md](./configuration/CONFIG_GUIDE.md)   | 20 min    |
| Use a predefined agent                             | [agents/README.md](./agents/README.md)                             | 15 min    |
| Set up an LLM provider                             | [llm/README.md](./llm/README.md)                                  | 15 min    |
| Add automatic retries to LLM calls                 | [llm/README.md](./llm/README.md)                                  | 10 min    |
| Track token usage and costs                        | [llm/README.md](./llm/README.md)                                  | 10 min    |
| Get structured JSON output from an LLM             | [llm/README.md](./llm/README.md)                                  | 10 min    |
| Build a graph-based workflow                        | [graph/README.md](./graph/README.md)                               | 20 min    |
| Compose subgraphs inside a parent graph             | [graph/README.md](./graph/README.md)                               | 15 min    |
| Run branches in parallel and join results           | [graph/README.md](./graph/README.md)                               | 15 min    |
| Export a graph to DOT, Mermaid, or JSON             | [graph/README.md](./graph/README.md)                               | 10 min    |
| Add human-in-the-loop approval steps                | [graph/README.md](./graph/README.md)                               | 15 min    |
| Use async conditional edges                         | [graph/README.md](./graph/README.md)                               | 10 min    |
| Enable RAG / vector search                          | [rag/README.md](./rag/README.md)                                  | 20 min    |
| Use Pinecone or Qdrant as a vector store            | [rag/README.md](./rag/README.md)                                  | 15 min    |
| Split documents into chunks                          | [rag/README.md](./rag/README.md)                                  | 10 min    |
| Build an ingestion pipeline                          | [rag/README.md](./rag/README.md)                                  | 15 min    |
| Use HuggingFace embeddings                           | [rag/README.md](./rag/README.md)                                  | 10 min    |
| Rerank search results                                | [rag/README.md](./rag/README.md)                                  | 10 min    |
| Build a chat-focused graph (MessageGraph)            | [graph/README.md](./graph/README.md)                               | 10 min    |
| Add ReAct-style tool execution (ToolNode)            | [graph/README.md](./graph/README.md)                               | 15 min    |
| Cache LLM responses                                  | [llm/README.md](./llm/README.md)                                  | 5 min     |
| Set up LLM fallback providers                        | [llm/README.md](./llm/README.md)                                  | 10 min    |
| Use prompt templates                                 | [llm/README.md](./llm/README.md)                                  | 10 min    |
| Parse structured output from LLM responses           | [llm/README.md](./llm/README.md)                                  | 10 min    |
| Use token-budget memory management                   | [memory/README.md](./memory/README.md)                             | 10 min    |
| Summarize old conversation messages                  | [memory/README.md](./memory/README.md)                             | 10 min    |
| Add conversation memory                             | [memory/README.md](./memory/README.md)                             | 10 min    |
| Persist checkpoints to disk                         | [memory/README.md](./memory/README.md)                             | 10 min    |
| Auto-evaluate and self-correct output               | [evaluation/README.md](./evaluation/README.md)                     | 15 min    |
| Connect tools and APIs                              | [tools/README.md](./tools/README.md)                               | 20 min    |
| Use MCP with reconnection and resources             | [tools/README.md](./tools/README.md)                               | 15 min    |
| Manage state with reducers and scoping              | [state/README.md](./state/README.md)                               | 15 min    |
| Let the LLM decide the next step                    | [planning/README.md](./planning/README.md)                         | 15 min    |
| Orchestrate multiple agents with a Supervisor       | [agents/README.md](./agents/README.md)                             | 20 min    |
| Write custom handler functions                      | [handlers/README.md](./handlers/README.md)                         | 30 min    |
| Use proc macros (#[node], #[register_handler])      | [macros/README.md](./macros/README.md)                             | 15 min    |
| Monitor execution with real-time events             | [observability/README.md](./observability/README.md)               | 15 min    |
| Export traces to OpenTelemetry                       | [observability/README.md](./observability/README.md)               | 15 min    |
| Write tests for agents and workflows                | [testing/README.md](./testing/README.md)                           | 30 min    |
| Learn advanced patterns                             | [development/DEVELOPER_GUIDE.md](./development/DEVELOPER_GUIDE.md) | 1 hour   |
| Copy a starter config template                      | [config.yaml.template](./config.yaml.template)                     | 1 min     |

---

## Learning Paths

### Beginner -- Your First Agent (30 minutes)

Build a working agent from scratch with minimal configuration.

1. [QUICKSTART.md](./QUICKSTART.md) -- Run a "hello world" agent in 5 minutes
2. [config.yaml.template](./config.yaml.template) -- Copy the template and make it yours
3. [configuration/CONFIG_GUIDE.md](./configuration/CONFIG_GUIDE.md) -- Understand the configuration options
4. [agents/README.md](./agents/README.md) -- Pick a predefined agent type
5. Browse the examples in `../examples/`

### Intermediate -- Custom Workflows (2-3 hours)

Go beyond predefined agents. Build multi-step workflows with tools, memory, and evaluation.

1. [FEATURES.md](./FEATURES.md) -- Survey the full feature set
2. [llm/README.md](./llm/README.md) -- Configure your LLM provider, enable retries and cost tracking
3. [graph/README.md](./graph/README.md) -- Build graph-based workflows with conditional routing
4. [state/README.md](./state/README.md) -- Manage data flow with reducers and scoped state
5. [memory/README.md](./memory/README.md) -- Add checkpointing (in-memory or file-based)
6. [tools/README.md](./tools/README.md) -- Wire up tools and MCP integrations
7. [rag/README.md](./rag/README.md) -- Add retrieval-augmented generation
8. [handlers/README.md](./handlers/README.md) -- Write your own handler functions
9. [evaluation/README.md](./evaluation/README.md) -- Auto-grade output quality and retry

### Advanced -- Production Systems (ongoing)

Design production-grade agent architectures with multi-agent orchestration, observability, and testing.

1. [graph/README.md](./graph/README.md) -- Subgraphs, parallel execution, human-in-the-loop
2. [agents/README.md](./agents/README.md) -- Supervisor multi-agent orchestration
3. [macros/README.md](./macros/README.md) -- Use `#[node]`, `#[register_handler]`, `#[derive(State)]`
4. [observability/README.md](./observability/README.md) -- Event broadcasting and OpenTelemetry export
5. [planning/README.md](./planning/README.md) -- Dynamic, LLM-driven planning
6. [testing/README.md](./testing/README.md) -- Testing strategies for agents and graphs
7. [development/DEVELOPER_GUIDE.md](./development/DEVELOPER_GUIDE.md) -- Advanced patterns, plugin system, middleware
8. Run `cargo doc --open` for the full API reference

---

## Feature Highlights

### Graph Engine

The graph compiler is the heart of Flowgentra AI. Define nodes, edges, and conditional branches, then let the compiler validate and execute your workflow.

| Feature                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| Graph Compiler             | Validates structure, detects cycles, ensures reachability              |
| SubgraphNode               | Compose entire graphs as single nodes inside a parent graph            |
| ParallelExecutor           | Run branches concurrently with configurable join strategies            |
| Async Conditional Edges    | Route execution with async functions that inspect state at runtime     |
| Graph Export               | Serialize graph structure to DOT, Mermaid, or JSON for visualization   |
| Human-in-the-Loop          | Interrupt execution before or after a node, resume with modified state |
| MessageGraphBuilder        | Convenience wrapper for chat-focused graphs with message accumulation  |
| ToolNode                   | Prebuilt node for automatic tool call execution from LLM responses     |
| tools_condition            | Router that directs to tool node or end based on tool calls in state   |

### Agents and Multi-Agent Systems

| Feature                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| ZeroShotReAct              | Reason-and-act loop without examples                                   |
| FewShotReAct               | Reason-and-act with in-context examples                                |
| Conversational Agent       | Multi-turn dialogue with memory                                        |
| Supervisor                 | Orchestrate multiple agents with intelligent routing                   |

### LLM Integration

| Feature                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| Multi-Provider Support     | OpenAI, Anthropic, Mistral, Groq, Azure, HuggingFace, Ollama          |
| RetryLLM             | Automatic retry with exponential backoff for transient failures        |
| Token Counting             | Track input/output tokens and manage context window limits             |
| Cost Tracking              | Estimate USD cost per LLM call based on provider pricing               |
| Anthropic Tool Calling     | Native `input_schema` format for Anthropic function calling            |
| HuggingFace SSE Streaming  | Real server-sent-event streaming for HuggingFace models                |
| ResponseFormat             | Structured output / JSON mode for deterministic parsing                |
| CachedLLM            | Hash-based response caching to avoid redundant API calls               |
| FallbackLLM          | Try multiple providers in sequence until one succeeds                  |
| PromptTemplate             | String interpolation with `{variable}` syntax and partial formatting   |
| ChatPromptTemplate         | Multi-message prompt builder with system/user/assistant templates      |
| OutputParser               | Parse JSON, lists, and structured data from LLM text responses         |

### State Management

| Feature                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| PlainState / SharedState   | Simple and thread-safe state containers                                |
| ScopedState                | Namespaced state isolation per node                                    |
| JsonReducer                | Per-field merge strategies via ReducerConfig                           |
| #[derive(State)]           | Derive macro for ergonomic state definitions                           |

### Tools and MCP

| Feature                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| Custom Tools               | Define tools as Rust functions with typed schemas                      |
| MCP Transports             | SSE, Stdio, and Docker-based MCP connections                          |
| Reconnecting Client        | Factory-pattern MCP client that auto-reconnects on failure             |
| MCP Resources and Prompts  | Full MCP protocol support including resources and prompt templates     |

### Retrieval-Augmented Generation

| Feature                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| PineconeStore              | Real REST API integration with Pinecone                                |
| QdrantStore                | Real REST API integration with Qdrant                                  |
| ChromaStore                | ChromaDB vector store support                                          |
| InMemoryStore              | Lightweight in-process store for development and testing               |
| Text Splitters             | Recursive, Markdown, Code, HTML, and Token-based document chunking     |
| Retriever                  | End-to-end pipeline: embed → search → rerank → dedup                   |
| HuggingFace Embeddings     | Inference API + self-hosted TEI with auto dimension detection           |
| Cross-Encoder Reranker     | HuggingFace cross-encoder model for result reranking                   |
| Document Loaders           | Load PDF, text, Markdown, JSON, CSV, HTML files                        |
| Ingestion Pipeline         | Load → split → embed → index in one call                               |
| Hybrid Search              | Combine semantic + BM25 keyword matching                               |
| RAG Evaluation             | Hit rate, MRR, and NDCG metrics for retrieval quality                  |

### Memory and Checkpointing

| Feature                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| InMemoryCheckpointer       | Fast, ephemeral checkpoint storage                                     |
| FileCheckpointer           | Persist checkpoints to disk as JSON for durable recovery               |
| TokenBufferMemory          | Sliding window memory managed by token budget                          |
| SummaryMemory              | LLM-based summarization of older conversation messages                 |

### Evaluation and Self-Correction

| Feature                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| Auto-Evaluation            | LLM-powered quality grading of agent output                           |
| Self-Correction            | Automatic retry loop when evaluation falls below threshold             |

### Planning

| Feature                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| Dynamic Planning           | LLM-driven step selection based on current state                       |
| Adaptive Workflows         | Plans that evolve as new information becomes available                  |

### Observability

| Feature                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| Event Broadcaster          | Real-time execution events delivered via broadcast channel             |
| OpenTelemetry Export       | OTLP-compatible spans for distributed tracing                          |
| Logging and Tracing        | Configurable log levels with structured output                         |

### Macros

| Feature                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| #[node]                    | Proc macro that generates node factory functions                       |
| #[register_handler]        | Register handler functions for the graph runtime                       |
| #[derive(State)]           | Derive state traits automatically                                      |

### Infrastructure

| Feature                    | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| Plugin System              | Extend the framework with custom plugins                               |
| Middleware Pipeline         | Composable middleware for logging, caching, rate limiting, auth        |
| Graph Validation           | Compile-time and runtime structural checks                             |

---

## Key Concepts

### config.yaml

Every Flowgentra AI project starts with a configuration file that declares your LLM provider, workflow graph, state schema, tools, and middleware. Copy the template to get started:

```bash
cp docs/config.yaml.template config.yaml
```

### Handlers

Handlers are async Rust functions that process state at each node of the graph:

```rust
pub async fn my_handler(mut state: State) -> Result<State> {
    // Read from state, call an LLM, update state
    Ok(state)
}
```

### State Flow

State is a JSON-compatible object that moves through the graph. Each handler receives the current state, transforms it, and passes it to the next node:

```
START --> Node A --> Node B --> Node C --> END
         state0     state1     state2     state3
```

With JsonReducer and ScopedState, you have fine-grained control over how state fields are merged and which nodes can see which data.

---

## Need Help?

**Not sure where to start?** Open [QUICKSTART.md](./QUICKSTART.md) and follow along.

**Looking for a specific feature?** Use the navigation table above to jump directly to the right guide.

**Building something advanced?** The [Developer Guide](./development/DEVELOPER_GUIDE.md) covers internals, plugin authoring, and production patterns.

**Want the full API reference?** Run `cargo doc --open` from the project root.

---

## File Reference

**Core Guides**

- `README.md` -- Main documentation hub (you are here)
- `QUICKSTART.md` -- Five-minute quickstart
- `FEATURES.md` -- Complete feature overview
- `config.yaml.template` -- Starter configuration template

**Feature Guides**

- `agents/` -- Predefined agents and Supervisor multi-agent orchestration
- `graph/` -- Graph engine, subgraphs, parallel execution, export, human-in-the-loop
- `llm/` -- LLM providers, retries, token counting, cost tracking, structured output
- `rag/` -- RAG with Pinecone, Qdrant, Chroma, and in-memory stores
- `memory/` -- Conversation memory, in-memory and file-based checkpointing
- `state/` -- State management, reducers, scoped state
- `tools/` -- Custom tools, MCP integration, reconnecting client, resources and prompts
- `evaluation/` -- Auto-evaluation and self-correction
- `planning/` -- Dynamic and adaptive planning
- `handlers/` -- Custom handler development
- `macros/` -- Proc macros: `#[node]`, `#[register_handler]`, `#[derive(State)]`
- `observability/` -- Event broadcasting, OpenTelemetry export, logging
- `testing/` -- Testing strategies and examples

**Configuration and Development**

- `configuration/` -- Complete config.yaml reference and examples
- `development/` -- Advanced patterns, plugin system, middleware internals
