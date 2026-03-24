# FlowgentraAI Features Guide

Everything FlowgentraAI can do, organized by category with practical examples.

---

## Graph Engine

The graph engine is the core of FlowgentraAI. Build workflows as directed graphs with nodes (processing steps) and edges (transitions).

### StateGraph Builder

Build workflows programmatically with type-safe state:

```rust
use flowgentra_ai::core::state_graph::StateGraphBuilder;
use flowgentra_ai::core::state::PlainState;

let graph = StateGraphBuilder::new()
    .add_fn("step1", process_input)
    .add_fn("step2", generate_output)
    .set_entry_point("step1")
    .add_edge("step1", "step2")
    .add_edge("step2", "__end__")
    .compile()?;

let result = graph.run(initial_state).await?;
```

### Conditional Routing

Route execution based on state at runtime:

```rust
builder.add_conditional_edge("classify", |state| {
    let category = state.get("category")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    Ok(category.to_string())
})
```

### Async Conditional Edges

For routing decisions that need async operations (API calls, DB lookups):

```rust
builder.add_async_conditional_edge("check", Box::new(|state| {
    Box::pin(async move {
        let result = call_external_api(&state).await?;
        Ok(result.next_step)
    })
}))
```

### Subgraph Composition

Nest entire graphs as single nodes inside a parent graph:

```rust
let inner_graph = StateGraphBuilder::new()
    .add_fn("a", step_a)
    .add_fn("b", step_b)
    .set_entry_point("a")
    .add_edge("a", "b")
    .add_edge("b", "__end__")
    .compile()?;

let outer = StateGraphBuilder::new()
    .add_subgraph("inner", inner_graph)
    .add_fn("final", finalize)
    .set_entry_point("inner")
    .add_edge("inner", "final")
    .add_edge("final", "__end__")
    .compile()?;
```

### Parallel Execution

Run branches concurrently and merge results:

```rust
use flowgentra_ai::core::runtime::parallel::{ParallelExecutor, JoinType};

let executor = ParallelExecutor::new()
    .with_join_type(JoinType::WaitAll)
    .with_timeout(Duration::from_secs(30))
    .with_continue_on_error(true);

let result = executor.execute(branches, state).await?;
```

Join strategies: `WaitAll`, `WaitAny`, `WaitCount(n)`, `WaitTimeout`.

### Graph Export

Visualize your graph in multiple formats:

```rust
let graph = builder.compile()?;

// Graphviz DOT
let dot = graph.to_dot();

// Mermaid (for Markdown docs)
let mermaid = graph.to_mermaid();

// JSON (for custom tooling)
let json = graph.to_json();
```

### Human-in-the-Loop

Pause execution for human review, then resume with modified state:

```rust
let graph = StateGraphBuilder::new()
    .add_fn("draft", draft_response)
    .add_fn("send", send_response)
    .set_entry_point("draft")
    .interrupt_before("send")  // Pause here for approval
    .add_edge("draft", "send")
    .add_edge("send", "__end__")
    .compile()?;

// Later, resume with edits
let result = graph.resume_with_state("thread-1", updated_state).await?;
```

### MessageGraphBuilder

A convenience wrapper for chat-focused workflows with automatic message accumulation:

```rust
use flowgentra_ai::prelude::*;

let graph = MessageGraphBuilder::new()
    .add_fn("echo", |state: &PlainState| {
        let messages = MessageGraphBuilder::get_messages(state);
        let last = messages.last().map(|m| m.content.clone()).unwrap_or_default();
        let mut s = state.clone();
        s = MessageGraphBuilder::add_message(s, Message::assistant(format!("Echo: {}", last)));
        Box::pin(async move { Ok(s) })
    })
    .set_entry_point("echo")
    .add_edge("echo", "__end__")
    .compile()?;

let state = MessageGraphBuilder::initial_state(vec![Message::user("Hello")]);
let result = graph.invoke(state).await?;
```

### ToolNode

Prebuilt node that automatically executes tool calls from LLM responses:

```rust
use flowgentra_ai::prelude::*;
use std::sync::Arc;

// Create a tool executor
let tool_node = create_tool_node(Arc::new(|name, args| {
    Box::pin(async move {
        match name.as_str() {
            "calculator" => Ok(format!("Result: {}", args)),
            _ => Err(format!("Unknown tool: {}", name)),
        }
    })
}));

// Route based on whether tool calls exist
builder.add_conditional_edge("agent", tools_condition("tools"));
// Routes to "tools" if tool_calls present, "__end__" otherwise

// Store tool calls from LLM response into state
let state = store_tool_calls(state, &llm_response);
```

---

## State Management

### PlainState and SharedState

Two state containers for different needs:

```rust
// Owned state (single-threaded, fast)
let mut state = PlainState::new();
state.set("key", json!("value"));

// Thread-safe state (concurrent access)
let state = SharedState::new();
state.set("key", json!("value"));
```

### ScopedState

Namespaced state isolation prevents key collisions between nodes:

```rust
use flowgentra_ai::core::state::ScopedState;

let scoped = ScopedState::new(shared_state.clone(), "my_node");
scoped.set("counter", json!(1));
// Actually stored as "my_node.counter"
```

### JsonReducer and ReducerConfig

Control how state fields merge when multiple nodes update them:

```rust
use flowgentra_ai::core::reducer::{ReducerConfig, JsonReducer};

let config = ReducerConfig::default()
    .field("messages", JsonReducer::Append)        // Append arrays
    .field("total_cost", JsonReducer::Sum)          // Sum numbers
    .field("settings", JsonReducer::DeepMerge)      // Deep merge objects
    .field("best_score", JsonReducer::Max)           // Keep maximum
    .field("unique_tags", JsonReducer::AppendUnique); // Append without duplicates
```

Available reducers: `Overwrite`, `Append`, `Sum`, `DeepMerge`, `Max`, `Min`, `AppendUnique`.

---

## Agents

### Predefined Agent Types

| Agent | Pattern | Best For |
|-------|---------|----------|
| **ZeroShotReAct** | Think + act without examples | General problem-solving |
| **FewShotReAct** | Learn from provided examples | Classification, structured output |
| **Conversational** | Multi-turn with memory | Chatbots, assistants |

```rust
use flowgentra_ai::core::agents::{AgentBuilder, AgentType};

let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_name("researcher")
    .with_llm_config("gpt-4")
    .with_tool(search_tool)
    .build()?;
```

### Supervisor (Multi-Agent Orchestration)

Route tasks to specialized sub-agents:

```rust
use flowgentra_ai::core::agents::Supervisor;

let supervisor = Supervisor::new(
    router_fn,          // Decides which agent handles each task
    named_agent_graphs, // Map of agent name -> compiled StateGraph
    max_rounds,         // Prevent infinite loops
);

let result = supervisor.run(initial_state).await?;
```

The router function inspects the state and returns the name of the agent that should handle the current step.

---

## LLM Integration

### Supported Providers

| Provider | Auth | Streaming | Tool Calling |
|----------|------|-----------|--------------|
| OpenAI | Bearer token | SSE | Function calling |
| Anthropic | x-api-key | SSE | input_schema format |
| Mistral | Bearer token | SSE | Function calling |
| Groq | Bearer token | SSE | Function calling |
| Azure OpenAI | api-key | SSE | Function calling |
| HuggingFace | Bearer token | Real SSE (TGI) | -- |
| Ollama | None (local) | NDJSON | -- |

### RetryLLMClient

Automatic retry with exponential backoff:

```rust
use flowgentra_ai::core::llm::RetryLLMClient;

let client = RetryLLMClient::new(inner_client)
    .with_max_retries(3)
    .with_initial_delay(Duration::from_millis(500));

// Automatically retries on transient failures
let response = client.chat(messages).await?;
```

### Token Counting and Cost Tracking

```rust
use flowgentra_ai::core::llm::token_counter::{estimate_tokens, context_window, ContextWindow};
use flowgentra_ai::core::llm::model_pricing;

// Estimate tokens
let tokens = estimate_tokens("Hello, how are you?"); // ~5 tokens

// Check context window limits
let max = context_window("gpt-4"); // Some(8192)

// Truncate messages to fit
let ctx = ContextWindow { max_tokens: 8192, reserve_for_completion: 1024 };
let trimmed = ctx.truncate(&messages);

// Track costs
let (input_price, output_price) = model_pricing("gpt-4").unwrap();
let cost = usage.estimated_cost("gpt-4"); // USD
```

### Structured Output (ResponseFormat)

Get deterministic JSON responses:

```rust
use flowgentra_ai::core::llm::{LLMConfig, ResponseFormat};

// Simple JSON mode
let config = LLMConfig::new(provider, model, key)
    .with_response_format(ResponseFormat::Json);

// JSON with schema enforcement (OpenAI)
let config = LLMConfig::new(provider, model, key)
    .with_response_format(ResponseFormat::JsonSchema {
        name: "analysis".into(),
        schema: json!({
            "type": "object",
            "properties": {
                "sentiment": { "type": "string" },
                "score": { "type": "number" }
            },
            "required": ["sentiment", "score"]
        }),
    });
```

### Anthropic Tool Calling

Native support for Anthropic's `input_schema` format:

```rust
// Tools are automatically formatted for Anthropic:
// { "name": "...", "description": "...", "input_schema": {...} }
// Response parsing handles tool_use content blocks
let response = client.chat_with_tools(messages, &tools).await?;
```

### CachedLLMClient

Cache LLM responses to avoid redundant API calls:

```rust
use flowgentra_ai::prelude::*;
use std::sync::Arc;

let cached = CachedLLMClient::new(inner_client)
    .with_max_entries(1000);

// Second call with identical messages returns cached response
let response = cached.chat(messages.clone()).await?;
let cached_response = cached.chat(messages).await?; // cache hit

println!("Cache size: {}", cached.cache_size());
cached.clear_cache();
```

### FallbackLLMClient

Try multiple LLM providers in order until one succeeds:

```rust
use flowgentra_ai::prelude::*;

let client = FallbackLLMClient::new(primary_client)
    .with_fallback(secondary_client)
    .with_fallback(tertiary_client);

// Tries primary first, then secondary, then tertiary
let response = client.chat(messages).await?;
```

### PromptTemplate

Template strings with `{variable}` interpolation:

```rust
use flowgentra_ai::core::llm::prompt_template::PromptTemplate;

let template = PromptTemplate::new("Summarize {text} in {language}");
let result = template.format(&[
    ("text", "Rust is a systems language"),
    ("language", "French"),
])?;
// "Summarize Rust is a systems language in French"

// Partial formatting
let partial = template.partial(&[("language", "French")])?;
let result = partial.format(&[("text", "some content")])?;
```

### ChatPromptTemplate

Build multi-message prompts with templates:

```rust
use flowgentra_ai::core::llm::prompt_template::ChatPromptTemplate;

let prompt = ChatPromptTemplate::new()
    .system("You are a {role} expert.")
    .user("Explain {topic} simply.");

let messages = prompt.format_messages(&[
    ("role", "Rust"),
    ("topic", "ownership"),
])?;
```

### OutputParser

Parse structured data from LLM text responses:

```rust
use flowgentra_ai::core::llm::output_parser::{JsonOutputParser, ListOutputParser};

// Extract JSON from freeform text
let parser = JsonOutputParser::new();
let value = parser.parse("Here's the result: ```json\n{\"score\": 95}\n```")?;

// Parse lists
let parser = ListOutputParser::comma_separated();
let items = parser.parse("apples, oranges, bananas")?;
// ["apples", "oranges", "bananas"]
```

---

## Tools and MCP

### Local Tools

```rust
let tool = ToolSpec::new("search", "Search the web")
    .with_parameter("query", "string")
    .required("query");
```

### MCP Transports

Connect via SSE, Stdio, or Docker:

```yaml
mcp:
  tools:
    - name: web_search
      type: sse
      url: "http://localhost:8000"
      timeout: 30

    - name: local_tool
      type: stdio
      command: "/usr/bin/my-tool"

    - name: containerized
      type: docker
      image: "my-tool:latest"
```

### Reconnecting MCP Client

Auto-reconnects on connection failure:

```rust
use flowgentra_ai::core::mcp::ReconnectingMCPClient;

let client = ReconnectingMCPClient::new(factory_fn)
    .with_max_reconnects(5);

// Automatically reconnects if the connection drops
let tools = client.list_tools().await?;
```

### MCP Resources and Prompts

Full MCP protocol support beyond just tool calls:

```rust
// List and read resources
let resources = client.list_resources().await?;
let content = client.read_resource("file:///path/to/doc.md").await?;

// List and get prompt templates
let prompts = client.list_prompts().await?;
let result = client.get_prompt("summarize", args).await?;
```

---

## RAG (Retrieval-Augmented Generation)

### Vector Stores

| Store | Type | Best For |
|-------|------|----------|
| **PineconeStore** | Cloud (REST API) | Production, managed service |
| **QdrantStore** | Self-hosted (REST API) | Privacy, self-managed |
| **ChromaStore** | Local/self-hosted | Development, small datasets |
| **InMemoryStore** | In-process | Testing, prototyping |

```rust
use flowgentra_ai::core::rag::{RAGConfig, PineconeStore, QdrantStore};

// Pinecone
let store = PineconeStore::new(api_key, index_host);
store.upsert(vectors).await?;
let results = store.query(query_vector, top_k).await?;

// Qdrant
let store = QdrantStore::new(url, collection);
store.upsert(points).await?;
let results = store.query(query_vector, top_k).await?;
```

### Text Splitters

Split documents into chunks for indexing:

```rust
use flowgentra_ai::prelude::*;

// Recursive character splitter (general purpose)
let splitter = RecursiveCharacterTextSplitter::new(500, 50);
let chunks = splitter.split("Long document text...");

// Markdown-aware splitter (preserves heading structure)
let splitter = MarkdownTextSplitter::new(500, 50);

// Code splitter (respects language syntax)
let splitter = CodeTextSplitter::new(Language::Rust, 500, 50);

// HTML splitter (splits on tags)
let splitter = HTMLTextSplitter::new(500, 50);

// Token-based splitter
let splitter = TokenTextSplitter::new(128, 20);
```

Supported code languages: Rust, Python, JavaScript, TypeScript, Go, Java, C, CPP, Ruby.

### Retriever

End-to-end retrieval pipeline that chains embed → search → rerank → dedup:

```rust
use flowgentra_ai::prelude::*;

let retriever = Retriever::new(
    vector_store,        // Arc<dyn VectorStoreBackend>
    embeddings_provider, // Arc<dyn EmbeddingsProvider>
)
.with_top_k(10)
.with_reranker(reranker);

let results = retriever.retrieve("What is Rust?").await?;
```

### HuggingFace Embeddings

Generate embeddings via HuggingFace Inference API or self-hosted TEI:

```rust
use flowgentra_ai::prelude::*;

let embeddings = HuggingFaceEmbeddings::new(
    "sentence-transformers/all-MiniLM-L6-v2",
    "hf_your_api_key",
);

// Auto-detects dimension (384 for MiniLM, 768 for mpnet, 1024 for bge-large)
let vector = embeddings.embed("Hello world").await?;

// Self-hosted TEI server
let embeddings = HuggingFaceEmbeddings::new("model", "key")
    .with_endpoint("http://localhost:8080/embed");
```

### Cross-Encoder Reranker

Rerank search results using a cross-encoder model:

```rust
use flowgentra_ai::prelude::*;

let reranker = CrossEncoderReranker::new(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "hf_your_api_key",
)
.with_top_k(5);

let reranked = reranker.rerank("query", results).await?;
```

### Document Loaders

Load documents from various file formats:

```rust
use flowgentra_ai::prelude::*;

// Single file
let doc = load_document("path/to/file.pdf")?;

// Entire directory
let docs = load_directory("path/to/docs/")?;
```

Supported formats: PDF, plain text, Markdown, JSON, CSV, HTML.

### Ingestion Pipeline

Load, split, embed, and index documents in one pipeline:

```rust
use flowgentra_ai::prelude::*;

let pipeline = IngestionPipeline::new(
    splitter,    // Box<dyn TextSplitter>
    embeddings,  // Arc<dyn EmbeddingsProvider>
    store,       // Arc<dyn VectorStoreBackend>
);

let stats = pipeline.ingest(documents).await?;
println!("Indexed {} chunks from {} documents", stats.chunks, stats.documents);
```

---

## Memory and Checkpointing

### InMemoryCheckpointer

Fast, ephemeral checkpoint storage for development:

```rust
let checkpointer = InMemoryCheckpointer::new();
checkpointer.save("thread-1", &state)?;
let restored = checkpointer.load("thread-1")?;
```

### FileCheckpointer

Persist checkpoints to disk for durable recovery:

```rust
use flowgentra_ai::core::state_graph::FileCheckpointer;

let checkpointer = FileCheckpointer::new("./checkpoints");
checkpointer.save("thread-1", &state)?;

// Later, even after restart:
let restored = checkpointer.load("thread-1")?;
```

### Conversation Memory

Remember chat history across turns:

```yaml
memory:
  conversation:
    enabled: true
    buffer_window: 20  # Last 20 messages
```

### TokenBufferMemory

Sliding window that respects token budgets instead of message counts:

```rust
use flowgentra_ai::prelude::*;

let mut memory = TokenBufferMemory::new(4000); // 4000 token budget
memory.add_message(Message::system("You are helpful."));
memory.add_message(Message::user("Hello!"));

// Oldest non-system messages are dropped when budget exceeded
let messages = memory.messages();
```

### SummaryMemory

Summarize older messages with an LLM to keep context compact:

```rust
use flowgentra_ai::prelude::*;

let config = SummaryConfig {
    buffer_size: 10,
    max_summary_tokens: 500,
};

let memory = SummaryMemory::new(config, |messages| {
    Box::pin(async move {
        // Call your LLM to summarize
        Ok("Summary of conversation so far...".to_string())
    })
});
```

---

## Observability

### Event Broadcaster

Stream real-time execution events:

```rust
use flowgentra_ai::core::observability::{EventBroadcaster, ExecutionEvent};

let broadcaster = EventBroadcaster::new(100); // buffer size
let mut rx = broadcaster.subscribe();

// In another task:
while let Ok(event) = rx.recv().await {
    match event {
        ExecutionEvent::NodeStarted { name, .. } => println!("Starting: {}", name),
        ExecutionEvent::NodeCompleted { name, duration, .. } => {
            println!("{} took {:?}", name, duration);
        }
        ExecutionEvent::NodeFailed { name, error, .. } => {
            eprintln!("{} failed: {}", name, error);
        }
        _ => {}
    }
}
```

### OpenTelemetry Export

Export traces in OTLP format for Jaeger, Datadog, Honeycomb, etc.:

```rust
use flowgentra_ai::core::observability::otel::{trace_to_otel_spans, export_to_otlp};

let spans = trace_to_otel_spans(&execution_trace);
export_to_otlp("http://localhost:4318/v1/traces", &spans).await?;
```

### Execution Replay

Step through past executions with full state snapshots:

```rust
use flowgentra_ai::core::observability::ReplayMode;

let replay = ReplayMode::new(trace);
let state_at_step_3 = replay.state_at(3);
let diff = replay.diff_states(2, 3); // What changed between steps
```

---

## Evaluation and Self-Correction

Automatically grade agent output and retry when quality is low:

```yaml
evaluation:
  enabled: true
  min_confidence: 0.8   # Retry if score < 80%
  max_retries: 3
  scoring:
    metrics: [relevance, completeness, accuracy]
    weights: [0.5, 0.3, 0.2]
```

---

## Dynamic Planning

Let the LLM decide what step to execute next based on current state:

```yaml
graph:
  planner:
    enabled: true
    max_plan_steps: 5
    prompt_template: |
      Current state: {current_state}
      Available actions: {available_nodes}
      What should we do next?
```

---

## Macros

### #[node] Proc Macro

Generate node factory functions from plain async functions:

```rust
use flowgentra_ai_macros::node;

#[node]
async fn my_processor(state: PlainState) -> Result<PlainState> {
    // your logic
    Ok(state)
}

// Generates: fn my_processor_node() -> impl Node<PlainState>
let graph = StateGraphBuilder::new()
    .add_node("processor", my_processor_node())
    .compile()?;
```

### #[register_handler]

Auto-register handlers for config-driven graph discovery:

```rust
#[register_handler]
pub async fn validate_input(state: State) -> Result<State> {
    Ok(state)
}
// Referenced in config.yaml as handler: "validate_input"
```

### #[derive(State)]

Derive the State trait for custom types automatically.

---

## Infrastructure

### Plugin System

Extend the framework with custom lifecycle hooks:

```rust
// Plugins can hook into: initialize, on_handler_start,
// on_handler_end, on_error, shutdown
```

### Middleware Pipeline

Composable middleware for cross-cutting concerns:
- Logging
- Caching (with TTL)
- Rate limiting (RPM + burst)
- Authentication

### Graph Validation

The compiler validates your graph at build time:
- All referenced nodes exist
- No orphaned/unreachable nodes
- START and END properly connected
- Cycle detection (unless intentional)

---

## Complete Feature Checklist

### Graph Engine
- [x] StateGraph builder with type-safe state
- [x] Conditional routing (sync and async)
- [x] Subgraph composition (SubgraphNode)
- [x] Parallel execution with join strategies
- [x] Graph export (DOT, Mermaid, JSON)
- [x] Human-in-the-loop (interrupt/resume)
- [x] Placeholder node detection
- [x] MessageGraphBuilder (chat-focused graph wrapper)
- [x] ToolNode (automatic tool call execution from state)
- [x] tools_condition (route to tools or end based on tool calls)

### State
- [x] PlainState (owned) and SharedState (thread-safe)
- [x] ScopedState (namespaced per node)
- [x] JsonReducer with 7 strategies
- [x] ReducerConfig for per-field merge control
- [x] MergeStrategy (Default, Replace, Merge)

### Agents
- [x] ZeroShotReAct, FewShotReAct, Conversational
- [x] Supervisor (multi-agent orchestration)

### LLM
- [x] 7 providers (OpenAI, Anthropic, Mistral, Groq, Azure, HuggingFace, Ollama)
- [x] RetryLLMClient with exponential backoff
- [x] CachedLLMClient (hash-based response caching)
- [x] FallbackLLMClient (try multiple providers in order)
- [x] Token counting and context window management
- [x] Cost tracking with model pricing table
- [x] Anthropic native tool calling (input_schema)
- [x] HuggingFace real SSE streaming (TGI)
- [x] Structured output (ResponseFormat: Text/Json/JsonSchema)
- [x] PromptTemplate ({variable} interpolation, partial formatting)
- [x] ChatPromptTemplate (multi-message prompt builder)
- [x] FewShotPromptTemplate (prefix + examples + suffix)
- [x] OutputParser trait (JsonOutputParser, ListOutputParser, StructuredOutputParser)

### Tools and MCP
- [x] Local tools (ToolSpec)
- [x] MCP transports (SSE, Stdio, Docker)
- [x] Reconnecting MCP client (auto-reconnect)
- [x] MCP resources and prompts protocol

### RAG
- [x] PineconeStore (REST API)
- [x] QdrantStore (REST API)
- [x] ChromaStore
- [x] InMemoryStore
- [x] Text splitters (Recursive, Markdown, Code, HTML, Token)
- [x] Retriever orchestrator (embed → search → rerank → dedup)
- [x] Cross-encoder reranker (HuggingFace API)
- [x] HuggingFace embeddings (Inference API + self-hosted TEI)
- [x] OpenAI embeddings
- [x] Mistral embeddings
- [x] Ollama embeddings
- [x] Cached embeddings (in-memory dedup)
- [x] Document loaders (PDF, text, Markdown, JSON, CSV, HTML)
- [x] Ingestion pipeline (load → split → embed → index)
- [x] Hybrid search (semantic + BM25 keyword matching)
- [x] Rerankers (RRF, LLM-based, cross-encoder, no-op)
- [x] Deduplication (by ID, by similarity threshold)
- [x] RAG evaluation (hit rate, MRR, NDCG)

### Memory
- [x] InMemoryCheckpointer
- [x] FileCheckpointer (JSON on disk)
- [x] Conversation memory with buffer window
- [x] TokenBufferMemory (token-budget sliding window)
- [x] SummaryMemory (LLM-based summarization of old messages)

### Observability
- [x] Event broadcaster (tokio broadcast channel)
- [x] OpenTelemetry export (OTLP-compatible)
- [x] Execution tracing and replay
- [x] State snapshots at each step
- [x] Configurable log levels

### Macros
- [x] #[node] (factory function generation)
- [x] #[register_handler] (auto-registration)
- [x] #[derive(State)]

### Evaluation
- [x] Auto-evaluation with quality scoring
- [x] Self-correction with retry loop
- [x] Configurable metrics and weights

### Planning
- [x] Dynamic LLM-driven step selection
- [x] Adaptive workflows

### Infrastructure
- [x] Plugin system with lifecycle hooks
- [x] Middleware pipeline (logging, cache, rate limit, auth)
- [x] Graph validation and compilation
