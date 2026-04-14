//! FlowgentraAI: A Rust library for building AI agents with graphs
//!
//! Build AI agent workflows using a declarative graph structure with:
//! - **Typed State**: Define state as a Rust struct with `#[derive(State)]` — compile-time schema
//! - **Per-field Reducers**: Control merge behavior with `#[reducer(Append)]`, etc.
//! - **Nodes**: Async functions that return partial state updates
//! - **Edges**: Connections with optional conditional logic
//! - **LLM Integration**: Built-in support for OpenAI, Anthropic, Mistral, etc.
//! - **Auto-Discovery**: Automatic handler registration via `#[register_handler]`
//!
//! # Quick Start
//!
//! ```ignore
//! use flowgentra_ai::prelude::*;
//! use serde::{Serialize, Deserialize};
//! use std::sync::Arc;
//!
//! #[derive(State, Clone, Debug, Serialize, Deserialize)]
//! struct MyState {
//!     input: String,
//!     output: Option<String>,
//! }
//!
//! // Build a graph — use #[node] + add_node, or FunctionNode directly
//! #[node]
//! async fn process(state: &MyState, _ctx: &Context) {
//!     update! { output: state.input.to_uppercase() }
//! }
//!
//! let graph = StateGraph::<MyState>::builder()
//!     .add_node("process", process_node())
//!     .set_entry_point("process")
//!     .add_edge("process", "__end__")
//!     .compile()?;
//! ```

pub mod core;

/// Build a partial state update with minimal syntax.
///
/// Instead of calling `MyStateUpdate::new().field(val).other(val2)`,
/// write `update! { field: val, other: val2 }`.
///
/// - The update type is inferred from context — no need to name it.
/// - Values are **not** wrapped in `Some(...)` — the macro does that automatically.
///
/// # Example — inside `#[node]` (no `Ok`, no return type needed)
///
/// ```ignore
/// #[node]
/// async fn process(state: &MyState, _ctx: &Context) {
///     update! { result: state.input.to_uppercase(), count: state.count + 1 }
/// }
/// ```
#[macro_export]
macro_rules! update {
    ($($field:ident : $value:expr),* $(,)?) => {{
        let mut __update = ::std::default::Default::default();
        $( __update.$field = ::std::option::Option::Some($value); )*
        __update
    }};
}

pub use crate::prelude::*;
pub use core::agent::{
    from_config_path, from_config_path_with_extra_handlers, Agent, ArcHandler, Handler,
    HandlerEntry, HandlerRegistry,
};
pub use core::config::AgentConfig;
pub use core::error::{FlowgentraError, Result};
pub use core::graph::Graph;
pub use core::AgentRuntime;
pub use flowgentra_ai_macros as macros;
pub use flowgentra_ai_macros::{register_handler, State};

/// Prelude module for commonly used types and macros.
///
/// ```no_run
/// use flowgentra_ai::prelude::*;
/// ```
pub mod prelude {
    pub use crate::core::agent::from_config_path;
    pub use crate::core::agent::{
        from_config_path_with_extra_handlers, Agent, ArcHandler, Handler, HandlerEntry,
        HandlerRegistry,
    };
    pub use crate::core::{
        config::{
            AgentConfig, EmbeddingsConfig, PdfSettings, RAGGraphConfig, RetrievalSettings,
            StateField, VectorStoreConfig,
        },
        error::{FlowgentraError, Result},
        graph::{
            routing::{ComparisonOp, Condition, RoutingCondition},
            Graph,
        },
        llm::{
            create_llm_client, CachedLLMClient, FallbackLLMClient, LLMConfig, LLMProvider,
            Message, MessageRole, ResponseFormat, ToolCall, ToolDefinition,
        },
        mcp::MCPConfig,
        memory::{
            BufferWindowConfig, Checkpoint, CheckpointMetadata, Checkpointer, ConversationMemory,
            ConversationMemoryConfig, InMemoryConversationMemory, MemoryCheckpointer, MemoryConfig,
            SummaryConfig, SummaryMemory, TokenBufferMemory,
        },
        node::{Node, NodeFunction},
        reducer::{Append, Max, MergeMap, Min, Overwrite, Reducer, Sum},
        runtime::{AgentRuntime, CloneStats, OptimizedState},
        state::{Context, State},
        state_graph::{
            create_tool_node, node::FunctionNode, store_tool_calls, tools_condition,
            MessageGraphBuilder, MessageState, MessageStateUpdate, StateGraph, StateGraphBuilder,
            ToolState, ToolStateUpdate,
        },
    };

    // Channel-based dynamic state
    pub use crate::core::state::{
        apply_channel_reducer, Channel, ChannelType, Checkpointer as StateCheckpointer, DynState,
        DynStateUpdate, FieldSchema, FileCheckpointer as StateFileCheckpointer,
        MemoryCheckpointer as StateMemoryCheckpointer, StateSnapshot,
    };

    pub use crate::core::rag::{
        bm25_score, chunk_text, chunk_text_by_tokens, dedup_by_id, dedup_by_similarity,
        estimate_tokens, evaluate, extract_and_chunk, extract_text, hit_rate, hybrid_merge,
        load_directory, load_document, mean_ndcg, mrr, CachedEmbeddings, ChromaStore,
        ChunkMetadata, CodeTextSplitter, CrossEncoderReranker, Document, EmbeddingError,
        EmbeddingModel, Embeddings, EmbeddingsProvider, EvalQuery, EvalResults, FileType,
        HTMLTextSplitter, HuggingFaceEmbeddings, InMemoryVectorStore, IngestionPipeline,
        IngestionStats, LLMReranker, Language, LoadedDocument, MarkdownTextSplitter,
        MetadataFilter, MistralEmbeddings, MockEmbeddings, NoopReranker, OllamaEmbeddings,
        OpenAIEmbeddings, PdfDocument, QueryExpander, QueryResult as EvalQueryResult, RAGConfig,
        RAGHandlers, RAGNodeConfig, RRFReranker, RecursiveCharacterTextSplitter, RerankStrategy,
        Reranker, RetrievalConfig, Retriever, RetrieverStrategy, SearchResult, TextChunk,
        TextSplitter, TokenTextSplitter, VectorStore, VectorStoreBackend, VectorStoreError,
        VectorStoreType,
    };

    pub use crate::register_handler;
    // Macros for state definition and updates
    pub use crate::update;
    pub use flowgentra_ai_macros::State;

    pub use crate::core::node::evaluation_node::evaluate_output_score;
}
