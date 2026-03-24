//! ## Project Hygiene
//!
//! - Enforce `cargo fmt` and `clippy` in CI.
//! - Maintain a `CHANGELOG.md` for releases.
//! - Keep `CONTRIBUTING.md` up to date.
//! FlowgentraAI: A Rust library for building AI agents with graphs
//!
//! Build AI agent workflows using a declarative graph structure with:
//! - **Nodes**: Computational steps (handlers)
//! - **Edges**: Connections with optional conditional logic
//! - **State**: Shared JSON data flowing between nodes
//! - **LLM Integration**: Built-in support for OpenAI, Anthropic, Mistral, etc.
//! - **Auto-Discovery**: Automatic handler registration via `#[register_handler]`
//!
//! # Quick Start
//!
//! ```ignore
//! use flowgentra_ai::prelude::*;
//! use serde_json::json;
//!
//! pub async fn my_handler(state: SharedState) -> Result<SharedState> {
//!     let input = state.get("input").and_then(|v| v.as_str().map(|s| s.to_string())).unwrap_or_default();
//!     state.set("output", json!(input.to_uppercase()));
//!     Ok(state)
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let mut agent = from_config_path("config.yaml")?;
//!     agent.state.set("input", json!("hello"));
//!     let result = agent.run().await?;
//!     Ok(())
//! }
//! ```
//!
//! # Architecture
//!
//! The library is organized by layer:
//! - [`core::agent`] - High-level agent API and handler registration
//! - [`core::runtime`] - Execution engine and graph orchestration
//! - [`core::graph`] - Graph structure (nodes and edges)
//! - [`core::state`] - State management and validation
//! - [`core::config`] - YAML configuration loading
//! - [`core::llm`] - LLM provider integration
//! - [`core::mcp`] - Model Context Protocol support
//!
//! For most use cases, import from the [`prelude`] module.

pub mod core;

pub use crate::prelude::*;
// Re-export from core submodules
pub use core::agent::{
    from_config_path, Agent, ArcHandler, Handler, HandlerEntry, HandlerRegistry,
};
pub use core::config::AgentConfig;
pub use core::error::{FlowgentraError, Result};
pub use core::graph::{Graph, GraphBuilder};
pub use core::AgentRuntime;
// These are re-exported via prelude; remove direct pub use to avoid unresolved import errors.
pub use flowgentra_ai_macros::register_handler;

/// Prelude module for commonly used types and macros.
///
/// Import everything you need for building agents:
///
/// ```no_run
/// use flowgentra_ai::prelude::*;
/// ```
///
/// This re-exports:
/// - `#[register_handler]` - Attribute macro for handler registration
/// - `from_config_path()` - Create agents with auto-discovered handlers
/// - `Agent`, `State`, `Result` - Core types
/// - `AgentConfig`, `LLMConfig` - Configuration types
/// - And other commonly used utilities
pub mod prelude {
    pub use crate::core::agent::from_config_path;
    pub use crate::core::agent::{Agent, ArcHandler, Handler, HandlerEntry, HandlerRegistry};
    pub use crate::core::{
        config::{AgentConfig, StateField, RAGGraphConfig, VectorStoreConfig, EmbeddingsConfig, RetrievalSettings, PdfSettings},
        error::{FlowgentraError, Result},
        graph::{
            routing::{ComparisonOp, Condition, RoutingCondition},
            Graph, GraphBuilder,
        },
        llm::{LLMConfig, LLMProvider, Message, MessageRole, ResponseFormat, ToolCall, ToolDefinition},
        mcp::MCPConfig,
        memory::{
            BufferWindowConfig, Checkpoint, CheckpointMetadata, Checkpointer, ConversationMemory,
            ConversationMemoryConfig, InMemoryConversationMemory, MemoryCheckpointer, MemoryConfig,
        },
        node::{Node, NodeFunction},
        runtime::{AgentRuntime, CloneStats, OptimizedState},
        state::{PlainState, SharedState, StateExt, TypedState},
        State,
    };

    pub use crate::core::rag::{
        // Core types
        ChromaStore, Document, EmbeddingError, EmbeddingModel, Embeddings, InMemoryVectorStore,
        MetadataFilter, MistralEmbeddings, OllamaEmbeddings, OpenAIEmbeddings, PdfDocument,
        QueryExpander, RAGConfig, RAGHandlers, RAGNodeConfig, RetrievalConfig, RetrieverStrategy,
        SearchResult, VectorStore, VectorStoreError, VectorStoreType,
        // PDF utilities
        chunk_text, chunk_text_by_tokens, estimate_tokens, extract_and_chunk, extract_text,
        // New features
        CachedEmbeddings, dedup_by_id, dedup_by_similarity, FileType, LoadedDocument,
        load_document, load_directory, IngestionPipeline, IngestionStats,
        LLMReranker, NoopReranker, RRFReranker, RerankStrategy, Reranker,
        hybrid_merge, bm25_score,
        EvalQuery, EvalResults, QueryResult as EvalQueryResult, evaluate, hit_rate, mean_ndcg, mrr,
    };

    pub use crate::register_handler;

    // ── Library utility functions users can call inside their own handlers ──
    // These are the same functions the built-in nodes use internally, exposed
    // so users can compose their own logic without re-implementing them.
    pub use crate::core::node::evaluation_node::evaluate_output_score;
}
