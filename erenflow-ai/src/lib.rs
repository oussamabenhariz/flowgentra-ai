//! ErenFlowAI: A LangGraph-inspired Rust library for building AI agents with graphs
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
//! ```no_run
//! use erenflow_ai::prelude::*;
//! use serde_json::json;
//!
//! #[register_handler]
//! pub async fn my_handler(mut state: State) -> Result<State> {
//!     let input = state.get("input").and_then(|v| v.as_str()).unwrap_or("");
//!     state.set("output", json!(input.to_uppercase()));
//!     Ok(state)
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let mut agent = from_config_path("config.yaml")?;
//!     let mut state = State::new();
//!     state.set("input", json!("hello"));
//!     let result = agent.run(state).await?;
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

// Re-export the register_handler macro
pub use erenflow_ai_macros::register_handler;
pub use core::agent::{from_config_path, ArcHandler, HandlerEntry};
pub use core::config::AgentConfig;
pub use core::error::{ErenFlowError, Result};
pub use core::graph::Graph;
pub use core::runtime::AgentRuntime;
pub use core::state::State;

/// Prelude module for commonly used types and macros.
///
/// Import everything you need for building agents:
///
/// ```no_run
/// use erenflow_ai::prelude::*;
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
        config::AgentConfig,
        error::{ErenFlowError, Result},
        graph::Graph,
        llm::{LLMConfig, LLMProvider, Message, MessageRole},
        mcp::MCPConfig,
        memory::{
            BufferWindowConfig, Checkpoint, CheckpointMetadata, Checkpointer, ConversationMemory,
            ConversationMemoryConfig, InMemoryConversationMemory, MemoryCheckpointer, MemoryConfig,
        },
        node::{Node, NodeFunction},
        rag::{
            EmbeddingModel, Embeddings, RAGConfig, RetrievalConfig, RetrieverStrategy, VectorStore,
        },
        runtime::AgentRuntime,
        state::State,
    };
    pub use crate::register_handler;
}
