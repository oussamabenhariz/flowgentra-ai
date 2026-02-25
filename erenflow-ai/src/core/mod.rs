//! # ErenFlowAI Core Modules
//!
//! This module provides the foundational building blocks for creating intelligent agents.
//!
//! ## Module Overview
//!
//! - **[error]** - Error types and result handling
//! - **[state]** - State management across node execution
//! - **[llm]** - LLM provider integration (OpenAI, Anthropic, Mistral, etc.)
//! - **[mcp]** - Model Context Protocol support for external tools
//! - **[node]** - Individual computational steps with handlers and conditions
//! - **[config]** - YAML configuration loading and validation
//! - **[graph]** - Graph data structure and operations
//! - **[runtime]** - Execution engine that orchestrates node execution
//! - **[evaluation]** - Auto-evaluation, scoring, grading, and self-correction
//! - **[agent]** - High-level API for easy agent creation and execution
//!
//! ## Quick Start
//!
//! ```ignore
//! use erenflow_ai::prelude::*;
//! use serde_json::json;
//! use std::collections::HashMap;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // 1. Define your handlers
//!     let mut handlers = HashMap::new();
//!     handlers.insert("my_handler".to_string(), Box::new(|state| {
//!         Box::pin(async move {
//!             // Your logic here
//!             Ok(state)
//!         })
//!     }) as crate::core::agent::Handler);
//!
//!     // 2. Create agent from config
//!     let mut agent = Agent::from_config("config.yaml", handlers, HashMap::new())?;
//!
//!     // 3. Prepare initial state
//!     let mut state = State::new();
//!     state.set("input", json!("your input"));
//!
//!     // 4. Execute
//!     let result = agent.run(state).await?;
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! The framework follows a modular, layered architecture:
//!
//! ```text
//!     Agent (User-facing API)
//!        ↓
//!     Runtime (Execution Engine)
//!        ↓
//!     Graph (Workflow Structure)
//!        ├─ Nodes (Computation)
//!        └─ Edges (Connections)
//!        ↓
//!     Supporting Services
//!        ├─ State (Data)
//!        ├─ LLM (AI)
//!        └─ MCP (Tools)
//! ```

// Core modules - organized by dependency layer
pub mod agent;
pub mod agents;
pub mod builder;
pub mod config;
pub mod error;
pub mod evaluation;
pub mod graph;
pub mod health;
pub mod llm;
pub mod mcp;
pub mod memory;
pub mod middleware;
pub mod node;
#[cfg(feature = "observability")]
pub mod observability;
pub mod rag;
pub mod runtime;
pub mod state;
pub mod tools;
pub mod utils;

// Re-export from submodules for backward compatibility
pub use builder::builders;
pub use graph::routing;
pub use node::{advanced_nodes, builtin_nodes, nodes_trait};
pub use runtime::{context, parallel};
#[cfg(feature = "visualization")]
pub use utils::visualization;
pub use utils::{debug, tracing};

// Re-export commonly used types for convenience
pub use agent::Agent;
pub use config::{AgentConfig, PlannerConfig};
pub use error::{ErenFlowError, Result};
pub use evaluation::{
    AutoEvaluationMiddleware, ConfidenceConfig, ConfidenceLevel, ConfidenceScore, ConfidenceScorer,
    EvaluationPolicy, EvaluationResult, EvaluationResultBuilder, GradeResult, LLMGrader, NodeScore,
    NodeScorer, RetryConfig, RetryPolicy, RetryResult, ScoringCriteria,
};
pub use state::state_validation::{FieldType, FieldValidator, StateSchema};
pub use state::State;
pub use utils::debug::{DebugConfig, ExecutionDebugInfo};
pub use utils::tracing::{init_tracing, ExecutionEvent, ExecutionTrace};

pub use advanced_nodes::{
    JoinNodeConfig, JoinType, LoopNodeConfig, MergeStrategy, ParallelNodeConfig, SubgraphNodeConfig,
};
pub use builders::AgentConfigBuilder;
pub use builtin_nodes::{
    ConditionalRouter, HumanInTheLoopNode, LLMNode, RetryNode, TimeoutNode, ToolNode,
};
pub use context::ExecutionContext;
pub use graph::Graph;
pub use health::{AgentHealth, ComponentHealth, HealthStatus};
pub use llm::{LLMConfig, LLMProvider, Message, MessageRole};
pub use mcp::{MCPConfig, MCPConnectionType};
pub use memory::{
    BufferWindowConfig, Checkpoint, CheckpointMetadata, Checkpointer, CheckpointerConfig,
    ConversationMemory, ConversationMemoryConfig, InMemoryConversationMemory, MemoryCheckpointer,
    MemoryConfig,
};
pub use middleware::{
    ExecutionContext as MiddlewareContext, Middleware, MiddlewarePipeline, MiddlewareResult,
};
pub use node::{Edge, EdgeConfig, Node, NodeConfig};
pub use nodes_trait::{
    ConditionalRouterConfig, HumanInTheLoopConfig, LLMNodeConfig, NodeOutput, PluggableNode,
    RetryNodeConfig, TimeoutNodeConfig, ToolNodeConfig,
};
#[cfg(feature = "observability-ui")]
pub use observability::TracingUIServer;
#[cfg(feature = "observability")]
pub use observability::{
    record_token_usage, ExecutionTrace, FailureSnapshot, NodeTiming, ObservabilityMiddleware,
    PathSegment, ReplayMode, TOKEN_USAGE_STATE_KEY,
};
pub use parallel::{BranchResult, BranchSync, ParallelExecutor};
pub use rag::{
    Document, EmbeddingModel, Embeddings, RAGConfig, RAGHandlers, RAGNodeConfig, RetrievalConfig,
    RetrieverStrategy, SearchResult, VectorStore, VectorStoreError,
};
pub use routing::{ComparisonOp, Condition, ConditionBuilder, FieldTypeCheck};
pub use runtime::AgentRuntime;
pub use tools::{
    builtin::{CalculatorTool, FilesTool, SearchTool, WebRequestTool},
    JsonSchema, Tool, ToolCallRequest, ToolCallResult, ToolDefinition, ToolRegistry,
};
#[cfg(feature = "visualization")]
pub use visualization::{
    visualize_graph, visualize_graph_with_execution, ExecutionOverlay, VisualizationConfig,
};
