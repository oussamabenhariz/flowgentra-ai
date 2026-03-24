//! ## API Consistency
//
//! - Use consistent naming for all public APIs (e.g., `run`, `execute`, `merge_state`).
//! - Only expose what’s needed; keep internals private.
//! - Clearly mark and document any deprecated APIs.
//! # FlowgentraAI Core Modules
//
//! This module provides the foundational building blocks for creating intelligent agents.
//
//! ## Module Overview
//
//! - **[error]** - Error types and result handling
//! - **[state]** - Typed state management across node execution
//! - **[reducer]** - Field-wise reducer system for state merging
//! - **[macros]** - Procedural macros for state/reducer generation
//! - **[llm]** - LLM provider integration (OpenAI, Anthropic, Mistral, etc.)
//! - **[mcp]** - Model Context Protocol support for external tools
//! - **[node]** - Individual computational steps with handlers and conditions
//! - **[config]** - YAML configuration loading and validation
//! - **[graph]** - Graph data structure and operations
//! - **[runtime]** - Execution engine that orchestrates node execution
//! - **[evaluation]** - Auto-evaluation, scoring, grading, and self-correction
//! - **[agent]** - High-level API for easy agent creation and execution
pub mod reducer;

//
// ## Architecture
//
// The framework follows a modular, layered architecture:
//
// ```text
//     Agent (User-facing API)
//        ↓
//     Runtime (Execution Engine)
//        ↓
//     Graph (Workflow Structure)
//        ├─ Nodes (Computation)
//        └─ Edges (Connections)
//        ↓
//     Supporting Services
//        ├─ State (Data)
//        ├─ LLM (AI)
//        └─ MCP (Tools)
/// ```
// Core modules - organized by dependency layer
pub mod agent;
pub mod agents;
pub mod builder;
pub mod config;
pub mod error;
pub mod evaluation;
pub mod graph;
pub mod llm;
pub mod mcp;
pub mod memory;
pub mod middleware;
pub mod node;
pub mod observability;
pub mod plugins;
pub mod rag;
pub mod runtime;
pub mod state;
pub mod state_graph;
pub mod tools;
pub mod utils;
pub mod validation;

// Re-export from submodules for backward compatibility
pub use graph::routing;
pub use node::{advanced_nodes, builtin_nodes, nodes_trait};
pub use reducer::{JsonReducer, ReducerConfig};
pub use runtime::{context, parallel};
pub use state_graph::{
    FunctionNode, InMemoryCheckpointer, StateGraph, StateGraphBuilder, StateGraphError,
    StateUpdate, END, START,
};
pub use utils::visualization;
pub use utils::{debug, tracing};

// Re-export commonly used types for convenience
pub use agent::Agent;
pub use agent::MemoryAwareAgent;
pub use agent::MemoryStats;
pub use config::{AgentConfig, EvaluationConfig, GradingConfig, PlannerConfig, ScoringConfig};
pub use error::{FlowgentraError, Result};
pub use state::state_validation::{FieldType, FieldValidator, StateSchema};
pub use state::State;
pub use utils::debug::{DebugConfig, ExecutionDebugInfo};
pub use utils::tracing::{init_tracing, ExecutionEvent, ExecutionTrace as TracingExecutionTrace};
pub use validation::ValidationError;

pub use advanced_nodes::{
    JoinNodeConfig, JoinType, LoopNodeConfig, MergeStrategy, ParallelNodeConfig, SubgraphNodeConfig,
};
pub use context::ExecutionContext;
pub use graph::Graph;
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
pub use node::builtin_nodes::{
    ConditionalRouter, HumanInTheLoopNode, LLMNode, RetryNode, TimeoutNode, ToolNode,
};
pub use node::{Edge, EdgeConfig, Node, NodeConfig};
pub use nodes_trait::{
    ConditionalRouterConfig, HumanInTheLoopConfig, LLMNodeConfig, NodeOutput, PluggableNode,
    RetryNodeConfig, TimeoutNodeConfig, ToolNodeConfig,
};
pub use observability::{
    record_token_usage, FailureSnapshot, GraphVisualizer, NodeTiming, ObservabilityMiddleware,
    PathSegment, ReplayMode, TOKEN_USAGE_STATE_KEY,
};
pub use parallel::{BranchResult, BranchSync, ParallelExecutor};
pub use plugins::{Plugin, PluginContext, PluginId, PluginRegistry};
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
pub use visualization::{
    visualize_graph, visualize_graph_with_execution, ExecutionOverlay, VisualizationConfig,
};
