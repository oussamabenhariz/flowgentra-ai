//! State graph module - graph execution engine
//!
//! # Overview
//!
//! The state graph system provides a Rust-idiomatic way to build AI agent workflows
//! It supports:
//!
//! - **Typed state** flowing through a directed graph
//! - **Nodes** as async functions that transform state
//! - **Conditional routing** based on state
//! - **Checkpointing** for fault tolerance and time-travel debugging
//! - **Human-in-the-loop** with breakpoints
//! - **Parallel execution** of independent nodes
//!
//! # Design philosophy
//!
//! - **Type-safe by default**: State type is checked at compile time
//! - **Async-first**: All node execution is async-compatible
//! - **Zero-copy where possible**: Use Arc<> for shared state
//! - **Ergonomic**: Builder pattern for graph construction
//! - **Observable**: Checkpointing and logging at every step

pub mod checkpoint;
pub mod edge;
pub mod error;
pub mod executor;
pub mod file_checkpointer;
pub mod message_graph;
pub mod node;
pub mod tool_node;

// Re-export public API
pub use checkpoint::{Checkpoint, Checkpointer, InMemoryCheckpointer};
pub use edge::{Edge, FixedEdge, END, START};
pub use error::{Result, StateGraphError};
pub use executor::{StateGraph, StateGraphBuilder, SubgraphNode};
pub use file_checkpointer::FileCheckpointer;
pub use message_graph::MessageGraphBuilder;
pub use node::{FunctionNode, MergeStrategy, Node, StateUpdate, UpdateNode};
pub use tool_node::{create_tool_node, store_tool_calls, tools_condition};

// Re-export common types
pub type NodeFn<S> = Box<
    dyn Fn(&S) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<S>> + Send>>
        + Send
        + Sync,
>;
pub type RouterFn<S> = Box<dyn Fn(&S) -> Result<String> + Send + Sync>;
