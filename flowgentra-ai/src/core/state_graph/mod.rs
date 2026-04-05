//! State graph module - graph execution engine
//!
//! # Overview
//!
//! The state graph system provides a Rust-idiomatic way to build AI agent workflows.
//! It supports:
//!
//! - **Typed state** flowing through a directed graph (compile-time schema via `#[derive(State)]`)
//! - **Nodes** as async functions that return partial updates (`S::Update`)
//! - **Per-field reducers** controlling how updates are merged (`#[reducer(Append)]`)
//! - **Conditional routing** based on state
//! - **Checkpointing** for fault tolerance and time-travel debugging
//! - **Human-in-the-loop** with breakpoints
//!
//! # Design philosophy
//!
//! - **Type-safe by default**: no node can add arbitrary keys — the struct IS the schema
//! - **Async-first**: all node execution is async-compatible
//! - **Ergonomic**: builder pattern for graph construction
//! - **Observable**: checkpointing and logging at every step

pub mod checkpoint;
pub mod edge;
pub mod error;
pub mod executor;
pub mod file_checkpointer;
pub mod message_graph;
pub mod node;
pub mod tool_node;

// Re-export public API
pub use checkpoint::{Checkpoint, CheckpointMigrator, Checkpointer, InMemoryCheckpointer};
pub use edge::{Edge, FixedEdge, END, START};
pub use error::{Result, StateGraphError};
pub use executor::{StateGraph, StateGraphBuilder, SubgraphNode};
pub use file_checkpointer::FileCheckpointer;
pub use message_graph::{MessageGraphBuilder, MessageState, MessageStateUpdate};
pub use node::{FunctionNode, Node};
pub use tool_node::{
    create_tool_node, store_tool_calls, tools_condition, ToolCallInfo, ToolExecutorFn, ToolResult,
    ToolState, ToolStateUpdate,
};

// Re-export common types
pub type NodeFn<S> = Box<
    dyn Fn(
            &S,
            &crate::core::state::Context,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<<S as crate::core::state::State>::Update>>
                    + Send,
            >,
        > + Send
        + Sync,
>;
pub type RouterFn<S> = Box<dyn Fn(&S) -> Result<String> + Send + Sync>;
