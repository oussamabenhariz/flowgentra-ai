//! State graph module - LangGraph-inspired graph execution engine
//!
//! # Overview
//!
//! The state graph system provides a Rust-idiomatic way to build AI agent workflows
//! inspired by LangGraph (Python). It supports:
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

pub mod error;
pub mod node;
pub mod edge;
pub mod checkpoint;
pub mod executor;
pub mod file_checkpointer;

// Re-export public API
pub use error::{StateGraphError, Result};
pub use node::{Node, FunctionNode, UpdateNode, StateUpdate, MergeStrategy};
pub use edge::{Edge, FixedEdge, START, END};
pub use checkpoint::{Checkpoint, Checkpointer, InMemoryCheckpointer};
pub use file_checkpointer::FileCheckpointer;
pub use executor::{StateGraph, StateGraphBuilder};

// Re-export common types
pub type NodeFn<S> = Box<dyn Fn(&S) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<S>> + Send>> + Send + Sync>;
pub type RouterFn<S> = Box<dyn Fn(&S) -> Result<String> + Send + Sync>;
