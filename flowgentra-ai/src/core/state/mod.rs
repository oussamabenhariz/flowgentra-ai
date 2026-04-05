//! # Typed State System
//!
//! State in FlowgentraAI is a user-defined Rust struct — like LangGraph's `TypedDict`.
//!
//! - **Compile-time schema**: the struct fields ARE the schema. No node can add arbitrary keys.
//! - **Per-field reducers**: annotate fields with `#[reducer(Append)]`, `#[reducer(Sum)]`, etc.
//! - **Partial updates**: nodes return `S::Update` (auto-generated struct with all fields `Option<T>`).
//!
//! # Example
//!
//! ```ignore
//! use flowgentra_ai::prelude::*;
//! use serde::{Serialize, Deserialize};
//!
//! #[derive(State, Clone, Debug, Serialize, Deserialize)]
//! struct MyState {
//!     query: String,
//!
//!     #[reducer(Append)]
//!     messages: Vec<Message>,
//!
//!     result: Option<String>,
//! }
//! ```

use serde::de::DeserializeOwned;
use serde::Serialize;

/// Core trait for typed graph state.
///
/// Implement via `#[derive(State)]` — do not implement manually.
pub trait State: Clone + Serialize + DeserializeOwned + Send + Sync + 'static {
    /// Partial update type — all fields wrapped in `Option<T>`.
    type Update: Default + Clone + Send + Sync;

    /// Apply a partial update to this state using per-field reducers.
    fn apply_update(&mut self, update: Self::Update);
}

/// Optional trait for validating state structs
pub trait ValidateState {
    fn validate(&self) -> Result<(), String>;
}

// Context for framework-managed resources (LLM, MCP, RAG)
pub mod context;
pub use context::Context;

// Channel primitives — ChannelType, FieldSchema, Channel, apply_channel_reducer
pub mod channel;
pub use channel::{apply_channel_reducer, Channel, ChannelType, FieldSchema};

// StateSnapshot — immutable capture of state at a single step
pub mod snapshot;
pub use snapshot::StateSnapshot;

// Checkpointers — MemoryCheckpointer, FileCheckpointer, Checkpointer trait
pub mod checkpointer;
pub use checkpointer::{Checkpointer, FileCheckpointer, MemoryCheckpointer};

// DynState — channel-backed JSON state
pub mod dyn_state;
pub use dyn_state::{DynState, DynStateUpdate};

// Legacy dynamic module — kept for compile compatibility, re-exports DynState
pub mod dynamic;

// State validation (schema-based)
pub mod state_validation;
pub(crate) use state_validation::*;

// Memory management helpers
pub mod state_management;
pub use state_management::{CompressionManager, CustomState, MessageHistory};

// Removed modules (kept as empty files for compilation)
pub mod scoped;
pub mod shared;
pub mod state_ext;
pub mod typed;
