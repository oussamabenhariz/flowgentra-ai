//! # Memory: checkpointer and conversation memory
//!
//! - **Checkpointer**: Persist state per thread for resume and multi-turn.
//! - **Conversation memory**: Message history per thread with optional buffer/window.
//!
//! Configure via config.yaml under `memory:` or set programmatically with
//! `agent.with_checkpointer(...)` and `agent.with_conversation_memory(...)`.

mod checkpointer;
mod config;
mod conversation;

pub use checkpointer::{Checkpoint, CheckpointMetadata, Checkpointer, MemoryCheckpointer};
pub use config::{CheckpointerConfig, ConversationMemoryConfig, MemoryConfig};
pub use conversation::{BufferWindowConfig, ConversationMemory, InMemoryConversationMemory};
