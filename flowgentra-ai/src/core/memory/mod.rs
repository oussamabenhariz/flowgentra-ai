//! # Memory: checkpointer and conversation memory
//!
//! - **Checkpointer**: Persist state per thread for resume and multi-turn.
//! - **Conversation memory**: Message history per thread with optional buffer/window.
//!
//! Configure via config.yaml under `memory:` or set programmatically with
//! `agent.with_checkpointer(...)` and `agent.with_conversation_memory(...)`.

pub mod async_checkpointer;
mod checkpointer;
mod config;
mod conversation;
mod postgres_checkpointer;
mod redis_checkpointer;
mod sqlite_checkpointer;
mod summary;

pub use checkpointer::{
    Checkpoint, CheckpointMetadata, Checkpointer, GenericCheckpointer, MemoryCheckpointer,
};
pub use config::{CheckpointerConfig, ConversationMemoryConfig, MemoryConfig};
pub use conversation::{BufferWindowConfig, ConversationMemory, InMemoryConversationMemory};
pub use summary::{SummaryConfig, SummaryMemory, TokenBufferMemory};

// Async checkpointer trait and in-memory implementation
pub use async_checkpointer::{
    AsyncCheckpointer, AsyncMemoryCheckpointer, CheckpointHistoryEntry, NamespacedCheckpointer,
};

// Async SQLite checkpointer
#[cfg(feature = "sqlite")]
pub use async_checkpointer::sqlite_async::AsyncSqliteCheckpointer;

// Async Postgres checkpointer
#[cfg(feature = "postgres")]
pub use async_checkpointer::postgres_async::AsyncPostgresCheckpointer;

// Async Redis checkpointer
#[cfg(feature = "redis-store")]
pub use async_checkpointer::redis_async::AsyncRedisCheckpointer;

// Async MongoDB checkpointer
#[cfg(feature = "mongodb-store")]
pub use async_checkpointer::mongo_async::AsyncMongoCheckpointer;

// Async MySQL checkpointer
#[cfg(feature = "mysql")]
pub use async_checkpointer::mysql_async::AsyncMysqlCheckpointer;

// Sync checkpointers (original, block_in_place based)
#[cfg(feature = "sqlite")]
pub use sqlite_checkpointer::SqliteCheckpointer;

#[cfg(feature = "postgres")]
pub use postgres_checkpointer::PostgresCheckpointer;

#[cfg(feature = "redis-store")]
pub use redis_checkpointer::RedisCheckpointer;
