//! # Checkpointer
//!
//! Persists graph state per thread so runs can be resumed and multi-turn conversations
//! can maintain context.

use crate::core::error::Result;
use crate::core::state::DynState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

/// Metadata stored with each checkpoint (e.g. last node name, timestamp).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CheckpointMetadata {
    /// Last executed node name
    pub last_node: Option<String>,
    /// Execution path so far (node names)
    pub execution_path: Vec<String>,
    /// Optional custom metadata
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// A checkpoint is a saved state plus metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// State as JSON for serialization (use state() to get State).
    #[serde(rename = "state")]
    state_value: serde_json::Value,
    pub metadata: CheckpointMetadata,
}

impl Checkpoint {
    pub fn new(state: &DynState, metadata: CheckpointMetadata) -> Result<Self> {
        let state_value = state.to_value();
        Ok(Checkpoint {
            state_value,
            metadata,
        })
    }

    /// Get the state from this checkpoint.
    pub fn state(&self) -> Result<DynState> {
        DynState::from_json(self.state_value.clone())
    }
}

/// Saves and loads graph state keyed by thread id.
///
/// Implementations can use in-memory storage, SQLite, Redis, etc.
pub trait Checkpointer: Send + Sync {
    /// Load the latest checkpoint for the thread, if any.
    fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>>;

    /// List all thread ids that have at least one checkpoint (optional).
    fn list_threads(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
}

/// Trait for generic checkpoint saving.
pub trait GenericCheckpointer: Send + Sync {
    /// Save state and metadata for the given thread.
    fn save(&self, thread_id: &str, state: &DynState, metadata: &CheckpointMetadata) -> Result<()>;
}

/// Combined trait for both loading and saving checkpoints (used for trait objects).
#[allow(dead_code)]
pub trait CheckpointStore: Checkpointer + GenericCheckpointer {}

/// In-memory checkpointer for development and single-process use.
///
/// Data is lost when the process exits.
pub struct MemoryCheckpointer {
    store: RwLock<HashMap<String, Checkpoint>>,
}

impl Default for MemoryCheckpointer {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryCheckpointer {
    pub fn new() -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
        }
    }
}

impl Checkpointer for MemoryCheckpointer {
    fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        let guard = self
            .store
            .read()
            .map_err(|e| crate::core::error::FlowgentraError::StateError(e.to_string()))?;
        Ok(guard.get(thread_id).cloned())
    }

    fn list_threads(&self) -> Result<Vec<String>> {
        let guard = self
            .store
            .read()
            .map_err(|e| crate::core::error::FlowgentraError::StateError(e.to_string()))?;
        Ok(guard.keys().cloned().collect())
    }
}

impl GenericCheckpointer for MemoryCheckpointer {
    fn save(&self, thread_id: &str, state: &DynState, metadata: &CheckpointMetadata) -> Result<()> {
        let checkpoint = Checkpoint::new(state, metadata.clone())?;
        self.store
            .write()
            .map_err(|e| crate::core::error::FlowgentraError::StateError(e.to_string()))?
            .insert(thread_id.to_string(), checkpoint);
        Ok(())
    }
}

// Blanket impl: any type that implements both traits implements CheckpointStore
impl<C: Checkpointer + GenericCheckpointer + ?Sized> CheckpointStore for C {}
