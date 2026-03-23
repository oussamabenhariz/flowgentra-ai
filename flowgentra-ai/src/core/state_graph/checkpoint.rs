//! Checkpointing system for fault tolerance and time-travel debugging

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::core::state::State;
use super::error::Result;

/// Checkpoint metadata
#[derive(Debug, Clone)]
pub struct Checkpoint<S: State> {
    /// Unique identifier for this execution thread
    pub thread_id: String,

    /// Super-step number (incremented after each node execution)
    pub step: usize,

    /// Node that just executed
    pub node_name: String,

    /// Full state snapshot
    pub state: S,

    /// When checkpoint was created
    pub timestamp: i64,

    /// Metadata (custom key-value pairs)
    pub metadata: HashMap<String, String>,
}

impl<S: State> Checkpoint<S> {
    pub fn new(
        thread_id: String,
        step: usize,
        node_name: String,
        state: S,
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        Self {
            thread_id,
            step,
            node_name,
            state,
            timestamp,
            metadata: HashMap::new(),
        }
    }
}

/// Trait for persisting and retrieving state checkpoints
#[async_trait]
pub trait Checkpointer<S: State>: Send + Sync {
    /// Save a checkpoint
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<()>;

    /// Load a specific checkpoint
    async fn load(&self, thread_id: &str, step: usize) -> Result<Option<Checkpoint<S>>>;

    /// Load the latest checkpoint for a thread
    async fn load_latest(&self, thread_id: &str) -> Result<Option<Checkpoint<S>>>;

    /// List all checkpoints for a thread
    async fn list_checkpoints(&self, thread_id: &str) -> Result<Vec<(usize, i64)>>;

    /// Delete a checkpoint
    async fn delete(&self, thread_id: &str, step: usize) -> Result<()>;

    /// Delete all checkpoints for a thread
    async fn delete_thread(&self, thread_id: &str) -> Result<()>;
}

/// In-memory checkpointer (default, for testing and single-session use)
pub struct InMemoryCheckpointer<S: State> {
    /// thread_id -> step -> Checkpoint
    storage: Arc<RwLock<HashMap<String, HashMap<usize, Checkpoint<S>>>>>,
}

impl<S: State> InMemoryCheckpointer<S> {
    pub fn new() -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl<S: State> Default for InMemoryCheckpointer<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: State> Clone for InMemoryCheckpointer<S> {
    fn clone(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
        }
    }
}

#[async_trait]
impl<S: State + Send + Sync> Checkpointer<S> for InMemoryCheckpointer<S> {
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<()> {
        let mut storage = self.storage.write().await;
        storage
            .entry(checkpoint.thread_id.clone())
            .or_insert_with(HashMap::new)
            .insert(checkpoint.step, checkpoint.clone());
        Ok(())
    }

    async fn load(&self, thread_id: &str, step: usize) -> Result<Option<Checkpoint<S>>> {
        let storage = self.storage.read().await;
        Ok(storage
            .get(thread_id)
            .and_then(|steps| steps.get(&step))
            .cloned())
    }

    async fn load_latest(&self, thread_id: &str) -> Result<Option<Checkpoint<S>>> {
        let storage = self.storage.read().await;
        Ok(storage
            .get(thread_id)
            .and_then(|steps| {
                steps
                    .iter()
                    .max_by_key(|(step, _)| *step)
                    .map(|(_, cp)| cp.clone())
            }))
    }

    async fn list_checkpoints(&self, thread_id: &str) -> Result<Vec<(usize, i64)>> {
        let storage = self.storage.read().await;
        let mut checkpoints: Vec<(usize, i64)> = storage
            .get(thread_id)
            .map(|steps| {
                steps
                    .iter()
                    .map(|(step, cp)| (*step, cp.timestamp))
                    .collect()
            })
            .unwrap_or_default();

        checkpoints.sort_by_key(|(step, _)| *step);
        Ok(checkpoints)
    }

    async fn delete(&self, thread_id: &str, step: usize) -> Result<()> {
        let mut storage = self.storage.write().await;
        if let Some(steps) = storage.get_mut(thread_id) {
            steps.remove(&step);
        }
        Ok(())
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        let mut storage = self.storage.write().await;
        storage.remove(thread_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use crate::core::state::PlainState;

    #[tokio::test]
    async fn test_in_memory_checkpointer() {
        let checkpointer = InMemoryCheckpointer::new();
        let mut state = PlainState::new();
        state.set("test", json!("value"));

        let cp = Checkpoint::new("thread1".to_string(), 0, "node1".to_string(), state.clone());
        checkpointer.save(&cp).await.unwrap();

        let loaded = checkpointer.load("thread1", 0).await.unwrap();
        assert!(loaded.is_some());

        let latest = checkpointer.load_latest("thread1").await.unwrap();
        assert!(latest.is_some());
    }
}
