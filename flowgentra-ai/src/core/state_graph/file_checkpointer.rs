//! File-based checkpointer — persists state checkpoints as JSON files on disk.
//!
//! Each thread gets its own directory, and each checkpoint is a separate JSON file.
//!
//! # Directory layout
//! ```text
//! <base_dir>/
//!   <thread_id>/
//!     step_0000.json
//!     step_0001.json
//!     ...
//! ```

use async_trait::async_trait;
use std::path::{Path, PathBuf};

use super::checkpoint::{Checkpoint, Checkpointer};
use super::error::{Result, StateGraphError};
use crate::core::state::State;

/// File-based checkpointer that persists checkpoints as JSON files.
pub struct FileCheckpointer {
    base_dir: PathBuf,
}

impl FileCheckpointer {
    /// Create a new file checkpointer rooted at the given directory.
    pub fn new(base_dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_dir)?;
        Ok(Self { base_dir })
    }

    fn thread_dir(&self, thread_id: &str) -> PathBuf {
        self.base_dir.join(thread_id)
    }

    fn step_file(&self, thread_id: &str, step: usize) -> PathBuf {
        self.thread_dir(thread_id)
            .join(format!("step_{:04}.json", step))
    }

    async fn scan_steps(&self, thread_id: &str) -> Result<Vec<usize>> {
        let dir = self.thread_dir(thread_id);
        if !dir.exists() {
            return Ok(Vec::new());
        }

        let mut steps = Vec::new();
        let mut entries = tokio::fs::read_dir(&dir)
            .await
            .map_err(|e| StateGraphError::CheckpointError(format!("ReadDir: {}", e)))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| StateGraphError::CheckpointError(format!("ReadDir: {}", e)))?
        {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(step) = parse_step_filename(&name) {
                steps.push(step);
            }
        }

        steps.sort();
        Ok(steps)
    }
}

fn parse_step_filename(name: &str) -> Option<usize> {
    let stem = name.strip_prefix("step_")?.strip_suffix(".json")?;
    stem.parse().ok()
}

/// Serializable checkpoint representation.
#[derive(serde::Serialize, serde::Deserialize)]
struct CheckpointFile {
    thread_id: String,
    step: usize,
    node_name: String,
    state: serde_json::Value,
    timestamp: i64,
    metadata: std::collections::HashMap<String, String>,
}

#[async_trait]
impl<S: State + Send + Sync + serde::Serialize + serde::de::DeserializeOwned> Checkpointer<S>
    for FileCheckpointer
{
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<()> {
        let dir = self.thread_dir(&checkpoint.thread_id);
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| StateGraphError::CheckpointError(format!("Create dir: {}", e)))?;

        let file = CheckpointFile {
            thread_id: checkpoint.thread_id.clone(),
            step: checkpoint.step,
            node_name: checkpoint.node_name.clone(),
            state: serde_json::to_value(&checkpoint.state)
                .map_err(|e| StateGraphError::CheckpointError(format!("Serialize: {}", e)))?,
            timestamp: checkpoint.timestamp,
            metadata: checkpoint.metadata.clone(),
        };

        let json = serde_json::to_string_pretty(&file)
            .map_err(|e| StateGraphError::CheckpointError(format!("Serialize: {}", e)))?;

        let path = self.step_file(&checkpoint.thread_id, checkpoint.step);
        tokio::fs::write(&path, json)
            .await
            .map_err(|e| StateGraphError::CheckpointError(format!("Write: {}", e)))?;

        tracing::debug!(thread_id = %checkpoint.thread_id, step = checkpoint.step, "Checkpoint saved to file");
        Ok(())
    }

    async fn load(&self, thread_id: &str, step: usize) -> Result<Option<Checkpoint<S>>> {
        let path = self.step_file(thread_id, step);
        if !path.exists() {
            return Ok(None);
        }

        let json = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| StateGraphError::CheckpointError(format!("Read: {}", e)))?;

        let file: CheckpointFile = serde_json::from_str(&json)
            .map_err(|e| StateGraphError::CheckpointError(format!("Deserialize: {}", e)))?;

        let state: S = serde_json::from_value(file.state)
            .map_err(|e| StateGraphError::CheckpointError(format!("State deserialize: {}", e)))?;

        Ok(Some(Checkpoint {
            thread_id: file.thread_id,
            step: file.step,
            node_name: file.node_name,
            state,
            timestamp: file.timestamp,
            metadata: file.metadata,
            schema_version: "1.0".to_string(),
        }))
    }

    async fn load_latest(&self, thread_id: &str) -> Result<Option<Checkpoint<S>>> {
        let steps = self.scan_steps(thread_id).await?;
        match steps.last() {
            Some(&step) => self.load(thread_id, step).await,
            None => Ok(None),
        }
    }

    async fn list_checkpoints(&self, thread_id: &str) -> Result<Vec<(usize, i64)>> {
        let steps = self.scan_steps(thread_id).await?;
        let mut results = Vec::new();
        for step in steps {
            if let Some(cp) = <Self as Checkpointer<S>>::load(self, thread_id, step).await? {
                results.push((cp.step, cp.timestamp));
            }
        }
        Ok(results)
    }

    async fn delete(&self, thread_id: &str, step: usize) -> Result<()> {
        let path = self.step_file(thread_id, step);
        if path.exists() {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|e| StateGraphError::CheckpointError(format!("Delete: {}", e)))?;
        }
        Ok(())
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        let dir = self.thread_dir(thread_id);
        if dir.exists() {
            tokio::fs::remove_dir_all(&dir)
                .await
                .map_err(|e| StateGraphError::CheckpointError(format!("DeleteThread: {}", e)))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state_graph::message_graph::MessageState;
    use crate::core::llm::Message;

    #[tokio::test]
    async fn test_file_checkpointer_save_load() {
        let tmp = tempfile::tempdir().unwrap();
        let cp = FileCheckpointer::new(tmp.path()).unwrap();

        let state = MessageState::new(vec![Message::user("hello")]);

        let checkpoint = Checkpoint::new("thread1".to_string(), 0, "node1".to_string(), state);
        cp.save(&checkpoint).await.unwrap();

        let loaded: Checkpoint<MessageState> = cp.load("thread1", 0).await.unwrap().unwrap();
        assert_eq!(loaded.node_name, "node1");
        assert_eq!(loaded.state.messages.len(), 1);
    }

    #[tokio::test]
    async fn test_file_checkpointer_latest() {
        let tmp = tempfile::tempdir().unwrap();
        let cp = FileCheckpointer::new(tmp.path()).unwrap();

        for step in 0..3 {
            let state = MessageState::new(vec![Message::user(format!("step {}", step))]);
            let checkpoint =
                Checkpoint::new("thread1".to_string(), step, format!("node_{}", step), state);
            cp.save(&checkpoint).await.unwrap();
        }

        let latest: Checkpoint<MessageState> = cp.load_latest("thread1").await.unwrap().unwrap();
        assert_eq!(latest.step, 2);
    }

    #[tokio::test]
    async fn test_file_checkpointer_delete() {
        let tmp = tempfile::tempdir().unwrap();
        let cp = FileCheckpointer::new(tmp.path()).unwrap();

        let state = MessageState::empty();
        let checkpoint = Checkpoint::new("t1".to_string(), 0, "n1".to_string(), state);
        cp.save(&checkpoint).await.unwrap();

        <FileCheckpointer as Checkpointer<MessageState>>::delete(&cp, "t1", 0)
            .await
            .unwrap();
        let deleted: Option<Checkpoint<MessageState>> = cp.load("t1", 0).await.unwrap();
        assert!(deleted.is_none());
    }
}
