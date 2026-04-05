//! # Checkpointers — persist and retrieve state history by thread ID.
//!
//! A checkpointer stores `StateSnapshot`s keyed by `(thread_id, step_id)`.
//! The graph executor calls `save()` after every step and `load_latest()` on
//! resume, enabling interrupt/resume and time-travel workflows.
//!
//! Two built-in implementations are provided:
//!
//! | Type                | Storage          | Use case                    |
//! |---------------------|------------------|-----------------------------|
//! | `MemoryCheckpointer`| In-process `HashMap` | Tests, single-run agents |
//! | `FileCheckpointer`  | JSON files on disk   | Long-running, restartable  |
//!
//! # Example
//!
//! ```ignore
//! use flowgentra_ai::core::state::{DynState, MemoryCheckpointer};
//! use serde_json::json;
//!
//! let cp = MemoryCheckpointer::new();
//! let state = DynState::new();
//! state.set("step", json!(1));
//!
//! cp.save("thread-1", &state.snapshot("step-1"));
//! let snap = cp.load_latest("thread-1").unwrap();
//! assert_eq!(snap.get("step"), Some(&json!(1)));
//! ```

use crate::core::error::{FlowgentraError, Result};
use crate::core::state::snapshot::StateSnapshot;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

// ── Checkpointer trait ────────────────────────────────────────────────────────

/// Trait for persisting and retrieving state snapshots.
///
/// All methods are synchronous; async wrappers can be layered on top if needed.
pub trait Checkpointer: Send + Sync {
    /// Persist a snapshot for the given thread.
    fn save(&self, thread_id: &str, snapshot: &StateSnapshot) -> Result<()>;

    /// Load the most recent snapshot for a thread, or `None` if none exists.
    fn load_latest(&self, thread_id: &str) -> Option<StateSnapshot>;

    /// Load a specific snapshot by `step_id`.
    fn load(&self, thread_id: &str, step_id: &str) -> Option<StateSnapshot>;

    /// Return all snapshots for a thread, oldest first.
    fn list(&self, thread_id: &str) -> Vec<StateSnapshot>;

    /// Delete all snapshots for a thread.
    fn delete_thread(&self, thread_id: &str) -> Result<()>;

    /// Return all known thread IDs.
    fn thread_ids(&self) -> Vec<String>;
}

// ── MemoryCheckpointer ────────────────────────────────────────────────────────

/// In-process checkpointer backed by a `HashMap`.
///
/// Thread-safe via `Arc<RwLock<...>>`.  Snapshots are lost when the process exits.
#[derive(Clone)]
pub struct MemoryCheckpointer {
    // thread_id → ordered list of snapshots (oldest first)
    store: Arc<RwLock<HashMap<String, Vec<StateSnapshot>>>>,
}

impl MemoryCheckpointer {
    pub fn new() -> Self {
        MemoryCheckpointer {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for MemoryCheckpointer {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MemoryCheckpointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let store = self.store.read().unwrap_or_else(|p| p.into_inner());
        let threads: usize = store.len();
        let total: usize = store.values().map(|v| v.len()).sum();
        write!(f, "MemoryCheckpointer(threads={threads}, snapshots={total})")
    }
}

impl Checkpointer for MemoryCheckpointer {
    fn save(&self, thread_id: &str, snapshot: &StateSnapshot) -> Result<()> {
        let mut guard = self.store.write().map_err(|_| {
            FlowgentraError::StateError("MemoryCheckpointer: lock poisoned".to_string())
        })?;
        guard
            .entry(thread_id.to_string())
            .or_default()
            .push(snapshot.clone());
        Ok(())
    }

    fn load_latest(&self, thread_id: &str) -> Option<StateSnapshot> {
        self.store
            .read()
            .ok()?
            .get(thread_id)?
            .last()
            .cloned()
    }

    fn load(&self, thread_id: &str, step_id: &str) -> Option<StateSnapshot> {
        self.store
            .read()
            .ok()?
            .get(thread_id)?
            .iter()
            .find(|s| s.step_id == step_id)
            .cloned()
    }

    fn list(&self, thread_id: &str) -> Vec<StateSnapshot> {
        self.store
            .read()
            .ok()
            .and_then(|g| g.get(thread_id).cloned())
            .unwrap_or_default()
    }

    fn delete_thread(&self, thread_id: &str) -> Result<()> {
        let mut guard = self.store.write().map_err(|_| {
            FlowgentraError::StateError("MemoryCheckpointer: lock poisoned".to_string())
        })?;
        guard.remove(thread_id);
        Ok(())
    }

    fn thread_ids(&self) -> Vec<String> {
        self.store
            .read()
            .ok()
            .map(|g| g.keys().cloned().collect())
            .unwrap_or_default()
    }
}

// ── FileCheckpointer ──────────────────────────────────────────────────────────

/// Disk-based checkpointer: each snapshot is stored as a JSON file at
/// `<base_dir>/<thread_id>/<step_id>.json`.
///
/// The directory tree is created on first write.
#[derive(Clone, Debug)]
pub struct FileCheckpointer {
    base_dir: PathBuf,
}

impl FileCheckpointer {
    /// Create a new `FileCheckpointer` rooted at `base_dir`.
    ///
    /// The directory will be created on first `save()` call if it does not exist.
    pub fn new(base_dir: impl AsRef<Path>) -> Self {
        FileCheckpointer {
            base_dir: base_dir.as_ref().to_path_buf(),
        }
    }

    fn thread_dir(&self, thread_id: &str) -> PathBuf {
        self.base_dir.join(thread_id)
    }

    fn snapshot_path(&self, thread_id: &str, step_id: &str) -> PathBuf {
        // Replace path-unsafe characters in step_id with '_'
        let safe_step: String = step_id
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' })
            .collect();
        self.thread_dir(thread_id).join(format!("{}.json", safe_step))
    }
}

impl Checkpointer for FileCheckpointer {
    fn save(&self, thread_id: &str, snapshot: &StateSnapshot) -> Result<()> {
        let dir = self.thread_dir(thread_id);
        std::fs::create_dir_all(&dir).map_err(|e| {
            FlowgentraError::StateError(format!(
                "FileCheckpointer: failed to create dir {}: {}",
                dir.display(),
                e
            ))
        })?;

        let path = self.snapshot_path(thread_id, &snapshot.step_id);
        let json = serde_json::to_string_pretty(snapshot).map_err(FlowgentraError::from)?;
        std::fs::write(&path, json).map_err(|e| {
            FlowgentraError::StateError(format!(
                "FileCheckpointer: failed to write {}: {}",
                path.display(),
                e
            ))
        })?;
        Ok(())
    }

    fn load_latest(&self, thread_id: &str) -> Option<StateSnapshot> {
        // "Latest" = lexicographically last filename (relies on step_ids being
        //  sortable, or we use `created_at` from the parsed snapshot).
        let mut all = self.list(thread_id);
        if all.is_empty() {
            return None;
        }
        all.sort_by_key(|s| s.created_at);
        all.into_iter().last()
    }

    fn load(&self, thread_id: &str, step_id: &str) -> Option<StateSnapshot> {
        let path = self.snapshot_path(thread_id, step_id);
        let data = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&data).ok()
    }

    fn list(&self, thread_id: &str) -> Vec<StateSnapshot> {
        let dir = self.thread_dir(thread_id);
        let read_dir = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(_) => return vec![],
        };

        let mut snaps = Vec::new();
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if let Ok(data) = std::fs::read_to_string(&path) {
                if let Ok(snap) = serde_json::from_str::<StateSnapshot>(&data) {
                    snaps.push(snap);
                }
            }
        }
        snaps.sort_by_key(|s| s.created_at);
        snaps
    }

    fn delete_thread(&self, thread_id: &str) -> Result<()> {
        let dir = self.thread_dir(thread_id);
        if dir.exists() {
            std::fs::remove_dir_all(&dir).map_err(|e| {
                FlowgentraError::StateError(format!(
                    "FileCheckpointer: failed to remove {}: {}",
                    dir.display(),
                    e
                ))
            })?;
        }
        Ok(())
    }

    fn thread_ids(&self) -> Vec<String> {
        let read_dir = match std::fs::read_dir(&self.base_dir) {
            Ok(rd) => rd,
            Err(_) => return vec![],
        };
        read_dir
            .flatten()
            .filter(|e| e.path().is_dir())
            .filter_map(|e| e.file_name().into_string().ok())
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    fn make_snap(step_id: &str, field: &str, val: serde_json::Value) -> StateSnapshot {
        let mut state = HashMap::new();
        state.insert(field.to_string(), val);
        StateSnapshot::new(step_id, state)
    }

    // ── MemoryCheckpointer ──────────────────────────────────────────────────

    #[test]
    fn memory_save_and_load_latest() {
        let cp = MemoryCheckpointer::new();
        cp.save("t1", &make_snap("s1", "v", json!(1))).unwrap();
        cp.save("t1", &make_snap("s2", "v", json!(2))).unwrap();

        let snap = cp.load_latest("t1").unwrap();
        assert_eq!(snap.step_id, "s2");
        assert_eq!(snap.get("v"), Some(&json!(2)));
    }

    #[test]
    fn memory_load_by_step_id() {
        let cp = MemoryCheckpointer::new();
        cp.save("t1", &make_snap("step-a", "x", json!(10))).unwrap();
        cp.save("t1", &make_snap("step-b", "x", json!(20))).unwrap();

        let snap = cp.load("t1", "step-a").unwrap();
        assert_eq!(snap.get("x"), Some(&json!(10)));
    }

    #[test]
    fn memory_list_ordered() {
        let cp = MemoryCheckpointer::new();
        cp.save("t1", &make_snap("s1", "v", json!(1))).unwrap();
        cp.save("t1", &make_snap("s2", "v", json!(2))).unwrap();
        cp.save("t1", &make_snap("s3", "v", json!(3))).unwrap();

        let list = cp.list("t1");
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].step_id, "s1");
    }

    #[test]
    fn memory_delete_thread() {
        let cp = MemoryCheckpointer::new();
        cp.save("t1", &make_snap("s1", "v", json!(1))).unwrap();
        cp.delete_thread("t1").unwrap();
        assert!(cp.load_latest("t1").is_none());
    }

    #[test]
    fn memory_thread_ids() {
        let cp = MemoryCheckpointer::new();
        cp.save("thread-a", &make_snap("s1", "v", json!(1))).unwrap();
        cp.save("thread-b", &make_snap("s1", "v", json!(2))).unwrap();

        let mut ids = cp.thread_ids();
        ids.sort();
        assert_eq!(ids, vec!["thread-a", "thread-b"]);
    }

    // ── FileCheckpointer ────────────────────────────────────────────────────

    #[test]
    fn file_save_and_load_latest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cp = FileCheckpointer::new(dir.path());

        cp.save("t1", &make_snap("step-1", "val", json!("hello"))).unwrap();
        cp.save("t1", &make_snap("step-2", "val", json!("world"))).unwrap();

        let snap = cp.load_latest("t1").unwrap();
        assert_eq!(snap.get("val"), Some(&json!("world")));
    }

    #[test]
    fn file_load_by_step_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cp = FileCheckpointer::new(dir.path());
        cp.save("t1", &make_snap("s-abc", "n", json!(42))).unwrap();

        let snap = cp.load("t1", "s-abc").unwrap();
        assert_eq!(snap.step_id, "s-abc");
        assert_eq!(snap.get("n"), Some(&json!(42)));
    }

    #[test]
    fn file_delete_thread() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cp = FileCheckpointer::new(dir.path());
        cp.save("t1", &make_snap("s1", "v", json!(1))).unwrap();
        cp.delete_thread("t1").unwrap();
        assert!(cp.load_latest("t1").is_none());
    }
}
