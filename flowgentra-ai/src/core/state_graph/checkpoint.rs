//! Checkpointing system for fault tolerance and time-travel debugging
//!
//! ## Schema Evolution
//!
//! Every `Checkpoint` carries a `schema_version` string. When you change your
//! state struct in a way that is not backward-compatible, bump the version so
//! old checkpoints are not silently misinterpreted.
//!
//! ```ignore
//! // Register a migration from "1.0" → "2.0":
//! let mut migrator = CheckpointMigrator::new("2.0");
//! migrator.register("1.0", |mut raw| {
//!     // e.g. rename a field
//!     if let Some(obj) = raw.as_object_mut() {
//!         if let Some(v) = obj.remove("old_field") {
//!             obj.insert("new_field".to_string(), v);
//!         }
//!     }
//!     raw
//! });
//!
//! // Use in your checkpointer:
//! let raw_state = serde_json::to_value(&checkpoint.state)?;
//! let migrated = migrator.migrate("1.0", raw_state)?;
//! let state: MyState = serde_json::from_value(migrated)?;
//! ```

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::error::Result;
use crate::core::state::State;

/// Checkpoint metadata
#[derive(Debug, Clone)]
pub struct Checkpoint<S: State> {
    pub thread_id: String,
    pub step: usize,
    pub node_name: String,
    pub state: S,
    pub timestamp: i64,
    pub metadata: HashMap<String, String>,
    /// Schema version of the state struct at the time this checkpoint was saved.
    ///
    /// Set this to the current version of your state struct so that future
    /// readers can detect and migrate stale checkpoints.
    /// Defaults to `"1.0"`.
    pub schema_version: String,
}

impl<S: State> Checkpoint<S> {
    pub fn new(thread_id: String, step: usize, node_name: String, state: S) -> Self {
        Self::with_version(thread_id, step, node_name, state, "1.0")
    }

    /// Create a checkpoint with an explicit schema version.
    pub fn with_version(
        thread_id: String,
        step: usize,
        node_name: String,
        state: S,
        schema_version: impl Into<String>,
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
            schema_version: schema_version.into(),
        }
    }

    /// Add a metadata entry to this checkpoint.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ─── CheckpointMigrator ────────────────────────────────────────────────────

/// Applies a chain of schema migrations to raw JSON state.
///
/// Register one migration function per version transition.
/// `migrate()` applies all applicable transitions in order until the
/// target version is reached.
///
/// # Example
///
/// ```ignore
/// let mut migrator = CheckpointMigrator::new("2.0");
/// migrator.register("1.0", |mut raw| {
///     // rename old_field → new_field
///     if let Some(obj) = raw.as_object_mut() {
///         if let Some(v) = obj.remove("old_field") {
///             obj.insert("new_field".to_string(), v);
///         }
///     }
///     raw
/// });
/// let migrated = migrator.migrate("1.0", raw_json)?;
/// ```
pub struct CheckpointMigrator {
    /// Target (current) schema version.
    target_version: String,
    /// Ordered list of (from_version, migration_fn).
    #[allow(clippy::type_complexity)]
    migrations: Vec<(String, Box<dyn Fn(Value) -> Value + Send + Sync>)>,
}

impl CheckpointMigrator {
    /// Create a migrator whose target version is `target_version`.
    pub fn new(target_version: impl Into<String>) -> Self {
        Self {
            target_version: target_version.into(),
            migrations: Vec::new(),
        }
    }

    /// Register a migration from `from_version` to the next version.
    ///
    /// Migrations are applied in registration order.
    pub fn register(
        &mut self,
        from_version: impl Into<String>,
        migration: impl Fn(Value) -> Value + Send + Sync + 'static,
    ) {
        self.migrations
            .push((from_version.into(), Box::new(migration)));
    }

    /// Apply all necessary migrations to bring `raw` from `current_version`
    /// up to the target version.
    ///
    /// Returns the migrated JSON value. If `current_version` already matches
    /// the target, returns `raw` unchanged.
    pub fn migrate(
        &self,
        current_version: &str,
        mut raw: Value,
    ) -> std::result::Result<Value, String> {
        if current_version == self.target_version {
            return Ok(raw);
        }

        let start_idx = self
            .migrations
            .iter()
            .position(|(from, _)| from == current_version)
            .ok_or_else(|| {
                format!(
                    "No migration registered for version '{}' → '{}'. \
                     Register a migration with CheckpointMigrator::register().",
                    current_version, self.target_version
                )
            })?;

        for (_, migration) in &self.migrations[start_idx..] {
            raw = migration(raw);
        }

        Ok(raw)
    }

    /// Check if a migration path exists from `from` to the target version.
    pub fn can_migrate(&self, from_version: &str) -> bool {
        from_version == self.target_version
            || self.migrations.iter().any(|(v, _)| v == from_version)
    }

    /// Get the target schema version.
    pub fn target_version(&self) -> &str {
        &self.target_version
    }
}

/// Trait for persisting and retrieving state checkpoints
#[async_trait]
pub trait Checkpointer<S: State>: Send + Sync {
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<()>;
    async fn load(&self, thread_id: &str, step: usize) -> Result<Option<Checkpoint<S>>>;
    async fn load_latest(&self, thread_id: &str) -> Result<Option<Checkpoint<S>>>;
    async fn list_checkpoints(&self, thread_id: &str) -> Result<Vec<(usize, i64)>>;
    async fn delete(&self, thread_id: &str, step: usize) -> Result<()>;
    async fn delete_thread(&self, thread_id: &str) -> Result<()>;
}

/// In-memory checkpointer (default)
pub struct InMemoryCheckpointer<S: State> {
    #[allow(clippy::type_complexity)]
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
        Ok(storage.get(thread_id).and_then(|steps| {
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
