//! Async Checkpointer Trait & Wrappers
//!
//! Provides a fully `async` checkpointing interface — no `block_in_place`, no
//! `Handle::block_on`. All methods are `async fn` so they compose naturally
//! with the rest of the async runtime.
//!
//! ## Components
//!
//! | Type | Description |
//! |---|---|
//! | `AsyncCheckpointer` | Core async trait (save / load / list_threads / list_history) |
//! | `NamespacedCheckpointer` | Wraps any `AsyncCheckpointer` and scopes keys to `(namespace, thread_id)` |
//! | `CheckpointHistoryEntry` | A single historical checkpoint record |
//!
//! ## Namespace support
//!
//! LangGraph identifies checkpoints with a 3-tuple
//! `(namespace, thread_id, checkpoint_id)`. Namespacing lets multiple tenants,
//! agents, or experiments share the same backing store without key collisions.
//!
//! ```rust,ignore
//! let namespaced = NamespacedCheckpointer::new(inner, "tenant_42");
//! namespaced.save("thread-1", &state, &meta).await?;
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::error::Result;
use crate::core::memory::checkpointer::{Checkpoint, CheckpointMetadata};
use crate::core::state::DynState;

// ── History entry ─────────────────────────────────────────────────────────────

/// A single checkpoint record in history (includes its unix timestamp).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointHistoryEntry {
    pub thread_id: String,
    pub namespace: Option<String>,
    pub saved_at: i64,
    pub metadata: CheckpointMetadata,
    pub checkpoint: Checkpoint,
}

// ── Core async trait ──────────────────────────────────────────────────────────

/// Fully async checkpointing interface.
///
/// Implement this trait to persist graph state across process restarts,
/// support multi-turn conversations, and enable workflow resumption.
#[async_trait]
pub trait AsyncCheckpointer: Send + Sync {
    /// Save the current state for a thread.
    async fn save(
        &self,
        thread_id: &str,
        state: &DynState,
        metadata: &CheckpointMetadata,
    ) -> Result<()>;

    /// Load the latest checkpoint for a thread. Returns `None` if no checkpoint exists.
    async fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>>;

    /// List all thread ids that have at least one checkpoint.
    async fn list_threads(&self) -> Result<Vec<String>> {
        Ok(vec![])
    }

    /// Return the full history of checkpoints for a thread, newest first.
    ///
    /// The default implementation returns only the latest checkpoint; backends
    /// that store full history (SQLite, Postgres, MongoDB) should override this.
    async fn list_history(&self, thread_id: &str) -> Result<Vec<CheckpointHistoryEntry>> {
        let entry = self.load(thread_id).await?;
        match entry {
            None => Ok(vec![]),
            Some(cp) => Ok(vec![CheckpointHistoryEntry {
                thread_id: thread_id.to_string(),
                namespace: None,
                saved_at: chrono::Utc::now().timestamp(),
                metadata: cp.metadata.clone(),
                checkpoint: cp,
            }]),
        }
    }

    /// Delete all checkpoints for a thread.
    async fn delete_thread(&self, thread_id: &str) -> Result<()>;
}

// ── In-memory async checkpointer ─────────────────────────────────────────────

/// Async in-memory checkpointer with full history support.
///
/// Each `save()` appends to an ordered list; `load()` returns the latest.
/// Suitable for testing and single-process workflows.
pub struct AsyncMemoryCheckpointer {
    // thread_id → list of (saved_at, checkpoint) ordered oldest → newest
    store: tokio::sync::RwLock<HashMap<String, Vec<CheckpointHistoryEntry>>>,
}

impl AsyncMemoryCheckpointer {
    pub fn new() -> Self {
        Self {
            store: tokio::sync::RwLock::new(HashMap::new()),
        }
    }
}

impl Default for AsyncMemoryCheckpointer {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AsyncCheckpointer for AsyncMemoryCheckpointer {
    async fn save(
        &self,
        thread_id: &str,
        state: &DynState,
        metadata: &CheckpointMetadata,
    ) -> Result<()> {
        let cp = Checkpoint::new(state, metadata.clone())?;
        let entry = CheckpointHistoryEntry {
            thread_id: thread_id.to_string(),
            namespace: None,
            saved_at: chrono::Utc::now().timestamp(),
            metadata: metadata.clone(),
            checkpoint: cp,
        };
        let mut guard = self.store.write().await;
        guard.entry(thread_id.to_string()).or_default().push(entry);
        Ok(())
    }

    async fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        let guard = self.store.read().await;
        Ok(guard
            .get(thread_id)
            .and_then(|v| v.last())
            .map(|e| e.checkpoint.clone()))
    }

    async fn list_threads(&self) -> Result<Vec<String>> {
        Ok(self.store.read().await.keys().cloned().collect())
    }

    async fn list_history(&self, thread_id: &str) -> Result<Vec<CheckpointHistoryEntry>> {
        let guard = self.store.read().await;
        let mut entries = guard
            .get(thread_id)
            .cloned()
            .unwrap_or_default();
        entries.reverse(); // newest first
        Ok(entries)
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        self.store.write().await.remove(thread_id);
        Ok(())
    }
}

// ── Namespaced wrapper ────────────────────────────────────────────────────────

/// Scopes every checkpoint key to `<namespace>:<thread_id>`.
///
/// Multiple tenants, agents, or environments can share the same underlying
/// store without key collisions.
///
/// ```rust,ignore
/// let inner = AsyncMemoryCheckpointer::new();
/// let agent_a = NamespacedCheckpointer::new(Arc::new(inner), "agent_a");
/// agent_a.save("thread-1", &state, &meta).await?;
/// // stored under key "agent_a:thread-1"
/// ```
pub struct NamespacedCheckpointer<C: AsyncCheckpointer> {
    inner: Arc<C>,
    namespace: String,
}

impl<C: AsyncCheckpointer> NamespacedCheckpointer<C> {
    pub fn new(inner: Arc<C>, namespace: impl Into<String>) -> Self {
        Self {
            inner,
            namespace: namespace.into(),
        }
    }

    fn scoped_key(&self, thread_id: &str) -> String {
        format!("{}:{}", self.namespace, thread_id)
    }
}

#[async_trait]
impl<C: AsyncCheckpointer + 'static> AsyncCheckpointer for NamespacedCheckpointer<C> {
    async fn save(
        &self,
        thread_id: &str,
        state: &DynState,
        metadata: &CheckpointMetadata,
    ) -> Result<()> {
        self.inner.save(&self.scoped_key(thread_id), state, metadata).await
    }

    async fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        self.inner.load(&self.scoped_key(thread_id)).await
    }

    async fn list_threads(&self) -> Result<Vec<String>> {
        let prefix = format!("{}:", self.namespace);
        let all = self.inner.list_threads().await?;
        Ok(all
            .into_iter()
            .filter_map(|t| t.strip_prefix(&prefix).map(str::to_string))
            .collect())
    }

    async fn list_history(&self, thread_id: &str) -> Result<Vec<CheckpointHistoryEntry>> {
        let mut entries = self
            .inner
            .list_history(&self.scoped_key(thread_id))
            .await?;
        let prefix = format!("{}:", self.namespace);
        for e in &mut entries {
            e.thread_id = e
                .thread_id
                .strip_prefix(&prefix)
                .unwrap_or(&e.thread_id)
                .to_string();
            e.namespace = Some(self.namespace.clone());
        }
        Ok(entries)
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        self.inner.delete_thread(&self.scoped_key(thread_id)).await
    }
}

// ── Async SQLite checkpointer ─────────────────────────────────────────────────

#[cfg(feature = "sqlite")]
pub mod sqlite_async {
    use super::*;
    use crate::core::error::FlowgentraError;
    use sqlx::SqlitePool;

    /// Fully async SQLite checkpointer with complete history.
    pub struct AsyncSqliteCheckpointer {
        pool: SqlitePool,
    }

    impl AsyncSqliteCheckpointer {
        pub async fn new(url: &str) -> Result<Self> {
            let pool = SqlitePool::connect(url).await.map_err(|e| {
                FlowgentraError::StateError(format!("SQLite connect: {e}"))
            })?;
            sqlx::query(
                "CREATE TABLE IF NOT EXISTS checkpoints (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id     TEXT    NOT NULL,
                    state_json    TEXT    NOT NULL,
                    metadata_json TEXT    NOT NULL,
                    saved_at      INTEGER NOT NULL
                )",
            )
            .execute(&pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("SQLite schema: {e}")))?;
            Ok(Self { pool })
        }
    }

    #[async_trait]
    impl AsyncCheckpointer for AsyncSqliteCheckpointer {
        async fn save(
            &self,
            thread_id: &str,
            state: &DynState,
            metadata: &CheckpointMetadata,
        ) -> Result<()> {
            let sj = serde_json::to_string(&state.to_value())
                .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
            let mj = serde_json::to_string(metadata)
                .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
            let ts = chrono::Utc::now().timestamp();
            sqlx::query(
                "INSERT INTO checkpoints (thread_id, state_json, metadata_json, saved_at)
                 VALUES (?, ?, ?, ?)",
            )
            .bind(thread_id)
            .bind(&sj)
            .bind(&mj)
            .bind(ts)
            .execute(&self.pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("SQLite save: {e}")))?;
            Ok(())
        }

        async fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
            let row: Option<(String, String)> = sqlx::query_as(
                "SELECT state_json, metadata_json FROM checkpoints
                 WHERE thread_id = ? ORDER BY id DESC LIMIT 1",
            )
            .bind(thread_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("SQLite load: {e}")))?;

            match row {
                None => Ok(None),
                Some((sj, mj)) => {
                    let sv: serde_json::Value = serde_json::from_str(&sj)
                        .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                    let meta: CheckpointMetadata = serde_json::from_str(&mj)
                        .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                    let state = DynState::from_json(sv)?;
                    Ok(Some(Checkpoint::new(&state, meta)?))
                }
            }
        }

        async fn list_threads(&self) -> Result<Vec<String>> {
            let rows: Vec<(String,)> = sqlx::query_as(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id",
            )
            .fetch_all(&self.pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("SQLite list: {e}")))?;
            Ok(rows.into_iter().map(|(t,)| t).collect())
        }

        async fn list_history(
            &self,
            thread_id: &str,
        ) -> Result<Vec<CheckpointHistoryEntry>> {
            let rows: Vec<(String, String, i64)> = sqlx::query_as(
                "SELECT state_json, metadata_json, saved_at FROM checkpoints
                 WHERE thread_id = ? ORDER BY id DESC",
            )
            .bind(thread_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("SQLite history: {e}")))?;

            let mut entries = Vec::new();
            for (sj, mj, ts) in rows {
                let sv: serde_json::Value = serde_json::from_str(&sj)
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let meta: CheckpointMetadata = serde_json::from_str(&mj)
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let state = DynState::from_json(sv)?;
                let cp = Checkpoint::new(&state, meta.clone())?;
                entries.push(CheckpointHistoryEntry {
                    thread_id: thread_id.to_string(),
                    namespace: None,
                    saved_at: ts,
                    metadata: meta,
                    checkpoint: cp,
                });
            }
            Ok(entries)
        }

        async fn delete_thread(&self, thread_id: &str) -> Result<()> {
            sqlx::query("DELETE FROM checkpoints WHERE thread_id = ?")
                .bind(thread_id)
                .execute(&self.pool)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("SQLite delete: {e}")))?;
            Ok(())
        }
    }
}

// ── Async Redis checkpointer ──────────────────────────────────────────────────

#[cfg(feature = "redis-store")]
pub mod redis_async {
    use super::*;
    use crate::core::error::FlowgentraError;
    use redis::aio::ConnectionManager;
    use redis::AsyncCommands;

    /// Async Redis checkpointer. Stores only the latest checkpoint per thread
    /// (no history). Keys: `checkpoint:<thread_id>` (JSON blob).
    pub struct AsyncRedisCheckpointer {
        manager: ConnectionManager,
        ttl_secs: Option<u64>,
    }

    impl AsyncRedisCheckpointer {
        pub async fn new(url: &str, ttl_secs: Option<u64>) -> Result<Self> {
            let client = redis::Client::open(url)
                .map_err(|e| FlowgentraError::StateError(format!("Redis client: {e}")))?;
            let manager = ConnectionManager::new(client)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Redis connect: {e}")))?;
            Ok(Self { manager, ttl_secs })
        }

        fn key(thread_id: &str) -> String {
            format!("checkpoint:{thread_id}")
        }

        fn index_key() -> &'static str {
            "checkpoint:__index__"
        }
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    struct RedisEntry {
        state_json: String,
        metadata_json: String,
        saved_at: i64,
    }

    #[async_trait]
    impl AsyncCheckpointer for AsyncRedisCheckpointer {
        async fn save(
            &self,
            thread_id: &str,
            state: &DynState,
            metadata: &CheckpointMetadata,
        ) -> Result<()> {
            let entry = RedisEntry {
                state_json: serde_json::to_string(&state.to_value())
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?,
                metadata_json: serde_json::to_string(metadata)
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?,
                saved_at: chrono::Utc::now().timestamp(),
            };
            let blob = serde_json::to_string(&entry)
                .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
            let key = Self::key(thread_id);
            let mut mgr = self.manager.clone();
            if let Some(ttl) = self.ttl_secs {
                let _: () = mgr
                    .set_ex(&key, &blob, ttl)
                    .await
                    .map_err(|e| FlowgentraError::StateError(format!("Redis set_ex: {e}")))?;
            } else {
                let _: () = mgr
                    .set(&key, &blob)
                    .await
                    .map_err(|e| FlowgentraError::StateError(format!("Redis set: {e}")))?;
            }
            let _: () = mgr
                .sadd(Self::index_key(), thread_id)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Redis sadd: {e}")))?;
            Ok(())
        }

        async fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
            let mut mgr = self.manager.clone();
            let blob: Option<String> = mgr
                .get(Self::key(thread_id))
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Redis get: {e}")))?;
            match blob {
                None => Ok(None),
                Some(b) => {
                    let entry: RedisEntry = serde_json::from_str(&b)
                        .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                    let sv: serde_json::Value = serde_json::from_str(&entry.state_json)
                        .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                    let meta: CheckpointMetadata = serde_json::from_str(&entry.metadata_json)
                        .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                    let state = DynState::from_json(sv)?;
                    Ok(Some(Checkpoint::new(&state, meta)?))
                }
            }
        }

        async fn list_threads(&self) -> Result<Vec<String>> {
            let mut mgr = self.manager.clone();
            let members: Vec<String> = mgr
                .smembers(Self::index_key())
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Redis smembers: {e}")))?;
            Ok(members)
        }

        async fn delete_thread(&self, thread_id: &str) -> Result<()> {
            let mut mgr = self.manager.clone();
            let _: () = mgr
                .del(Self::key(thread_id))
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Redis del: {e}")))?;
            let _: () = mgr
                .srem(Self::index_key(), thread_id)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Redis srem: {e}")))?;
            Ok(())
        }
    }
}

// ── Async MongoDB checkpointer ────────────────────────────────────────────────

#[cfg(feature = "mongodb-store")]
pub mod mongo_async {
    use super::*;
    use crate::core::error::FlowgentraError;
    use mongodb::{bson::doc, options::FindOptions, Client};

    /// Async MongoDB checkpointer with full history.
    ///
    /// Collection: `checkpoints`. Documents: `{ thread_id, state_json, metadata_json, saved_at }`.
    pub struct AsyncMongoCheckpointer {
        collection: mongodb::Collection<mongodb::bson::Document>,
    }

    impl AsyncMongoCheckpointer {
        pub async fn new(
            url: &str,
            db_name: &str,
            collection_name: &str,
        ) -> Result<Self> {
            let client = Client::with_uri_str(url)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Mongo connect: {e}")))?;
            let db = client.database(db_name);
            let collection = db.collection(collection_name);
            // Ensure index on thread_id for fast lookups
            let index = mongodb::IndexModel::builder()
                .keys(doc! { "thread_id": 1, "saved_at": -1 })
                .build();
            collection
                .create_index(index, None)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Mongo index: {e}")))?;
            Ok(Self { collection })
        }
    }

    #[async_trait]
    impl AsyncCheckpointer for AsyncMongoCheckpointer {
        async fn save(
            &self,
            thread_id: &str,
            state: &DynState,
            metadata: &CheckpointMetadata,
        ) -> Result<()> {
            let sj = serde_json::to_string(&state.to_value())
                .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
            let mj = serde_json::to_string(metadata)
                .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
            let ts = chrono::Utc::now().timestamp();
            let doc = doc! {
                "thread_id": thread_id,
                "state_json": sj,
                "metadata_json": mj,
                "saved_at": ts,
            };
            self.collection
                .insert_one(doc, None)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Mongo insert: {e}")))?;
            Ok(())
        }

        async fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
            let opts = FindOptions::builder()
                .sort(doc! { "saved_at": -1 })
                .limit(1)
                .build();
            let mut cursor = self
                .collection
                .find(doc! { "thread_id": thread_id }, opts)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Mongo find: {e}")))?;
            use futures::TryStreamExt;
            if let Some(document) = cursor.try_next().await
                .map_err(|e| FlowgentraError::StateError(format!("Mongo cursor: {e}")))?
            {
                let sj = document.get_str("state_json")
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let mj = document.get_str("metadata_json")
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let sv: serde_json::Value = serde_json::from_str(sj)
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let meta: CheckpointMetadata = serde_json::from_str(mj)
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let state = DynState::from_json(sv)?;
                return Ok(Some(Checkpoint::new(&state, meta)?));
            }
            Ok(None)
        }

        async fn list_threads(&self) -> Result<Vec<String>> {
            let threads = self
                .collection
                .distinct("thread_id", None, None)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Mongo distinct: {e}")))?;
            Ok(threads
                .into_iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect())
        }

        async fn list_history(
            &self,
            thread_id: &str,
        ) -> Result<Vec<CheckpointHistoryEntry>> {
            let opts = FindOptions::builder()
                .sort(doc! { "saved_at": -1 })
                .build();
            let mut cursor = self
                .collection
                .find(doc! { "thread_id": thread_id }, opts)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Mongo find: {e}")))?;
            use futures::TryStreamExt;
            let mut entries = Vec::new();
            while let Some(document) = cursor.try_next().await
                .map_err(|e| FlowgentraError::StateError(format!("Mongo cursor: {e}")))?
            {
                let sj = document.get_str("state_json")
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let mj = document.get_str("metadata_json")
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let ts = document.get_i64("saved_at").unwrap_or(0);
                let sv: serde_json::Value = serde_json::from_str(sj)
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let meta: CheckpointMetadata = serde_json::from_str(mj)
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let state = DynState::from_json(sv)?;
                let cp = Checkpoint::new(&state, meta.clone())?;
                entries.push(CheckpointHistoryEntry {
                    thread_id: thread_id.to_string(),
                    namespace: None,
                    saved_at: ts,
                    metadata: meta,
                    checkpoint: cp,
                });
            }
            Ok(entries)
        }

        async fn delete_thread(&self, thread_id: &str) -> Result<()> {
            self.collection
                .delete_many(doc! { "thread_id": thread_id }, None)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Mongo delete: {e}")))?;
            Ok(())
        }
    }
}

// ── Async MySQL checkpointer ──────────────────────────────────────────────────

#[cfg(feature = "mysql")]
pub mod mysql_async {
    use super::*;
    use crate::core::error::FlowgentraError;
    use sqlx::MySqlPool;

    /// Async MySQL checkpointer with full history.
    pub struct AsyncMysqlCheckpointer {
        pool: MySqlPool,
    }

    impl AsyncMysqlCheckpointer {
        pub async fn new(url: &str) -> Result<Self> {
            let pool = MySqlPool::connect(url).await.map_err(|e| {
                FlowgentraError::StateError(format!("MySQL connect: {e}"))
            })?;
            sqlx::query(
                "CREATE TABLE IF NOT EXISTS checkpoints (
                    id            BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    thread_id     VARCHAR(255) NOT NULL,
                    state_json    LONGTEXT     NOT NULL,
                    metadata_json TEXT         NOT NULL,
                    saved_at      BIGINT       NOT NULL,
                    INDEX idx_thread_id (thread_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4",
            )
            .execute(&pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("MySQL schema: {e}")))?;
            Ok(Self { pool })
        }
    }

    #[async_trait]
    impl AsyncCheckpointer for AsyncMysqlCheckpointer {
        async fn save(
            &self,
            thread_id: &str,
            state: &DynState,
            metadata: &CheckpointMetadata,
        ) -> Result<()> {
            let sj = serde_json::to_string(&state.to_value())
                .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
            let mj = serde_json::to_string(metadata)
                .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
            let ts = chrono::Utc::now().timestamp();
            sqlx::query(
                "INSERT INTO checkpoints (thread_id, state_json, metadata_json, saved_at)
                 VALUES (?, ?, ?, ?)",
            )
            .bind(thread_id)
            .bind(&sj)
            .bind(&mj)
            .bind(ts)
            .execute(&self.pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("MySQL save: {e}")))?;
            Ok(())
        }

        async fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
            let row: Option<(String, String)> = sqlx::query_as(
                "SELECT state_json, metadata_json FROM checkpoints
                 WHERE thread_id = ? ORDER BY id DESC LIMIT 1",
            )
            .bind(thread_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("MySQL load: {e}")))?;

            match row {
                None => Ok(None),
                Some((sj, mj)) => {
                    let sv: serde_json::Value = serde_json::from_str(&sj)
                        .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                    let meta: CheckpointMetadata = serde_json::from_str(&mj)
                        .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                    let state = DynState::from_json(sv)?;
                    Ok(Some(Checkpoint::new(&state, meta)?))
                }
            }
        }

        async fn list_threads(&self) -> Result<Vec<String>> {
            let rows: Vec<(String,)> = sqlx::query_as(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id",
            )
            .fetch_all(&self.pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("MySQL list: {e}")))?;
            Ok(rows.into_iter().map(|(t,)| t).collect())
        }

        async fn list_history(
            &self,
            thread_id: &str,
        ) -> Result<Vec<CheckpointHistoryEntry>> {
            let rows: Vec<(String, String, i64)> = sqlx::query_as(
                "SELECT state_json, metadata_json, saved_at FROM checkpoints
                 WHERE thread_id = ? ORDER BY id DESC",
            )
            .bind(thread_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("MySQL history: {e}")))?;

            let mut entries = Vec::new();
            for (sj, mj, ts) in rows {
                let sv: serde_json::Value = serde_json::from_str(&sj)
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let meta: CheckpointMetadata = serde_json::from_str(&mj)
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let state = DynState::from_json(sv)?;
                let cp = Checkpoint::new(&state, meta.clone())?;
                entries.push(CheckpointHistoryEntry {
                    thread_id: thread_id.to_string(),
                    namespace: None,
                    saved_at: ts,
                    metadata: meta,
                    checkpoint: cp,
                });
            }
            Ok(entries)
        }

        async fn delete_thread(&self, thread_id: &str) -> Result<()> {
            sqlx::query("DELETE FROM checkpoints WHERE thread_id = ?")
                .bind(thread_id)
                .execute(&self.pool)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("MySQL delete: {e}")))?;
            Ok(())
        }
    }
}

// ── Async Postgres checkpointer ───────────────────────────────────────────────

#[cfg(feature = "postgres")]
pub mod postgres_async {
    use super::*;
    use crate::core::error::FlowgentraError;
    use sqlx::PgPool;

    pub struct AsyncPostgresCheckpointer {
        pool: PgPool,
    }

    impl AsyncPostgresCheckpointer {
        pub async fn new(url: &str) -> Result<Self> {
            let pool = PgPool::connect(url).await.map_err(|e| {
                FlowgentraError::StateError(format!("Postgres connect: {e}"))
            })?;
            sqlx::query(
                "CREATE TABLE IF NOT EXISTS checkpoints (
                    id            BIGSERIAL PRIMARY KEY,
                    thread_id     TEXT    NOT NULL,
                    state_json    TEXT    NOT NULL,
                    metadata_json TEXT    NOT NULL,
                    saved_at      BIGINT  NOT NULL
                )",
            )
            .execute(&pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("Postgres schema: {e}")))?;
            sqlx::query(
                "CREATE INDEX IF NOT EXISTS cp_thread_idx ON checkpoints(thread_id)",
            )
            .execute(&pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("Postgres index: {e}")))?;
            Ok(Self { pool })
        }
    }

    #[async_trait]
    impl AsyncCheckpointer for AsyncPostgresCheckpointer {
        async fn save(
            &self,
            thread_id: &str,
            state: &DynState,
            metadata: &CheckpointMetadata,
        ) -> Result<()> {
            let sj = serde_json::to_string(&state.to_value())
                .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
            let mj = serde_json::to_string(metadata)
                .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
            let ts = chrono::Utc::now().timestamp();
            sqlx::query(
                "INSERT INTO checkpoints (thread_id, state_json, metadata_json, saved_at)
                 VALUES ($1, $2, $3, $4)",
            )
            .bind(thread_id)
            .bind(&sj)
            .bind(&mj)
            .bind(ts)
            .execute(&self.pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("Postgres save: {e}")))?;
            Ok(())
        }

        async fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
            let row: Option<(String, String)> = sqlx::query_as(
                "SELECT state_json, metadata_json FROM checkpoints
                 WHERE thread_id = $1 ORDER BY id DESC LIMIT 1",
            )
            .bind(thread_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("Postgres load: {e}")))?;

            match row {
                None => Ok(None),
                Some((sj, mj)) => {
                    let sv: serde_json::Value = serde_json::from_str(&sj)
                        .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                    let meta: CheckpointMetadata = serde_json::from_str(&mj)
                        .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                    let state = DynState::from_json(sv)?;
                    Ok(Some(Checkpoint::new(&state, meta)?))
                }
            }
        }

        async fn list_threads(&self) -> Result<Vec<String>> {
            let rows: Vec<(String,)> = sqlx::query_as(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id",
            )
            .fetch_all(&self.pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("Postgres list: {e}")))?;
            Ok(rows.into_iter().map(|(t,)| t).collect())
        }

        async fn list_history(
            &self,
            thread_id: &str,
        ) -> Result<Vec<CheckpointHistoryEntry>> {
            let rows: Vec<(String, String, i64)> = sqlx::query_as(
                "SELECT state_json, metadata_json, saved_at FROM checkpoints
                 WHERE thread_id = $1 ORDER BY id DESC",
            )
            .bind(thread_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("Postgres history: {e}")))?;

            let mut entries = Vec::new();
            for (sj, mj, ts) in rows {
                let sv: serde_json::Value = serde_json::from_str(&sj)
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let meta: CheckpointMetadata = serde_json::from_str(&mj)
                    .map_err(|e| FlowgentraError::StateError(e.to_string()))?;
                let state = DynState::from_json(sv)?;
                let cp = Checkpoint::new(&state, meta.clone())?;
                entries.push(CheckpointHistoryEntry {
                    thread_id: thread_id.to_string(),
                    namespace: None,
                    saved_at: ts,
                    metadata: meta,
                    checkpoint: cp,
                });
            }
            Ok(entries)
        }

        async fn delete_thread(&self, thread_id: &str) -> Result<()> {
            sqlx::query("DELETE FROM checkpoints WHERE thread_id = $1")
                .bind(thread_id)
                .execute(&self.pool)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Postgres delete: {e}")))?;
            Ok(())
        }
    }
}
