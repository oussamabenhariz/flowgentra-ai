//! SQLite-backed Checkpointer
//!
//! Persists graph checkpoints in a local SQLite database. Survives process restarts
//! and works in single-process deployments without an external database server.
//!
//! # Schema
//!
//! Creates one table: `checkpoints(thread_id TEXT, state_json TEXT, metadata_json TEXT,
//! saved_at INTEGER)`. Only the latest checkpoint per thread is meaningful for resume;
//! all rows are kept so you can replay or inspect history.
//!
//! # Feature flag
//!
//! Requires the `sqlite` Cargo feature (`sqlx/sqlite`).
//!
//! # Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::memory::SqliteCheckpointer;
//!
//! let cp = SqliteCheckpointer::new("sqlite:///var/data/checkpoints.db").await?;
//! graph.compile(checkpointer: cp).run_with_thread("thread-1", state).await?;
//! ```

#[cfg(feature = "sqlite")]
mod inner {
    use crate::core::error::{FlowgentraError, Result};
    use crate::core::memory::checkpointer::{
        Checkpoint, CheckpointMetadata, Checkpointer, GenericCheckpointer,
    };
    use crate::core::state::DynState;
    use sqlx::SqlitePool;

    /// SQLite-backed checkpointer.
    ///
    /// Creates the `checkpoints` table on first use if it does not exist.
    pub struct SqliteCheckpointer {
        pool: SqlitePool,
    }

    impl SqliteCheckpointer {
        /// Connect to a SQLite database and initialise the schema.
        ///
        /// `url` follows the sqlx format: `sqlite:///absolute/path.db`
        /// or `sqlite::memory:` for an in-process database.
        pub async fn new(url: &str) -> Result<Self> {
            let pool = SqlitePool::connect(url)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("SQLite connect error: {e}")))?;

            sqlx::query(
                "CREATE TABLE IF NOT EXISTS checkpoints (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id   TEXT    NOT NULL,
                    state_json  TEXT    NOT NULL,
                    metadata_json TEXT  NOT NULL,
                    saved_at    INTEGER NOT NULL
                )",
            )
            .execute(&pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("SQLite schema error: {e}")))?;

            Ok(Self { pool })
        }

        /// Delete all checkpoints for a thread (useful for cleanup / testing).
        pub async fn delete_thread(&self, thread_id: &str) -> Result<()> {
            sqlx::query("DELETE FROM checkpoints WHERE thread_id = ?")
                .bind(thread_id)
                .execute(&self.pool)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("SQLite delete error: {e}")))?;
            Ok(())
        }

        /// Return all thread ids that have at least one saved checkpoint.
        pub async fn thread_ids(&self) -> Result<Vec<String>> {
            let rows: Vec<(String,)> =
                sqlx::query_as("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")
                    .fetch_all(&self.pool)
                    .await
                    .map_err(|e| FlowgentraError::StateError(format!("SQLite query error: {e}")))?;
            Ok(rows.into_iter().map(|(id,)| id).collect())
        }
    }

    impl Checkpointer for SqliteCheckpointer {
        fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
            // We need a sync interface — block_in_place is safe inside a tokio multi-thread
            // runtime and avoids pulling in a separate sync executor.
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let row: Option<(String, String)> = sqlx::query_as(
                        "SELECT state_json, metadata_json
                         FROM checkpoints
                         WHERE thread_id = ?
                         ORDER BY id DESC
                         LIMIT 1",
                    )
                    .bind(thread_id)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(|e| FlowgentraError::StateError(format!("SQLite load error: {e}")))?;

                    match row {
                        None => Ok(None),
                        Some((state_json, metadata_json)) => {
                            let state_value: serde_json::Value = serde_json::from_str(&state_json)
                                .map_err(|e| {
                                    FlowgentraError::StateError(format!(
                                        "Deserialize state error: {e}"
                                    ))
                                })?;
                            let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)
                                .map_err(|e| {
                                    FlowgentraError::StateError(format!(
                                        "Deserialize metadata error: {e}"
                                    ))
                                })?;

                            let state = DynState::from_json(state_value)?;
                            let cp = Checkpoint::new(&state, metadata)?;
                            Ok(Some(cp))
                        }
                    }
                })
            })
        }

        fn list_threads(&self) -> Result<Vec<String>> {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async { self.thread_ids().await })
            })
        }
    }

    impl GenericCheckpointer for SqliteCheckpointer {
        fn save(
            &self,
            thread_id: &str,
            state: &DynState,
            metadata: &CheckpointMetadata,
        ) -> Result<()> {
            let state_json = serde_json::to_string(&state.to_value())
                .map_err(|e| FlowgentraError::StateError(format!("Serialize state error: {e}")))?;
            let metadata_json = serde_json::to_string(metadata).map_err(|e| {
                FlowgentraError::StateError(format!("Serialize metadata error: {e}"))
            })?;
            let saved_at = chrono::Utc::now().timestamp();

            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    sqlx::query(
                        "INSERT INTO checkpoints (thread_id, state_json, metadata_json, saved_at)
                         VALUES (?, ?, ?, ?)",
                    )
                    .bind(thread_id)
                    .bind(&state_json)
                    .bind(&metadata_json)
                    .bind(saved_at)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| FlowgentraError::StateError(format!("SQLite save error: {e}")))?;
                    Ok(())
                })
            })
        }
    }
}

#[cfg(feature = "sqlite")]
pub use inner::SqliteCheckpointer;
