//! PostgreSQL-backed Checkpointer
//!
//! Stores graph checkpoints in a Postgres table. Suitable for distributed deployments
//! where multiple workers share the same database and need durable, queryable state.
//!
//! # Schema
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS checkpoints (
//!     id           BIGSERIAL PRIMARY KEY,
//!     thread_id    TEXT    NOT NULL,
//!     state_json   TEXT    NOT NULL,
//!     metadata_json TEXT   NOT NULL,
//!     saved_at     BIGINT  NOT NULL
//! );
//! CREATE INDEX IF NOT EXISTS checkpoints_thread_idx ON checkpoints(thread_id);
//! ```
//!
//! # Feature flag
//!
//! Requires the `postgres` Cargo feature (`sqlx/postgres`).
//!
//! # Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::memory::PostgresCheckpointer;
//!
//! let cp = PostgresCheckpointer::new(
//!     "postgresql://user:pass@localhost/mydb"
//! ).await?;
//! graph.compile(checkpointer: cp).run_with_thread("thread-1", state).await?;
//! ```

#[cfg(feature = "postgres")]
mod inner {
    use crate::core::error::{FlowgentraError, Result};
    use crate::core::memory::checkpointer::{
        Checkpoint, CheckpointMetadata, Checkpointer, GenericCheckpointer,
    };
    use crate::core::state::DynState;
    use sqlx::PgPool;

    /// PostgreSQL-backed checkpointer.
    pub struct PostgresCheckpointer {
        pool: PgPool,
    }

    impl PostgresCheckpointer {
        /// Connect to PostgreSQL and initialise the schema.
        pub async fn new(url: &str) -> Result<Self> {
            let pool = PgPool::connect(url)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Postgres connect error: {e}")))?;

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
            .map_err(|e| FlowgentraError::StateError(format!("Postgres schema error: {e}")))?;

            sqlx::query(
                "CREATE INDEX IF NOT EXISTS checkpoints_thread_idx ON checkpoints(thread_id)",
            )
            .execute(&pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("Postgres index error: {e}")))?;

            Ok(Self { pool })
        }

        /// Delete all checkpoints for a given thread.
        pub async fn delete_thread(&self, thread_id: &str) -> Result<()> {
            sqlx::query("DELETE FROM checkpoints WHERE thread_id = $1")
                .bind(thread_id)
                .execute(&self.pool)
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Postgres delete error: {e}")))?;
            Ok(())
        }

        /// Return all distinct thread ids that have checkpoints.
        pub async fn thread_ids(&self) -> Result<Vec<String>> {
            let rows: Vec<(String,)> = sqlx::query_as(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id",
            )
            .fetch_all(&self.pool)
            .await
            .map_err(|e| FlowgentraError::StateError(format!("Postgres query error: {e}")))?;
            Ok(rows.into_iter().map(|(id,)| id).collect())
        }
    }

    impl Checkpointer for PostgresCheckpointer {
        fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let row: Option<(String, String)> = sqlx::query_as(
                        "SELECT state_json, metadata_json
                         FROM checkpoints
                         WHERE thread_id = $1
                         ORDER BY id DESC
                         LIMIT 1",
                    )
                    .bind(thread_id)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(|e| {
                        FlowgentraError::StateError(format!("Postgres load error: {e}"))
                    })?;

                    match row {
                        None => Ok(None),
                        Some((state_json, metadata_json)) => {
                            let state_value: serde_json::Value =
                                serde_json::from_str(&state_json).map_err(|e| {
                                    FlowgentraError::StateError(format!(
                                        "Deserialize state error: {e}"
                                    ))
                                })?;
                            let metadata: CheckpointMetadata =
                                serde_json::from_str(&metadata_json).map_err(|e| {
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
                tokio::runtime::Handle::current()
                    .block_on(async { self.thread_ids().await })
            })
        }
    }

    impl GenericCheckpointer for PostgresCheckpointer {
        fn save(
            &self,
            thread_id: &str,
            state: &DynState,
            metadata: &CheckpointMetadata,
        ) -> Result<()> {
            let state_json = serde_json::to_string(&state.to_value()).map_err(|e| {
                FlowgentraError::StateError(format!("Serialize state error: {e}"))
            })?;
            let metadata_json = serde_json::to_string(metadata).map_err(|e| {
                FlowgentraError::StateError(format!("Serialize metadata error: {e}"))
            })?;
            let saved_at = chrono::Utc::now().timestamp();

            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    sqlx::query(
                        "INSERT INTO checkpoints (thread_id, state_json, metadata_json, saved_at)
                         VALUES ($1, $2, $3, $4)",
                    )
                    .bind(thread_id)
                    .bind(&state_json)
                    .bind(&metadata_json)
                    .bind(saved_at)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| {
                        FlowgentraError::StateError(format!("Postgres save error: {e}"))
                    })?;
                    Ok(())
                })
            })
        }
    }
}

#[cfg(feature = "postgres")]
pub use inner::PostgresCheckpointer;
