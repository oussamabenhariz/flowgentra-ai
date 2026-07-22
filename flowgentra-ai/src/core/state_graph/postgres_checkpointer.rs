//! Postgres-backed checkpointer — durable execution shared across processes.
//!
//! Implements the same [`Checkpointer`] trait as [`FileCheckpointer`](super::file_checkpointer::FileCheckpointer)
//! and [`SqliteCheckpointer`](super::sqlite_checkpointer::SqliteCheckpointer), so graphs switch
//! backends without code changes:
//!
//! ```ignore
//! let cp = PostgresCheckpointer::connect("postgres://user:pass@localhost/mydb").await?;
//! let graph = StateGraph::<MyState>::builder()
//!     .set_checkpointer(Arc::new(cp))
//!     /* … */
//!     .compile()?;
//! ```
//!
//! Unlike SQLite (single file, single process) or the in-memory default
//! (single process), Postgres lets multiple processes/replicas resume the
//! same thread — the natural choice for a horizontally-scaled service. One
//! table holds all threads.

use async_trait::async_trait;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};

use super::checkpoint::{Checkpoint, Checkpointer};
use super::error::{Result, StateGraphError};
use crate::core::state::State;

/// Current on-disk schema version written with each checkpoint row.
const SCHEMA_VERSION: &str = "1.0";

fn db_err(context: &str, e: impl std::fmt::Display) -> StateGraphError {
    StateGraphError::CheckpointError(format!("PostgresCheckpointer {context}: {e}"))
}

/// Checkpointer that persists checkpoints in a Postgres database.
#[derive(Clone)]
pub struct PostgresCheckpointer {
    pool: PgPool,
}

impl PostgresCheckpointer {
    /// Connect to a Postgres URL (e.g. `postgres://user:pass@localhost/mydb`)
    /// and create the checkpoint table if missing.
    pub async fn connect(url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(url)
            .await
            .map_err(|e| db_err("connect", e))?;
        let cp = Self { pool };
        cp.init().await?;
        Ok(cp)
    }

    /// Wrap an existing pool (shared with other application tables).
    pub async fn from_pool(pool: PgPool) -> Result<Self> {
        let cp = Self { pool };
        cp.init().await?;
        Ok(cp)
    }

    async fn init(&self) -> Result<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS flowgentra_checkpoints (
                thread_id      TEXT    NOT NULL,
                step           BIGINT  NOT NULL,
                node_name      TEXT    NOT NULL,
                state_json     TEXT    NOT NULL,
                timestamp      BIGINT  NOT NULL,
                metadata_json  TEXT    NOT NULL DEFAULT '{}',
                schema_version TEXT    NOT NULL DEFAULT '1.0',
                PRIMARY KEY (thread_id, step)
            )",
        )
        .execute(&self.pool)
        .await
        .map_err(|e| db_err("create table", e))?;
        Ok(())
    }

    fn row_to_checkpoint<S: State + serde::de::DeserializeOwned>(
        row: &sqlx::postgres::PgRow,
    ) -> Result<Checkpoint<S>> {
        let thread_id: String = row.get("thread_id");
        let step: i64 = row.get("step");
        let state_json: String = row.get("state_json");
        let metadata_json: String = row.get("metadata_json");

        let state: S = serde_json::from_str(&state_json).map_err(|e| {
            db_err(
                "state deserialize (state type changed since checkpoint was written?)",
                e,
            )
        })?;
        let metadata = serde_json::from_str(&metadata_json).unwrap_or_default();

        Ok(Checkpoint {
            thread_id,
            step: step as usize,
            node_name: row.get("node_name"),
            state,
            timestamp: row.get("timestamp"),
            metadata,
            schema_version: row.get("schema_version"),
        })
    }
}

#[async_trait]
impl<S: State + Send + Sync + serde::Serialize + serde::de::DeserializeOwned> Checkpointer<S>
    for PostgresCheckpointer
{
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<()> {
        let state_json =
            serde_json::to_string(&checkpoint.state).map_err(|e| db_err("state serialize", e))?;
        let metadata_json = serde_json::to_string(&checkpoint.metadata)
            .map_err(|e| db_err("metadata serialize", e))?;

        sqlx::query(
            "INSERT INTO flowgentra_checkpoints
             (thread_id, step, node_name, state_json, timestamp, metadata_json, schema_version)
             VALUES ($1, $2, $3, $4, $5, $6, $7)
             ON CONFLICT (thread_id, step) DO UPDATE SET
                node_name = EXCLUDED.node_name,
                state_json = EXCLUDED.state_json,
                timestamp = EXCLUDED.timestamp,
                metadata_json = EXCLUDED.metadata_json,
                schema_version = EXCLUDED.schema_version",
        )
        .bind(&checkpoint.thread_id)
        .bind(checkpoint.step as i64)
        .bind(&checkpoint.node_name)
        .bind(&state_json)
        .bind(checkpoint.timestamp)
        .bind(&metadata_json)
        .bind(SCHEMA_VERSION)
        .execute(&self.pool)
        .await
        .map_err(|e| db_err("save", e))?;
        Ok(())
    }

    async fn load(&self, thread_id: &str, step: usize) -> Result<Option<Checkpoint<S>>> {
        let row =
            sqlx::query("SELECT * FROM flowgentra_checkpoints WHERE thread_id = $1 AND step = $2")
                .bind(thread_id)
                .bind(step as i64)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| db_err("load", e))?;

        row.as_ref().map(Self::row_to_checkpoint).transpose()
    }

    async fn load_latest(&self, thread_id: &str) -> Result<Option<Checkpoint<S>>> {
        let row = sqlx::query(
            "SELECT * FROM flowgentra_checkpoints WHERE thread_id = $1
             ORDER BY step DESC LIMIT 1",
        )
        .bind(thread_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| db_err("load_latest", e))?;

        row.as_ref().map(Self::row_to_checkpoint).transpose()
    }

    async fn list_checkpoints(&self, thread_id: &str) -> Result<Vec<(usize, i64)>> {
        let rows = sqlx::query(
            "SELECT step, timestamp FROM flowgentra_checkpoints
             WHERE thread_id = $1 ORDER BY step ASC",
        )
        .bind(thread_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| db_err("list", e))?;

        Ok(rows
            .iter()
            .map(|r| (r.get::<i64, _>("step") as usize, r.get("timestamp")))
            .collect())
    }

    async fn delete(&self, thread_id: &str, step: usize) -> Result<()> {
        sqlx::query("DELETE FROM flowgentra_checkpoints WHERE thread_id = $1 AND step = $2")
            .bind(thread_id)
            .bind(step as i64)
            .execute(&self.pool)
            .await
            .map_err(|e| db_err("delete", e))?;
        Ok(())
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        sqlx::query("DELETE FROM flowgentra_checkpoints WHERE thread_id = $1")
            .bind(thread_id)
            .execute(&self.pool)
            .await
            .map_err(|e| db_err("delete_thread", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::llm::Message;
    use crate::core::state_graph::message_graph::MessageState;

    // These tests require a reachable Postgres instance and are gated behind
    // the FLOWGENTRA_TEST_POSTGRES_URL env var so `cargo test` doesn't fail
    // in environments without a database (CI sets this var explicitly).
    fn test_url() -> Option<String> {
        std::env::var("FLOWGENTRA_TEST_POSTGRES_URL").ok()
    }

    async fn mem_cp() -> Option<PostgresCheckpointer> {
        let url = test_url()?;
        Some(PostgresCheckpointer::connect(&url).await.unwrap())
    }

    #[tokio::test]
    async fn save_load_round_trip() {
        let Some(cp) = mem_cp().await else {
            eprintln!("skipping: FLOWGENTRA_TEST_POSTGRES_URL not set");
            return;
        };
        let thread = uuid::Uuid::new_v4().to_string();
        let state = MessageState::new(vec![Message::user("hello")]);
        let checkpoint = Checkpoint::new(thread.clone(), 0, "n1".into(), state);
        cp.save(&checkpoint).await.unwrap();

        let loaded: Checkpoint<MessageState> = cp.load(&thread, 0).await.unwrap().unwrap();
        assert_eq!(loaded.node_name, "n1");
        assert_eq!(loaded.state.messages.len(), 1);
        assert_eq!(loaded.schema_version, "1.0");

        <PostgresCheckpointer as Checkpointer<MessageState>>::delete_thread(&cp, &thread)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn load_latest_and_list() {
        let Some(cp) = mem_cp().await else {
            eprintln!("skipping: FLOWGENTRA_TEST_POSTGRES_URL not set");
            return;
        };
        let thread = uuid::Uuid::new_v4().to_string();
        for step in 0..3 {
            let state = MessageState::new(vec![Message::user(format!("s{step}"))]);
            cp.save(&Checkpoint::new(
                thread.clone(),
                step,
                format!("n{step}"),
                state,
            ))
            .await
            .unwrap();
        }
        let latest: Checkpoint<MessageState> = cp.load_latest(&thread).await.unwrap().unwrap();
        assert_eq!(latest.step, 2);
        let listed =
            <PostgresCheckpointer as Checkpointer<MessageState>>::list_checkpoints(&cp, &thread)
                .await
                .unwrap();
        assert_eq!(listed.len(), 3);

        <PostgresCheckpointer as Checkpointer<MessageState>>::delete_thread(&cp, &thread)
            .await
            .unwrap();
    }
}
