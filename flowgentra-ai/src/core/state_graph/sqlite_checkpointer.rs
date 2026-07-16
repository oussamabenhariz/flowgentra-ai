//! SQLite-backed checkpointer — durable execution in a single file.
//!
//! Implements the same [`Checkpointer`] trait as [`FileCheckpointer`] and the
//! in-memory default, so graphs switch backends without code changes:
//!
//! ```ignore
//! let cp = SqliteCheckpointer::connect("sqlite://checkpoints.db").await?;
//! let graph = StateGraph::<MyState>::builder()
//!     .set_checkpointer(Arc::new(cp))
//!     /* … */
//!     .compile()?;
//! ```
//!
//! Writes are transactional (SQLite guarantees atomicity), so a crash cannot
//! leave a torn checkpoint. One table holds all threads.

use async_trait::async_trait;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::{Row, SqlitePool};

use super::checkpoint::{Checkpoint, Checkpointer};
use super::error::{Result, StateGraphError};
use crate::core::state::State;

/// Current on-disk schema version written with each checkpoint row.
const SCHEMA_VERSION: &str = "1.0";

fn db_err(context: &str, e: impl std::fmt::Display) -> StateGraphError {
    StateGraphError::CheckpointError(format!("SqliteCheckpointer {context}: {e}"))
}

/// Checkpointer that persists checkpoints in a SQLite database.
#[derive(Clone)]
pub struct SqliteCheckpointer {
    pool: SqlitePool,
}

impl SqliteCheckpointer {
    /// Connect to a SQLite database URL (e.g. `sqlite://checkpoints.db` or
    /// `sqlite::memory:`) and create the checkpoint table if missing.
    ///
    /// For file URLs the database file is created if it does not exist.
    pub async fn connect(url: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&Self::creatable_url(url))
            .await
            .map_err(|e| db_err("connect", e))?;
        let cp = Self { pool };
        cp.init().await?;
        Ok(cp)
    }

    /// Wrap an existing pool (shared with other application tables).
    pub async fn from_pool(pool: SqlitePool) -> Result<Self> {
        let cp = Self { pool };
        cp.init().await?;
        Ok(cp)
    }

    fn creatable_url(url: &str) -> String {
        // sqlx requires ?mode=rwc to create a missing file DB.
        if url.contains(":memory:") || url.contains("mode=") {
            url.to_string()
        } else if url.contains('?') {
            format!("{url}&mode=rwc")
        } else {
            format!("{url}?mode=rwc")
        }
    }

    async fn init(&self) -> Result<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS flowgentra_checkpoints (
                thread_id      TEXT    NOT NULL,
                step           INTEGER NOT NULL,
                node_name      TEXT    NOT NULL,
                state_json     TEXT    NOT NULL,
                timestamp      INTEGER NOT NULL,
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
        row: &sqlx::sqlite::SqliteRow,
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
    for SqliteCheckpointer
{
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<()> {
        let state_json =
            serde_json::to_string(&checkpoint.state).map_err(|e| db_err("state serialize", e))?;
        let metadata_json = serde_json::to_string(&checkpoint.metadata)
            .map_err(|e| db_err("metadata serialize", e))?;

        sqlx::query(
            "INSERT OR REPLACE INTO flowgentra_checkpoints
             (thread_id, step, node_name, state_json, timestamp, metadata_json, schema_version)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
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
        let row = sqlx::query(
            "SELECT * FROM flowgentra_checkpoints WHERE thread_id = ? AND step = ?",
        )
        .bind(thread_id)
        .bind(step as i64)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| db_err("load", e))?;

        row.as_ref().map(Self::row_to_checkpoint).transpose()
    }

    async fn load_latest(&self, thread_id: &str) -> Result<Option<Checkpoint<S>>> {
        let row = sqlx::query(
            "SELECT * FROM flowgentra_checkpoints WHERE thread_id = ?
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
             WHERE thread_id = ? ORDER BY step ASC",
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
        sqlx::query("DELETE FROM flowgentra_checkpoints WHERE thread_id = ? AND step = ?")
            .bind(thread_id)
            .bind(step as i64)
            .execute(&self.pool)
            .await
            .map_err(|e| db_err("delete", e))?;
        Ok(())
    }

    async fn delete_thread(&self, thread_id: &str) -> Result<()> {
        sqlx::query("DELETE FROM flowgentra_checkpoints WHERE thread_id = ?")
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

    async fn mem_cp() -> SqliteCheckpointer {
        SqliteCheckpointer::connect("sqlite::memory:").await.unwrap()
    }

    #[tokio::test]
    async fn save_load_round_trip() {
        let cp = mem_cp().await;
        let state = MessageState::new(vec![Message::user("hello")]);
        let checkpoint = Checkpoint::new("t1".into(), 0, "n1".into(), state);
        cp.save(&checkpoint).await.unwrap();

        let loaded: Checkpoint<MessageState> = cp.load("t1", 0).await.unwrap().unwrap();
        assert_eq!(loaded.node_name, "n1");
        assert_eq!(loaded.state.messages.len(), 1);
        assert_eq!(loaded.schema_version, "1.0");
    }

    #[tokio::test]
    async fn load_latest_and_list() {
        let cp = mem_cp().await;
        for step in 0..3 {
            let state = MessageState::new(vec![Message::user(format!("s{step}"))]);
            cp.save(&Checkpoint::new("t1".into(), step, format!("n{step}"), state))
                .await
                .unwrap();
        }
        let latest: Checkpoint<MessageState> = cp.load_latest("t1").await.unwrap().unwrap();
        assert_eq!(latest.step, 2);
        let listed = <SqliteCheckpointer as Checkpointer<MessageState>>::list_checkpoints(
            &cp, "t1",
        )
        .await
        .unwrap();
        assert_eq!(listed.len(), 3);
    }

    #[tokio::test]
    async fn save_is_idempotent_per_step() {
        let cp = mem_cp().await;
        let s1 = MessageState::new(vec![Message::user("first")]);
        let s2 = MessageState::new(vec![Message::user("second")]);
        cp.save(&Checkpoint::new("t1".into(), 0, "n".into(), s1))
            .await
            .unwrap();
        cp.save(&Checkpoint::new("t1".into(), 0, "n".into(), s2))
            .await
            .unwrap();
        let loaded: Checkpoint<MessageState> = cp.load("t1", 0).await.unwrap().unwrap();
        assert_eq!(loaded.state.messages[0].content, "second");
    }

    #[tokio::test]
    async fn delete_thread_removes_all() {
        let cp = mem_cp().await;
        let state = MessageState::empty();
        cp.save(&Checkpoint::new("t1".into(), 0, "n".into(), state))
            .await
            .unwrap();
        <SqliteCheckpointer as Checkpointer<MessageState>>::delete_thread(&cp, "t1")
            .await
            .unwrap();
        let loaded: Option<Checkpoint<MessageState>> = cp.load("t1", 0).await.unwrap();
        assert!(loaded.is_none());
    }
}
