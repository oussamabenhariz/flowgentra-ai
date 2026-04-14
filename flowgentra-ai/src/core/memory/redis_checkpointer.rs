//! Redis-backed Checkpointer
//!
//! Stores the latest checkpoint for each thread as a Redis hash under the key
//! `checkpoint:<thread_id>`. Also maintains a Redis Set `checkpoint:threads`
//! that tracks all known thread ids for fast listing.
//!
//! Optional TTL lets you expire stale checkpoints automatically.
//!
//! # Feature flag
//!
//! Requires the `redis-store` Cargo feature.
//!
//! # Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::memory::RedisCheckpointer;
//!
//! // No TTL (checkpoints live forever)
//! let cp = RedisCheckpointer::new("redis://127.0.0.1/", None).await?;
//!
//! // Expire checkpoints after 7 days
//! let cp = RedisCheckpointer::new("redis://127.0.0.1/", Some(60 * 60 * 24 * 7)).await?;
//! ```

#[cfg(feature = "redis-store")]
mod inner {
    use crate::core::error::{FlowgentraError, Result};
    use crate::core::memory::checkpointer::{
        Checkpoint, CheckpointMetadata, Checkpointer, GenericCheckpointer,
    };
    use crate::core::state::DynState;
    use redis::AsyncCommands;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// Redis-backed checkpointer.
    ///
    /// Stores each thread's latest checkpoint as a Redis hash:
    /// - field `state`    → JSON-encoded state value
    /// - field `metadata` → JSON-encoded CheckpointMetadata
    pub struct RedisCheckpointer {
        client: Arc<Mutex<redis::aio::Connection>>,
        ttl_seconds: Option<usize>,
    }

    impl RedisCheckpointer {
        /// Connect to Redis and return a new checkpointer.
        ///
        /// `ttl_seconds`: if `Some(n)`, each checkpoint key expires after `n` seconds.
        pub async fn new(url: &str, ttl_seconds: Option<usize>) -> Result<Self> {
            let client = redis::Client::open(url)
                .map_err(|e| FlowgentraError::StateError(format!("Redis client error: {e}")))?;
            let conn = client
                .get_async_connection()
                .await
                .map_err(|e| FlowgentraError::StateError(format!("Redis connect error: {e}")))?;
            Ok(Self {
                client: Arc::new(Mutex::new(conn)),
                ttl_seconds,
            })
        }

        fn key(thread_id: &str) -> String {
            format!("checkpoint:{}", thread_id)
        }

        const THREADS_SET: &'static str = "checkpoint:threads";
    }

    impl Checkpointer for RedisCheckpointer {
        fn load(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
            let client = Arc::clone(&self.client);
            let key = Self::key(thread_id);

            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    let mut conn = client.lock().await;

                    let (state_json, metadata_json): (Option<String>, Option<String>) =
                        redis::pipe()
                            .hget(&key, "state")
                            .hget(&key, "metadata")
                            .query_async(&mut *conn)
                            .await
                            .map_err(|e| {
                                FlowgentraError::StateError(format!("Redis load error: {e}"))
                            })?;

                    match (state_json, metadata_json) {
                        (Some(sj), Some(mj)) => {
                            let state_value: serde_json::Value =
                                serde_json::from_str(&sj).map_err(|e| {
                                    FlowgentraError::StateError(format!(
                                        "Deserialize state error: {e}"
                                    ))
                                })?;
                            let metadata: CheckpointMetadata =
                                serde_json::from_str(&mj).map_err(|e| {
                                    FlowgentraError::StateError(format!(
                                        "Deserialize metadata error: {e}"
                                    ))
                                })?;
                            let state = DynState::from_json(state_value)?;
                            let cp = Checkpoint::new(&state, metadata)?;
                            Ok(Some(cp))
                        }
                        _ => Ok(None),
                    }
                })
            })
        }

        fn list_threads(&self) -> Result<Vec<String>> {
            let client = Arc::clone(&self.client);
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    let mut conn = client.lock().await;
                    let members: Vec<String> = conn
                        .smembers(Self::THREADS_SET)
                        .await
                        .map_err(|e| {
                            FlowgentraError::StateError(format!("Redis smembers error: {e}"))
                        })?;
                    Ok(members)
                })
            })
        }
    }

    impl GenericCheckpointer for RedisCheckpointer {
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
            let key = Self::key(thread_id);
            let ttl = self.ttl_seconds;
            let client = Arc::clone(&self.client);
            let thread_id = thread_id.to_string();

            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    let mut conn = client.lock().await;

                    // Store state and metadata as hash fields
                    conn.hset_multiple::<_, _, _, ()>(
                        &key,
                        &[("state", &state_json), ("metadata", &metadata_json)],
                    )
                    .await
                    .map_err(|e| {
                        FlowgentraError::StateError(format!("Redis hset error: {e}"))
                    })?;

                    // Optionally set TTL
                    if let Some(secs) = ttl {
                        conn.expire::<_, ()>(&key, secs as i64)
                            .await
                            .map_err(|e| {
                                FlowgentraError::StateError(format!("Redis expire error: {e}"))
                            })?;
                    }

                    // Track thread id in the threads set
                    conn.sadd::<_, _, ()>(Self::THREADS_SET, &thread_id)
                        .await
                        .map_err(|e| {
                            FlowgentraError::StateError(format!("Redis sadd error: {e}"))
                        })?;

                    Ok(())
                })
            })
        }
    }
}

#[cfg(feature = "redis-store")]
pub use inner::RedisCheckpointer;
