//! Node-level result caching keyed on the input state.
//!
//! Wrap any node in [`CachedNode`] to memoize its update by input-state hash:
//! two invocations with identical state skip the second execution. Ideal for
//! expensive deterministic nodes (LLM calls with temperature 0, retrieval,
//! embeddings).
//!
//! ```ignore
//! let cached = CachedNode::new(expensive_node, 256, Some(Duration::from_secs(600)));
//! builder = builder.add_node("expensive", Arc::new(cached));
//! ```
//!
//! Cache key = hash of the serialized full state, so any state change is a
//! miss. Do not cache nodes with side effects or nondeterministic output you
//! want fresh each run.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::error::Result;
use super::node::Node;
use crate::core::state::{Context, State};

struct CacheEntry<U> {
    update: U,
    inserted: Instant,
}

/// Wraps a node with an input-state-keyed result cache (bounded, optional TTL).
pub struct CachedNode<S: State> {
    inner: Arc<dyn Node<S>>,
    max_entries: usize,
    ttl: Option<Duration>,
    cache: Mutex<HashMap<u64, CacheEntry<S::Update>>>,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

impl<S: State> CachedNode<S> {
    /// Wrap `inner` with a cache of at most `max_entries` results.
    /// `ttl = None` means entries never expire (until evicted by size).
    pub fn new(inner: Arc<dyn Node<S>>, max_entries: usize, ttl: Option<Duration>) -> Self {
        Self {
            inner,
            max_entries: max_entries.max(1),
            ttl,
            cache: Mutex::new(HashMap::new()),
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// (hits, misses) since creation.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.hits.load(std::sync::atomic::Ordering::Relaxed),
            self.misses.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    fn state_key(state: &S) -> Result<u64> {
        let value = serde_json::to_value(state).map_err(|e| {
            super::error::StateGraphError::SerializationError(format!(
                "CachedNode: cannot hash state: {e}"
            ))
        })?;
        let mut hasher = DefaultHasher::new();
        // Canonical (key-sorted) hash: DynState serializes an inner HashMap,
        // whose iteration order is nondeterministic — a plain string hash
        // would randomly miss the cache for identical states.
        hash_value_canonical(&value, &mut hasher);
        Ok(hasher.finish())
    }
}

/// Hash a JSON value with object keys visited in sorted order, so logically
/// equal values always hash equally regardless of map iteration order.
fn hash_value_canonical(value: &serde_json::Value, hasher: &mut DefaultHasher) {
    use serde_json::Value;
    match value {
        Value::Null => 0u8.hash(hasher),
        Value::Bool(b) => (1u8, b).hash(hasher),
        Value::Number(n) => (2u8, n.to_string()).hash(hasher),
        Value::String(s) => (3u8, s).hash(hasher),
        Value::Array(items) => {
            (4u8, items.len()).hash(hasher);
            for item in items {
                hash_value_canonical(item, hasher);
            }
        }
        Value::Object(map) => {
            (5u8, map.len()).hash(hasher);
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for key in keys {
                key.hash(hasher);
                hash_value_canonical(&map[key], hasher);
            }
        }
    }
}

#[async_trait]
impl<S: State + Send + Sync + 'static> Node<S> for CachedNode<S> {
    async fn execute(&self, state: &S, ctx: &Context) -> Result<S::Update> {
        use std::sync::atomic::Ordering;
        let key = Self::state_key(state)?;

        {
            let mut cache = self.cache.lock().await;
            if let Some(entry) = cache.get(&key) {
                let expired = self
                    .ttl
                    .map(|ttl| entry.inserted.elapsed() > ttl)
                    .unwrap_or(false);
                if !expired {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    tracing::debug!(node = self.inner.name(), "cache hit");
                    return Ok(entry.update.clone());
                }
                cache.remove(&key);
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let update = self.inner.execute(state, ctx).await?;

        let mut cache = self.cache.lock().await;
        if cache.len() >= self.max_entries {
            // Evict the oldest entry (simple O(n) scan — caches are small).
            if let Some(&oldest) = cache.iter().min_by_key(|(_, e)| e.inserted).map(|(k, _)| k) {
                cache.remove(&oldest);
            }
        }
        cache.insert(
            key,
            CacheEntry {
                update: update.clone(),
                inserted: Instant::now(),
            },
        );
        Ok(update)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn is_parallelizable(&self) -> bool {
        self.inner.is_parallelizable()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::llm::Message;
    use crate::core::state_graph::message_graph::{MessageState, MessageStateUpdate};
    use crate::core::state_graph::node::FunctionNode;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn counting_node(counter: Arc<AtomicU64>) -> Arc<dyn Node<MessageState>> {
        Arc::new(FunctionNode::new(
            "counting",
            move |_state: &MessageState, _ctx: &Context| {
                let counter = Arc::clone(&counter);
                Box::pin(async move {
                    counter.fetch_add(1, Ordering::Relaxed);
                    Ok(MessageStateUpdate {
                        messages: Some(vec![Message::assistant("ran")]),
                    })
                })
            },
        ))
    }

    #[tokio::test]
    async fn identical_state_hits_cache() {
        let calls = Arc::new(AtomicU64::new(0));
        let cached = CachedNode::new(counting_node(Arc::clone(&calls)), 16, None);
        let state = MessageState::new(vec![Message::user("same")]);
        let ctx = Context::new();

        let u1 = cached.execute(&state, &ctx).await.unwrap();
        let u2 = cached.execute(&state, &ctx).await.unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(u1.messages.unwrap()[0].content, "ran");
        assert!(u2.messages.is_some());
        assert_eq!(cached.stats(), (1, 1));
    }

    #[tokio::test]
    async fn different_state_misses() {
        let calls = Arc::new(AtomicU64::new(0));
        let cached = CachedNode::new(counting_node(Arc::clone(&calls)), 16, None);
        let ctx = Context::new();

        cached
            .execute(&MessageState::new(vec![Message::user("a")]), &ctx)
            .await
            .unwrap();
        cached
            .execute(&MessageState::new(vec![Message::user("b")]), &ctx)
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn ttl_expiry_reruns() {
        let calls = Arc::new(AtomicU64::new(0));
        let cached = CachedNode::new(
            counting_node(Arc::clone(&calls)),
            16,
            Some(Duration::from_millis(20)),
        );
        let state = MessageState::new(vec![Message::user("x")]);
        let ctx = Context::new();

        cached.execute(&state, &ctx).await.unwrap();
        tokio::time::sleep(Duration::from_millis(40)).await;
        cached.execute(&state, &ctx).await.unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn dyn_state_key_is_order_independent() {
        use crate::core::state::{DynState, DynStateUpdate};

        let calls = Arc::new(AtomicU64::new(0));
        let calls2 = Arc::clone(&calls);
        let node: Arc<dyn Node<DynState>> = Arc::new(FunctionNode::new(
            "n",
            move |_state: &DynState, _ctx: &Context| {
                let calls = Arc::clone(&calls2);
                Box::pin(async move {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(DynStateUpdate::new())
                })
            },
        ));
        let cached = CachedNode::new(node, 16, None);
        let ctx = Context::new();

        // Same logical content, keys inserted in opposite orders — the two
        // underlying HashMaps may iterate differently.
        let s1 = DynState::new();
        for k in ["a", "b", "c", "d", "e", "f", "g", "h"] {
            s1.set_raw(k, serde_json::json!(k));
        }
        let s2 = DynState::new();
        for k in ["h", "g", "f", "e", "d", "c", "b", "a"] {
            s2.set_raw(k, serde_json::json!(k));
        }

        cached.execute(&s1, &ctx).await.unwrap();
        cached.execute(&s2, &ctx).await.unwrap();
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "logically identical states must share a cache key"
        );
    }

    #[tokio::test]
    async fn size_bound_evicts() {
        let calls = Arc::new(AtomicU64::new(0));
        let cached = CachedNode::new(counting_node(Arc::clone(&calls)), 2, None);
        let ctx = Context::new();

        for text in ["a", "b", "c"] {
            cached
                .execute(&MessageState::new(vec![Message::user(text)]), &ctx)
                .await
                .unwrap();
        }
        // "a" was evicted; running it again is a miss.
        cached
            .execute(&MessageState::new(vec![Message::user("a")]), &ctx)
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 4);
    }
}
