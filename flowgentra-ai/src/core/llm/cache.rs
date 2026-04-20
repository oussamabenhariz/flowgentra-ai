//! LLM Response Caching — avoid redundant API calls for identical prompts
//!
//! Provides in-memory and file-based caching for LLM responses.
//! Caches by message content hash to avoid duplicate API calls during development.

use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::Mutex;

use super::{Message, TokenUsage, ToolDefinition, LLM};

/// In-memory LLM response cache.
///
/// Wraps any `LLM` and caches responses by message content hash.
/// Cache hits avoid API calls entirely.  Uses an LRU eviction policy so that
/// only the *least recently used* entry is dropped when the cache is full —
/// not all entries at once (which would cause a thundering herd of API calls).
///
/// # Example
/// ```ignore
/// let client = CachedLLM::new(llm);
/// // First call hits the API
/// let r1 = client.chat(messages.clone()).await?;
/// // Second call with same messages returns cached response
/// let r2 = client.chat(messages).await?;
/// ```
pub struct CachedLLM {
    inner: std::sync::Arc<dyn LLM>,
    /// LruCache is wrapped in Mutex (not RwLock) because `get` promotes an
    /// entry to most-recently-used and therefore requires exclusive access.
    cache: Mutex<LruCache<u64, CachedResponse>>,
}

#[derive(Clone)]
struct CachedResponse {
    message: Message,
    usage: Option<TokenUsage>,
}

impl CachedLLM {
    pub fn new(inner: std::sync::Arc<dyn LLM>) -> Self {
        Self {
            inner,
            cache: Mutex::new(LruCache::new(NonZeroUsize::new(1000).unwrap())),
        }
    }

    /// Set the maximum number of cached entries.
    pub fn with_max_entries(self, max: usize) -> Self {
        let cap = NonZeroUsize::new(max).unwrap_or(NonZeroUsize::new(1).unwrap());
        // Drain existing entries into a new cache with the new capacity.
        let old = self.cache.into_inner().unwrap_or_else(|p| p.into_inner());
        let mut new_cache = LruCache::new(cap);
        // Re-insert in LRU order (oldest first) so the new cache respects the cap.
        let entries: Vec<_> = old.iter().map(|(k, v)| (*k, v.clone())).collect();
        for (k, v) in entries {
            new_cache.put(k, v);
        }
        Self {
            inner: self.inner,
            cache: Mutex::new(new_cache),
        }
    }

    /// Get the number of cached entries.
    pub fn cache_size(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }

    /// Clear all cached entries.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    fn hash_messages(messages: &[Message]) -> u64 {
        // Use blake3 (collision-resistant, stable across Rust versions) instead of
        // DefaultHasher (non-crypto, version-unstable) to prevent cache poisoning.
        let mut hasher = blake3::Hasher::new();
        for msg in messages {
            hasher.update(format!("{:?}", msg.role).as_bytes());
            hasher.update(b"\x00"); // separator
            hasher.update(msg.content.as_bytes());
            hasher.update(b"\x01"); // separator
        }
        let hash = hasher.finalize();
        // Truncate to u64 — still 64 bits of collision resistance
        u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap())
    }

    fn get_cached(&self, key: u64) -> Option<CachedResponse> {
        self.cache
            .lock()
            .ok()
            .and_then(|mut cache| cache.get(&key).cloned())
    }

    fn set_cached(&self, key: u64, response: CachedResponse) {
        if let Ok(mut cache) = self.cache.lock() {
            // LruCache::put automatically evicts the least-recently-used entry
            // when the cache is at capacity — no thundering herd.
            cache.put(key, response);
        }
    }
}

#[async_trait::async_trait]
impl LLM for CachedLLM {
    async fn chat(&self, messages: Vec<Message>) -> crate::core::error::Result<Message> {
        let key = Self::hash_messages(&messages);

        if let Some(cached) = self.get_cached(key) {
            return Ok(cached.message);
        }

        let response = self.inner.chat(messages).await?;

        self.set_cached(
            key,
            CachedResponse {
                message: response.clone(),
                usage: None,
            },
        );

        Ok(response)
    }

    async fn chat_with_usage(
        &self,
        messages: Vec<Message>,
    ) -> crate::core::error::Result<(Message, Option<TokenUsage>)> {
        let key = Self::hash_messages(&messages);

        if let Some(cached) = self.get_cached(key) {
            return Ok((cached.message, cached.usage));
        }

        let (response, usage) = self.inner.chat_with_usage(messages).await?;

        self.set_cached(
            key,
            CachedResponse {
                message: response.clone(),
                usage: usage.clone(),
            },
        );

        Ok((response, usage))
    }

    async fn chat_with_tools(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
    ) -> crate::core::error::Result<Message> {
        // Don't cache tool calls — they may have side effects
        self.inner.chat_with_tools(messages, tools).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<Message>,
    ) -> crate::core::error::Result<tokio::sync::mpsc::Receiver<String>> {
        // Can't cache streaming responses
        self.inner.chat_stream(messages).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_consistency() {
        let msgs = vec![Message::user("hello"), Message::assistant("hi")];
        let h1 = CachedLLM::hash_messages(&msgs);
        let h2 = CachedLLM::hash_messages(&msgs);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different_messages() {
        let msgs1 = vec![Message::user("hello")];
        let msgs2 = vec![Message::user("world")];
        let h1 = CachedLLM::hash_messages(&msgs1);
        let h2 = CachedLLM::hash_messages(&msgs2);
        assert_ne!(h1, h2);
    }
}
