//! LLM Response Caching — avoid redundant API calls for identical prompts
//!
//! Provides in-memory and file-based caching for LLM responses.
//! Caches by message content hash to avoid duplicate API calls during development.

use std::collections::HashMap;
use std::sync::RwLock;

use super::{LLMClient, Message, TokenUsage, ToolDefinition};

/// In-memory LLM response cache.
///
/// Wraps any `LLMClient` and caches responses by message content hash.
/// Cache hits avoid API calls entirely.
///
/// # Example
/// ```ignore
/// let client = CachedLLMClient::new(llm_client);
/// // First call hits the API
/// let r1 = client.chat(messages.clone()).await?;
/// // Second call with same messages returns cached response
/// let r2 = client.chat(messages).await?;
/// ```
pub struct CachedLLMClient {
    inner: std::sync::Arc<dyn LLMClient>,
    cache: RwLock<HashMap<u64, CachedResponse>>,
    max_entries: usize,
}

#[derive(Clone)]
struct CachedResponse {
    message: Message,
    usage: Option<TokenUsage>,
}

impl CachedLLMClient {
    pub fn new(inner: std::sync::Arc<dyn LLMClient>) -> Self {
        Self {
            inner,
            cache: RwLock::new(HashMap::new()),
            max_entries: 1000,
        }
    }

    /// Set the maximum number of cached entries.
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Get the number of cached entries.
    pub fn cache_size(&self) -> usize {
        self.cache.read().map(|c| c.len()).unwrap_or(0)
    }

    /// Clear all cached entries.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    fn hash_messages(messages: &[Message]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for msg in messages {
            format!("{:?}", msg.role).hash(&mut hasher);
            msg.content.hash(&mut hasher);
        }
        hasher.finish()
    }

    fn get_cached(&self, key: u64) -> Option<CachedResponse> {
        self.cache
            .read()
            .ok()
            .and_then(|cache| cache.get(&key).cloned())
    }

    fn set_cached(&self, key: u64, response: CachedResponse) {
        if let Ok(mut cache) = self.cache.write() {
            // Simple eviction: clear when at capacity
            if cache.len() >= self.max_entries {
                cache.clear();
            }
            cache.insert(key, response);
        }
    }
}

#[async_trait::async_trait]
impl LLMClient for CachedLLMClient {
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
        let h1 = CachedLLMClient::hash_messages(&msgs);
        let h2 = CachedLLMClient::hash_messages(&msgs);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different_messages() {
        let msgs1 = vec![Message::user("hello")];
        let msgs2 = vec![Message::user("world")];
        let h1 = CachedLLMClient::hash_messages(&msgs1);
        let h2 = CachedLLMClient::hash_messages(&msgs2);
        assert_ne!(h1, h2);
    }
}
