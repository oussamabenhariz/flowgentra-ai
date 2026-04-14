//! Embedding Cache — avoids redundant API calls for previously-seen text
//!
//! Uses a content-hash (FNV-style) of the input text as the cache key.
//! Thread-safe via `DashMap`. Purely in-memory (no disk persistence).

use dashmap::DashMap;
use std::sync::Arc;

use async_trait::async_trait;

use super::embeddings::{EmbeddingError, EmbeddingsProvider};

/// Content-hash key for cache lookups.
///
/// Uses blake3 instead of DefaultHasher for collision resistance and
/// cross-version stability.
fn content_hash(text: &str) -> u64 {
    let hash = blake3::hash(text.as_bytes());
    u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap())
}

/// Caching wrapper around any `EmbeddingsProvider`.
///
/// Caches embeddings by content hash so identical texts are only embedded once.
pub struct CachedEmbeddings {
    inner: Arc<dyn EmbeddingsProvider>,
    cache: DashMap<u64, Vec<f32>>,
}

impl CachedEmbeddings {
    /// Wrap an existing provider with a cache
    pub fn new(provider: Arc<dyn EmbeddingsProvider>) -> Self {
        Self {
            inner: provider,
            cache: DashMap::new(),
        }
    }

    /// Number of cached entries
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

#[async_trait]
impl EmbeddingsProvider for CachedEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let key = content_hash(text);

        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached.clone());
        }

        let embedding = self.inner.embed(text).await?;
        self.cache.insert(key, embedding.clone());
        Ok(embedding)
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();

        // Check cache for each text
        for (i, text) in texts.iter().enumerate() {
            let key = content_hash(text);
            if let Some(cached) = self.cache.get(&key) {
                results.push(Some(cached.clone()));
            } else {
                results.push(None);
                uncached_indices.push(i);
                uncached_texts.push(*text);
            }
        }

        // Batch-embed only uncached texts
        if !uncached_texts.is_empty() {
            let new_embeddings = self.inner.embed_batch(uncached_texts.clone()).await?;

            for (idx, embedding) in uncached_indices.iter().zip(new_embeddings) {
                let key = content_hash(texts[*idx]);
                self.cache.insert(key, embedding.clone());
                results[*idx] = Some(embedding);
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    fn get_dimension(&self) -> usize {
        self.inner.get_dimension()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rag::embeddings::MockEmbeddings;

    #[tokio::test]
    async fn test_cache_hit() {
        let mock = Arc::new(MockEmbeddings::new(64));
        let cached = CachedEmbeddings::new(mock);

        let e1 = cached.embed("hello world").await.unwrap();
        assert_eq!(cached.cache_size(), 1);

        let e2 = cached.embed("hello world").await.unwrap();
        assert_eq!(cached.cache_size(), 1); // no new entry
        assert_eq!(e1, e2);
    }

    #[tokio::test]
    async fn test_cache_miss() {
        let mock = Arc::new(MockEmbeddings::new(64));
        let cached = CachedEmbeddings::new(mock);

        cached.embed("text a").await.unwrap();
        cached.embed("text b").await.unwrap();
        assert_eq!(cached.cache_size(), 2);
    }

    #[tokio::test]
    async fn test_batch_partial_cache() {
        let mock = Arc::new(MockEmbeddings::new(64));
        let cached = CachedEmbeddings::new(mock);

        // Pre-cache one text
        cached.embed("already cached").await.unwrap();
        assert_eq!(cached.cache_size(), 1);

        let results = cached
            .embed_batch(vec!["already cached", "new text"])
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(cached.cache_size(), 2);
    }

    #[test]
    fn test_clear_cache() {
        let mock = Arc::new(MockEmbeddings::new(64));
        let cached = CachedEmbeddings::new(mock);
        cached.cache.insert(123, vec![1.0, 2.0]);
        assert_eq!(cached.cache_size(), 1);
        cached.clear_cache();
        assert_eq!(cached.cache_size(), 0);
    }
}
