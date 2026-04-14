//! Time-Weighted Retriever
//!
//! Mirrors LangChain's `TimeWeightedVectorStoreRetriever`. Blends semantic
//! similarity with a **recency decay** — documents accessed or created more
//! recently receive a higher combined score.
//!
//! ## Scoring formula
//!
//! ```text
//! combined_score = semantic_score * (1 - decay_rate)^hours_since_last_access
//! ```
//!
//! A `decay_rate` of `0.01` means a document accessed 24 hours ago retains
//! ~79 % of its base score. A rate of `0.1` decays to ~9 % over the same period.
//!
//! ## Timestamps
//!
//! Each document must carry a Unix timestamp in its metadata under a configurable
//! key (default: `"last_accessed_at"`). When a document is returned by this
//! retriever its timestamp is **updated to now** (access-tracking).
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{TimeWeightedRetriever, TimeWeightedConfig};
//! use serde_json::json;
//!
//! let retriever = TimeWeightedRetriever::new(store, embeddings, TimeWeightedConfig {
//!     decay_rate: 0.01,
//!     top_k: 5,
//!     ..Default::default()
//! });
//!
//! // Documents must have "last_accessed_at" (Unix timestamp) in metadata
//! retriever.retrieve("What is Rust?").await?;
//! ```

use std::sync::Arc;

use serde_json::json;

use super::embeddings::Embeddings;
use super::vector_db::{Document, SearchResult, VectorStoreBackend, VectorStoreError};

/// Configuration for the time-weighted retriever.
#[derive(Debug, Clone)]
pub struct TimeWeightedConfig {
    /// Exponential decay rate per hour (0 < decay_rate < 1).
    /// Higher = faster decay, lower = slower decay.
    pub decay_rate: f32,
    /// Number of candidates to fetch from the vector store before applying decay.
    pub fetch_k: usize,
    /// Number of results to return after decay re-ranking.
    pub top_k: usize,
    /// Minimum combined score to include in results.
    pub score_threshold: f32,
    /// Metadata key that stores the Unix timestamp (seconds).
    pub timestamp_key: String,
    /// If true, update the timestamp of returned documents to now.
    pub update_on_access: bool,
}

impl Default for TimeWeightedConfig {
    fn default() -> Self {
        Self {
            decay_rate: 0.01,
            fetch_k: 50,
            top_k: 5,
            score_threshold: 0.0,
            timestamp_key: "last_accessed_at".to_string(),
            update_on_access: true,
        }
    }
}

/// Retriever that weights semantic scores by document recency.
pub struct TimeWeightedRetriever {
    store: Arc<dyn VectorStoreBackend>,
    embeddings: Arc<Embeddings>,
    config: TimeWeightedConfig,
}

impl TimeWeightedRetriever {
    pub fn new(
        store: Arc<dyn VectorStoreBackend>,
        embeddings: Arc<Embeddings>,
        config: TimeWeightedConfig,
    ) -> Self {
        Self {
            store,
            embeddings,
            config,
        }
    }

    /// Compute the decay multiplier for a timestamp.
    ///
    /// `hours_elapsed` = (now - last_accessed) / 3600
    fn decay_multiplier(&self, timestamp_secs: i64) -> f32 {
        let now = chrono::Utc::now().timestamp();
        let hours_elapsed = ((now - timestamp_secs).max(0) as f32) / 3600.0;
        (1.0 - self.config.decay_rate).powf(hours_elapsed)
    }

    /// Retrieve documents weighted by recency.
    ///
    /// Documents without a valid timestamp are treated as having been last
    /// accessed at epoch (very old), giving them minimal decay multiplier.
    pub async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, VectorStoreError> {
        let query_embedding = self
            .embeddings
            .embed(query)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Embedding error: {e}")))?;

        // Fetch more candidates than needed so decay re-ranking has room to work
        let mut results = self
            .store
            .search(query_embedding, self.config.fetch_k, None)
            .await?;

        let ts_key = &self.config.timestamp_key;

        // Apply decay to each result's score
        let now = chrono::Utc::now().timestamp();
        let mut weighted: Vec<SearchResult> = results
            .iter_mut()
            .map(|r| {
                let ts = r
                    .metadata
                    .get(ts_key)
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let decay = self.decay_multiplier(ts);
                SearchResult {
                    id: r.id.clone(),
                    text: r.text.clone(),
                    score: r.score * decay,
                    metadata: r.metadata.clone(),
                }
            })
            .filter(|r| r.score >= self.config.score_threshold)
            .collect();

        // Sort by combined score descending
        weighted.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        weighted.truncate(self.config.top_k);

        // Update timestamps on accessed documents if configured
        if self.config.update_on_access {
            for result in &weighted {
                // Best-effort: don't propagate update errors to the caller
                if let Ok(mut doc) = self.store.get(&result.id).await {
                    doc.metadata.insert(ts_key.clone(), json!(now));
                    let _ = self.store.update(doc).await;
                }
            }
        }

        Ok(weighted)
    }

    /// Add a document to the store, stamping it with the current time.
    ///
    /// If the document already has a timestamp in its metadata it is kept as-is;
    /// this method only sets `last_accessed_at` if it is not already present.
    pub async fn add_document(&self, mut doc: Document) -> Result<(), VectorStoreError> {
        doc.metadata
            .entry(self.config.timestamp_key.clone())
            .or_insert_with(|| json!(chrono::Utc::now().timestamp()));
        self.store.index(doc).await
    }

    /// Add many documents, stamping each with the current time.
    pub async fn add_documents(&self, docs: Vec<Document>) -> Result<(), VectorStoreError> {
        for doc in docs {
            self.add_document(doc).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rag::{embeddings::Embeddings, vector_db::InMemoryVectorStore};
    use serde_json::json;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_decay_multiplier() {
        let store = Arc::new(InMemoryVectorStore::new());
        let embeddings = Arc::new(Embeddings::mock(8));
        let retriever = TimeWeightedRetriever::new(
            store,
            embeddings,
            TimeWeightedConfig {
                decay_rate: 0.5,
                ..Default::default()
            },
        );

        let now = chrono::Utc::now().timestamp();
        // Just created → multiplier close to 1
        let m = retriever.decay_multiplier(now);
        assert!(m > 0.99, "Expected near 1.0, got {m}");

        // 2 hours ago → multiplier = 0.5^2 = 0.25
        let two_hours_ago = now - 7200;
        let m2 = retriever.decay_multiplier(two_hours_ago);
        assert!((m2 - 0.25).abs() < 0.01, "Expected ~0.25, got {m2}");
    }

    #[tokio::test]
    async fn test_add_and_retrieve() {
        let store = Arc::new(InMemoryVectorStore::new());
        let embeddings = Arc::new(Embeddings::mock(16));
        let retriever = TimeWeightedRetriever::new(
            store,
            embeddings.clone(),
            TimeWeightedConfig {
                decay_rate: 0.0,          // no decay → scores unchanged
                top_k: 2,
                update_on_access: false,
                ..Default::default()
            },
        );

        let now = chrono::Utc::now().timestamp();

        // Recent document
        let mut meta = HashMap::new();
        meta.insert("last_accessed_at".to_string(), json!(now));
        let emb = embeddings.embed("Rust memory safety").await.unwrap();
        retriever
            .add_document(Document {
                id: "d1".to_string(),
                text: "Rust memory safety and ownership".to_string(),
                embedding: Some(emb),
                metadata: meta,
            })
            .await
            .unwrap();

        let results = retriever.retrieve("memory safety").await.unwrap();
        assert!(!results.is_empty());
    }
}
