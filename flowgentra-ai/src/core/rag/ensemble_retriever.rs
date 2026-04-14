//! Ensemble Retriever
//!
//! Mirrors LangChain's `EnsembleRetriever`. Combines results from **multiple
//! independent retrievers** using Reciprocal Rank Fusion (RRF). Each retriever
//! can have a different weight so high-quality sources are preferred.
//!
//! ## Why ensemble?
//!
//! A single BM25 retriever misses semantic matches. A single dense retriever
//! misses exact-term matches. Combining them captures both — typically beating
//! either method alone on benchmarks.
//!
//! ## Retriever trait
//!
//! Any type implementing [`AsyncRetriever`] can participate in the ensemble.
//! The crate ships two ready-to-use implementations:
//!
//! - [`VectorRetriever`] — wraps any `VectorStoreBackend` + `Embeddings`
//! - [`Bm25RetrieverAdapter`] — wraps `Bm25Retriever` as an async retriever
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{
//!     EnsembleRetriever, VectorRetriever, Bm25RetrieverAdapter, Bm25Retriever,
//!     Bm25Config, EnsembleConfig,
//! };
//!
//! let dense = VectorRetriever::new(store, embeddings, 10);
//! let sparse = Bm25RetrieverAdapter::new(bm25_retriever);
//!
//! let ensemble = EnsembleRetriever::new(
//!     vec![
//!         (Box::new(dense),  0.7),   // 70 % weight to dense
//!         (Box::new(sparse), 0.3),   // 30 % weight to BM25
//!     ],
//!     EnsembleConfig { top_k: 5, ..Default::default() },
//! );
//!
//! let results = ensemble.retrieve("Rust memory safety").await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use super::bm25_retriever::Bm25Retriever;
use super::embeddings::Embeddings;
use super::vector_db::{SearchResult, VectorStoreBackend, VectorStoreError};

// ── Retriever trait ──────────────────────────────────────────────────────────

/// Any source that can retrieve `SearchResult` items for a query.
#[async_trait]
pub trait AsyncRetriever: Send + Sync {
    async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, VectorStoreError>;
}

// ── VectorRetriever (dense) ──────────────────────────────────────────────────

/// Dense retriever backed by a `VectorStoreBackend`.
pub struct VectorRetriever {
    store: Arc<dyn VectorStoreBackend>,
    embeddings: Arc<Embeddings>,
    top_k: usize,
}

impl VectorRetriever {
    pub fn new(
        store: Arc<dyn VectorStoreBackend>,
        embeddings: Arc<Embeddings>,
        top_k: usize,
    ) -> Self {
        Self {
            store,
            embeddings,
            top_k,
        }
    }
}

#[async_trait]
impl AsyncRetriever for VectorRetriever {
    async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, VectorStoreError> {
        let embedding = self
            .embeddings
            .embed(query)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Embedding error: {e}")))?;
        self.store.search(embedding, self.top_k, None).await
    }
}

// ── Bm25RetrieverAdapter (sparse) ────────────────────────────────────────────

/// Wraps a synchronous `Bm25Retriever` as an `AsyncRetriever`.
pub struct Bm25RetrieverAdapter {
    inner: Arc<Bm25Retriever>,
}

impl Bm25RetrieverAdapter {
    pub fn new(retriever: Bm25Retriever) -> Self {
        Self {
            inner: Arc::new(retriever),
        }
    }
}

#[async_trait]
impl AsyncRetriever for Bm25RetrieverAdapter {
    async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, VectorStoreError> {
        Ok(self.inner.retrieve(query))
    }
}

// ── EnsembleConfig ───────────────────────────────────────────────────────────

/// Configuration for the ensemble retriever.
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Number of final results to return.
    pub top_k: usize,
    /// RRF constant k (higher = less sensitivity to rank differences; default 60).
    pub rrf_k: usize,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            rrf_k: 60,
        }
    }
}

// ── EnsembleRetriever ────────────────────────────────────────────────────────

/// Retriever that combines multiple sources using Reciprocal Rank Fusion.
///
/// Each retriever is run concurrently and its ranked list is fused into a
/// single ranking via weighted RRF:
///
/// ```text
/// rrf_score(doc, retriever_i) = weight_i / (rrf_k + rank_of_doc_in_list_i)
/// final_score(doc) = Σ rrf_score(doc, i)
/// ```
pub struct EnsembleRetriever {
    /// `(retriever, weight)` pairs. Weights do not need to sum to 1.
    retrievers: Vec<(Box<dyn AsyncRetriever>, f32)>,
    config: EnsembleConfig,
}

impl EnsembleRetriever {
    /// Create a new ensemble retriever.
    ///
    /// `retrievers` is a list of `(retriever, weight)`. Weights are used as
    /// multipliers on each retriever's RRF scores; they do not need to sum to 1.
    pub fn new(retrievers: Vec<(Box<dyn AsyncRetriever>, f32)>, config: EnsembleConfig) -> Self {
        Self { retrievers, config }
    }

    /// Retrieve and fuse results from all retrievers.
    ///
    /// All retrievers are called concurrently. Failed retrievers are silently
    /// skipped so a single unhealthy source does not break the ensemble.
    pub async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, VectorStoreError> {
        // Run all retrievers concurrently using join_all
        let futures: Vec<_> = self
            .retrievers
            .iter()
            .map(|(r, _)| r.retrieve(query))
            .collect();

        let all_results = futures::future::join_all(futures).await;

        // Compute weighted RRF scores
        let rrf_k = self.config.rrf_k as f32;
        let mut scores: HashMap<String, (f32, String, HashMap<String, serde_json::Value>)> =
            HashMap::new();

        for (results_opt, (_retriever, weight)) in
            all_results.into_iter().zip(self.retrievers.iter())
        {
            let results = match results_opt {
                Ok(r) => r,
                Err(_) => continue, // skip failing retrievers
            };

            for (rank, result) in results.into_iter().enumerate() {
                let rrf_score = weight / (rrf_k + rank as f32 + 1.0);
                let entry = scores
                    .entry(result.id.clone())
                    .or_insert_with(|| (0.0, result.text.clone(), result.metadata.clone()));
                entry.0 += rrf_score;
            }
        }

        let mut fused: Vec<SearchResult> = scores
            .into_iter()
            .map(|(id, (score, text, metadata))| SearchResult {
                id,
                text,
                score,
                metadata,
            })
            .collect();

        fused.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        fused.truncate(self.config.top_k);

        Ok(fused)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rag::{
        bm25_retriever::Bm25Config,
        embeddings::Embeddings,
        vector_db::{Document, InMemoryVectorStore},
    };
    use std::collections::HashMap;

    async fn make_vector_retriever() -> VectorRetriever {
        let embeddings = Arc::new(Embeddings::mock(16));
        let store = Arc::new(InMemoryVectorStore::new());
        for (id, text) in [("r1", "Rust memory safety"), ("r2", "Python data science")] {
            let emb = embeddings.embed(text).await.unwrap();
            store
                .index(Document {
                    id: id.to_string(),
                    text: text.to_string(),
                    embedding: Some(emb),
                    metadata: HashMap::new(),
                })
                .await
                .unwrap();
        }
        VectorRetriever::new(store, embeddings, 5)
    }

    fn make_bm25_retriever() -> Bm25RetrieverAdapter {
        let retriever = Bm25Retriever::from_texts(
            vec![
                ("b1", "Rust ownership borrow checker memory"),
                ("b2", "Python pandas numpy"),
            ],
            Bm25Config::default(),
        );
        Bm25RetrieverAdapter::new(retriever)
    }

    #[tokio::test]
    async fn test_ensemble_combines_sources() {
        let dense = make_vector_retriever().await;
        let sparse = make_bm25_retriever();

        let ensemble = EnsembleRetriever::new(
            vec![(Box::new(dense), 0.7), (Box::new(sparse), 0.3)],
            EnsembleConfig {
                top_k: 4,
                ..Default::default()
            },
        );

        let results = ensemble.retrieve("rust memory").await.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 4);
        // All scores should be positive RRF scores
        for r in &results {
            assert!(r.score > 0.0);
        }
    }
}
