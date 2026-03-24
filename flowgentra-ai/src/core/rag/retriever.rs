//! Retriever Orchestrator — executes retrieval strategies end-to-end
//!
//! Ties together vector stores, embeddings, hybrid search, reranking, and dedup
//! into a single `retrieve()` call. This is the main entry point for RAG retrieval.

use std::sync::Arc;

use super::dedup::dedup_by_similarity;
use super::embeddings::Embeddings;
use super::hybrid::hybrid_merge;
use super::reranker::Reranker;
use super::retrievers::{RetrievalConfig, RetrieverStrategy};
use super::vector_db::{SearchResult, VectorStoreBackend, VectorStoreError};

/// Orchestrates the full retrieval pipeline: embed → search → hybrid → rerank → dedup.
pub struct Retriever {
    store: Arc<dyn VectorStoreBackend>,
    embeddings: Arc<Embeddings>,
    reranker: Option<Arc<dyn Reranker>>,
    config: RetrievalConfig,
    dedup_threshold: Option<f32>,
}

impl Retriever {
    /// Create a new retriever with a vector store backend, embeddings provider, and config.
    pub fn new(
        store: Arc<dyn VectorStoreBackend>,
        embeddings: Arc<Embeddings>,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            store,
            embeddings,
            reranker: None,
            config,
            dedup_threshold: None,
        }
    }

    /// Attach a reranker to the pipeline.
    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Enable deduplication with a similarity threshold (e.g., 0.85).
    pub fn with_dedup(mut self, threshold: f32) -> Self {
        self.dedup_threshold = Some(threshold);
        self
    }

    /// Execute the full retrieval pipeline for a single query.
    pub async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, VectorStoreError> {
        // Step 1: Embed the query
        let query_embedding = self
            .embeddings
            .embed(query)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Embedding error: {}", e)))?;

        // Step 2: Vector search (over-fetch for reranking)
        let mut results = self
            .store
            .search(query_embedding, self.config.top_k * 2, None)
            .await?;

        // Step 3: Hybrid merge if configured
        if let RetrieverStrategy::Hybrid { keyword_weight } = &self.config.strategy {
            results = hybrid_merge(results, query, *keyword_weight);
        }

        // Step 4: Rerank if configured
        if let Some(reranker) = &self.reranker {
            results = reranker
                .rerank(query, results)
                .await
                .map_err(|e| VectorStoreError::Unknown(format!("Rerank error: {}", e)))?;
        }

        // Step 5: Filter by similarity threshold
        results.retain(|r| r.score >= self.config.similarity_threshold);

        // Step 6: Dedup if configured
        if let Some(threshold) = self.dedup_threshold {
            results = dedup_by_similarity(results, threshold);
        }

        // Step 7: Truncate to top_k
        results.truncate(self.config.top_k);

        Ok(results)
    }

    /// Multi-query retrieval: expand the query into variants, retrieve for each, merge.
    pub async fn retrieve_multi_query<F, Fut>(
        &self,
        query: &str,
        expand_fn: F,
        max_variants: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError>
    where
        F: FnOnce(String) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<String>, String>>,
    {
        // Expand query
        let mut queries = vec![query.to_string()];
        if let Ok(variants) = expand_fn(query.to_string()).await {
            for v in variants {
                let trimmed = v.trim().to_string();
                if !trimmed.is_empty() && !queries.contains(&trimmed) {
                    queries.push(trimmed);
                }
                if queries.len() >= max_variants {
                    break;
                }
            }
        }

        // Retrieve for each variant
        let mut all_results = Vec::new();
        for q in &queries {
            match self.retrieve(q).await {
                Ok(results) => all_results.extend(results),
                Err(_) => continue,
            }
        }

        // Dedup across results from different queries
        all_results = dedup_by_similarity(all_results, 0.9);

        // Sort by score and truncate
        all_results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(self.config.top_k);

        Ok(all_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rag::vector_db::{Document, InMemoryVectorStore};

    async fn setup_store() -> (Arc<InMemoryVectorStore>, Arc<Embeddings>) {
        let embeddings = Arc::new(Embeddings::mock(64));
        let store = Arc::new(InMemoryVectorStore::new());

        // Index some documents
        for (id, text) in [
            ("doc1", "Rust programming language is fast and safe"),
            ("doc2", "Python is great for data science"),
            ("doc3", "JavaScript runs in the browser"),
            ("doc4", "Rust has a borrow checker for memory safety"),
        ] {
            let emb = embeddings.embed(text).await.unwrap();
            let doc = Document {
                id: id.to_string(),
                text: text.to_string(),
                embedding: Some(emb),
                metadata: std::collections::HashMap::new(),
            };
            store.index(doc).await.unwrap();
        }

        (store, embeddings)
    }

    #[tokio::test]
    async fn test_basic_retrieval() {
        let (store, embeddings) = setup_store().await;
        let config = RetrievalConfig::default().with_top_k(2).with_threshold(0.0);
        let retriever = Retriever::new(store, embeddings, config);

        let results = retriever.retrieve("Rust programming").await.unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
    }

    #[tokio::test]
    async fn test_hybrid_retrieval() {
        let (store, embeddings) = setup_store().await;
        let config = RetrievalConfig::new(RetrieverStrategy::hybrid(0.3))
            .with_top_k(3)
            .with_threshold(0.0);
        let retriever = Retriever::new(store, embeddings, config);

        let results = retriever.retrieve("Rust memory safety").await.unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_retrieval_with_dedup() {
        let (store, embeddings) = setup_store().await;
        let config = RetrievalConfig::default().with_top_k(4).with_threshold(0.0);
        let retriever = Retriever::new(store, embeddings, config).with_dedup(0.85);

        let results = retriever.retrieve("programming language").await.unwrap();
        assert!(results.len() <= 4);
    }
}
