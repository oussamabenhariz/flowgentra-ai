//! Score-Threshold Retriever
//!
//! Mirrors LangChain's `ScoreThresholdRetriever`. Wraps any [`AsyncRetriever`]
//! and drops results whose similarity score falls below a configurable
//! threshold, ensuring the LLM only sees high-confidence matches.

use async_trait::async_trait;

use super::ensemble_retriever::AsyncRetriever;
use super::vector_db::{SearchResult, VectorStoreError};

/// Wraps a retriever and filters results by minimum similarity score.
pub struct ScoreThresholdRetriever {
    inner: Box<dyn AsyncRetriever>,
    min_score: f32,
    /// Hard cap on returned documents (applied after score filter).
    top_k: usize,
}

impl ScoreThresholdRetriever {
    pub fn new(inner: impl AsyncRetriever + 'static, min_score: f32, top_k: usize) -> Self {
        Self {
            inner: Box::new(inner),
            min_score,
            top_k,
        }
    }
}

#[async_trait]
impl AsyncRetriever for ScoreThresholdRetriever {
    async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut results = self.inner.retrieve(query).await?;
        results.retain(|r| r.score >= self.min_score);
        results.truncate(self.top_k);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rag::{
        embeddings::Embeddings,
        ensemble_retriever::VectorRetriever,
        vector_db::{Document, InMemoryVectorStore, VectorStoreBackend},
    };
    use std::{collections::HashMap, sync::Arc};

    #[tokio::test]
    async fn test_score_threshold() {
        let embeddings = Arc::new(Embeddings::mock(16));
        let store = Arc::new(InMemoryVectorStore::new());

        let emb = embeddings.embed("hello world").await.unwrap();
        store
            .index(Document {
                id: "d1".into(),
                text: "hello world".into(),
                embedding: Some(emb),
                metadata: HashMap::new(),
            })
            .await
            .unwrap();

        let base = VectorRetriever::new(store, embeddings, 10);
        // threshold = 0.0 → keep everything
        let retriever = ScoreThresholdRetriever::new(base, 0.0, 5);
        let results = retriever.retrieve("hello").await.unwrap();
        assert!(!results.is_empty());
    }
}
