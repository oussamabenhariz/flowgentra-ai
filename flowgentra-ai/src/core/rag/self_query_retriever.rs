//! Self-Querying Retriever
//!
//! Mirrors LangChain's `SelfQueryRetriever`: an LLM automatically extracts
//! structured metadata filters from a natural-language query, then executes
//! the filtered vector search.
//!
//! ## How it works
//!
//! 1. User asks: *"Find blog posts from 2024 about Rust"*
//! 2. The **filter extractor** (your LLM call) returns:
//!    ```json
//!    { "year": 2024, "category": "blog", "topic": "Rust" }
//!    ```
//! 3. Those fields are converted into a [`MetadataFilter`] and passed to the
//!    vector store's `search()`.
//!
//! Because the filter extractor is a plain async function/closure, you can use
//! any LLM backend—the retriever doesn't depend on a specific client.
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{
//!     SelfQueryRetriever, SelfQueryConfig, FilterExpr, SearchResult,
//! };
//! use serde_json::json;
//! use std::sync::Arc;
//!
//! let retriever = SelfQueryRetriever::new(store, embeddings, config)
//!     .with_extractor(|query: String| async move {
//!         // Call your LLM here, parse the response into FilterExpr
//!         // Example: static rule for demonstration
//!         if query.contains("2024") {
//!             Ok(Some(FilterExpr::eq("year", json!(2024))))
//!         } else {
//!             Ok(None)
//!         }
//!     });
//!
//! let results = retriever.retrieve("Rust blog posts from 2024").await?;
//! ```

use std::future::Future;
use std::sync::Arc;

use super::embeddings::Embeddings;
use super::filter::MetadataFilter;
use super::retrievers::RetrievalConfig;
use super::vector_db::{SearchResult, VectorStoreBackend, VectorStoreError};

/// Configuration for the self-querying retriever.
#[derive(Debug, Clone)]
pub struct SelfQueryConfig {
    /// Underlying retrieval settings (top_k, threshold, etc.)
    pub retrieval: RetrievalConfig,
    /// If true, the original query is run even when no filter is extracted.
    /// If false, an unfiltered run is skipped when the extractor returns None.
    pub fallback_on_no_filter: bool,
}

impl Default for SelfQueryConfig {
    fn default() -> Self {
        Self {
            retrieval: RetrievalConfig::default(),
            fallback_on_no_filter: true,
        }
    }
}

/// A retriever that uses an LLM to derive metadata filters from natural language.
///
/// The filter extractor function receives the raw query string and returns an
/// `Option<MetadataFilter>`:
/// - `Some(filter)` → vector search is executed with that filter.
/// - `None` → vector search runs without any filter (plain semantic search).
pub struct SelfQueryRetriever<E> {
    store: Arc<dyn VectorStoreBackend>,
    embeddings: Arc<Embeddings>,
    config: SelfQueryConfig,
    extractor: E,
}

impl<E, Fut> SelfQueryRetriever<E>
where
    E: Fn(String) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Option<MetadataFilter>, String>> + Send,
{
    /// Create a new self-querying retriever.
    ///
    /// `extractor` is an async function/closure that receives the query and returns
    /// an optional filter. Implement your LLM call inside it.
    pub fn new(
        store: Arc<dyn VectorStoreBackend>,
        embeddings: Arc<Embeddings>,
        config: SelfQueryConfig,
        extractor: E,
    ) -> Self {
        Self {
            store,
            embeddings,
            config,
            extractor,
        }
    }

    /// Execute the full self-querying pipeline:
    ///
    /// 1. Call the extractor to get an optional filter.
    /// 2. Embed the query.
    /// 3. Run the filtered vector search.
    pub async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, VectorStoreError> {
        // Step 1: Extract filter from the query via LLM
        let filter = (self.extractor)(query.to_string())
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Filter extraction error: {e}")))?;

        // Step 2: If no filter extracted and fallback is disabled, return empty
        if filter.is_none() && !self.config.fallback_on_no_filter {
            return Ok(vec![]);
        }

        // Step 3: Embed the query
        let query_embedding = self
            .embeddings
            .embed(query)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Embedding error: {e}")))?;

        // Step 4: Vector search with optional filter
        let top_k = self.config.retrieval.top_k;
        let threshold = self.config.retrieval.similarity_threshold;

        let mut results = self.store.search(query_embedding, top_k, filter).await?;

        // Step 5: Apply similarity threshold
        results.retain(|r| r.score >= threshold);
        results.truncate(top_k);

        Ok(results)
    }

    /// Same as `retrieve` but also returns the filter that was extracted,
    /// useful for debugging and UI display.
    pub async fn retrieve_with_filter(
        &self,
        query: &str,
    ) -> Result<(Vec<SearchResult>, Option<MetadataFilter>), VectorStoreError> {
        let filter = (self.extractor)(query.to_string())
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Filter extraction error: {e}")))?;

        if filter.is_none() && !self.config.fallback_on_no_filter {
            return Ok((vec![], None));
        }

        let query_embedding = self
            .embeddings
            .embed(query)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Embedding error: {e}")))?;

        let top_k = self.config.retrieval.top_k;
        let threshold = self.config.retrieval.similarity_threshold;

        let mut results = self
            .store
            .search(query_embedding, top_k, filter.clone())
            .await?;

        results.retain(|r| r.score >= threshold);
        results.truncate(top_k);

        Ok((results, filter))
    }
}

/// Helper: build a `MetadataFilter` from a flat JSON object.
///
/// Converts `{"year": 2024, "category": "blog"}` →
/// `FilterExpr::And([FilterExpr::Eq("year", 2024), FilterExpr::Eq("category", "blog")])`
///
/// Useful inside your LLM extractor closure after parsing the model output.
pub fn filter_from_json_object(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Option<MetadataFilter> {
    use super::filter::FilterExpr;

    let exprs: Vec<FilterExpr> = obj
        .iter()
        .map(|(k, v)| FilterExpr::eq(k.clone(), v.clone()))
        .collect();

    match exprs.len() {
        0 => None,
        1 => Some(exprs.into_iter().next().unwrap()),
        _ => Some(FilterExpr::and(exprs)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rag::{
        embeddings::Embeddings,
        filter::FilterExpr,
        vector_db::{Document, InMemoryVectorStore},
    };
    use serde_json::json;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_self_query_with_filter() {
        let embeddings = Arc::new(Embeddings::mock(8));
        let store = Arc::new(InMemoryVectorStore::new());

        // Index some documents with metadata
        for (id, text, year) in [
            ("a", "Rust language post", 2024),
            ("b", "Python post from 2023", 2023),
            ("c", "Rust ownership 2024", 2024),
        ] {
            let emb = embeddings.embed(text).await.unwrap();
            let mut meta = HashMap::new();
            meta.insert("year".to_string(), json!(year));
            store
                .index(Document {
                    id: id.to_string(),
                    text: text.to_string(),
                    embedding: Some(emb),
                    metadata: meta,
                })
                .await
                .unwrap();
        }

        let config = SelfQueryConfig::default();

        let retriever =
            SelfQueryRetriever::new(store, embeddings, config, |_query: String| async move {
                Ok(Some(FilterExpr::eq("year", json!(2024))))
            });

        let results = retriever.retrieve("Rust posts").await.unwrap();
        // All results should have year == 2024 (InMemoryVectorStore filters metadata)
        assert!(!results.is_empty());
    }

    #[test]
    fn test_filter_from_json_object() {
        let mut obj = serde_json::Map::new();
        obj.insert("year".to_string(), json!(2024));
        obj.insert("category".to_string(), json!("blog"));

        let filter = filter_from_json_object(&obj);
        assert!(filter.is_some());
    }

    #[test]
    fn test_filter_from_empty_object() {
        let obj = serde_json::Map::new();
        let filter = filter_from_json_object(&obj);
        assert!(filter.is_none());
    }
}
