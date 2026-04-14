//! Multi-Vector Retriever
//!
//! Mirrors LangChain's `MultiVectorRetriever`. Stores **multiple embeddings** per
//! parent document — each representing a different "view" of the same content:
//!
//! | View | Description | When to use |
//! |------|-------------|-------------|
//! | `Chunk` | Small text fragments of the parent | General purpose |
//! | `Summary` | LLM-generated summary of the parent | Long documents |
//! | `HypotheticalQuestions` | Questions the document would answer | Q&A retrieval |
//! | `Custom(tag)` | Any other embedding you compute | Domain-specific |
//!
//! Each view is indexed as a separate vector entry with the parent id in metadata.
//! At query time, all views are searched in parallel and the best-matching
//! parent documents are fetched from the doc store.
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{
//!     MultiVectorRetriever, MultiVectorConfig, VectorView, ParentDocument,
//! };
//! use std::sync::Arc;
//!
//! let retriever = MultiVectorRetriever::new(store, embeddings, config);
//!
//! // Add documents with a summary view
//! retriever.add_with_views(doc, vec![
//!     VectorView::Chunk,
//!     VectorView::Summary("Short intro to Rust".into()),
//! ]).await?;
//!
//! let parents = retriever.retrieve("memory safety").await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use serde_json::json;

use super::embeddings::Embeddings;
use super::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};
use super::vector_db::{Document, SearchResult, VectorStoreBackend, VectorStoreError};

const PARENT_ID_KEY: &str = "__parent_id__";
const VIEW_TAG_KEY: &str = "__view__";

/// Describes how a document representation was produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorView {
    /// Small chunk of the parent's raw text.
    Chunk,
    /// LLM-generated or hand-written summary.
    Summary(String),
    /// Hypothetical question(s) the document could answer.
    HypotheticalQuestions(Vec<String>),
    /// Any custom text representation, identified by a tag.
    Custom { tag: String, text: String },
}

impl VectorView {
    /// Extract the text(s) to embed for this view.
    pub fn texts(&self) -> Vec<String> {
        match self {
            VectorView::Chunk => vec![], // handled separately via splitter
            VectorView::Summary(s) => vec![s.clone()],
            VectorView::HypotheticalQuestions(qs) => qs.clone(),
            VectorView::Custom { text, .. } => vec![text.clone()],
        }
    }

    /// Tag stored in metadata to identify the view type.
    pub fn tag(&self) -> &str {
        match self {
            VectorView::Chunk => "chunk",
            VectorView::Summary(_) => "summary",
            VectorView::HypotheticalQuestions(_) => "hypothetical",
            VectorView::Custom { tag, .. } => tag.as_str(),
        }
    }
}

/// Configuration for the multi-vector retriever.
#[derive(Debug, Clone)]
pub struct MultiVectorConfig {
    /// Characters per chunk when using `VectorView::Chunk`.
    pub chunk_size: usize,
    /// Overlap between consecutive chunks.
    pub chunk_overlap: usize,
    /// Number of vector entries to retrieve during search.
    pub top_k: usize,
    /// Minimum similarity score.
    pub similarity_threshold: f32,
    /// Maximum number of distinct parent documents to return.
    pub max_parents: usize,
}

impl Default for MultiVectorConfig {
    fn default() -> Self {
        Self {
            chunk_size: 400,
            chunk_overlap: 50,
            top_k: 20,
            similarity_threshold: 0.0,
            max_parents: 5,
        }
    }
}

/// A parent document (same structure as in `parent_doc_retriever`).
#[derive(Debug, Clone)]
pub struct MultiVectorParent {
    pub id: String,
    pub text: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Retriever that indexes multiple embedding views per parent document.
pub struct MultiVectorRetriever {
    vector_store: Arc<dyn VectorStoreBackend>,
    embeddings: Arc<Embeddings>,
    config: MultiVectorConfig,
    parent_store: Arc<DashMap<String, MultiVectorParent>>,
}

impl MultiVectorRetriever {
    pub fn new(
        vector_store: Arc<dyn VectorStoreBackend>,
        embeddings: Arc<Embeddings>,
        config: MultiVectorConfig,
    ) -> Self {
        Self {
            vector_store,
            embeddings,
            config,
            parent_store: Arc::new(DashMap::new()),
        }
    }

    /// Index a parent document using the specified views.
    ///
    /// `views` determines which representations are embedded and stored.
    /// If `views` is empty, defaults to `[VectorView::Chunk]`.
    pub async fn add_with_views(
        &self,
        parent: MultiVectorParent,
        views: Vec<VectorView>,
    ) -> Result<(), VectorStoreError> {
        let views = if views.is_empty() {
            vec![VectorView::Chunk]
        } else {
            views
        };

        self.parent_store.insert(parent.id.clone(), parent.clone());

        let splitter =
            RecursiveCharacterTextSplitter::new(self.config.chunk_size, self.config.chunk_overlap);

        for view in &views {
            let chunk_texts: Vec<String> = if matches!(view, VectorView::Chunk) {
                splitter
                    .split_text(&parent.text)
                    .into_iter()
                    .map(|c| c.text)
                    .collect()
            } else {
                view.texts()
            };

            for (i, text) in chunk_texts.iter().enumerate() {
                let entry_id = format!("{}__{}__{}", parent.id, view.tag(), i);

                let embedding = self
                    .embeddings
                    .embed(text.as_str())
                    .await
                    .map_err(|e| VectorStoreError::Unknown(format!("Embedding error: {e}")))?;

                let mut metadata = parent.metadata.clone();
                metadata.insert(PARENT_ID_KEY.to_string(), json!(parent.id));
                metadata.insert(VIEW_TAG_KEY.to_string(), json!(view.tag()));
                metadata.insert("view_index".to_string(), json!(i));

                self.vector_store
                    .index(Document {
                        id: entry_id,
                        text: text.clone(),
                        embedding: Some(embedding),
                        metadata,
                    })
                    .await?;
            }
        }

        Ok(())
    }

    /// Retrieve full parent documents for a query.
    ///
    /// All views are searched simultaneously; results are deduplicated by parent id.
    pub async fn retrieve(&self, query: &str) -> Result<Vec<MultiVectorParent>, VectorStoreError> {
        let query_embedding = self
            .embeddings
            .embed(query)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Embedding error: {e}")))?;

        let results = self
            .vector_store
            .search(query_embedding, self.config.top_k, None)
            .await?;

        let mut parent_scores: HashMap<String, f32> = HashMap::new();
        for result in &results {
            if result.score < self.config.similarity_threshold {
                continue;
            }
            if let Some(pid) = result.metadata.get(PARENT_ID_KEY).and_then(|v| v.as_str()) {
                let entry = parent_scores.entry(pid.to_string()).or_insert(0.0_f32);
                if result.score > *entry {
                    *entry = result.score;
                }
            }
        }

        let mut sorted: Vec<(String, f32)> = parent_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(self.config.max_parents);

        let mut parents = Vec::new();
        for (pid, _) in sorted {
            if let Some(p) = self.parent_store.get(&pid) {
                parents.push(p.clone());
            }
        }
        Ok(parents)
    }

    /// Retrieve as `SearchResult` for pipeline compatibility.
    pub async fn retrieve_as_results(
        &self,
        query: &str,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let query_embedding = self
            .embeddings
            .embed(query)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Embedding error: {e}")))?;

        let results = self
            .vector_store
            .search(query_embedding, self.config.top_k, None)
            .await?;

        let mut parent_scores: HashMap<String, f32> = HashMap::new();
        for result in &results {
            if result.score < self.config.similarity_threshold {
                continue;
            }
            if let Some(pid) = result.metadata.get(PARENT_ID_KEY).and_then(|v| v.as_str()) {
                let entry = parent_scores.entry(pid.to_string()).or_insert(0.0_f32);
                if result.score > *entry {
                    *entry = result.score;
                }
            }
        }

        let mut sorted: Vec<(String, f32)> = parent_scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(self.config.max_parents);

        let mut search_results = Vec::new();
        for (pid, score) in sorted {
            if let Some(p) = self.parent_store.get(&pid) {
                search_results.push(SearchResult {
                    id: p.id.clone(),
                    text: p.text.clone(),
                    score,
                    metadata: p.metadata.clone(),
                });
            }
        }
        Ok(search_results)
    }

    /// Number of parent documents indexed.
    pub fn parent_count(&self) -> usize {
        self.parent_store.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rag::{embeddings::Embeddings, vector_db::InMemoryVectorStore};

    #[tokio::test]
    async fn test_multi_vector_chunk_and_summary() {
        let embeddings = Arc::new(Embeddings::mock(16));
        let store = Arc::new(InMemoryVectorStore::new());
        let retriever = MultiVectorRetriever::new(store, embeddings, MultiVectorConfig::default());

        let parent = MultiVectorParent {
            id: "doc1".to_string(),
            text: "Rust is a systems language emphasising safety memory efficiency.".to_string(),
            metadata: HashMap::new(),
        };

        retriever
            .add_with_views(
                parent,
                vec![
                    VectorView::Chunk,
                    VectorView::Summary("Safe systems language".to_string()),
                ],
            )
            .await
            .unwrap();

        assert_eq!(retriever.parent_count(), 1);
        let results = retriever.retrieve("memory safety").await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc1");
    }
}
