//! Parent Document Retriever
//!
//! Mirrors LangChain's `ParentDocumentRetriever`. Solves the trade-off between
//! chunk granularity (small = precise retrieval) and context richness
//! (large = complete information):
//!
//! - **Index** small chunks → vector store (for precise retrieval)
//! - **Store** full parent documents → in-memory or external doc store
//! - **Return** full parents when a child chunk matches
//!
//! ## How it works
//!
//! ```text
//! Ingest:
//!   parent doc  ──split──►  child chunks ──embed──►  vector store
//!                        ──store parent─►  doc store (keyed by parent_id)
//!
//! Retrieve:
//!   query ──embed──► vector search → child chunks
//!                 → look up parent_id in each chunk's metadata
//!                 → fetch full parent from doc store
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{
//!     ParentDocumentRetriever, ParentDocConfig,
//!     RecursiveCharacterTextSplitter, Document, Embeddings,
//! };
//! use std::sync::Arc;
//!
//! let retriever = ParentDocumentRetriever::new(store, embeddings, ParentDocConfig::default());
//!
//! // Ingest a batch of full documents (they are split internally)
//! retriever.add_documents(docs).await?;
//!
//! // Retrieve full parents for a query
//! let parents = retriever.retrieve("What is Rust?").await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use serde_json::json;

use super::embeddings::Embeddings;
use super::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};
use super::vector_db::{Document, SearchResult, VectorStoreBackend, VectorStoreError};

/// Metadata key used to link a child chunk back to its parent document.
const PARENT_ID_KEY: &str = "__parent_id__";

/// Configuration for the parent document retriever.
#[derive(Debug, Clone)]
pub struct ParentDocConfig {
    /// Maximum characters per child chunk.
    pub child_chunk_size: usize,
    /// Overlap between consecutive child chunks (characters).
    pub child_chunk_overlap: usize,
    /// How many child chunk matches to retrieve before deduplicating parents.
    pub child_top_k: usize,
    /// Minimum similarity score for child chunks to be included.
    pub similarity_threshold: f32,
    /// Maximum number of distinct parent documents to return.
    pub max_parents: usize,
}

impl Default for ParentDocConfig {
    fn default() -> Self {
        Self {
            child_chunk_size: 400,
            child_chunk_overlap: 50,
            child_top_k: 20,
            similarity_threshold: 0.0,
            max_parents: 5,
        }
    }
}

/// A document stored in the parent doc store.
#[derive(Debug, Clone)]
pub struct ParentDocument {
    pub id: String,
    pub text: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Retriever that indexes child chunks but returns full parent documents.
pub struct ParentDocumentRetriever {
    /// Vector store for child chunk embeddings.
    vector_store: Arc<dyn VectorStoreBackend>,
    embeddings: Arc<Embeddings>,
    config: ParentDocConfig,
    /// In-process store mapping parent_id → ParentDocument.
    parent_store: Arc<DashMap<String, ParentDocument>>,
}

impl ParentDocumentRetriever {
    /// Create a new retriever with the given vector store and embeddings.
    pub fn new(
        vector_store: Arc<dyn VectorStoreBackend>,
        embeddings: Arc<Embeddings>,
        config: ParentDocConfig,
    ) -> Self {
        Self {
            vector_store,
            embeddings,
            config,
            parent_store: Arc::new(DashMap::new()),
        }
    }

    /// Ingest a list of parent documents.
    ///
    /// Each document is split into child chunks. Child chunks are embedded and
    /// indexed in the vector store. The full parent is stored in the doc store.
    pub async fn add_documents(
        &self,
        documents: Vec<ParentDocument>,
    ) -> Result<(), VectorStoreError> {
        let splitter = RecursiveCharacterTextSplitter::new(
            self.config.child_chunk_size,
            self.config.child_chunk_overlap,
        );

        for parent in documents {
            // Store the full parent
            self.parent_store
                .insert(parent.id.clone(), parent.clone());

            // Split into child chunks
            let chunks = splitter.split_text(&parent.text);

            for (i, chunk) in chunks.iter().enumerate() {
                let chunk_id = format!("{}__child_{}", parent.id, i);
                let chunk_text = &chunk.text;

                // Embed the chunk
                let embedding = self
                    .embeddings
                    .embed(chunk_text)
                    .await
                    .map_err(|e| VectorStoreError::Unknown(format!("Embedding error: {e}")))?;

                // Build chunk document with parent_id in metadata
                let mut metadata = parent.metadata.clone();
                metadata.insert(PARENT_ID_KEY.to_string(), json!(parent.id));
                metadata.insert("chunk_index".to_string(), json!(i));

                let chunk_doc = Document {
                    id: chunk_id,
                    text: chunk_text.clone(),
                    embedding: Some(embedding),
                    metadata,
                };

                self.vector_store.index(chunk_doc).await?;
            }
        }

        Ok(())
    }

    /// Retrieve full parent documents for a query.
    ///
    /// 1. Embed the query and retrieve matching child chunks.
    /// 2. Look up the parent_id in each chunk's metadata.
    /// 3. Fetch each unique parent from the doc store.
    /// 4. Return up to `max_parents` parents, ordered by best child-chunk score.
    pub async fn retrieve(
        &self,
        query: &str,
    ) -> Result<Vec<ParentDocument>, VectorStoreError> {
        let query_embedding = self
            .embeddings
            .embed(query)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Embedding error: {e}")))?;

        let child_results = self
            .vector_store
            .search(query_embedding, self.config.child_top_k, None)
            .await?;

        // Collect unique parent ids, tracking best score per parent
        let mut parent_scores: HashMap<String, f32> = HashMap::new();
        for result in &child_results {
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

        // Sort by best score descending, take up to max_parents
        let mut sorted_parents: Vec<(String, f32)> = parent_scores.into_iter().collect();
        sorted_parents.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_parents.truncate(self.config.max_parents);

        // Fetch full parent documents
        let mut parents = Vec::new();
        for (parent_id, _score) in sorted_parents {
            if let Some(parent) = self.parent_store.get(&parent_id) {
                parents.push(parent.clone());
            }
        }

        Ok(parents)
    }

    /// Retrieve as `SearchResult` for compatibility with pipelines expecting that type.
    ///
    /// Returns `SearchResult` where `text` is the full parent text and `score` is
    /// the best child-chunk similarity score.
    pub async fn retrieve_as_results(
        &self,
        query: &str,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let query_embedding = self
            .embeddings
            .embed(query)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Embedding error: {e}")))?;

        let child_results = self
            .vector_store
            .search(query_embedding, self.config.child_top_k, None)
            .await?;

        let mut parent_scores: HashMap<String, f32> = HashMap::new();
        for result in &child_results {
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

        let mut sorted_parents: Vec<(String, f32)> = parent_scores.into_iter().collect();
        sorted_parents.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_parents.truncate(self.config.max_parents);

        let mut results = Vec::new();
        for (parent_id, score) in sorted_parents {
            if let Some(parent) = self.parent_store.get(&parent_id) {
                results.push(SearchResult {
                    id: parent.id.clone(),
                    text: parent.text.clone(),
                    score,
                    metadata: parent.metadata.clone(),
                });
            }
        }

        Ok(results)
    }

    /// Number of parent documents currently stored.
    pub fn parent_count(&self) -> usize {
        self.parent_store.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rag::{embeddings::Embeddings, vector_db::InMemoryVectorStore};

    #[tokio::test]
    async fn test_parent_doc_retriever() {
        let embeddings = Arc::new(Embeddings::mock(16));
        let store = Arc::new(InMemoryVectorStore::new());
        let config = ParentDocConfig {
            child_chunk_size: 50,
            child_chunk_overlap: 10,
            child_top_k: 10,
            similarity_threshold: 0.0,
            max_parents: 3,
        };
        let retriever = ParentDocumentRetriever::new(store, embeddings, config);

        let docs = vec![
            ParentDocument {
                id: "doc1".to_string(),
                text:
                    "Rust is a systems programming language focused on safety and performance. \
                     It prevents memory bugs at compile time using ownership rules."
                        .to_string(),
                metadata: HashMap::new(),
            },
            ParentDocument {
                id: "doc2".to_string(),
                text:
                    "Python is a high-level scripting language known for its simplicity. \
                     It is widely used in data science and machine learning."
                        .to_string(),
                metadata: HashMap::new(),
            },
        ];

        retriever.add_documents(docs).await.unwrap();
        assert_eq!(retriever.parent_count(), 2);

        let parents = retriever.retrieve("memory safety programming").await.unwrap();
        assert!(!parents.is_empty());
        // Should return the Rust doc since it mentions memory safety
        assert!(parents.iter().any(|p| p.id == "doc1"));
    }
}
