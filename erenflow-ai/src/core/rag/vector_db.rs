//! Vector Database Abstractions and Implementations
//!
//! Provides a unified interface for multiple vector stores.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for vector stores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAGConfig {
    pub store_type: VectorStoreType,
    pub api_key: Option<String>,
    pub endpoint: Option<String>,
    pub index_name: String,
    pub embedding_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VectorStoreType {
    #[serde(rename = "pinecone")]
    Pinecone,
    #[serde(rename = "weaviate")]
    Weaviate,
    #[serde(rename = "chroma")]
    Chroma,
    #[serde(rename = "milvus")]
    Milvus,
    #[serde(rename = "qdrant")]
    Qdrant,
    #[serde(rename = "memory")]
    Memory,
}

impl RAGConfig {
    pub fn pinecone(index: &str, api_key: &str) -> Result<Self, VectorStoreError> {
        Ok(Self {
            store_type: VectorStoreType::Pinecone,
            api_key: Some(api_key.to_string()),
            endpoint: Some("https://api.pinecone.io".to_string()),
            index_name: index.to_string(),
            embedding_dim: 1536, // OpenAI default
        })
    }

    pub fn weaviate(endpoint: &str) -> Result<Self, VectorStoreError> {
        Ok(Self {
            store_type: VectorStoreType::Weaviate,
            api_key: None,
            endpoint: Some(endpoint.to_string()),
            index_name: "Documents".to_string(),
            embedding_dim: 1536,
        })
    }

    pub fn chroma(endpoint: &str) -> Result<Self, VectorStoreError> {
        Ok(Self {
            store_type: VectorStoreType::Chroma,
            api_key: None,
            endpoint: Some(endpoint.to_string()),
            index_name: "documents".to_string(),
            embedding_dim: 1536,
        })
    }

    pub fn memory(embedding_dim: usize) -> Result<Self, VectorStoreError> {
        Ok(Self {
            store_type: VectorStoreType::Memory,
            api_key: None,
            endpoint: None,
            index_name: "memory".to_string(),
            embedding_dim,
        })
    }
}

/// Error types for vector store operations
#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("Document not found: {0}")]
    NotFound(String),

    #[error("Embedding error: {0}")]
    EmbeddingError(String),

    #[error("Query error: {0}")]
    QueryError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// A document to be indexed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: HashMap<String, Value>,
}

impl Document {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            embedding: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn add_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Search result from vector store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub metadata: HashMap<String, Value>,
}

/// Abstraction for vector store operations
#[async_trait]
pub trait VectorStoreBackend: Send + Sync {
    /// Index a document with its embedding
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError>;

    /// Retrieve documents by semantic similarity
    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError>;

    /// Delete a document
    async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError>;

    /// Update a document
    async fn update(&self, doc: Document) -> Result<(), VectorStoreError>;

    /// Get document by ID
    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError>;

    /// List all documents
    async fn list(&self) -> Result<Vec<Document>, VectorStoreError>;

    /// Clear all documents
    async fn clear(&self) -> Result<(), VectorStoreError>;
}

/// Main vector store interface
pub struct VectorStore {
    backend: Arc<dyn VectorStoreBackend>,
    config: RAGConfig,
}

impl VectorStore {
    pub fn new(backend: Arc<dyn VectorStoreBackend>, config: RAGConfig) -> Self {
        Self { backend, config }
    }

    pub async fn index_document(
        &self,
        id: impl Into<String>,
        text: impl Into<String>,
        metadata: Value,
    ) -> Result<(), VectorStoreError> {
        let mut doc = Document::new(id, text);
        if let Value::Object(map) = metadata {
            doc.metadata = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        }
        self.backend.index(doc).await
    }

    pub async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        self.backend.search(query_embedding, top_k).await
    }

    pub async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError> {
        self.backend.delete(doc_id).await
    }

    pub async fn update(&self, id: &str, text: &str) -> Result<(), VectorStoreError> {
        let doc = Document::new(id, text);
        self.backend.update(doc).await
    }

    pub async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        self.backend.get(doc_id).await
    }

    pub async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        self.backend.list().await
    }

    pub async fn clear(&self) -> Result<(), VectorStoreError> {
        self.backend.clear().await
    }

    pub fn config(&self) -> &RAGConfig {
        &self.config
    }
}

// ============================================================================
// In-Memory Implementation (always available)
// ============================================================================

pub struct InMemoryVectorStore {
    documents: std::sync::Mutex<HashMap<String, Document>>,
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self {
            documents: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl InMemoryVectorStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }
}

#[async_trait]
impl VectorStoreBackend for InMemoryVectorStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let mut docs = self.documents.lock().unwrap();
        docs.insert(doc.id.clone(), doc);
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let docs = self.documents.lock().unwrap();

        let mut results: Vec<SearchResult> = docs
            .values()
            .filter_map(|doc| {
                doc.embedding.as_ref().map(|emb| {
                    let score = InMemoryVectorStore::cosine_similarity(&query_embedding, emb);
                    SearchResult {
                        id: doc.id.clone(),
                        text: doc.text.clone(),
                        score,
                        metadata: doc.metadata.clone(),
                    }
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);

        Ok(results)
    }

    async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError> {
        let mut docs = self.documents.lock().unwrap();
        docs.remove(doc_id);
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        let mut docs = self.documents.lock().unwrap();
        docs.insert(doc.id.clone(), doc);
        Ok(())
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let docs = self.documents.lock().unwrap();
        docs.get(doc_id)
            .cloned()
            .ok_or_else(|| VectorStoreError::NotFound(doc_id.to_string()))
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        let docs = self.documents.lock().unwrap();
        Ok(docs.values().cloned().collect())
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let mut docs = self.documents.lock().unwrap();
        docs.clear();
        Ok(())
    }
}

// ============================================================================
// Pinecone Implementation (when feature enabled)
// ============================================================================

#[cfg(feature = "rag")]
pub struct PineconeStore {
    config: RAGConfig,
}

#[cfg(feature = "rag")]
impl PineconeStore {
    pub async fn new(config: RAGConfig) -> Result<Self, VectorStoreError> {
        if config.store_type != VectorStoreType::Pinecone {
            return Err(VectorStoreError::ConfigError(
                "Config type must be Pinecone".to_string(),
            ));
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> &RAGConfig {
        &self.config
    }
}

#[cfg(feature = "rag")]
#[async_trait]
impl VectorStoreBackend for PineconeStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        // Implement Pinecone indexing
        Ok(())
    }

    async fn search(
        &self,
        _query_embedding: Vec<f32>,
        _top_k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        // Implement Pinecone search
        Ok(vec![])
    }

    async fn delete(&self, _doc_id: &str) -> Result<(), VectorStoreError> {
        Ok(())
    }

    async fn update(&self, _doc: Document) -> Result<(), VectorStoreError> {
        Ok(())
    }

    async fn get(&self, _doc_id: &str) -> Result<Document, VectorStoreError> {
        Err(VectorStoreError::NotFound("Document not found".to_string()))
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        Ok(vec![])
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        Ok(())
    }
}
