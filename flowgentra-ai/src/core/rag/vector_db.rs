//! Vector Database Abstractions and Implementations
//!
//! Provides a unified interface for multiple vector stores.

use async_trait::async_trait;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub use super::filter::{FilterExpr, MetadataFilter};

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
            embedding_dim: 1536,
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

    pub fn qdrant(
        endpoint: &str,
        collection: &str,
        embedding_dim: usize,
    ) -> Result<Self, VectorStoreError> {
        Ok(Self {
            store_type: VectorStoreType::Qdrant,
            api_key: None,
            endpoint: Some(endpoint.to_string()),
            index_name: collection.to_string(),
            embedding_dim,
        })
    }

    pub fn milvus(
        endpoint: &str,
        collection: &str,
        embedding_dim: usize,
    ) -> Result<Self, VectorStoreError> {
        Ok(Self {
            store_type: VectorStoreType::Milvus,
            api_key: None,
            endpoint: Some(endpoint.to_string()),
            index_name: collection.to_string(),
            embedding_dim,
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

    #[error("Not implemented: {0}")]
    NotImplemented(String),

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

// MetadataFilter is re-exported from filter.rs above.

/// Abstraction for vector store operations
#[async_trait]
pub trait VectorStoreBackend: Send + Sync {
    /// Index a document with its embedding
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError>;

    /// Retrieve documents by semantic similarity, with optional metadata filter
    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
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

    /// Whether this backend supports the `list()` operation.
    ///
    /// Cloud vector DBs (Pinecone, Qdrant) do not expose a list-all endpoint
    /// in their standard APIs. Check this before calling `list()` to avoid a
    /// runtime `NotImplemented` error.
    fn supports_list(&self) -> bool {
        true
    }

    /// Batch search — run multiple queries in parallel.
    ///
    /// Default implementation runs searches sequentially. Backends can override
    /// for true parallel execution.
    async fn search_batch(
        &self,
        queries: Vec<Vec<f32>>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<Vec<SearchResult>>, VectorStoreError> {
        let mut all_results = Vec::with_capacity(queries.len());
        for query in queries {
            let results = self.search(query, top_k, filter.clone()).await?;
            all_results.push(results);
        }
        Ok(all_results)
    }
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

    /// Index a document along with its pre-computed embedding.
    ///
    /// Use this instead of `index_document` when you have already computed the
    /// embedding externally (e.g. via `IngestionPipeline`) to avoid a second
    /// embedding call inside the backend.
    pub async fn index_document_with_embedding(
        &self,
        id: impl Into<String>,
        text: impl Into<String>,
        metadata: Value,
        embedding: Vec<f32>,
    ) -> Result<(), VectorStoreError> {
        let mut doc = Document::new(id, text);
        if let Value::Object(map) = metadata {
            doc.metadata = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        }
        doc.embedding = Some(embedding);
        self.backend.index(doc).await
    }

    pub async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        self.backend.search(query_embedding, top_k, filter).await
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

    pub async fn search_batch(
        &self,
        queries: Vec<Vec<f32>>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<Vec<SearchResult>>, VectorStoreError> {
        self.backend.search_batch(queries, top_k, filter).await
    }

    /// Returns `false` for backends that do not implement `list()`.
    /// Check this before calling `list()` to avoid a runtime error.
    pub fn supports_list(&self) -> bool {
        self.backend.supports_list()
    }

    pub fn config(&self) -> &RAGConfig {
        &self.config
    }
}

// ============================================================================
// In-Memory Implementation (always available)
// ============================================================================

pub struct InMemoryVectorStore {
    pub(crate) documents: DashMap<String, Document>,
}

impl Default for InMemoryVectorStore {
    fn default() -> Self {
        Self {
            documents: DashMap::new(),
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

    /// Evaluate a `FilterExpr` against a document's metadata map.
    fn matches_filter(metadata: &HashMap<String, Value>, filter: &FilterExpr) -> bool {
        match filter {
            FilterExpr::Eq(k, v) => metadata.get(k) == Some(v),
            FilterExpr::Ne(k, v) => metadata.get(k) != Some(v),
            FilterExpr::Gt(k, v) => {
                Self::cmp_values(metadata.get(k), v) == Some(std::cmp::Ordering::Greater)
            }
            FilterExpr::Lt(k, v) => {
                Self::cmp_values(metadata.get(k), v) == Some(std::cmp::Ordering::Less)
            }
            FilterExpr::Gte(k, v) => matches!(
                Self::cmp_values(metadata.get(k), v),
                Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)
            ),
            FilterExpr::Lte(k, v) => matches!(
                Self::cmp_values(metadata.get(k), v),
                Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)
            ),
            FilterExpr::In(k, vs) => metadata.get(k).map(|v| vs.contains(v)).unwrap_or(false),
            FilterExpr::And(exprs) => exprs.iter().all(|e| Self::matches_filter(metadata, e)),
            FilterExpr::Or(exprs) => exprs.iter().any(|e| Self::matches_filter(metadata, e)),
        }
    }

    /// Compare two JSON values for ordering (numbers and strings only).
    fn cmp_values(a: Option<&Value>, b: &Value) -> Option<std::cmp::Ordering> {
        let a = a?;
        match (a, b) {
            (Value::Number(an), Value::Number(bn)) => an.as_f64()?.partial_cmp(&bn.as_f64()?),
            (Value::String(as_), Value::String(bs)) => Some(as_.cmp(bs)),
            _ => None,
        }
    }
}

#[async_trait]
impl VectorStoreBackend for InMemoryVectorStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.documents.insert(doc.id.clone(), doc);
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut results: Vec<SearchResult> = self
            .documents
            .iter()
            .filter_map(|entry| {
                let doc = entry.value();

                // Apply metadata filter
                if let Some(ref f) = filter {
                    if !Self::matches_filter(&doc.metadata, f) {
                        return None;
                    }
                }

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
        self.documents.remove(doc_id);
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.documents.insert(doc.id.clone(), doc);
        Ok(())
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        self.documents
            .get(doc_id)
            .map(|entry| entry.value().clone())
            .ok_or_else(|| VectorStoreError::NotFound(doc_id.to_string()))
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        Ok(self
            .documents
            .iter()
            .map(|entry| entry.value().clone())
            .collect())
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        self.documents.clear();
        Ok(())
    }
}

// ============================================================================
// Pinecone Implementation — NOT YET IMPLEMENTED
// ============================================================================

/// Pinecone vector store using the Pinecone REST API.
///
/// Requires `api_key` and `endpoint` (your Pinecone index host URL) in the config.
/// The `index_name` field is used as the Pinecone *namespace*.
pub struct PineconeStore {
    config: RAGConfig,
    client: reqwest::Client,
}

impl PineconeStore {
    pub async fn new(config: RAGConfig) -> Result<Self, VectorStoreError> {
        if config.store_type != VectorStoreType::Pinecone {
            return Err(VectorStoreError::ConfigError(
                "Config type must be Pinecone".to_string(),
            ));
        }
        if config.api_key.is_none() {
            return Err(VectorStoreError::ConfigError(
                "Pinecone requires an api_key".to_string(),
            ));
        }
        if config.endpoint.is_none() {
            return Err(VectorStoreError::ConfigError(
                "Pinecone requires an endpoint (your index host URL)".to_string(),
            ));
        }
        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    fn api_key(&self) -> &str {
        self.config.api_key.as_deref().unwrap_or("")
    }

    fn base_url(&self) -> &str {
        self.config
            .endpoint
            .as_deref()
            .unwrap_or("https://api.pinecone.io")
    }

    /// Convert a `FilterExpr` into Pinecone's metadata filter JSON.
    ///
    /// Pinecone format: `{"field": {"$eq": value}}` for leaves,
    /// `{"$and": [...]}` / `{"$or": [...]}` for compound expressions.
    fn to_pinecone_filter(f: &FilterExpr) -> Value {
        match f {
            FilterExpr::Eq(k, v) => serde_json::json!({ k: { "$eq":  v } }),
            FilterExpr::Ne(k, v) => serde_json::json!({ k: { "$ne":  v } }),
            FilterExpr::Gt(k, v) => serde_json::json!({ k: { "$gt":  v } }),
            FilterExpr::Lt(k, v) => serde_json::json!({ k: { "$lt":  v } }),
            FilterExpr::Gte(k, v) => serde_json::json!({ k: { "$gte": v } }),
            FilterExpr::Lte(k, v) => serde_json::json!({ k: { "$lte": v } }),
            FilterExpr::In(k, vs) => serde_json::json!({ k: { "$in":  vs } }),
            FilterExpr::And(exprs) => serde_json::json!({
                "$and": exprs.iter().map(Self::to_pinecone_filter).collect::<Vec<_>>()
            }),
            FilterExpr::Or(exprs) => serde_json::json!({
                "$or": exprs.iter().map(Self::to_pinecone_filter).collect::<Vec<_>>()
            }),
        }
    }
}

#[async_trait]
impl VectorStoreBackend for PineconeStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc.embedding.ok_or_else(|| {
            VectorStoreError::EmbeddingError("Document must have an embedding to index".to_string())
        })?;

        let payload = serde_json::json!({
            "vectors": [{
                "id": doc.id,
                "values": embedding,
                "metadata": {
                    "text": doc.text,
                    "_meta": doc.metadata,
                }
            }],
            "namespace": self.config.index_name,
        });

        let resp = self
            .client
            .post(format!("{}/vectors/upsert", self.base_url()))
            .header("Api-Key", self.api_key())
            .json(&payload)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Pinecone upsert failed: {}",
                body
            )));
        }
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut payload = serde_json::json!({
            "vector": query_embedding,
            "topK": top_k,
            "includeMetadata": true,
            "namespace": self.config.index_name,
        });

        if let Some(f) = filter {
            payload["filter"] = Self::to_pinecone_filter(&f);
        }

        let resp = self
            .client
            .post(format!("{}/query", self.base_url()))
            .header("Api-Key", self.api_key())
            .json(&payload)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Pinecone query failed: {}",
                body
            )));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;

        let empty_vec = Vec::new();
        let matches = data
            .get("matches")
            .and_then(|m| m.as_array())
            .unwrap_or(&empty_vec);

        let results = matches
            .iter()
            .filter_map(|m| {
                let id = m.get("id")?.as_str()?.to_string();
                let score = m.get("score")?.as_f64()? as f32;
                let metadata = m.get("metadata").cloned().unwrap_or(serde_json::json!({}));
                let text = metadata
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let meta_map: HashMap<String, Value> = metadata
                    .get("_meta")
                    .and_then(|m| serde_json::from_value(m.clone()).ok())
                    .unwrap_or_default();
                Some(SearchResult {
                    id,
                    text,
                    score,
                    metadata: meta_map,
                })
            })
            .collect();

        Ok(results)
    }

    async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError> {
        let payload = serde_json::json!({
            "ids": [doc_id],
            "namespace": self.config.index_name,
        });

        let resp = self
            .client
            .post(format!("{}/vectors/delete", self.base_url()))
            .header("Api-Key", self.api_key())
            .json(&payload)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Pinecone delete failed: {}",
                body
            )));
        }
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        // Pinecone upsert is idempotent — updating is the same as indexing
        self.index(doc).await
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let payload = serde_json::json!({
            "ids": [doc_id],
            "namespace": self.config.index_name,
        });

        let resp = self
            .client
            .get(format!("{}/vectors/fetch", self.base_url()))
            .header("Api-Key", self.api_key())
            .query(&[("ids", doc_id), ("namespace", &self.config.index_name)])
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Pinecone fetch failed: {}",
                body
            )));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;

        let vectors = data
            .get("vectors")
            .and_then(|v| v.as_object())
            .ok_or_else(|| VectorStoreError::NotFound(doc_id.to_string()))?;

        let vec_data = vectors
            .get(doc_id)
            .ok_or_else(|| VectorStoreError::NotFound(doc_id.to_string()))?;

        let metadata = vec_data
            .get("metadata")
            .cloned()
            .unwrap_or(serde_json::json!({}));
        let text = metadata
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        let embedding = vec_data
            .get("values")
            .and_then(|v| serde_json::from_value::<Vec<f32>>(v.clone()).ok());
        let meta_map: HashMap<String, Value> = metadata
            .get("_meta")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();

        let _ = payload; // suppress unused warning
        Ok(Document {
            id: doc_id.to_string(),
            text,
            embedding,
            metadata: meta_map,
        })
    }

    fn supports_list(&self) -> bool {
        false
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        // Pinecone doesn't have a list-all endpoint in the standard API.
        // Users should track document IDs externally.
        Err(VectorStoreError::NotImplemented(
            "Pinecone does not support listing all documents. Track IDs externally.".into(),
        ))
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let payload = serde_json::json!({
            "deleteAll": true,
            "namespace": self.config.index_name,
        });

        let resp = self
            .client
            .post(format!("{}/vectors/delete", self.base_url()))
            .header("Api-Key", self.api_key())
            .json(&payload)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Pinecone clear failed: {}",
                body
            )));
        }
        Ok(())
    }
}

// ============================================================================
// Qdrant Implementation
// ============================================================================

/// Qdrant vector store using the Qdrant REST API.
///
/// Requires `endpoint` (e.g., "http://localhost:6333") in the config.
pub struct QdrantStore {
    config: RAGConfig,
    client: reqwest::Client,
}

impl QdrantStore {
    pub async fn new(config: RAGConfig) -> Result<Self, VectorStoreError> {
        if config.store_type != VectorStoreType::Qdrant {
            return Err(VectorStoreError::ConfigError(
                "Config type must be Qdrant".to_string(),
            ));
        }
        if config.endpoint.is_none() {
            return Err(VectorStoreError::ConfigError(
                "Qdrant requires an endpoint".to_string(),
            ));
        }
        Ok(Self {
            config,
            client: reqwest::Client::new(),
        })
    }

    fn base_url(&self) -> &str {
        self.config
            .endpoint
            .as_deref()
            .unwrap_or("http://localhost:6333")
    }

    fn collection(&self) -> &str {
        &self.config.index_name
    }

    /// Build a Qdrant top-level filter object from a `FilterExpr`.
    ///
    /// Qdrant format: `{"must": [...]}` / `{"should": [...]}` at the top level,
    /// with leaf conditions `{"key": k, "match": {"value": v}}`.
    fn to_qdrant_filter(f: &FilterExpr) -> Value {
        match f {
            FilterExpr::And(exprs) => serde_json::json!({
                "must": exprs.iter().map(Self::qdrant_condition).collect::<Vec<_>>()
            }),
            FilterExpr::Or(exprs) => serde_json::json!({
                "should": exprs.iter().map(Self::qdrant_condition).collect::<Vec<_>>()
            }),
            // Single leaf: wrap in must
            other => serde_json::json!({ "must": [Self::qdrant_condition(other)] }),
        }
    }

    /// Build a single Qdrant condition object (leaf or nested).
    fn qdrant_condition(f: &FilterExpr) -> Value {
        match f {
            FilterExpr::Eq(k, v) => serde_json::json!({ "key": k, "match":  { "value": v } }),
            FilterExpr::Ne(k, v) => serde_json::json!({ "key": k, "match":  { "except": [v] } }),
            FilterExpr::Gt(k, v) => serde_json::json!({ "key": k, "range":  { "gt":  v } }),
            FilterExpr::Lt(k, v) => serde_json::json!({ "key": k, "range":  { "lt":  v } }),
            FilterExpr::Gte(k, v) => serde_json::json!({ "key": k, "range":  { "gte": v } }),
            FilterExpr::Lte(k, v) => serde_json::json!({ "key": k, "range":  { "lte": v } }),
            FilterExpr::In(k, vs) => serde_json::json!({ "key": k, "match":  { "any":  vs } }),
            FilterExpr::And(exprs) => serde_json::json!({
                "filter": { "must": exprs.iter().map(Self::qdrant_condition).collect::<Vec<_>>() }
            }),
            FilterExpr::Or(exprs) => serde_json::json!({
                "filter": { "should": exprs.iter().map(Self::qdrant_condition).collect::<Vec<_>>() }
            }),
        }
    }
}

#[async_trait]
impl VectorStoreBackend for QdrantStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc.embedding.ok_or_else(|| {
            VectorStoreError::EmbeddingError("Document must have an embedding".to_string())
        })?;

        let mut payload_map = doc.metadata.clone();
        payload_map.insert("text".to_string(), serde_json::json!(doc.text));

        // Qdrant uses numeric or UUID point IDs. Use a hash of doc.id for numeric ID.
        let point_id =
            uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, doc.id.as_bytes()).to_string();

        let body = serde_json::json!({
            "points": [{
                "id": point_id,
                "vector": embedding,
                "payload": payload_map,
            }]
        });

        let resp = self
            .client
            .put(format!(
                "{}/collections/{}/points",
                self.base_url(),
                self.collection()
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Qdrant upsert failed: {}",
                text
            )));
        }
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut body = serde_json::json!({
            "vector": query_embedding,
            "limit": top_k,
            "with_payload": true,
        });

        if let Some(f) = filter {
            body["filter"] = Self::to_qdrant_filter(&f);
        }

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/search",
                self.base_url(),
                self.collection()
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Qdrant search failed: {}",
                text
            )));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;

        let empty_vec2 = Vec::new();
        let results_arr = data
            .get("result")
            .and_then(|r| r.as_array())
            .unwrap_or(&empty_vec2);

        let results = results_arr
            .iter()
            .filter_map(|r| {
                let id = r.get("id")?.to_string().trim_matches('"').to_string();
                let score = r.get("score")?.as_f64()? as f32;
                let payload = r.get("payload").cloned().unwrap_or(serde_json::json!({}));
                let text = payload
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let mut meta: HashMap<String, Value> =
                    serde_json::from_value(payload.clone()).unwrap_or_default();
                meta.remove("text");
                Some(SearchResult {
                    id,
                    text,
                    score,
                    metadata: meta,
                })
            })
            .collect();

        Ok(results)
    }

    async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError> {
        let point_id =
            uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, doc_id.as_bytes()).to_string();
        let body = serde_json::json!({
            "points": [point_id],
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/delete",
                self.base_url(),
                self.collection()
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Qdrant delete failed: {}",
                text
            )));
        }
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.index(doc).await
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let point_id =
            uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, doc_id.as_bytes()).to_string();
        let body = serde_json::json!({
            "ids": [point_id],
            "with_payload": true,
            "with_vector": true,
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points",
                self.base_url(),
                self.collection()
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(VectorStoreError::NotFound(doc_id.to_string()));
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;

        let point = data
            .get("result")
            .and_then(|r| r.as_array())
            .and_then(|a| a.first())
            .ok_or_else(|| VectorStoreError::NotFound(doc_id.to_string()))?;

        let payload = point
            .get("payload")
            .cloned()
            .unwrap_or(serde_json::json!({}));
        let text = payload
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        let embedding = point
            .get("vector")
            .and_then(|v| serde_json::from_value::<Vec<f32>>(v.clone()).ok());
        let mut meta: HashMap<String, Value> = serde_json::from_value(payload).unwrap_or_default();
        meta.remove("text");

        Ok(Document {
            id: doc_id.to_string(),
            text,
            embedding,
            metadata: meta,
        })
    }

    fn supports_list(&self) -> bool {
        false
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        Err(VectorStoreError::NotImplemented(
            "Qdrant list-all requires scroll API. Use search with empty filter instead.".into(),
        ))
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        // Delete and recreate the collection
        let _ = self
            .client
            .delete(format!(
                "{}/collections/{}",
                self.base_url(),
                self.collection()
            ))
            .send()
            .await;

        let body = serde_json::json!({
            "vectors": {
                "size": self.config.embedding_dim,
                "distance": "Cosine"
            }
        });

        let resp = self
            .client
            .put(format!(
                "{}/collections/{}",
                self.base_url(),
                self.collection()
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Qdrant clear failed: {}",
                text
            )));
        }
        Ok(())
    }
}

// ============================================================================
// Per-backend config structs (clear field names, no overloaded index_name)
// ============================================================================

/// Config for [`PineconeStore`] with explicit field names.
///
/// Converts into [`RAGConfig`] via `From<PineconeConfig>`.
#[derive(Debug, Clone)]
pub struct PineconeConfig {
    /// Pinecone API key
    pub api_key: String,
    /// Index host URL (e.g. `https://<index>-<project>.svc.pinecone.io`)
    pub endpoint: String,
    /// Namespace within the index (maps to `index_name` in RAGConfig)
    pub namespace: String,
    /// Embedding dimension (e.g. 1536 for OpenAI text-embedding-ada-002)
    pub embedding_dim: usize,
}

impl From<PineconeConfig> for RAGConfig {
    fn from(c: PineconeConfig) -> Self {
        Self {
            store_type: VectorStoreType::Pinecone,
            api_key: Some(c.api_key),
            endpoint: Some(c.endpoint),
            index_name: c.namespace,
            embedding_dim: c.embedding_dim,
        }
    }
}

/// Config for [`QdrantStore`] with explicit field names.
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    /// Qdrant base URL (e.g. `http://localhost:6333`)
    pub endpoint: String,
    /// Collection name (maps to `index_name` in RAGConfig)
    pub collection: String,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Optional API key for Qdrant Cloud
    pub api_key: Option<String>,
}

impl From<QdrantConfig> for RAGConfig {
    fn from(c: QdrantConfig) -> Self {
        Self {
            store_type: VectorStoreType::Qdrant,
            api_key: c.api_key,
            endpoint: Some(c.endpoint),
            index_name: c.collection,
            embedding_dim: c.embedding_dim,
        }
    }
}

/// Config for [`ChromaStore`] with explicit field names.
#[derive(Debug, Clone)]
pub struct ChromaConfig {
    /// ChromaDB base URL (e.g. `http://localhost:8000`)
    pub endpoint: String,
    /// Collection name (maps to `index_name` in RAGConfig)
    pub collection: String,
}

impl From<ChromaConfig> for RAGConfig {
    fn from(c: ChromaConfig) -> Self {
        Self {
            store_type: VectorStoreType::Chroma,
            api_key: None,
            endpoint: Some(c.endpoint),
            index_name: c.collection,
            embedding_dim: 1536,
        }
    }
}

/// Config for [`MilvusStore`] with explicit field names.
#[derive(Debug, Clone)]
pub struct MilvusConfig {
    /// Milvus REST endpoint (e.g. `http://localhost:19530`)
    pub endpoint: String,
    /// Collection name (maps to `index_name` in RAGConfig)
    pub collection: String,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Optional Bearer token
    pub api_key: Option<String>,
}

impl From<MilvusConfig> for RAGConfig {
    fn from(c: MilvusConfig) -> Self {
        Self {
            store_type: VectorStoreType::Milvus,
            api_key: c.api_key,
            endpoint: Some(c.endpoint),
            index_name: c.collection,
            embedding_dim: c.embedding_dim,
        }
    }
}

/// Config for [`WeaviateStore`] with explicit field names.
#[derive(Debug, Clone)]
pub struct WeaviateConfig {
    /// Weaviate base URL (e.g. `http://localhost:8080`)
    pub endpoint: String,
    /// Class name — auto-capitalised if needed (maps to `index_name` in RAGConfig)
    pub class: String,
    /// Optional Bearer API key
    pub api_key: Option<String>,
}

impl From<WeaviateConfig> for RAGConfig {
    fn from(c: WeaviateConfig) -> Self {
        Self {
            store_type: VectorStoreType::Weaviate,
            api_key: c.api_key,
            endpoint: Some(c.endpoint),
            index_name: c.class,
            embedding_dim: 1536,
        }
    }
}
