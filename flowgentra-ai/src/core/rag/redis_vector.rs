//! Redis Vector Store — uses the RediSearch module (Redis Stack / Redis Cloud).
//!
//! Enabled with the `redis-store` Cargo feature.
//!
//! Documents are stored as Redis Hashes. A RediSearch HNSW index on the
//! `embedding` field enables KNN similarity search. Requires Redis Stack or
//! a Redis instance with the RediSearch module loaded.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{RedisVectorConfig, RedisVectorStore, VectorStoreBackend};
//! use flowgentra_ai::core::rag::Document;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = RedisVectorStore::connect(RedisVectorConfig {
//!         url: "redis://localhost:6379".into(),
//!         index_name: "docs".into(),
//!         key_prefix: "doc".into(),
//!         embedding_dim: 1536,
//!     }).await?;
//!
//!     let mut doc = Document::new("doc-1", "Hello world");
//!     doc.embedding = Some(vec![0.1_f32; 1536]);
//!     store.index(doc).await?;
//!
//!     let results = store.search(vec![0.1_f32; 1536], 5, None).await?;
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use redis::AsyncCommands;
use serde_json::Value;
use std::collections::HashMap;

use super::filter::FilterExpr;
use super::vector_db::{Document, MetadataFilter, SearchResult, VectorStoreBackend, VectorStoreError};

/// Configuration for [`RedisVectorStore`].
#[derive(Debug, Clone)]
pub struct RedisVectorConfig {
    /// Redis connection URL (e.g. `redis://localhost:6379`)
    pub url: String,
    /// RediSearch index name
    pub index_name: String,
    /// Key prefix for document hashes (e.g. `"doc"` → keys like `doc:id-1`)
    pub key_prefix: String,
    /// Embedding dimension
    pub embedding_dim: usize,
}

/// Redis vector store using RediSearch HNSW index.
pub struct RedisVectorStore {
    client: redis::Client,
    config: RedisVectorConfig,
}

impl RedisVectorStore {
    /// Connect to Redis and create the RediSearch vector index if it doesn't exist.
    pub async fn connect(config: RedisVectorConfig) -> Result<Self, VectorStoreError> {
        let client = redis::Client::open(config.url.as_str())
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        // Verify connection
        let mut conn = client
            .get_async_connection()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        let store = Self { client, config };
        store.ensure_index(&mut conn).await?;
        Ok(store)
    }

    async fn conn(&self) -> Result<redis::aio::Connection, VectorStoreError> {
        self.client
            .get_async_connection()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))
    }

    fn doc_key(&self, id: &str) -> String {
        format!("{}:{}", self.config.key_prefix, id)
    }

    /// Encode a Vec<f32> as raw bytes (little-endian IEEE 754) for Redis vector fields.
    fn encode_embedding(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Create the RediSearch HNSW index (no-op if it already exists).
    async fn ensure_index(&self, conn: &mut redis::aio::Connection) -> Result<(), VectorStoreError> {
        let dim = self.config.embedding_dim;
        let idx = &self.config.index_name;
        let prefix = format!("{}:", self.config.key_prefix);

        // FT.CREATE idx ON HASH PREFIX 1 "doc:" SCHEMA ...
        let result: redis::RedisResult<()> = redis::cmd("FT.CREATE")
            .arg(idx)
            .arg("ON").arg("HASH")
            .arg("PREFIX").arg(1).arg(&prefix)
            .arg("SCHEMA")
            .arg("doc_id").arg("TAG")
            .arg("text").arg("TEXT")
            .arg("metadata").arg("TEXT")
            .arg("embedding").arg("VECTOR").arg("HNSW").arg(6)
                .arg("TYPE").arg("FLOAT32")
                .arg("DIM").arg(dim)
                .arg("DISTANCE_METRIC").arg("COSINE")
            .query_async(conn)
            .await;

        // Index already exists → ignore error
        if let Err(e) = result {
            if !e.to_string().contains("Index already exists") {
                return Err(VectorStoreError::ConnectionError(e.to_string()));
            }
        }
        Ok(())
    }

    /// Build RediSearch filter expression from a `FilterExpr`.
    fn filter_to_redis(f: &FilterExpr) -> String {
        match f {
            FilterExpr::Eq(k, v)  => format!("@{}:{{{}}}", k, v.as_str().unwrap_or("")),
            FilterExpr::Ne(k, v)  => format!("-@{}:{{{}}}", k, v.as_str().unwrap_or("")),
            FilterExpr::Gt(k, v)  => format!("@{}:[({}+inf]", k, v),
            FilterExpr::Lt(k, v)  => format!("@{}:[-inf ({}]", k, v),
            FilterExpr::Gte(k, v) => format!("@{}:[{} +inf]", k, v),
            FilterExpr::Lte(k, v) => format!("@{}:[-inf {}]", k, v),
            FilterExpr::In(k, vs) => {
                let vals: Vec<String> = vs.iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect();
                format!("@{}:{{{}}}", k, vals.join("|"))
            }
            FilterExpr::And(exprs) => {
                exprs.iter().map(Self::filter_to_redis).collect::<Vec<_>>().join(" ")
            }
            FilterExpr::Or(exprs) => {
                let parts: Vec<String> = exprs.iter().map(Self::filter_to_redis).collect();
                format!("({})", parts.join("|"))
            }
        }
    }
}

#[async_trait]
impl VectorStoreBackend for RedisVectorStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc.embedding.ok_or_else(|| {
            VectorStoreError::EmbeddingError("Document must have an embedding".into())
        })?;
        let mut conn = self.conn().await?;
        let key = self.doc_key(&doc.id);
        let metadata_json = serde_json::to_string(&doc.metadata)
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;
        let embedding_bytes = Self::encode_embedding(&embedding);

        redis::cmd("HSET")
            .arg(&key)
            .arg("doc_id").arg(&doc.id)
            .arg("text").arg(&doc.text)
            .arg("metadata").arg(&metadata_json)
            .arg("embedding").arg(&embedding_bytes)
            .query_async::<_, ()>(&mut conn)
            .await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut conn = self.conn().await?;
        let query_bytes = Self::encode_embedding(&query_embedding);

        let base_filter = filter
            .as_ref()
            .map(Self::filter_to_redis)
            .unwrap_or_else(|| "*".to_string());

        let knn_query = format!(
            "{}=>[KNN {} @embedding $vec AS __score]",
            base_filter, top_k
        );

        let raw: Vec<Value> = redis::cmd("FT.SEARCH")
            .arg(&self.config.index_name)
            .arg(&knn_query)
            .arg("PARAMS").arg(2).arg("vec").arg(&query_bytes)
            .arg("SORTBY").arg("__score").arg("ASC")
            .arg("RETURN").arg(4).arg("doc_id").arg("text").arg("metadata").arg("__score")
            .arg("DIALECT").arg(2)
            .query_async(&mut conn)
            .await
            .map_err(|e| VectorStoreError::QueryError(e.to_string()))?;

        // FT.SEARCH returns: [total_count, key1, [field, val, ...], key2, ...]
        let mut results = Vec::new();
        let mut i = 1usize; // skip total count
        while i + 1 < raw.len() {
            i += 1; // skip key
            if let Some(Value::Array(fields)) = raw.get(i) {
                let mut map: HashMap<String, String> = HashMap::new();
                let mut j = 0;
                while j + 1 < fields.len() {
                    let k = fields[j].as_str().unwrap_or("").to_string();
                    let v = fields[j + 1].as_str().unwrap_or("").to_string();
                    map.insert(k, v);
                    j += 2;
                }
                let id = map.get("doc_id").cloned().unwrap_or_default();
                let text = map.get("text").cloned().unwrap_or_default();
                let score: f32 = map.get("__score")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1.0);
                let similarity = 1.0 - score; // cosine distance → similarity
                let metadata: HashMap<String, Value> = map.get("metadata")
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or_default();
                results.push(SearchResult { id, text, score: similarity, metadata });
            }
            i += 1;
        }
        Ok(results)
    }

    async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError> {
        let mut conn = self.conn().await?;
        let key = self.doc_key(doc_id);
        conn.del::<_, ()>(&key)
            .await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.index(doc).await
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let mut conn = self.conn().await?;
        let key = self.doc_key(doc_id);
        let fields: HashMap<String, String> = conn
            .hgetall(&key)
            .await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;

        if fields.is_empty() {
            return Err(VectorStoreError::NotFound(doc_id.to_string()));
        }
        let text = fields.get("text").cloned().unwrap_or_default();
        let metadata: HashMap<String, Value> = fields
            .get("metadata")
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();
        Ok(Document { id: doc_id.to_string(), text, embedding: None, metadata })
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        let mut conn = self.conn().await?;
        let pattern = format!("{}:*", self.config.key_prefix);
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(&pattern)
            .query_async(&mut conn)
            .await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;

        let mut docs = Vec::new();
        for key in keys {
            let fields: HashMap<String, String> = conn
                .hgetall(&key)
                .await
                .unwrap_or_default();
            if fields.is_empty() { continue; }
            let id = fields.get("doc_id").cloned().unwrap_or_default();
            let text = fields.get("text").cloned().unwrap_or_default();
            let metadata: HashMap<String, Value> = fields
                .get("metadata")
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or_default();
            docs.push(Document { id, text, embedding: None, metadata });
        }
        Ok(docs)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let mut conn = self.conn().await?;
        let pattern = format!("{}:*", self.config.key_prefix);
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(&pattern)
            .query_async(&mut conn)
            .await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;
        if !keys.is_empty() {
            conn.del::<_, ()>(keys)
                .await
                .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;
        }
        Ok(())
    }
}
