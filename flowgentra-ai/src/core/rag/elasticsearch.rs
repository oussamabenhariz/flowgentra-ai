//! Elasticsearch Vector Store — uses the `dense_vector` field type and kNN search.
//!
//! No extra Cargo feature needed — uses the existing `reqwest` HTTP client.
//! Enable with the `elasticsearch-store` feature flag (currently a no-op marker).
//!
//! Compatible with Elasticsearch 8.x+ (kNN search was GA in 8.0).
//! Also works with **Elastic Cloud** — just set the endpoint to your Cloud URL.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{ElasticsearchConfig, ElasticsearchStore, VectorStoreBackend};
//! use flowgentra_ai::core::rag::Document;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = ElasticsearchStore::new(ElasticsearchConfig {
//!         endpoint: "http://localhost:9200".into(),
//!         index: "documents".into(),
//!         embedding_dim: 1536,
//!         api_key: None,
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
use serde_json::{json, Value};
use std::collections::HashMap;

use super::filter::FilterExpr;
use super::vector_db::{
    Document, MetadataFilter, SearchResult, VectorStoreBackend, VectorStoreError,
};

/// Configuration for [`ElasticsearchStore`].
#[derive(Debug, Clone)]
pub struct ElasticsearchConfig {
    /// Elasticsearch base URL (e.g. `http://localhost:9200`)
    pub endpoint: String,
    /// Index name
    pub index: String,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Optional API key for Elastic Cloud (`ApiKey <base64>` or just the key)
    pub api_key: Option<String>,
}

/// Elasticsearch vector store.
pub struct ElasticsearchStore {
    client: reqwest::Client,
    config: ElasticsearchConfig,
}

impl ElasticsearchStore {
    /// Create the store and ensure the index mapping exists.
    pub async fn new(config: ElasticsearchConfig) -> Result<Self, VectorStoreError> {
        let client = reqwest::Client::new();
        let store = Self { client, config };
        store.ensure_index().await?;
        Ok(store)
    }

    fn url(&self, path: &str) -> String {
        format!(
            "{}/{}{}",
            self.config.endpoint.trim_end_matches('/'),
            self.config.index,
            path
        )
    }

    fn add_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref key) = self.config.api_key {
            req.header("Authorization", format!("ApiKey {}", key))
        } else {
            req
        }
    }

    async fn ensure_index(&self) -> Result<(), VectorStoreError> {
        let mapping = json!({
            "mappings": {
                "properties": {
                    "text":      { "type": "text" },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.config.embedding_dim,
                        "index": true,
                        "similarity": "cosine"
                    },
                    "metadata":  { "type": "object", "dynamic": true }
                }
            }
        });

        let resp = self
            .add_auth(self.client.put(self.url("")))
            .json(&mapping)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        let status = resp.status();
        // 400 with "resource_already_exists_exception" is fine
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            if !body.contains("resource_already_exists_exception") {
                return Err(VectorStoreError::ApiError(format!(
                    "ES create index failed ({status}): {body}"
                )));
            }
        }
        Ok(())
    }

    /// Convert `FilterExpr` to an Elasticsearch query DSL fragment.
    fn filter_to_es(f: &FilterExpr) -> Value {
        match f {
            FilterExpr::Eq(k, v) => json!({ "term": { format!("metadata.{}", k): v } }),
            FilterExpr::Ne(k, v) => {
                json!({ "bool": { "must_not": [{ "term": { format!("metadata.{}", k): v } }] } })
            }
            FilterExpr::Gt(k, v) => json!({ "range": { format!("metadata.{}", k): { "gt": v } } }),
            FilterExpr::Lt(k, v) => json!({ "range": { format!("metadata.{}", k): { "lt": v } } }),
            FilterExpr::Gte(k, v) => {
                json!({ "range": { format!("metadata.{}", k): { "gte": v } } })
            }
            FilterExpr::Lte(k, v) => {
                json!({ "range": { format!("metadata.{}", k): { "lte": v } } })
            }
            FilterExpr::In(k, vs) => json!({ "terms": { format!("metadata.{}", k): vs } }),
            FilterExpr::And(exprs) => {
                json!({ "bool": { "must": exprs.iter().map(Self::filter_to_es).collect::<Vec<_>>() } })
            }
            FilterExpr::Or(exprs) => {
                json!({ "bool": { "should": exprs.iter().map(Self::filter_to_es).collect::<Vec<_>>() } })
            }
        }
    }
}

#[async_trait]
impl VectorStoreBackend for ElasticsearchStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc.embedding.ok_or_else(|| {
            VectorStoreError::EmbeddingError("Document must have an embedding".into())
        })?;

        let body = json!({
            "text":      doc.text,
            "embedding": embedding,
            "metadata":  doc.metadata,
        });

        let resp = self
            .add_auth(self.client.put(self.url(&format!("/_doc/{}", doc.id))))
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "ES index failed: {text}"
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
        let mut body = json!({
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": top_k * 10
            },
            "_source": ["text", "metadata"],
            "size": top_k
        });

        if let Some(f) = filter {
            body["knn"]["filter"] = Self::filter_to_es(&f);
        }

        let resp = self
            .add_auth(self.client.post(self.url("/_search")))
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::QueryError(format!(
                "ES search failed: {text}"
            )));
        }

        let data: Value = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;

        let empty = vec![];
        let hits = data["hits"]["hits"].as_array().unwrap_or(&empty);
        let results = hits
            .iter()
            .filter_map(|hit| {
                let id = hit["_id"].as_str()?.to_string();
                let score = hit["_score"].as_f64()? as f32;
                let src = &hit["_source"];
                let text = src["text"].as_str().unwrap_or("").to_string();
                let metadata: HashMap<String, Value> =
                    serde_json::from_value(src["metadata"].clone()).unwrap_or_default();
                Some(SearchResult {
                    id,
                    text,
                    score,
                    metadata,
                })
            })
            .collect();
        Ok(results)
    }

    async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError> {
        let resp = self
            .add_auth(self.client.delete(self.url(&format!("/_doc/{}", doc_id))))
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        if !resp.status().is_success() && resp.status().as_u16() != 404 {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "ES delete failed: {text}"
            )));
        }
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.index(doc).await
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let resp = self
            .add_auth(self.client.get(self.url(&format!("/_doc/{}", doc_id))))
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if resp.status().as_u16() == 404 {
            return Err(VectorStoreError::NotFound(doc_id.to_string()));
        }
        let data: Value = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;
        let src = &data["_source"];
        let text = src["text"].as_str().unwrap_or("").to_string();
        let metadata: HashMap<String, Value> =
            serde_json::from_value(src["metadata"].clone()).unwrap_or_default();
        Ok(Document {
            id: doc_id.to_string(),
            text,
            embedding: None,
            metadata,
        })
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        let body =
            json!({ "query": { "match_all": {} }, "size": 10000, "_source": ["text", "metadata"] });
        let resp = self
            .add_auth(self.client.post(self.url("/_search")))
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        let data: Value = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;
        let empty = vec![];
        let hits = data["hits"]["hits"].as_array().unwrap_or(&empty);
        let docs = hits
            .iter()
            .filter_map(|hit| {
                let id = hit["_id"].as_str()?.to_string();
                let src = &hit["_source"];
                let text = src["text"].as_str().unwrap_or("").to_string();
                let metadata: HashMap<String, Value> =
                    serde_json::from_value(src["metadata"].clone()).unwrap_or_default();
                Some(Document {
                    id,
                    text,
                    embedding: None,
                    metadata,
                })
            })
            .collect();
        Ok(docs)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let resp = self
            .add_auth(self.client.post(self.url("/_delete_by_query")))
            .json(&json!({ "query": { "match_all": {} } }))
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "ES clear failed: {text}"
            )));
        }
        Ok(())
    }
}
