//! OpenSearch Vector Store — uses the `knn_vector` field type and k-NN plugin.
//!
//! OpenSearch 2.x compatible. Works with **AWS OpenSearch Service** and self-hosted.
//! The filter conversion reuses the same Elasticsearch DSL structure since
//! OpenSearch's query language is near-identical to ES.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{OpenSearchConfig, OpenSearchStore, VectorStoreBackend};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = OpenSearchStore::new(OpenSearchConfig {
//!         endpoint: "http://localhost:9200".into(),
//!         index: "documents".into(),
//!         embedding_dim: 1536,
//!         username: Some("admin".into()),
//!         password: Some("admin".into()),
//!     }).await?;
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;

use super::filter::FilterExpr;
use super::vector_db::{Document, MetadataFilter, SearchResult, VectorStoreBackend, VectorStoreError};

/// Configuration for [`OpenSearchStore`].
#[derive(Debug, Clone)]
pub struct OpenSearchConfig {
    /// OpenSearch base URL (e.g. `http://localhost:9200`)
    pub endpoint: String,
    /// Index name
    pub index: String,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Optional HTTP Basic Auth username
    pub username: Option<String>,
    /// Optional HTTP Basic Auth password
    pub password: Option<String>,
}

/// OpenSearch vector store using the kNN plugin.
pub struct OpenSearchStore {
    client: reqwest::Client,
    config: OpenSearchConfig,
}

impl OpenSearchStore {
    /// Create the store and ensure the k-NN index mapping exists.
    pub async fn new(config: OpenSearchConfig) -> Result<Self, VectorStoreError> {
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
        match (&self.config.username, &self.config.password) {
            (Some(u), Some(p)) => req.basic_auth(u, Some(p)),
            _ => req,
        }
    }

    async fn ensure_index(&self) -> Result<(), VectorStoreError> {
        let mapping = json!({
            "settings": { "index": { "knn": true } },
            "mappings": {
                "properties": {
                    "text":      { "type": "text" },
                    "embedding": {
                        "type":            "knn_vector",
                        "dimension":       self.config.embedding_dim,
                        "method": {
                            "name":       "hnsw",
                            "space_type": "cosinesimil",
                            "engine":     "lucene"
                        }
                    },
                    "metadata":  { "type": "object", "dynamic": true }
                }
            }
        });

        let resp = self
            .add_auth(self.client.put(&self.url("")))
            .json(&mapping)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            if !body.contains("resource_already_exists_exception") {
                return Err(VectorStoreError::ApiError(format!(
                    "OpenSearch create index failed: {body}"
                )));
            }
        }
        Ok(())
    }

    fn filter_to_os(f: &FilterExpr) -> Value {
        // OpenSearch uses identical filter DSL to Elasticsearch
        match f {
            FilterExpr::Eq(k, v)  => json!({ "term":  { format!("metadata.{}", k): v } }),
            FilterExpr::Ne(k, v)  => json!({ "bool":  { "must_not": [{ "term": { format!("metadata.{}", k): v } }] } }),
            FilterExpr::Gt(k, v)  => json!({ "range": { format!("metadata.{}", k): { "gt":  v } } }),
            FilterExpr::Lt(k, v)  => json!({ "range": { format!("metadata.{}", k): { "lt":  v } } }),
            FilterExpr::Gte(k, v) => json!({ "range": { format!("metadata.{}", k): { "gte": v } } }),
            FilterExpr::Lte(k, v) => json!({ "range": { format!("metadata.{}", k): { "lte": v } } }),
            FilterExpr::In(k, vs) => json!({ "terms": { format!("metadata.{}", k): vs } }),
            FilterExpr::And(exprs) => json!({ "bool": { "must":   exprs.iter().map(Self::filter_to_os).collect::<Vec<_>>() } }),
            FilterExpr::Or(exprs)  => json!({ "bool": { "should": exprs.iter().map(Self::filter_to_os).collect::<Vec<_>>() } }),
        }
    }
}

#[async_trait]
impl VectorStoreBackend for OpenSearchStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc.embedding.ok_or_else(|| {
            VectorStoreError::EmbeddingError("Document must have an embedding".into())
        })?;
        let body = json!({ "text": doc.text, "embedding": embedding, "metadata": doc.metadata });
        let resp = self
            .add_auth(self.client.put(&self.url(&format!("/_doc/{}", doc.id))))
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!("OpenSearch index failed: {text}")));
        }
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let knn_clause = json!({
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": top_k
                }
            }
        });

        let query = if let Some(f) = filter {
            json!({ "bool": { "must": [knn_clause, Self::filter_to_os(&f)] } })
        } else {
            knn_clause
        };

        let body = json!({ "query": query, "size": top_k, "_source": ["text", "metadata"] });
        let resp = self
            .add_auth(self.client.post(&self.url("/_search")))
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::QueryError(format!("OpenSearch search failed: {text}")));
        }
        let data: Value = resp.json().await
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;
        let empty = vec![];
        let hits = data["hits"]["hits"].as_array().unwrap_or(&empty);
        let results = hits.iter().filter_map(|hit| {
            let id    = hit["_id"].as_str()?.to_string();
            let score = hit["_score"].as_f64()? as f32;
            let src   = &hit["_source"];
            let text  = src["text"].as_str().unwrap_or("").to_string();
            let metadata: HashMap<String, Value> =
                serde_json::from_value(src["metadata"].clone()).unwrap_or_default();
            Some(SearchResult { id, text, score, metadata })
        }).collect();
        Ok(results)
    }

    async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError> {
        let resp = self
            .add_auth(self.client.delete(&self.url(&format!("/_doc/{}", doc_id))))
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        if !resp.status().is_success() && resp.status().as_u16() != 404 {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!("OpenSearch delete failed: {text}")));
        }
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> { self.index(doc).await }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let resp = self
            .add_auth(self.client.get(&self.url(&format!("/_doc/{}", doc_id))))
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        if resp.status().as_u16() == 404 {
            return Err(VectorStoreError::NotFound(doc_id.to_string()));
        }
        let data: Value = resp.json().await
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;
        let src  = &data["_source"];
        let text = src["text"].as_str().unwrap_or("").to_string();
        let metadata: HashMap<String, Value> =
            serde_json::from_value(src["metadata"].clone()).unwrap_or_default();
        Ok(Document { id: doc_id.to_string(), text, embedding: None, metadata })
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        let body = json!({ "query": { "match_all": {} }, "size": 10000, "_source": ["text", "metadata"] });
        let resp = self.add_auth(self.client.post(&self.url("/_search"))).json(&body).send().await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        let data: Value = resp.json().await
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;
        let empty = vec![];
        Ok(data["hits"]["hits"].as_array().unwrap_or(&empty).iter().filter_map(|hit| {
            let id   = hit["_id"].as_str()?.to_string();
            let src  = &hit["_source"];
            let text = src["text"].as_str().unwrap_or("").to_string();
            let metadata: HashMap<String, Value> =
                serde_json::from_value(src["metadata"].clone()).unwrap_or_default();
            Some(Document { id, text, embedding: None, metadata })
        }).collect())
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let resp = self.add_auth(self.client.post(&self.url("/_delete_by_query")))
            .json(&json!({ "query": { "match_all": {} } })).send().await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!("OpenSearch clear failed: {text}")));
        }
        Ok(())
    }
}
