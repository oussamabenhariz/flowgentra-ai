//! Upstash Vector Store — serverless vector database via REST API.
//!
//! Pure HTTP backend, no extra deps. Upstash Vector is a serverless managed
//! vector database with a simple REST API.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{UpstashVectorConfig, UpstashVectorStore, VectorStoreBackend};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = UpstashVectorStore::new(UpstashVectorConfig {
//!         url:   "https://<endpoint>.upstash.io".into(),
//!         token: "your-token".into(),
//!     });
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

/// Configuration for [`UpstashVectorStore`].
#[derive(Debug, Clone)]
pub struct UpstashVectorConfig {
    /// Upstash Vector REST URL (from your Upstash console)
    pub url: String,
    /// Upstash Vector REST token
    pub token: String,
}

/// Upstash serverless vector store.
pub struct UpstashVectorStore {
    client: reqwest::Client,
    config: UpstashVectorConfig,
}

impl UpstashVectorStore {
    pub fn new(config: UpstashVectorConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.config.url.trim_end_matches('/'), path)
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req.header("Authorization", format!("Bearer {}", self.config.token))
    }

    async fn post<T: serde::Serialize>(
        &self,
        path: &str,
        body: &T,
        ctx: &str,
    ) -> Result<Value, VectorStoreError> {
        let resp = self
            .auth(self.client.post(self.url(path)))
            .json(body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(format!("{ctx}: {e}")))?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!("{ctx} failed: {text}")));
        }
        resp.json()
            .await
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))
    }
}

#[async_trait]
impl VectorStoreBackend for UpstashVectorStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc.embedding.ok_or_else(|| {
            VectorStoreError::EmbeddingError("Document must have an embedding".into())
        })?;
        let mut metadata = doc.metadata.clone();
        metadata.insert("__text__".to_string(), Value::String(doc.text.clone()));

        let body = json!([{
            "id":       doc.id,
            "vector":   embedding,
            "metadata": metadata,
        }]);
        self.post("/upsert", &body, "Upstash upsert").await?;
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut body = json!({
            "vector":          query_embedding,
            "topK":            top_k,
            "includeMetadata": true,
            "includeVectors":  false,
        });

        // Upstash filter syntax: simple key=value as string "field = 'value'"
        if let Some(f) = filter {
            body["filter"] = Value::String(upstash_filter(&f));
        }

        let data = self.post("/query", &body, "Upstash query").await?;
        let empty = vec![];
        let results = data["result"]
            .as_array()
            .unwrap_or(&empty)
            .iter()
            .filter_map(|item| {
                let id = item["id"].as_str()?.to_string();
                let score = item["score"].as_f64()? as f32;
                let meta = item["metadata"].as_object().cloned().unwrap_or_default();
                let text = meta
                    .get("__text__")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let metadata: HashMap<String, Value> =
                    meta.into_iter().filter(|(k, _)| k != "__text__").collect();
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
        self.post("/delete", &json!({ "ids": [doc_id] }), "Upstash delete")
            .await?;
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.index(doc).await
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let data = self
            .post(
                "/fetch",
                &json!({ "ids": [doc_id], "includeMetadata": true }),
                "Upstash fetch",
            )
            .await?;
        let empty = vec![];
        let item = data["result"]
            .as_array()
            .unwrap_or(&empty)
            .first()
            .and_then(|v| if v.is_null() { None } else { Some(v) })
            .ok_or_else(|| VectorStoreError::NotFound(doc_id.to_string()))?;

        let meta = item["metadata"].as_object().cloned().unwrap_or_default();
        let text = meta
            .get("__text__")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let metadata: HashMap<String, Value> =
            meta.into_iter().filter(|(k, _)| k != "__text__").collect();
        Ok(Document {
            id: doc_id.to_string(),
            text,
            embedding: None,
            metadata,
        })
    }

    fn supports_list(&self) -> bool {
        false
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        Err(VectorStoreError::NotImplemented(
            "Upstash Vector does not support listing all documents.".into(),
        ))
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        self.post("/reset", &json!({}), "Upstash reset").await?;
        Ok(())
    }
}

/// Convert a `FilterExpr` to an Upstash filter string.
/// Upstash uses SQL-like filter syntax: `field = 'value' AND field2 > 5`.
fn upstash_filter(f: &FilterExpr) -> String {
    match f {
        FilterExpr::Eq(k, v) => format!("{} = {}", k, upstash_val(v)),
        FilterExpr::Ne(k, v) => format!("{} != {}", k, upstash_val(v)),
        FilterExpr::Gt(k, v) => format!("{} > {}", k, upstash_val(v)),
        FilterExpr::Lt(k, v) => format!("{} < {}", k, upstash_val(v)),
        FilterExpr::Gte(k, v) => format!("{} >= {}", k, upstash_val(v)),
        FilterExpr::Lte(k, v) => format!("{} <= {}", k, upstash_val(v)),
        FilterExpr::In(k, vs) => {
            let vals: Vec<String> = vs.iter().map(upstash_val).collect();
            format!("{} IN ({})", k, vals.join(", "))
        }
        FilterExpr::And(exprs) => exprs
            .iter()
            .map(upstash_filter)
            .collect::<Vec<_>>()
            .join(" AND "),
        FilterExpr::Or(exprs) => {
            let parts: Vec<String> = exprs
                .iter()
                .map(|e| format!("({})", upstash_filter(e)))
                .collect();
            parts.join(" OR ")
        }
    }
}

fn upstash_val(v: &Value) -> String {
    match v {
        Value::String(s) => format!("'{}'", s.replace('\'', "\\'")),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => format!("'{}'", v),
    }
}
