//! DataStax Astra DB Vector Store — uses the Astra DB JSON API (Data API).
//!
//! Astra DB is managed Cassandra with a vector-capable JSON API.
//! Pure HTTP backend using the Data API v1.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{AstraDbConfig, AstraDbStore, VectorStoreBackend};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = AstraDbStore::new(AstraDbConfig {
//!         endpoint:   "https://<db-id>-<region>.apps.astra.datastax.com".into(),
//!         token:      "AstraCS:...".into(),
//!         keyspace:   "default_keyspace".into(),
//!         collection: "documents".into(),
//!         embedding_dim: 1536,
//!     }).await?;
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;

use super::vector_db::{Document, MetadataFilter, SearchResult, VectorStoreBackend, VectorStoreError};

/// Configuration for [`AstraDbStore`].
#[derive(Debug, Clone)]
pub struct AstraDbConfig {
    /// Astra DB endpoint URL
    pub endpoint: String,
    /// Astra DB application token (`AstraCS:...`)
    pub token: String,
    /// Keyspace name
    pub keyspace: String,
    /// Collection name
    pub collection: String,
    /// Embedding dimension
    pub embedding_dim: usize,
}

/// Astra DB vector store using the Data API.
pub struct AstraDbStore {
    client: reqwest::Client,
    config: AstraDbConfig,
}

impl AstraDbStore {
    /// Connect and ensure the collection exists with vector support.
    pub async fn new(config: AstraDbConfig) -> Result<Self, VectorStoreError> {
        let client = reqwest::Client::new();
        let store = Self { client, config };
        store.ensure_collection().await?;
        Ok(store)
    }

    fn collection_url(&self) -> String {
        format!(
            "{}/api/json/v1/{}/{}",
            self.config.endpoint.trim_end_matches('/'),
            self.config.keyspace,
            self.config.collection,
        )
    }

    fn headers(&self) -> [(&'static str, String); 2] {
        [
            ("Token", self.config.token.clone()),
            ("Content-Type", "application/json".to_string()),
        ]
    }

    async fn post_command(&self, body: &Value, ctx: &str) -> Result<Value, VectorStoreError> {
        let mut req = self.client.post(&self.collection_url()).json(body);
        for (k, v) in self.headers() {
            req = req.header(k, v);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(format!("{ctx}: {e}")))?;
        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!("{ctx} failed: {text}")));
        }
        let data: Value = resp.json().await
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;
        // Check for embedded errors
        if let Some(errors) = data.get("errors").and_then(|e| e.as_array()) {
            if !errors.is_empty() {
                return Err(VectorStoreError::ApiError(format!("{ctx}: {:?}", errors)));
            }
        }
        Ok(data)
    }

    async fn ensure_collection(&self) -> Result<(), VectorStoreError> {
        // Try to create collection with vector enabled; ignore "already exists"
        let create_url = format!(
            "{}/api/json/v1/{}",
            self.config.endpoint.trim_end_matches('/'),
            self.config.keyspace,
        );
        let body = json!({
            "createCollection": {
                "name": self.config.collection,
                "options": {
                    "vector": {
                        "dimension": self.config.embedding_dim,
                        "metric":    "cosine"
                    }
                }
            }
        });
        let mut req = self.client.post(&create_url).json(&body);
        for (k, v) in self.headers() {
            req = req.header(k, v);
        }
        let resp = req.send().await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        let data: Value = resp.json().await.unwrap_or(json!({}));
        if let Some(errors) = data.get("errors").and_then(|e| e.as_array()) {
            for err in errors {
                let msg = err["message"].as_str().unwrap_or("");
                if !msg.contains("already exists") {
                    return Err(VectorStoreError::ApiError(msg.to_string()));
                }
            }
        }
        Ok(())
    }

    /// Build an Astra DB filter object from a `FilterExpr`.
    fn build_filter(f: &MetadataFilter) -> Value {
        use super::filter::FilterExpr as F;
        match f {
            F::Eq(k, v)  => json!({ k: { "$eq":  v } }),
            F::Ne(k, v)  => json!({ k: { "$ne":  v } }),
            F::Gt(k, v)  => json!({ k: { "$gt":  v } }),
            F::Lt(k, v)  => json!({ k: { "$lt":  v } }),
            F::Gte(k, v) => json!({ k: { "$gte": v } }),
            F::Lte(k, v) => json!({ k: { "$lte": v } }),
            F::In(k, vs) => json!({ k: { "$in":  vs } }),
            F::And(exprs) => json!({ "$and": exprs.iter().map(Self::build_filter).collect::<Vec<_>>() }),
            F::Or(exprs)  => json!({ "$or":  exprs.iter().map(Self::build_filter).collect::<Vec<_>>() }),
        }
    }
}

#[async_trait]
impl VectorStoreBackend for AstraDbStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc.embedding.ok_or_else(|| {
            VectorStoreError::EmbeddingError("Document must have an embedding".into())
        })?;
        let mut document = doc.metadata.clone();
        document.insert("_id".to_string(), Value::String(doc.id.clone()));
        document.insert("text".to_string(), Value::String(doc.text.clone()));
        document.insert("$vector".to_string(), json!(embedding));

        self.post_command(
            &json!({ "insertOne": { "document": document } }),
            "Astra insertOne",
        ).await?;
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut find = json!({
            "sort":    { "$vector": query_embedding },
            "options": { "limit": top_k, "includeSimilarity": true },
            "projection": { "text": 1, "$vector": 0 }
        });
        if let Some(f) = filter {
            find["filter"] = Self::build_filter(&f);
        }

        let data = self.post_command(&json!({ "find": find }), "Astra find").await?;
        let empty = vec![];
        let docs = data["data"]["documents"].as_array().unwrap_or(&empty);
        let results = docs.iter().filter_map(|d| {
            let id    = d["_id"].as_str()?.to_string();
            let text  = d["text"].as_str().unwrap_or("").to_string();
            let score = d["$similarity"].as_f64().unwrap_or(0.0) as f32;
            let metadata: HashMap<String, Value> = d.as_object()?
                .iter()
                .filter(|(k, _)| !matches!(k.as_str(), "_id" | "text" | "$similarity" | "$vector"))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Some(SearchResult { id, text, score, metadata })
        }).collect();
        Ok(results)
    }

    async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError> {
        self.post_command(
            &json!({ "deleteOne": { "filter": { "_id": doc_id } } }),
            "Astra deleteOne",
        ).await?;
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.delete(&doc.id).await?;
        self.index(doc).await
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let data = self.post_command(
            &json!({ "findOne": { "filter": { "_id": doc_id } } }),
            "Astra findOne",
        ).await?;
        let d = data["data"]["document"]
            .as_object()
            .ok_or_else(|| VectorStoreError::NotFound(doc_id.to_string()))?;
        let text = d.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let metadata: HashMap<String, Value> = d.iter()
            .filter(|(k, _)| !matches!(k.as_str(), "_id" | "text" | "$vector"))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        Ok(Document { id: doc_id.to_string(), text, embedding: None, metadata })
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        let data = self.post_command(
            &json!({ "find": { "options": { "limit": 1000 }, "projection": { "text": 1, "$vector": 0 } } }),
            "Astra find all",
        ).await?;
        let empty = vec![];
        Ok(data["data"]["documents"].as_array().unwrap_or(&empty).iter().filter_map(|d| {
            let id   = d["_id"].as_str()?.to_string();
            let text = d["text"].as_str().unwrap_or("").to_string();
            let metadata: HashMap<String, Value> = d.as_object()?
                .iter()
                .filter(|(k, _)| !matches!(k.as_str(), "_id" | "text" | "$vector"))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Some(Document { id, text, embedding: None, metadata })
        }).collect())
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        self.post_command(
            &json!({ "deleteMany": { "filter": {} } }),
            "Astra deleteMany",
        ).await?;
        Ok(())
    }
}
