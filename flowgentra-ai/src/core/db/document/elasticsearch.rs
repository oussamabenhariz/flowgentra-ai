//! Elasticsearch document store backend for [`DocumentStore`].
//!
//! Uses the Elasticsearch REST API — no extra crate needed beyond `reqwest`.
//! Index name maps directly to `collection`.
//!
//! Enabled with the `elasticsearch-store` Cargo feature.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::db::document::elasticsearch::{
//!     ElasticsearchDocumentStore, ElasticsearchDocConfig,
//! };
//! use flowgentra_ai::core::db::document::DocumentStore;
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = ElasticsearchDocumentStore::new(ElasticsearchDocConfig {
//!         endpoint: "http://localhost:9200".into(),
//!         api_key:  None,
//!     });
//!     let id = store.insert("my_index", json!({"title": "Hello", "body": "World"})).await?;
//!     let docs = store.find("my_index", json!({"query": {"match": {"title": "Hello"}}})).await?;
//!     store.delete("my_index", &id).await?;
//!     Ok(())
//! }
//! ```
//!
//! ## `find` filter format
//!
//! The `filter` argument is passed **directly** as the Elasticsearch query body,
//! so you have full access to the query DSL:
//!
//! ```json
//! {"query": {"match_all": {}}}
//! {"query": {"term": {"status": "published"}}}
//! {"query": {"range": {"age": {"gte": 18}}}}
//! ```

use async_trait::async_trait;
use serde_json::{json, Value};
use uuid::Uuid;

use super::super::DbError;
use super::DocumentStore;

/// Configuration for [`ElasticsearchDocumentStore`].
#[derive(Debug, Clone)]
pub struct ElasticsearchDocConfig {
    /// Elasticsearch endpoint (e.g. `http://localhost:9200` or Elastic Cloud URL).
    pub endpoint: String,
    /// Optional API key for authentication (`Authorization: ApiKey <key>`).
    pub api_key: Option<String>,
}

/// Elasticsearch document store using the Elasticsearch REST API.
pub struct ElasticsearchDocumentStore {
    client: reqwest::Client,
    config: ElasticsearchDocConfig,
}

impl ElasticsearchDocumentStore {
    pub fn new(config: ElasticsearchDocConfig) -> Self {
        Self { client: reqwest::Client::new(), config }
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref key) = self.config.api_key {
            req.header("Authorization", format!("ApiKey {key}"))
        } else {
            req
        }
    }

    fn doc_url(&self, index: &str, id: &str) -> String {
        format!(
            "{}/{}/_doc/{}",
            self.config.endpoint.trim_end_matches('/'),
            index,
            id
        )
    }

    fn search_url(&self, index: &str) -> String {
        format!(
            "{}/{}/_search",
            self.config.endpoint.trim_end_matches('/'),
            index
        )
    }
}

#[async_trait]
impl DocumentStore for ElasticsearchDocumentStore {
    /// Index a document in Elasticsearch.
    ///
    /// If the document contains an `"id"` field, that is used as the ES `_id`;
    /// otherwise a UUID v4 is generated. The document is stored with `PUT` so
    /// it's idempotent — re-indexing the same ID overwrites the existing doc.
    async fn insert(&self, collection: &str, doc: Value) -> Result<String, DbError> {
        let id = doc.get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let url = self.doc_url(collection, &id);
        let resp = self
            .auth(self.client.put(&url))
            .header("Content-Type", "application/json")
            .json(&doc)
            .send()
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(DbError::Query(format!("Elasticsearch index error: {text}")));
        }
        Ok(id)
    }

    /// Search documents in the index using the full Elasticsearch query DSL.
    ///
    /// `filter` is the request body sent to `/{index}/_search`.
    /// Pass `{}` or `{"query": {"match_all": {}}}` to return all documents.
    ///
    /// Returns up to 10 documents by default (Elasticsearch default `size`).
    /// Add `"size": N` to your filter to change this.
    async fn find(&self, collection: &str, filter: Value) -> Result<Vec<Value>, DbError> {
        // Wrap bare filter objects without a "query" key into match_all + filter.
        let body = if filter.get("query").is_some() || filter.as_object().map_or(true, |o| o.is_empty()) {
            if filter.as_object().map_or(false, |o| o.is_empty()) {
                json!({"query": {"match_all": {}}})
            } else {
                filter
            }
        } else {
            // Treat top-level keys as term filters for convenience.
            let terms: Vec<Value> = filter
                .as_object()
                .map(|obj| {
                    obj.iter()
                        .map(|(k, v)| json!({"term": {k: v}}))
                        .collect()
                })
                .unwrap_or_default();
            json!({"query": {"bool": {"must": terms}}})
        };

        let url = self.search_url(collection);
        let resp = self
            .auth(self.client.post(&url))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(DbError::Query(format!("Elasticsearch search error: {text}")));
        }

        let data: Value = resp.json().await
            .map_err(|e| DbError::Serialization(e.to_string()))?;

        let hits = data["hits"]["hits"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let docs = hits
            .into_iter()
            .filter_map(|h| {
                let mut source = h["_source"].clone();
                // Inject the ES _id back into the document.
                if let (Some(id), Some(obj)) = (h["_id"].as_str(), source.as_object_mut()) {
                    obj.entry("id").or_insert_with(|| Value::String(id.to_string()));
                }
                if source.is_null() { None } else { Some(source) }
            })
            .collect();

        Ok(docs)
    }

    /// Delete the document with `_id = id` from the index.
    async fn delete(&self, collection: &str, id: &str) -> Result<(), DbError> {
        let url = self.doc_url(collection, id);
        let resp = self
            .auth(self.client.delete(&url))
            .send()
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        // 404 is acceptable — document was already gone.
        if !resp.status().is_success() && resp.status() != reqwest::StatusCode::NOT_FOUND {
            let text = resp.text().await.unwrap_or_default();
            return Err(DbError::Query(format!("Elasticsearch delete error: {text}")));
        }
        Ok(())
    }
}
