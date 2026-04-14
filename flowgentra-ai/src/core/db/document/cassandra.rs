//! Cassandra / Astra DB document store via the Stargate REST API.
//!
//! Uses Stargate's Document API (`/v2/namespaces/{keyspace}/collections/{collection}`)
//! so no native Cassandra driver is needed — just `reqwest`.
//!
//! Works with:
//! - DataStax Astra DB (managed Cassandra) — use the Astra REST URL and token
//! - Self-hosted Cassandra with Stargate sidecar
//!
//! Enabled with the `cassandra-store` Cargo feature.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::db::document::cassandra::{CassandraDocumentStore, CassandraConfig};
//! use flowgentra_ai::core::db::document::DocumentStore;
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = CassandraDocumentStore::new(CassandraConfig {
//!         endpoint:  "https://<id>-<region>.apps.astra.datastax.com".into(),
//!         keyspace:  "default_keyspace".into(),
//!         token:     std::env::var("ASTRA_TOKEN")?,
//!     });
//!     let id = store.insert("users", json!({"name": "Alice"})).await?;
//!     let docs = store.find("users", json!({"name": {"$eq": "Alice"}})).await?;
//!     store.delete("users", &id).await?;
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use super::super::DbError;
use super::DocumentStore;

/// Configuration for [`CassandraDocumentStore`].
#[derive(Debug, Clone)]
pub struct CassandraConfig {
    /// Stargate REST API base URL (no trailing slash).
    ///
    /// Astra: `https://<database-id>-<region>.apps.astra.datastax.com`
    /// Self-hosted: `http://localhost:8082`
    pub endpoint: String,
    /// Cassandra keyspace (Astra: `default_keyspace`)
    pub keyspace: String,
    /// Authentication token (Astra token or Stargate `X-Cassandra-Token`)
    pub token: String,
}

/// Cassandra document store using the Stargate Document REST API.
pub struct CassandraDocumentStore {
    client: reqwest::Client,
    config: CassandraConfig,
}

impl CassandraDocumentStore {
    pub fn new(config: CassandraConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }

    fn collection_url(&self, collection: &str) -> String {
        format!(
            "{}/v2/namespaces/{}/collections/{}",
            self.config.endpoint.trim_end_matches('/'),
            self.config.keyspace,
            collection
        )
    }

    fn document_url(&self, collection: &str, id: &str) -> String {
        format!("{}/{}", self.collection_url(collection), id)
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req.header("X-Cassandra-Token", &self.config.token)
    }
}

#[async_trait]
impl DocumentStore for CassandraDocumentStore {
    /// Insert a document into the Stargate collection.
    ///
    /// If the document has an `"id"` field, that value is used as the
    /// Stargate document ID; otherwise a UUID v4 is generated.
    async fn insert(&self, collection: &str, doc: Value) -> Result<String, DbError> {
        let id = doc
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let url = self.document_url(collection, &id);
        let resp = self
            .auth(self.client.put(&url))
            .json(&doc)
            .send()
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(DbError::Query(format!("Cassandra insert error: {text}")));
        }
        Ok(id)
    }

    /// Search the collection using a Stargate `where` filter.
    ///
    /// `filter` must be a Stargate Document API where-clause JSON, e.g.:
    /// `{"name": {"$eq": "Alice"}}` or `{"age": {"$gt": 25}}`.
    ///
    /// Returns up to 20 documents per page (Stargate default). For pagination
    /// you'll need to use the Stargate API directly.
    async fn find(&self, collection: &str, filter: Value) -> Result<Vec<Value>, DbError> {
        let where_str =
            serde_json::to_string(&filter).map_err(|e| DbError::Serialization(e.to_string()))?;

        let url = self.collection_url(collection);
        let resp = self
            .auth(self.client.get(&url))
            .query(&[("where", where_str.as_str())])
            .send()
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(DbError::Query(format!("Cassandra find error: {text}")));
        }

        let data: Value = resp
            .json()
            .await
            .map_err(|e| DbError::Serialization(e.to_string()))?;

        // Stargate returns {"data": {"<id>": {...}, ...}}
        let docs = data["data"]
            .as_object()
            .map(|map| map.values().map(|v| v.clone()).collect::<Vec<_>>())
            .unwrap_or_default();

        Ok(docs)
    }

    /// Delete the document identified by `id` from the collection.
    async fn delete(&self, collection: &str, id: &str) -> Result<(), DbError> {
        let url = self.document_url(collection, id);
        let resp = self
            .auth(self.client.delete(&url))
            .send()
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(DbError::Query(format!("Cassandra delete error: {text}")));
        }
        Ok(())
    }
}
