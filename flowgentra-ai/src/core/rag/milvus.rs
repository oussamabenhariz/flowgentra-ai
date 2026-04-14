//! Milvus Vector Store Backend
//!
//! Implements the `VectorStoreBackend` trait for Milvus via its RESTful API v2
//! (available in Milvus 2.3+).
//!
//! ## Configuration
//! - `endpoint`   — Milvus REST endpoint (e.g. `http://localhost:19530`)
//! - `index_name` — Collection name
//! - `api_key`    — optional token; sent as `Authorization: Bearer <token>`
//! - `embedding_dim` — vector dimension (required to create the collection schema)
//!
//! ## Collection schema
//! The adapter ensures a collection with three fields:
//! | Milvus field   | Type       | Role                    |
//! |----------------|------------|-------------------------|
//! | `id`           | VarChar PK | original document ID    |
//! | `text`         | VarChar    | document text           |
//! | `metadata_json`| VarChar    | JSON-encoded metadata   |
//! | `vector`       | FloatVector| embedding               |
//!
//! A flat IVF_FLAT index is created on `vector` so the collection is load-ready
//! out of the box.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use super::filter::FilterExpr;
use super::vector_db::{
    Document, MetadataFilter, RAGConfig, SearchResult, VectorStoreBackend, VectorStoreError,
    VectorStoreType,
};

// ─── MilvusStore ──────────────────────────────────────────────────────────────

/// Milvus vector store backend using the Milvus RESTful API v2.
pub struct MilvusStore {
    config: RAGConfig,
    client: reqwest::Client,
}

impl MilvusStore {
    /// Create a new `MilvusStore` and ensure the collection exists.
    pub async fn new(config: RAGConfig) -> Result<Self, VectorStoreError> {
        if config.store_type != VectorStoreType::Milvus {
            return Err(VectorStoreError::ConfigError(
                "Config type must be Milvus".to_string(),
            ));
        }
        if config.endpoint.is_none() {
            return Err(VectorStoreError::ConfigError(
                "Milvus requires an endpoint (e.g. http://localhost:19530)".to_string(),
            ));
        }

        let store = Self {
            config,
            client: reqwest::Client::new(),
        };

        store.ensure_collection().await?;
        Ok(store)
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    fn base_url(&self) -> &str {
        self.config
            .endpoint
            .as_deref()
            .unwrap_or("http://localhost:19530")
    }

    fn collection(&self) -> &str {
        &self.config.index_name
    }

    fn bearer_header(&self) -> Option<String> {
        self.config
            .api_key
            .as_deref()
            .map(|k| format!("Bearer {k}"))
    }

    /// POST to a Milvus REST v2 path, check success, return parsed JSON.
    async fn post(&self, path: &str, body: &Value, ctx: &str) -> Result<Value, VectorStoreError> {
        let url = format!("{}{path}", self.base_url());
        let mut req = self.client.post(&url).json(body);
        if let Some(ref auth) = self.bearer_header() {
            req = req.header("Authorization", auth);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(format!("{ctx}: {e}")))?;

        let status = resp.status();
        let data: Value = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::SerializationError(format!("{ctx} parse: {e}")))?;

        // Milvus REST API embeds its own error code in the JSON body even on 200.
        if let Some(code) = data.get("code").and_then(|c| c.as_i64()) {
            if code != 0 && code != 200 {
                let msg = data
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown");
                return Err(VectorStoreError::ApiError(format!(
                    "{ctx} failed (code={code}): {msg}"
                )));
            }
        } else if !status.is_success() {
            return Err(VectorStoreError::ApiError(format!(
                "{ctx} failed ({status}): {data}"
            )));
        }

        Ok(data)
    }

    /// Ensure the collection (and its index) exists; create if absent.
    async fn ensure_collection(&self) -> Result<(), VectorStoreError> {
        // Check if collection already exists.
        let check_body = serde_json::json!({
            "collectionName": self.collection()
        });
        let resp = self
            .post(
                "/v2/vectordb/collections/has",
                &check_body,
                "Milvus has-collection",
            )
            .await;

        let exists = resp
            .ok()
            .and_then(|v| v.get("data").and_then(|d| d.get("has")).and_then(|h| h.as_bool()))
            .unwrap_or(false);

        if exists {
            // Ensure it's loaded into memory (no-op if already loaded).
            let _ = self
                .post(
                    "/v2/vectordb/collections/load",
                    &serde_json::json!({ "collectionName": self.collection() }),
                    "Milvus load",
                )
                .await;
            return Ok(());
        }

        // Create collection with our fixed schema.
        let dim = self.config.embedding_dim.to_string();
        let create_body = serde_json::json!({
            "collectionName": self.collection(),
            "schema": {
                "autoId": false,
                "enableDynamicField": false,
                "fields": [
                    {
                        "fieldName": "id",
                        "dataType": "VarChar",
                        "isPrimary": true,
                        "params": { "max_length": "512" }
                    },
                    {
                        "fieldName": "text",
                        "dataType": "VarChar",
                        "params": { "max_length": "65535" }
                    },
                    {
                        "fieldName": "metadata_json",
                        "dataType": "VarChar",
                        "params": { "max_length": "65535" }
                    },
                    {
                        "fieldName": "vector",
                        "dataType": "FloatVector",
                        "params": { "dim": dim }
                    }
                ]
            },
            "indexParams": [
                {
                    "fieldName": "vector",
                    "metricType": "COSINE",
                    "indexType": "IVF_FLAT",
                    "params": { "nlist": "128" }
                }
            ]
        });

        self.post(
            "/v2/vectordb/collections/create",
            &create_body,
            "Milvus create collection",
        )
        .await?;

        // Load the new collection.
        self.post(
            "/v2/vectordb/collections/load",
            &serde_json::json!({ "collectionName": self.collection() }),
            "Milvus load",
        )
        .await?;

        Ok(())
    }

    /// Serialize metadata to a JSON string (stored in the `metadata_json` field).
    fn encode_metadata(metadata: &std::collections::HashMap<String, Value>) -> String {
        serde_json::to_string(metadata).unwrap_or_else(|_| "{}".to_string())
    }

    /// Parse the `metadata_json` field back into a HashMap.
    fn decode_metadata(s: &str) -> HashMap<String, Value> {
        serde_json::from_str(s).unwrap_or_default()
    }

    /// Convert a `FilterExpr` into a Milvus scalar filter expression string.
    ///
    /// Milvus format: `field == "value"`, `field > 5`, `(a) && (b)`, etc.
    fn build_filter(f: &FilterExpr) -> String {
        match f {
            FilterExpr::Eq(k, v)  => format!("{k} == {}", Self::milvus_val(v)),
            FilterExpr::Ne(k, v)  => format!("{k} != {}", Self::milvus_val(v)),
            FilterExpr::Gt(k, v)  => format!("{k} > {}",  Self::milvus_val(v)),
            FilterExpr::Lt(k, v)  => format!("{k} < {}",  Self::milvus_val(v)),
            FilterExpr::Gte(k, v) => format!("{k} >= {}", Self::milvus_val(v)),
            FilterExpr::Lte(k, v) => format!("{k} <= {}", Self::milvus_val(v)),
            FilterExpr::In(k, vs) => {
                let vals: Vec<String> = vs.iter().map(Self::milvus_val).collect();
                format!("{k} in [{}]", vals.join(", "))
            }
            FilterExpr::And(exprs) => exprs
                .iter()
                .map(|e| format!("({})", Self::build_filter(e)))
                .collect::<Vec<_>>()
                .join(" && "),
            FilterExpr::Or(exprs) => exprs
                .iter()
                .map(|e| format!("({})", Self::build_filter(e)))
                .collect::<Vec<_>>()
                .join(" || "),
        }
    }

    fn milvus_val(v: &Value) -> String {
        match v {
            Value::String(s) => format!("\"{s}\""),
            Value::Number(n) => n.to_string(),
            Value::Bool(b)   => b.to_string(),
            _                => format!("\"{v}\""),
        }
    }
}

// ─── VectorStoreBackend impl ──────────────────────────────────────────────────

#[async_trait]
impl VectorStoreBackend for MilvusStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc.embedding.ok_or_else(|| {
            VectorStoreError::EmbeddingError(
                "Document must have an embedding to index into Milvus".to_string(),
            )
        })?;

        let body = serde_json::json!({
            "collectionName": self.collection(),
            "data": [{
                "id":            doc.id,
                "text":          doc.text,
                "metadata_json": Self::encode_metadata(&doc.metadata),
                "vector":        embedding,
            }]
        });

        self.post("/v2/vectordb/entities/insert", &body, "Milvus insert")
            .await?;
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut body = serde_json::json!({
            "collectionName": self.collection(),
            "data": [query_embedding],
            "limit": top_k,
            "outputFields": ["id", "text", "metadata_json"],
            "annsField": "vector",
            "searchParams": { "metricType": "COSINE" }
        });

        if let Some(ref f) = filter {
            body["filter"] = Value::String(Self::build_filter(f));
        }

        let resp = self
            .post("/v2/vectordb/entities/search", &body, "Milvus search")
            .await?;

        let empty_vec = Vec::new();
        let hits = resp
            .get("data")
            .and_then(|d| d.as_array())
            .unwrap_or(&empty_vec);

        let results = hits
            .iter()
            .filter_map(|hit| {
                let id = hit.get("id")?.as_str()?.to_string();
                let text = hit
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let metadata = hit
                    .get("metadata_json")
                    .and_then(|v| v.as_str())
                    .map(|s| Self::decode_metadata(s))
                    .unwrap_or_default();
                // Milvus COSINE metric returns similarity directly (1 = identical).
                let score = hit
                    .get("distance")
                    .and_then(|s| s.as_f64())
                    .unwrap_or(0.0) as f32;
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
        let body = serde_json::json!({
            "collectionName": self.collection(),
            "filter": format!("id == \"{doc_id}\""),
        });
        self.post("/v2/vectordb/entities/delete", &body, "Milvus delete")
            .await?;
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        // Milvus upsert: insert with upsert flag, or delete + insert.
        // Use the upsert endpoint (available from Milvus 2.3.3+).
        let embedding = doc.embedding.ok_or_else(|| {
            VectorStoreError::EmbeddingError(
                "Document must have an embedding to update in Milvus".to_string(),
            )
        })?;

        let body = serde_json::json!({
            "collectionName": self.collection(),
            "data": [{
                "id":            doc.id,
                "text":          doc.text,
                "metadata_json": Self::encode_metadata(&doc.metadata),
                "vector":        embedding,
            }]
        });

        self.post("/v2/vectordb/entities/upsert", &body, "Milvus upsert")
            .await?;
        Ok(())
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let body = serde_json::json!({
            "collectionName": self.collection(),
            "id": [doc_id],
            "outputFields": ["id", "text", "metadata_json"],
        });

        let resp = self
            .post("/v2/vectordb/entities/get", &body, "Milvus get")
            .await?;

        let entity = resp
            .get("data")
            .and_then(|d| d.as_array())
            .and_then(|a| a.first())
            .ok_or_else(|| VectorStoreError::NotFound(doc_id.to_string()))?;

        let id = entity
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or(doc_id)
            .to_string();
        let text = entity
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let metadata = entity
            .get("metadata_json")
            .and_then(|v| v.as_str())
            .map(|s| Self::decode_metadata(s))
            .unwrap_or_default();

        Ok(Document {
            id,
            text,
            embedding: None, // Milvus doesn't return vectors in get by default
            metadata,
        })
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        // Use a query with no filter to retrieve all entities (up to a large limit).
        let body = serde_json::json!({
            "collectionName": self.collection(),
            "filter": "",
            "limit": 16384,
            "outputFields": ["id", "text", "metadata_json"],
        });

        let resp = self
            .post("/v2/vectordb/entities/query", &body, "Milvus list")
            .await?;

        let empty_vec = Vec::new();
        let entities = resp
            .get("data")
            .and_then(|d| d.as_array())
            .unwrap_or(&empty_vec);

        let docs = entities
            .iter()
            .map(|e| {
                let id = e
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let text = e
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let metadata = e
                    .get("metadata_json")
                    .and_then(|v| v.as_str())
                    .map(|s| Self::decode_metadata(s))
                    .unwrap_or_default();
                Document {
                    id,
                    text,
                    embedding: None,
                    metadata,
                }
            })
            .collect();

        Ok(docs)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        // Drop and recreate the collection.
        let drop_body = serde_json::json!({ "collectionName": self.collection() });
        let _ = self
            .post(
                "/v2/vectordb/collections/drop",
                &drop_body,
                "Milvus drop collection",
            )
            .await;

        self.ensure_collection().await
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_filter_eq() {
        let f = FilterExpr::eq("source", Value::String("pdf".to_string()));
        let expr = MilvusStore::build_filter(&f);
        assert_eq!(expr, r#"source == "pdf""#);
    }

    #[test]
    fn test_build_filter_gt() {
        let f = FilterExpr::gt("page", Value::Number(serde_json::Number::from(3)));
        let expr = MilvusStore::build_filter(&f);
        assert_eq!(expr, "page > 3");
    }

    #[test]
    fn test_build_filter_and() {
        let f = FilterExpr::and(vec![
            FilterExpr::eq("source", Value::String("pdf".to_string())),
            FilterExpr::gt("page", Value::Number(serde_json::Number::from(3))),
        ]);
        let expr = MilvusStore::build_filter(&f);
        assert!(expr.contains(r#"source == "pdf""#));
        assert!(expr.contains("page > 3"));
        assert!(expr.contains(" && "));
    }

    #[test]
    fn test_build_filter_in() {
        let f = FilterExpr::in_values(
            "category",
            vec![Value::String("a".into()), Value::String("b".into())],
        );
        let expr = MilvusStore::build_filter(&f);
        assert_eq!(expr, r#"category in ["a", "b"]"#);
    }

    #[test]
    fn test_encode_decode_metadata() {
        let mut meta = HashMap::new();
        meta.insert("k".to_string(), Value::String("v".to_string()));
        let encoded = MilvusStore::encode_metadata(&meta);
        let decoded = MilvusStore::decode_metadata(&encoded);
        assert_eq!(decoded.get("k").and_then(|v| v.as_str()), Some("v"));
    }
}
