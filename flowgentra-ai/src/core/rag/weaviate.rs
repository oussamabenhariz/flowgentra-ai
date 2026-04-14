//! Weaviate Vector Store Backend
//!
//! Implements the `VectorStoreBackend` trait for Weaviate via its REST v1 API.
//!
//! ## Configuration
//! - `endpoint`   — Weaviate base URL (e.g. `http://localhost:8080`)
//! - `index_name` — Weaviate *class* name (must begin with a capital letter; the
//!   adapter auto-capitalises if needed)
//! - `api_key`    — optional; sent as `Authorization: Bearer <key>`
//!
//! ## Storage layout
//! Each document is stored as a Weaviate object with three properties:
//! | Weaviate property | Content             |
//! |-------------------|---------------------|
//! | `doc_id`          | original doc ID     |
//! | `text`            | document text       |
//! | `metadata_json`   | JSON-encoded metadata |
//!
//! The Weaviate object UUID is derived deterministically from `doc_id` via UUID v5
//! (same strategy used by `QdrantStore`).

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

use super::filter::FilterExpr;
use super::vector_db::{
    Document, MetadataFilter, RAGConfig, SearchResult, VectorStoreBackend, VectorStoreError,
    VectorStoreType,
};

// ─── Internal response shapes ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct WeaviateObject {
    id: String,
    #[serde(default)]
    properties: Option<HashMap<String, Value>>,
    #[serde(default)]
    vector: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct WeaviateListResponse {
    objects: Option<Vec<WeaviateObject>>,
}

// ─── WeaviateStore ────────────────────────────────────────────────────────────

/// Weaviate vector store backend.
///
/// Connects to Weaviate via the REST v1 API and uses a GraphQL `nearVector` query
/// for semantic search. Metadata filtering is forwarded as a `where` clause.
pub struct WeaviateStore {
    config: RAGConfig,
    /// Weaviate class name (always starts with a capital letter)
    class_name: String,
    client: reqwest::Client,
}

impl WeaviateStore {
    /// Create a new `WeaviateStore` and ensure the target class exists.
    pub async fn new(config: RAGConfig) -> Result<Self, VectorStoreError> {
        if config.store_type != VectorStoreType::Weaviate {
            return Err(VectorStoreError::ConfigError(
                "Config type must be Weaviate".to_string(),
            ));
        }
        if config.endpoint.is_none() {
            return Err(VectorStoreError::ConfigError(
                "Weaviate requires an endpoint (e.g. http://localhost:8080)".to_string(),
            ));
        }

        let class_name = capitalise(&config.index_name);
        let store = Self {
            config,
            class_name,
            client: reqwest::Client::new(),
        };

        store.ensure_class().await?;
        Ok(store)
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    fn base_url(&self) -> &str {
        self.config
            .endpoint
            .as_deref()
            .unwrap_or("http://localhost:8080")
    }

    /// Build the `Authorization: Bearer …` header value, owned.
    fn bearer_header(&self) -> Option<String> {
        self.config
            .api_key
            .as_deref()
            .map(|k| format!("Bearer {k}"))
    }

    /// Attach optional bearer auth and POST JSON, returning parsed JSON.
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
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "{ctx} failed ({status}): {text}"
            )));
        }
        resp.json()
            .await
            .map_err(|e| VectorStoreError::SerializationError(format!("{ctx} parse: {e}")))
    }

    async fn post_ok(&self, path: &str, body: &Value, ctx: &str) -> Result<(), VectorStoreError> {
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
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "{ctx} failed ({status}): {text}"
            )));
        }
        Ok(())
    }

    /// GET, optional bearer, return `None` on 404.
    async fn get_opt(&self, path: &str, ctx: &str) -> Result<Option<Value>, VectorStoreError> {
        let url = format!("{}{path}", self.base_url());
        let mut req = self.client.get(&url);
        if let Some(ref auth) = self.bearer_header() {
            req = req.header("Authorization", auth);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(format!("{ctx}: {e}")))?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "{ctx} failed ({status}): {text}"
            )));
        }
        Ok(Some(resp.json().await.map_err(|e| {
            VectorStoreError::SerializationError(format!("{ctx} parse: {e}"))
        })?))
    }

    /// Derive a deterministic Weaviate UUID from a document ID string.
    fn point_uuid(doc_id: &str) -> String {
        uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, doc_id.as_bytes()).to_string()
    }

    /// Ensure the Weaviate class (collection) exists; create it if not.
    async fn ensure_class(&self) -> Result<(), VectorStoreError> {
        let path = format!("/v1/schema/{}", self.class_name);
        if self.get_opt(&path, "Weaviate check class").await?.is_some() {
            return Ok(());
        }

        // Create the class with `none` vectorizer (we supply our own vectors).
        let schema = serde_json::json!({
            "class": self.class_name,
            "vectorizer": "none",
            "properties": [
                {"name": "doc_id",        "dataType": ["text"]},
                {"name": "text",          "dataType": ["text"]},
                {"name": "metadata_json", "dataType": ["text"]}
            ]
        });

        self.post_ok("/v1/schema", &schema, "Weaviate create class")
            .await
    }

    /// Parse a Weaviate object's `properties` map into a `Document`.
    fn object_to_doc(obj: &WeaviateObject) -> Document {
        let props = obj.properties.clone().unwrap_or_default();
        let doc_id = props
            .get("doc_id")
            .and_then(|v| v.as_str())
            .unwrap_or(&obj.id)
            .to_string();
        let text = props
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let metadata: HashMap<String, Value> = props
            .get("metadata_json")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        Document {
            id: doc_id,
            text,
            embedding: obj.vector.clone(),
            metadata,
        }
    }

    /// Convert a `FilterExpr` into a Weaviate GraphQL `where` clause JSON.
    fn build_where_filter(f: &FilterExpr) -> Value {
        match f {
            FilterExpr::Eq(k, v) => weaviate_leaf("Equal", k, v),
            FilterExpr::Ne(k, v) => weaviate_leaf("NotEqual", k, v),
            FilterExpr::Gt(k, v) => weaviate_leaf("GreaterThan", k, v),
            FilterExpr::Lt(k, v) => weaviate_leaf("LessThan", k, v),
            FilterExpr::Gte(k, v) => weaviate_leaf("GreaterThanEqual", k, v),
            FilterExpr::Lte(k, v) => weaviate_leaf("LessThanEqual", k, v),
            // Weaviate has no native IN; expand as OR over equality checks.
            FilterExpr::In(k, vs) => serde_json::json!({
                "operator": "Or",
                "operands": vs.iter().map(|v| weaviate_leaf("Equal", k, v)).collect::<Vec<_>>()
            }),
            FilterExpr::And(exprs) => serde_json::json!({
                "operator": "And",
                "operands": exprs.iter().map(Self::build_where_filter).collect::<Vec<_>>()
            }),
            FilterExpr::Or(exprs) => serde_json::json!({
                "operator": "Or",
                "operands": exprs.iter().map(Self::build_where_filter).collect::<Vec<_>>()
            }),
        }
    }
}

// ─── VectorStoreBackend impl ──────────────────────────────────────────────────

#[async_trait]
impl VectorStoreBackend for WeaviateStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc.embedding.ok_or_else(|| {
            VectorStoreError::EmbeddingError(
                "Document must have an embedding to index into Weaviate".to_string(),
            )
        })?;

        let metadata_json =
            serde_json::to_string(&doc.metadata).unwrap_or_else(|_| "{}".to_string());

        let body = serde_json::json!({
            "class": self.class_name,
            "id": Self::point_uuid(&doc.id),
            "vector": embedding,
            "properties": {
                "doc_id": doc.id,
                "text":   doc.text,
                "metadata_json": metadata_json,
            }
        });

        self.post_ok("/v1/objects", &body, "Weaviate index").await
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        // Build where clause string for GraphQL (only if filter provided)
        let where_clause = filter
            .as_ref()
            .map(|f| {
                let w = Self::build_where_filter(f);
                format!(", where: {}", graphql_value(&w))
            })
            .unwrap_or_default();

        let gql = format!(
            r#"{{ Get {{ {class}(
                nearVector: {{vector: {vec}}}
                limit: {top_k}
                {where}
            ) {{
                doc_id text metadata_json
                _additional {{ id distance }}
            }} }} }}"#,
            class = self.class_name,
            vec = serde_json::to_string(&query_embedding)
                .unwrap_or_else(|_| "[]".to_string()),
            top_k = top_k,
            where = where_clause,
        );

        let body = serde_json::json!({ "query": gql });
        let resp = self.post("/v1/graphql", &body, "Weaviate search").await?;

        // Navigate: data.Get.<ClassName>[...]
        let empty_vec = Vec::new();
        let items = resp
            .get("data")
            .and_then(|d| d.get("Get"))
            .and_then(|g| g.get(&self.class_name))
            .and_then(|c| c.as_array())
            .unwrap_or(&empty_vec);

        let results = items
            .iter()
            .filter_map(|item| {
                let doc_id = item.get("doc_id")?.as_str()?.to_string();
                let text = item
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let metadata: HashMap<String, Value> = item
                    .get("metadata_json")
                    .and_then(|v| v.as_str())
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or_default();
                // Weaviate returns cosine distance (0 = identical); convert to similarity.
                let distance = item
                    .get("_additional")
                    .and_then(|a| a.get("distance"))
                    .and_then(|d| d.as_f64())
                    .unwrap_or(1.0) as f32;
                let score = 1.0 - distance;
                Some(SearchResult {
                    id: doc_id,
                    text,
                    score,
                    metadata,
                })
            })
            .collect();

        Ok(results)
    }

    async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError> {
        let uuid = Self::point_uuid(doc_id);
        let path = format!("/v1/objects/{}/{}", self.class_name, uuid);
        let url = format!("{}{path}", self.base_url());
        let mut req = self.client.delete(&url);
        if let Some(ref auth) = self.bearer_header() {
            req = req.header("Authorization", auth);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(format!("Weaviate delete: {e}")))?;
        let status = resp.status();
        if !status.is_success() && status != reqwest::StatusCode::NOT_FOUND {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Weaviate delete failed ({status}): {text}"
            )));
        }
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        // Weaviate upsert: delete then re-index (PUT /v1/objects/{class}/{id} replaces in full).
        let embedding = doc.embedding.clone().ok_or_else(|| {
            VectorStoreError::EmbeddingError(
                "Document must have an embedding to update in Weaviate".to_string(),
            )
        })?;

        let metadata_json =
            serde_json::to_string(&doc.metadata).unwrap_or_else(|_| "{}".to_string());
        let uuid = Self::point_uuid(&doc.id);
        let path = format!("/v1/objects/{}/{}", self.class_name, uuid);
        let url = format!("{}{path}", self.base_url());

        let body = serde_json::json!({
            "class": self.class_name,
            "id": uuid,
            "vector": embedding,
            "properties": {
                "doc_id": doc.id,
                "text":   doc.text,
                "metadata_json": metadata_json,
            }
        });

        let mut req = self.client.put(&url).json(&body);
        if let Some(ref auth) = self.bearer_header() {
            req = req.header("Authorization", auth);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(format!("Weaviate update: {e}")))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Weaviate update failed ({status}): {text}"
            )));
        }
        Ok(())
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let uuid = Self::point_uuid(doc_id);
        let path = format!("/v1/objects/{}/{}?include=vector", self.class_name, uuid);
        let data = self
            .get_opt(&path, "Weaviate get")
            .await?
            .ok_or_else(|| VectorStoreError::NotFound(doc_id.to_string()))?;

        let obj: WeaviateObject = serde_json::from_value(data).map_err(|e| {
            VectorStoreError::SerializationError(format!("Weaviate get parse: {e}"))
        })?;

        Ok(Self::object_to_doc(&obj))
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        let path = format!("/v1/objects?class={}&include=vector", self.class_name);
        let data = self
            .get_opt(&path, "Weaviate list")
            .await?
            .unwrap_or(serde_json::json!({ "objects": [] }));

        let response: WeaviateListResponse = serde_json::from_value(data).map_err(|e| {
            VectorStoreError::SerializationError(format!("Weaviate list parse: {e}"))
        })?;

        Ok(response
            .objects
            .unwrap_or_default()
            .iter()
            .map(Self::object_to_doc)
            .collect())
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        // Delete the entire class and recreate it — Weaviate's recommended approach.
        let path = format!("/v1/schema/{}", self.class_name);
        let url = format!("{}{path}", self.base_url());
        let mut req = self.client.delete(&url);
        if let Some(ref auth) = self.bearer_header() {
            req = req.header("Authorization", auth);
        }
        let _ = req.send().await; // ignore errors (class may not exist)

        self.ensure_class().await
    }
}

// ─── GraphQL helpers ──────────────────────────────────────────────────────────

/// Build a single Weaviate where-clause leaf node.
fn weaviate_leaf(operator: &str, path: &str, v: &Value) -> Value {
    let (type_field, typed_val) = match v {
        Value::String(s) => ("valueText", Value::String(s.clone())),
        Value::Number(n) => ("valueNumber", Value::Number(n.clone())),
        Value::Bool(b) => ("valueBoolean", Value::Bool(*b)),
        _ => ("valueText", Value::String(v.to_string())),
    };
    serde_json::json!({
        "path": [path],
        "operator": operator,
        type_field: typed_val,
    })
}

/// Render a `serde_json::Value` as an unquoted GraphQL value literal.
///
/// Weaviate's GraphQL `where` clauses require values without surrounding quotes
/// for numbers/booleans, and the overall argument structure without key quotes.
fn graphql_value(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let fields: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", k, graphql_value(v)))
                .collect();
            format!("{{{}}}", fields.join(", "))
        }
        Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(graphql_value).collect();
            format!("[{}]", items.join(", "))
        }
        Value::String(s) => format!("\"{s}\""),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
    }
}

// ─── Utility ──────────────────────────────────────────────────────────────────

/// Capitalise the first character of `s` (Weaviate class names must start with A–Z).
fn capitalise(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capitalise() {
        assert_eq!(capitalise("documents"), "Documents");
        assert_eq!(capitalise("Documents"), "Documents");
        assert_eq!(capitalise(""), "");
    }

    #[test]
    fn test_graphql_value_object() {
        let v = serde_json::json!({"operator": "Equal", "path": ["source"], "valueText": "pdf"});
        let s = graphql_value(&v);
        assert!(s.contains("operator"));
        assert!(s.contains("Equal"));
    }

    #[test]
    fn test_point_uuid_deterministic() {
        let a = WeaviateStore::point_uuid("doc-1");
        let b = WeaviateStore::point_uuid("doc-1");
        assert_eq!(a, b);
        assert_ne!(a, WeaviateStore::point_uuid("doc-2"));
    }
}
