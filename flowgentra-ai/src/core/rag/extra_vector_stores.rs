//! Additional Vector Store Backends
//!
//! HTTP-based implementations for:
//!
//! | Type | API | Notes |
//! |---|---|---|
//! | [`SingleStoreVectorStore`] | SingleStore HTTP | Requires SINGLESTORE_HOST + API key |
//! | [`AzureAISearchStore`] | Azure AI Search REST | Requires endpoint + admin key |
//! | [`VectaraStore`] | Vectara REST API | Requires customer_id + corpus_id + api_key |
//! | [`Neo4jVectorStore`] | Neo4j Bolt / HTTP | Requires `neo4j-store` feature |
//! | [`TurbopufferStore`] | Turbopuffer API | Requires `TURBOPUFFER_API_KEY` |
//!
//! All types implement [`VectorStoreBackend`].

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::filter::FilterExpr;
use super::vector_db::{Document, SearchResult, VectorStoreBackend, VectorStoreError};

// ── helpers ───────────────────────────────────────────────────────────────────

fn http_err(label: &str, e: impl std::fmt::Display) -> VectorStoreError {
    VectorStoreError::Unknown(format!("{label}: {e}"))
}

// ── SingleStore ───────────────────────────────────────────────────────────────

/// Vector store backed by SingleStore's vector similarity search.
///
/// Requires a SingleStore instance with a table containing `id TEXT`,
/// `text TEXT`, `embedding VECTOR(dim)`, and `metadata JSON` columns.
pub struct SingleStoreVectorStore {
    /// HTTP MySQL-compatible connection URL or REST endpoint.
    pub host: String,
    pub api_key: String,
    pub table: String,
    pub database: String,
}

impl SingleStoreVectorStore {
    pub fn new(
        host: impl Into<String>,
        api_key: impl Into<String>,
        database: impl Into<String>,
        table: impl Into<String>,
    ) -> Self {
        Self {
            host: host.into(),
            api_key: api_key.into(),
            database: database.into(),
            table: table.into(),
        }
    }

    async fn post(&self, endpoint: &str, body: Value) -> Result<Value, VectorStoreError> {
        let client = reqwest::Client::new();
        let url = format!("{}{}", self.host.trim_end_matches('/'), endpoint);
        let resp = client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| http_err("SingleStore HTTP", e))?;
        resp.json::<Value>()
            .await
            .map_err(|e| http_err("SingleStore JSON", e))
    }
}

#[async_trait]
impl VectorStoreBackend for SingleStoreVectorStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding_str = doc
            .embedding
            .as_ref()
            .map(|e| {
                format!(
                    "[{}]",
                    e.iter()
                        .map(|f| f.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .unwrap_or_default();

        let sql = format!(
            "INSERT INTO {}.{} (id, text, embedding, metadata) \
             VALUES ('{}', '{}', '{}', '{}') \
             ON DUPLICATE KEY UPDATE text=VALUES(text), embedding=VALUES(embedding), metadata=VALUES(metadata)",
            self.database, self.table,
            doc.id, doc.text.replace('\'', "\\'"),
            embedding_str,
            serde_json::to_string(&doc.metadata).unwrap_or_default()
        );
        self.post("/api/v2/exec", json!({ "sql": sql })).await?;
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        _filter: Option<FilterExpr>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let emb_str = format!(
            "[{}]",
            query_embedding
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        let sql = format!(
            "SELECT id, text, metadata, DOT_PRODUCT(embedding, '{}') AS score \
             FROM {}.{} ORDER BY score DESC LIMIT {}",
            emb_str, self.database, self.table, top_k
        );
        let resp = self.post("/api/v2/exec", json!({ "sql": sql })).await?;
        let rows = resp["results"].as_array().cloned().unwrap_or_default();
        Ok(rows
            .iter()
            .map(|r| SearchResult {
                id: r["id"].as_str().unwrap_or("").to_string(),
                text: r["text"].as_str().unwrap_or("").to_string(),
                score: r["score"].as_f64().unwrap_or(0.0) as f32,
                metadata: r["metadata"]
                    .as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default(),
            })
            .collect())
    }

    async fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        let sql = format!("DELETE FROM {}.{} WHERE id='{id}'", self.database, self.table);
        self.post("/api/v2/exec", json!({ "sql": sql })).await?;
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.index(doc).await
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let sql = format!("SELECT id, text, metadata FROM {}.{} WHERE id='{doc_id}'", self.database, self.table);
        let resp = self.post("/api/v2/exec", json!({ "sql": sql })).await?;
        let rows = resp["results"].as_array().cloned().unwrap_or_default();
        if let Some(r) = rows.first() {
            return Ok(Document {
                id: r["id"].as_str().unwrap_or("").to_string(),
                text: r["text"].as_str().unwrap_or("").to_string(),
                embedding: None,
                metadata: HashMap::new(),
            });
        }
        Err(VectorStoreError::Unknown(format!("Document {doc_id} not found")))
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        let sql = format!("SELECT id, text FROM {}.{}", self.database, self.table);
        let resp = self.post("/api/v2/exec", json!({ "sql": sql })).await?;
        Ok(resp["results"].as_array().cloned().unwrap_or_default().iter().map(|r| Document {
            id: r["id"].as_str().unwrap_or("").to_string(),
            text: r["text"].as_str().unwrap_or("").to_string(),
            embedding: None,
            metadata: HashMap::new(),
        }).collect())
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let sql = format!("DELETE FROM {}.{}", self.database, self.table);
        self.post("/api/v2/exec", json!({ "sql": sql })).await?;
        Ok(())
    }
}

// ── Azure AI Search ───────────────────────────────────────────────────────────

/// Vector store backed by Azure AI Search (formerly Azure Cognitive Search).
pub struct AzureAISearchStore {
    pub endpoint: String,
    pub index_name: String,
    pub api_key: String,
    pub api_version: String,
}

impl AzureAISearchStore {
    pub fn new(
        endpoint: impl Into<String>,
        index_name: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            index_name: index_name.into(),
            api_key: api_key.into(),
            api_version: "2024-07-01".to_string(),
        }
    }

    fn base_url(&self) -> String {
        format!(
            "{}/indexes/{}/docs",
            self.endpoint.trim_end_matches('/'),
            self.index_name
        )
    }

    async fn request(
        &self,
        method: reqwest::Method,
        url: &str,
        body: Option<Value>,
    ) -> Result<Value, VectorStoreError> {
        let client = reqwest::Client::new();
        let mut req = client
            .request(method, url)
            .query(&[("api-version", &self.api_version)])
            .header("api-key", &self.api_key)
            .header("Content-Type", "application/json");
        if let Some(b) = body {
            req = req.json(&b);
        }
        let resp = req.send().await.map_err(|e| http_err("Azure Search HTTP", e))?;
        resp.json::<Value>().await.map_err(|e| http_err("Azure Search JSON", e))
    }
}

#[async_trait]
impl VectorStoreBackend for AzureAISearchStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let mut fields = json!({
            "@search.action": "mergeOrUpload",
            "id": doc.id,
            "text": doc.text,
        });
        if let Some(emb) = doc.embedding {
            fields["embedding"] = json!(emb);
        }
        for (k, v) in &doc.metadata {
            fields[k] = v.clone();
        }
        let url = format!("{}/index", self.base_url());
        self.request(
            reqwest::Method::POST,
            &url,
            Some(json!({ "value": [fields] })),
        )
        .await?;
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        _filter: Option<FilterExpr>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let url = format!("{}/search", self.base_url());
        let body = json!({
            "vectorQueries": [{
                "kind": "vector",
                "vector": query_embedding,
                "fields": "embedding",
                "k": top_k,
            }],
            "select": "id,text",
            "top": top_k,
        });
        let resp = self
            .request(reqwest::Method::POST, &url, Some(body))
            .await?;
        let results = resp["value"].as_array().cloned().unwrap_or_default();
        Ok(results
            .iter()
            .map(|r| SearchResult {
                id: r["id"].as_str().unwrap_or("").to_string(),
                text: r["text"].as_str().unwrap_or("").to_string(),
                score: r["@search.score"].as_f64().unwrap_or(0.0) as f32,
                metadata: HashMap::new(),
            })
            .collect())
    }

    async fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        let url = format!("{}/index", self.base_url());
        self.request(
            reqwest::Method::POST,
            &url,
            Some(json!({ "value": [{"@search.action": "delete", "id": id}] })),
        )
        .await?;
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.index(doc).await
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let url = format!("{}/{}", self.base_url(), doc_id);
        let resp = self.request(reqwest::Method::GET, &url, None).await?;
        Ok(Document {
            id: resp["id"].as_str().unwrap_or(doc_id).to_string(),
            text: resp["text"].as_str().unwrap_or("").to_string(),
            embedding: None,
            metadata: HashMap::new(),
        })
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        let url = format!("{}/search", self.base_url());
        let body = json!({ "search": "*", "top": 1000 });
        let resp = self.request(reqwest::Method::POST, &url, Some(body)).await?;
        Ok(resp["value"].as_array().cloned().unwrap_or_default().iter().map(|r| Document {
            id: r["id"].as_str().unwrap_or("").to_string(),
            text: r["text"].as_str().unwrap_or("").to_string(),
            embedding: None,
            metadata: HashMap::new(),
        }).collect())
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        // Azure AI Search: delete all requires fetching all IDs first
        let docs = self.list().await?;
        for doc in docs {
            self.delete(&doc.id).await.ok();
        }
        Ok(())
    }
}

// ── Vectara ───────────────────────────────────────────────────────────────────

/// Vector store backed by Vectara's semantic search platform.
pub struct VectaraStore {
    pub customer_id: String,
    pub corpus_id: u64,
    pub api_key: String,
}

impl VectaraStore {
    pub fn new(
        customer_id: impl Into<String>,
        corpus_id: u64,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            customer_id: customer_id.into(),
            corpus_id,
            api_key: api_key.into(),
        }
    }

    async fn post(&self, endpoint: &str, body: Value) -> Result<Value, VectorStoreError> {
        let client = reqwest::Client::new();
        let url = format!("https://api.vectara.io{endpoint}");
        let resp = client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("customer-id", &self.customer_id)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| http_err("Vectara HTTP", e))?;
        resp.json::<Value>().await.map_err(|e| http_err("Vectara JSON", e))
    }
}

#[async_trait]
impl VectorStoreBackend for VectaraStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let body = json!({
            "customerId": self.customer_id.parse::<u64>().unwrap_or(0),
            "corpusId": self.corpus_id,
            "document": {
                "documentId": doc.id,
                "title": doc.metadata.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                "section": [{"text": doc.text}],
                "metadataJson": serde_json::to_string(&doc.metadata).unwrap_or_default(),
            }
        });
        self.post("/v1/index", body).await?;
        Ok(())
    }

    async fn search(
        &self,
        _query_embedding: Vec<f32>,
        _top_k: usize,
        _filter: Option<FilterExpr>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        // Vectara is text-in / results-out; embedding-based search uses their internal model
        // This requires the original query text which isn't available here;
        // for full support use VectaraRetriever which accepts a text query.
        Err(VectorStoreError::Unknown(
            "VectaraStore.search() requires text query — use VectaraStore.search_text() instead"
                .to_string(),
        ))
    }

    async fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        let body = json!({
            "customerId": self.customer_id.parse::<u64>().unwrap_or(0),
            "corpusId": self.corpus_id,
            "documentId": id,
        });
        self.post("/v1/delete-doc", body).await?;
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.index(doc).await
    }

    async fn get(&self, _doc_id: &str) -> Result<Document, VectorStoreError> {
        Err(VectorStoreError::Unknown("VectaraStore.get() not supported via REST API".into()))
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        Err(VectorStoreError::Unknown("VectaraStore.list() not supported via REST API".into()))
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        Err(VectorStoreError::Unknown("VectaraStore.clear() not supported via REST API — delete documents individually".into()))
    }
}

impl VectaraStore {
    /// Full text search (recommended — uses Vectara's internal embedding model).
    pub async fn search_text(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let body = json!({
            "query": [{
                "query": query,
                "numResults": top_k,
                "corpusKey": [{"customerId": self.customer_id.parse::<u64>().unwrap_or(0), "corpusId": self.corpus_id}],
            }]
        });
        let resp = self.post("/v1/query", body).await?;
        let responses = resp["responseSet"][0]["response"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        Ok(responses
            .iter()
            .enumerate()
            .map(|(i, r)| SearchResult {
                id: format!("vectara_{i}"),
                text: r["text"].as_str().unwrap_or("").to_string(),
                score: r["score"].as_f64().unwrap_or(0.0) as f32,
                metadata: HashMap::new(),
            })
            .collect())
    }
}

// ── Turbopuffer ───────────────────────────────────────────────────────────────

/// Vector store backed by Turbopuffer's serverless vector database.
pub struct TurbopufferStore {
    pub api_key: String,
    pub namespace: String,
}

impl TurbopufferStore {
    pub fn new(api_key: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            namespace: namespace.into(),
        }
    }

    fn base_url(&self) -> String {
        format!(
            "https://api.turbopuffer.com/v1/vectors/{}",
            self.namespace
        )
    }

    async fn request(
        &self,
        method: reqwest::Method,
        url: &str,
        body: Option<Value>,
    ) -> Result<Value, VectorStoreError> {
        let client = reqwest::Client::new();
        let mut req = client
            .request(method, url)
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/json");
        if let Some(b) = body {
            req = req.json(&b);
        }
        let resp = req.send().await.map_err(|e| http_err("Turbopuffer HTTP", e))?;
        resp.json::<Value>().await.map_err(|e| http_err("Turbopuffer JSON", e))
    }
}

#[async_trait]
impl VectorStoreBackend for TurbopufferStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc
            .embedding
            .as_ref()
            .map(|e| json!(e))
            .unwrap_or(json!([]));
        let attributes = serde_json::to_value(&doc.metadata).unwrap_or_default();
        let body = json!({
            "vectors": [{
                "id": doc.id,
                "vector": embedding,
                "attributes": {
                    "text": doc.text,
                    "metadata": attributes,
                }
            }]
        });
        self.request(reqwest::Method::POST, &self.base_url(), Some(body))
            .await?;
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        _filter: Option<FilterExpr>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let url = format!("{}/query", self.base_url());
        let body = json!({
            "vector": query_embedding,
            "top_k": top_k,
            "include_vectors": false,
            "include_attributes": ["text", "metadata"],
            "distance_metric": "cosine_distance",
        });
        let resp = self
            .request(reqwest::Method::POST, &url, Some(body))
            .await?;
        let items = resp.as_array().cloned().unwrap_or_default();
        Ok(items
            .iter()
            .map(|item| {
                let text = item["attributes"]["text"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                let dist = item["dist"].as_f64().unwrap_or(1.0) as f32;
                SearchResult {
                    id: item["id"].as_str().unwrap_or("").to_string(),
                    text,
                    score: 1.0 - dist, // convert distance to similarity
                    metadata: HashMap::new(),
                }
            })
            .collect())
    }

    async fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        let url = format!("{}/delete", self.base_url());
        self.request(
            reqwest::Method::POST,
            &url,
            Some(json!({ "ids": [id] })),
        )
        .await?;
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.index(doc).await
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let url = format!("{}/vectors", self.base_url());
        let body = json!({ "ids": [doc_id] });
        let resp = self.request(reqwest::Method::POST, &url, Some(body)).await?;
        let item = &resp[0];
        Ok(Document {
            id: item["id"].as_str().unwrap_or(doc_id).to_string(),
            text: item["attributes"]["text"].as_str().unwrap_or("").to_string(),
            embedding: None,
            metadata: HashMap::new(),
        })
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        let url = format!("{}/vectors", self.base_url());
        let resp = self.request(reqwest::Method::GET, &url, None).await?;
        Ok(resp.as_array().cloned().unwrap_or_default().iter().map(|item| Document {
            id: item["id"].as_str().unwrap_or("").to_string(),
            text: item["attributes"]["text"].as_str().unwrap_or("").to_string(),
            embedding: None,
            metadata: HashMap::new(),
        }).collect())
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let url = format!("{}/delete_all", self.base_url());
        self.request(reqwest::Method::DELETE, &url, None).await?;
        Ok(())
    }
}

// ── Neo4j Vector Store ────────────────────────────────────────────────────────

/// Vector store backed by Neo4j's vector index (Neo4j 5.11+).
///
/// Uses the Neo4j HTTP API (no extra feature required).
/// For the Bolt protocol, enable the `neo4j-store` feature.
pub struct Neo4jVectorStore {
    pub url: String,
    pub username: String,
    pub password: String,
    pub index_name: String,
    pub node_label: String,
    pub embedding_property: String,
    pub text_property: String,
}

impl Neo4jVectorStore {
    pub fn new(
        url: impl Into<String>,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        Self {
            url: url.into(),
            username: username.into(),
            password: password.into(),
            index_name: "vector_index".to_string(),
            node_label: "Document".to_string(),
            embedding_property: "embedding".to_string(),
            text_property: "text".to_string(),
        }
    }

    pub fn with_index(mut self, index_name: impl Into<String>) -> Self {
        self.index_name = index_name.into();
        self
    }

    pub fn with_node_label(mut self, label: impl Into<String>) -> Self {
        self.node_label = label.into();
        self
    }

    async fn cypher(&self, query: &str, params: Value) -> Result<Value, VectorStoreError> {
        let client = reqwest::Client::new();
        let url = format!("{}/db/neo4j/tx/commit", self.url.trim_end_matches('/'));
        let body = json!({
            "statements": [{ "statement": query, "parameters": params }]
        });
        let resp = client
            .post(&url)
            .basic_auth(&self.username, Some(&self.password))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| http_err("Neo4j HTTP", e))?;
        resp.json::<Value>().await.map_err(|e| http_err("Neo4j JSON", e))
    }
}

#[async_trait]
impl VectorStoreBackend for Neo4jVectorStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let query = format!(
            "MERGE (n:{} {{id: $id}}) \
             SET n.text = $text, n.embedding = $embedding, n.metadata = $metadata",
            self.node_label
        );
        let embedding = doc
            .embedding
            .as_ref()
            .map(|e| json!(e))
            .unwrap_or(json!([]));
        self.cypher(
            &query,
            json!({
                "id": doc.id,
                "text": doc.text,
                "embedding": embedding,
                "metadata": serde_json::to_string(&doc.metadata).unwrap_or_default(),
            }),
        )
        .await?;
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        _filter: Option<FilterExpr>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let query = format!(
            "CALL db.index.vector.queryNodes($index, $k, $embedding) \
             YIELD node, score \
             RETURN node.id AS id, node.text AS text, score",
        );
        let resp = self
            .cypher(
                &query,
                json!({
                    "index": self.index_name,
                    "k": top_k as i64,
                    "embedding": query_embedding,
                }),
            )
            .await?;

        let rows = resp["results"][0]["data"].as_array().cloned().unwrap_or_default();
        Ok(rows
            .iter()
            .map(|r| {
                let row = &r["row"];
                SearchResult {
                    id: row[0].as_str().unwrap_or("").to_string(),
                    text: row[1].as_str().unwrap_or("").to_string(),
                    score: row[2].as_f64().unwrap_or(0.0) as f32,
                    metadata: HashMap::new(),
                }
            })
            .collect())
    }

    async fn delete(&self, id: &str) -> Result<(), VectorStoreError> {
        let query = format!("MATCH (n:{} {{id: $id}}) DELETE n", self.node_label);
        self.cypher(&query, json!({ "id": id })).await?;
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.index(doc).await
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let query = format!(
            "MATCH (n:{} {{id: $id}}) RETURN n.id AS id, n.text AS text",
            self.node_label
        );
        let resp = self.cypher(&query, json!({ "id": doc_id })).await?;
        let rows = resp["results"][0]["data"].as_array().cloned().unwrap_or_default();
        if let Some(r) = rows.first() {
            let row = &r["row"];
            return Ok(Document {
                id: row[0].as_str().unwrap_or(doc_id).to_string(),
                text: row[1].as_str().unwrap_or("").to_string(),
                embedding: None,
                metadata: HashMap::new(),
            });
        }
        Err(VectorStoreError::Unknown(format!("Neo4j: document {doc_id} not found")))
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        let query = format!(
            "MATCH (n:{}) RETURN n.id AS id, n.text AS text LIMIT 1000",
            self.node_label
        );
        let resp = self.cypher(&query, json!({})).await?;
        let rows = resp["results"][0]["data"].as_array().cloned().unwrap_or_default();
        Ok(rows.iter().map(|r| {
            let row = &r["row"];
            Document {
                id: row[0].as_str().unwrap_or("").to_string(),
                text: row[1].as_str().unwrap_or("").to_string(),
                embedding: None,
                metadata: HashMap::new(),
            }
        }).collect())
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let query = format!("MATCH (n:{}) DELETE n", self.node_label);
        self.cypher(&query, json!({})).await?;
        Ok(())
    }
}
