//! ChromaDB Vector Store Backend
//!
//! Implements the `VectorStoreBackend` trait for ChromaDB,
//! communicating via ChromaDB's REST API.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::filter::FilterExpr;
use super::vector_db::{
    Document, MetadataFilter, RAGConfig, SearchResult, VectorStoreBackend, VectorStoreError,
};

/// ChromaDB vector store backend
pub struct ChromaStore {
    client: reqwest::Client,
    base_url: String,
    collection_name: String,
    collection_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChromaCollection {
    id: String,
    name: String,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ChromaAddRequest {
    ids: Vec<String>,
    documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embeddings: Option<Vec<Vec<f32>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadatas: Option<Vec<HashMap<String, serde_json::Value>>>,
}

#[derive(Debug, Deserialize)]
struct ChromaQueryResponse {
    ids: Vec<Vec<String>>,
    #[serde(default)]
    documents: Option<Vec<Vec<Option<String>>>>,
    #[serde(default)]
    distances: Option<Vec<Vec<f32>>>,
    #[serde(default)]
    #[allow(clippy::type_complexity)]
    metadatas: Option<Vec<Vec<Option<HashMap<String, serde_json::Value>>>>>,
}

#[derive(Debug, Deserialize)]
struct ChromaGetResponse {
    ids: Vec<String>,
    #[serde(default)]
    documents: Option<Vec<Option<String>>>,
    #[serde(default)]
    embeddings: Option<Vec<Option<Vec<f32>>>>,
    #[serde(default)]
    metadatas: Option<Vec<Option<HashMap<String, serde_json::Value>>>>,
}

impl ChromaStore {
    /// Create a new ChromaStore and ensure the collection exists
    pub async fn new(config: &RAGConfig) -> Result<Self, VectorStoreError> {
        let base_url = config
            .endpoint
            .as_deref()
            .unwrap_or("http://localhost:8000");

        let client = reqwest::Client::new();
        let collection_name = config.index_name.clone();

        let mut store = Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            collection_name,
            collection_id: None,
        };

        store.ensure_collection().await?;
        Ok(store)
    }

    /// Get or create the collection in ChromaDB
    async fn ensure_collection(&mut self) -> Result<(), VectorStoreError> {
        let url = format!("{}/api/v1/collections", self.base_url);

        // Try to get existing collection
        let resp = self.client.get(&url).send().await.map_err(|e| {
            VectorStoreError::ConnectionError(format!("ChromaDB unreachable: {}", e))
        })?;

        if resp.status().is_success() {
            let collections: Vec<ChromaCollection> = resp.json().await.map_err(|e| {
                VectorStoreError::ApiError(format!("Failed to parse collections: {}", e))
            })?;

            if let Some(col) = collections.iter().find(|c| c.name == self.collection_name) {
                self.collection_id = Some(col.id.clone());
                return Ok(());
            }
        }

        // Create collection
        let create_url = format!("{}/api/v1/collections", self.base_url);
        let body = serde_json::json!({
            "name": self.collection_name,
            "metadata": { "hnsw:space": "cosine" }
        });

        let resp = self
            .client
            .post(&create_url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                VectorStoreError::ConnectionError(format!("Failed to create collection: {}", e))
            })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Failed to create collection ({}): {}",
                status, text
            )));
        }

        let col: ChromaCollection = resp.json().await.map_err(|e| {
            VectorStoreError::ApiError(format!("Failed to parse created collection: {}", e))
        })?;

        self.collection_id = Some(col.id);
        Ok(())
    }

    fn collection_url(&self) -> Result<String, VectorStoreError> {
        let id = self.collection_id.as_ref().ok_or_else(|| {
            VectorStoreError::ConfigError("Collection not initialized".to_string())
        })?;
        Ok(format!("{}/api/v1/collections/{}", self.base_url, id))
    }
}

/// Convert a `FilterExpr` to a ChromaDB `where` clause JSON.
///
/// ChromaDB format: `{"field": {"$eq": value}}` for leaves,
/// `{"$and": [...]}` / `{"$or": [...]}` for compound expressions.
fn chroma_filter(f: &FilterExpr) -> serde_json::Value {
    match f {
        FilterExpr::Eq(k, v)  => serde_json::json!({ k: { "$eq":  v } }),
        FilterExpr::Ne(k, v)  => serde_json::json!({ k: { "$ne":  v } }),
        FilterExpr::Gt(k, v)  => serde_json::json!({ k: { "$gt":  v } }),
        FilterExpr::Lt(k, v)  => serde_json::json!({ k: { "$lt":  v } }),
        FilterExpr::Gte(k, v) => serde_json::json!({ k: { "$gte": v } }),
        FilterExpr::Lte(k, v) => serde_json::json!({ k: { "$lte": v } }),
        FilterExpr::In(k, vs) => serde_json::json!({ k: { "$in":  vs } }),
        FilterExpr::And(exprs) => serde_json::json!({
            "$and": exprs.iter().map(chroma_filter).collect::<Vec<_>>()
        }),
        FilterExpr::Or(exprs) => serde_json::json!({
            "$or": exprs.iter().map(chroma_filter).collect::<Vec<_>>()
        }),
    }
}

#[async_trait]
impl VectorStoreBackend for ChromaStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let url = format!("{}/add", self.collection_url()?);

        let body = ChromaAddRequest {
            ids: vec![doc.id],
            documents: vec![doc.text],
            embeddings: doc.embedding.map(|e| vec![e]),
            metadatas: if doc.metadata.is_empty() {
                None
            } else {
                Some(vec![doc.metadata])
            },
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(format!("Index failed: {}", e)))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Index failed: {}",
                text
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
        let url = format!("{}/query", self.collection_url()?);

        let mut body = serde_json::json!({
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "distances", "metadatas"]
        });

        // Convert FilterExpr to ChromaDB `where` clause
        if let Some(f) = filter {
            body["where"] = chroma_filter(&f);
        }

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(format!("Search failed: {e}")))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!("Search failed: {text}")));
        }

        let result: ChromaQueryResponse = resp.json().await.map_err(|e| {
            VectorStoreError::ApiError(format!("Parse search results failed: {}", e))
        })?;

        let mut results = Vec::new();

        if let Some(ids) = result.ids.first() {
            let documents = result.documents.as_ref().and_then(|d| d.first());
            let distances = result.distances.as_ref().and_then(|d| d.first());
            let metadatas = result.metadatas.as_ref().and_then(|m| m.first());

            for (i, id) in ids.iter().enumerate() {
                let text = documents
                    .and_then(|docs| docs.get(i))
                    .and_then(|d| d.clone())
                    .unwrap_or_default();

                // ChromaDB returns distances (lower = more similar for cosine).
                // Convert to similarity score: score = 1 - distance
                let distance = distances.and_then(|d| d.get(i).copied()).unwrap_or(1.0);
                let score = 1.0 - distance;

                let metadata = metadatas
                    .and_then(|m| m.get(i))
                    .and_then(|m| m.clone())
                    .unwrap_or_default();

                results.push(SearchResult {
                    id: id.clone(),
                    text,
                    score,
                    metadata,
                });
            }
        }

        Ok(results)
    }

    async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError> {
        let url = format!("{}/delete", self.collection_url()?);
        let body = serde_json::json!({ "ids": [doc_id] });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(format!("Delete failed: {}", e)))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Delete failed: {}",
                text
            )));
        }

        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        let url = format!("{}/update", self.collection_url()?);

        let body = serde_json::json!({
            "ids": [doc.id],
            "documents": [doc.text],
            "embeddings": doc.embedding.map(|e| vec![e]),
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(format!("Update failed: {}", e)))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Update failed: {}",
                text
            )));
        }

        Ok(())
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let url = format!("{}/get", self.collection_url()?);

        let body = serde_json::json!({
            "ids": [doc_id],
            "include": ["documents", "embeddings", "metadatas"]
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(format!("Get failed: {}", e)))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!("Get failed: {}", text)));
        }

        let result: ChromaGetResponse = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::ApiError(format!("Parse get result failed: {}", e)))?;

        if result.ids.is_empty() {
            return Err(VectorStoreError::NotFound(doc_id.to_string()));
        }

        let text = result
            .documents
            .as_ref()
            .and_then(|d| d.first())
            .and_then(|d| d.clone())
            .unwrap_or_default();

        let embedding = result
            .embeddings
            .as_ref()
            .and_then(|e| e.first())
            .and_then(|e| e.clone());

        let metadata = result
            .metadatas
            .as_ref()
            .and_then(|m| m.first())
            .and_then(|m| m.clone())
            .unwrap_or_default();

        Ok(Document {
            id: result.ids[0].clone(),
            text,
            embedding,
            metadata,
        })
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        let url = format!("{}/get", self.collection_url()?);

        let body = serde_json::json!({
            "include": ["documents", "metadatas"]
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(format!("List failed: {}", e)))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!("List failed: {}", text)));
        }

        let result: ChromaGetResponse = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::ApiError(format!("Parse list result failed: {}", e)))?;

        let mut docs = Vec::new();
        for (i, id) in result.ids.iter().enumerate() {
            let text = result
                .documents
                .as_ref()
                .and_then(|d| d.get(i))
                .and_then(|d| d.clone())
                .unwrap_or_default();

            let metadata = result
                .metadatas
                .as_ref()
                .and_then(|m| m.get(i))
                .and_then(|m| m.clone())
                .unwrap_or_default();

            docs.push(Document {
                id: id.clone(),
                text,
                embedding: None,
                metadata,
            });
        }

        Ok(docs)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        // Get all document IDs, then delete them
        let all_docs = self.list().await?;
        if all_docs.is_empty() {
            return Ok(());
        }

        let ids: Vec<&str> = all_docs.iter().map(|d| d.id.as_str()).collect();
        let url = format!("{}/delete", self.collection_url()?);
        let body = serde_json::json!({ "ids": ids });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::ConnectionError(format!("Clear failed: {}", e)))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(VectorStoreError::ApiError(format!(
                "Clear failed: {}",
                text
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chroma_store_types() {
        // Just verify types compile
        let _req = ChromaAddRequest {
            ids: vec!["test".to_string()],
            documents: vec!["hello".to_string()],
            embeddings: None,
            metadatas: None,
        };
    }
}
