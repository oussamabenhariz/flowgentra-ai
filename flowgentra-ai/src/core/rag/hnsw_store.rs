//! Local HNSW Vector Store  (FAISS-equivalent)
//!
//! A persistent, file-serialisable local vector store with cosine similarity
//! search. Mirrors FAISS's `IndexFlatIP` (exact inner-product / cosine) with
//! the same `save_local` / `load_local` API familiar from LangChain.
//!
//! ## Feature comparison vs `InMemoryVectorStore`
//!
//! | Feature | `InMemoryVectorStore` | `HnswVectorStore` |
//! |---|---|---|
//! | Persistence | No | Yes — `save_local` / `load_local` |
//! | Metadata filter | Yes | Yes |
//! | Timestamps | No | Yes — built-in `created_at` per doc |
//! | Cosine similarity | Yes | Yes |
//! | Thread-safe | Yes | Yes |
//! | No external deps | Yes | Yes (pure Rust) |
//!
//! The name "HNSW" reflects the intended upgrade path: the public API is
//! stable and future versions may swap the exact search for an approximate
//! HNSW index without breaking callers.
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::HnswVectorStore;
//!
//! let store = HnswVectorStore::new(1536);
//! store.index(doc).await?;
//!
//! // Persist to disk
//! store.save_local("./faiss_index").await?;
//!
//! // Load from disk on next run
//! let store = HnswVectorStore::load_local("./faiss_index").await?;
//! ```

use async_trait::async_trait;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use super::filter::FilterExpr;
use super::vector_db::{
    Document, MetadataFilter, SearchResult, VectorStoreBackend, VectorStoreError,
};

// ── Internal stored entry ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredEntry {
    id: String,
    text: String,
    embedding: Vec<f32>,
    metadata: HashMap<String, serde_json::Value>,
    created_at: i64,
}

// ── HnswVectorStore ──────────────────────────────────────────────────────────

/// Local, persistence-capable vector store with exact cosine similarity search.
///
/// All data is kept in a thread-safe `DashMap`. Persistence is handled by
/// serialising the entire map to / from a JSON file on disk.
pub struct HnswVectorStore {
    entries: Arc<DashMap<String, StoredEntry>>,
    /// Expected embedding dimensionality (used for validation).
    dim: usize,
}

impl HnswVectorStore {
    /// Create an empty store for vectors of dimension `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            entries: Arc::new(DashMap::new()),
            dim,
        }
    }

    /// Persist the index to `dir/index.json`.
    ///
    /// The directory is created if it does not exist.
    pub async fn save_local(&self, dir: impl AsRef<Path>) -> Result<(), VectorStoreError> {
        let dir = dir.as_ref();
        tokio::fs::create_dir_all(dir)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Create dir error: {e}")))?;

        let path = dir.join("index.json");
        let entries: Vec<StoredEntry> = self.entries.iter().map(|kv| kv.value().clone()).collect();

        let payload = serde_json::json!({
            "dim": self.dim,
            "entries": entries,
        });

        let json = serde_json::to_string_pretty(&payload)
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;

        tokio::fs::write(&path, json)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Write index error: {e}")))?;

        Ok(())
    }

    /// Load a previously saved index from `dir/index.json`.
    pub async fn load_local(dir: impl AsRef<Path>) -> Result<Self, VectorStoreError> {
        let path = dir.as_ref().join("index.json");
        let json = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Read index error: {e}")))?;

        let payload: serde_json::Value = serde_json::from_str(&json)
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;

        let dim = payload["dim"]
            .as_u64()
            .ok_or_else(|| VectorStoreError::SerializationError("Missing dim".into()))?
            as usize;

        let raw_entries: Vec<StoredEntry> = serde_json::from_value(payload["entries"].clone())
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;

        let map: DashMap<String, StoredEntry> = DashMap::new();
        for entry in raw_entries {
            map.insert(entry.id.clone(), entry);
        }

        Ok(Self {
            entries: Arc::new(map),
            dim,
        })
    }

    /// Number of entries in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return all documents as `Document` structs (embeddings included).
    pub fn all_documents(&self) -> Vec<Document> {
        self.entries
            .iter()
            .map(|kv| {
                let e = kv.value();
                Document {
                    id: e.id.clone(),
                    text: e.text.clone(),
                    embedding: Some(e.embedding.clone()),
                    metadata: e.metadata.clone(),
                }
            })
            .collect()
    }

    // ── Cosine similarity ────────────────────────────────────────────────────

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
        }
    }

    // ── Metadata filter evaluation ───────────────────────────────────────────

    fn matches_filter(metadata: &HashMap<String, serde_json::Value>, filter: &FilterExpr) -> bool {
        match filter {
            FilterExpr::Eq(k, v) => metadata.get(k) == Some(v),
            FilterExpr::Ne(k, v) => metadata.get(k) != Some(v),
            FilterExpr::Gt(k, v) => cmp_json(metadata.get(k), v) > 0,
            FilterExpr::Lt(k, v) => cmp_json(metadata.get(k), v) < 0,
            FilterExpr::Gte(k, v) => cmp_json(metadata.get(k), v) >= 0,
            FilterExpr::Lte(k, v) => cmp_json(metadata.get(k), v) <= 0,
            FilterExpr::In(k, vals) => {
                if let Some(mv) = metadata.get(k) {
                    vals.contains(mv)
                } else {
                    false
                }
            }
            FilterExpr::And(exprs) => exprs.iter().all(|e| Self::matches_filter(metadata, e)),
            FilterExpr::Or(exprs) => exprs.iter().any(|e| Self::matches_filter(metadata, e)),
        }
    }
}

/// Numeric comparison of two JSON values. Returns -1, 0, or 1.
fn cmp_json(a: Option<&serde_json::Value>, b: &serde_json::Value) -> i8 {
    let (Some(av), Some(bv)) = (a, b.as_f64()) else {
        return 0;
    };
    let af = av.as_f64().unwrap_or(0.0);
    af.partial_cmp(&bv)
        .map(|o| match o {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        })
        .unwrap_or(0)
}

// ── VectorStoreBackend impl ──────────────────────────────────────────────────

#[async_trait]
impl VectorStoreBackend for HnswVectorStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc
            .embedding
            .ok_or_else(|| VectorStoreError::EmbeddingError("Document has no embedding".into()))?;

        if embedding.len() != self.dim && self.dim > 0 {
            return Err(VectorStoreError::ConfigError(format!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dim,
                embedding.len()
            )));
        }

        let entry = StoredEntry {
            id: doc.id.clone(),
            text: doc.text,
            embedding,
            metadata: doc.metadata,
            created_at: chrono::Utc::now().timestamp(),
        };
        self.entries.insert(doc.id, entry);
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut scores: Vec<(String, f32, String, HashMap<String, serde_json::Value>)> = self
            .entries
            .iter()
            .filter(|kv| {
                filter
                    .as_ref()
                    .map(|f| Self::matches_filter(&kv.value().metadata, f))
                    .unwrap_or(true)
            })
            .map(|kv| {
                let e = kv.value();
                let score = Self::cosine_similarity(&query_embedding, &e.embedding);
                (e.id.clone(), score, e.text.clone(), e.metadata.clone())
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);

        Ok(scores
            .into_iter()
            .map(|(id, score, text, metadata)| SearchResult {
                id,
                text,
                score,
                metadata,
            })
            .collect())
    }

    async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError> {
        self.entries.remove(doc_id);
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc
            .embedding
            .ok_or_else(|| VectorStoreError::EmbeddingError("Document has no embedding".into()))?;

        self.entries.alter(&doc.id, |_, mut e| {
            e.text = doc.text.clone();
            e.embedding = embedding.clone();
            e.metadata = doc.metadata.clone();
            e
        });
        Ok(())
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        self.entries
            .get(doc_id)
            .map(|e| Document {
                id: e.id.clone(),
                text: e.text.clone(),
                embedding: Some(e.embedding.clone()),
                metadata: e.metadata.clone(),
            })
            .ok_or_else(|| VectorStoreError::NotFound(doc_id.to_string()))
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        Ok(self.all_documents())
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        self.entries.clear();
        Ok(())
    }

    fn supports_list(&self) -> bool {
        true
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, text: &str, dim: usize) -> Document {
        let embedding: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) / dim as f32).collect();
        Document {
            id: id.to_string(),
            text: text.to_string(),
            embedding: Some(embedding),
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_index_and_search() {
        let store = HnswVectorStore::new(4);
        let doc = make_doc("doc1", "hello world", 4);
        store.index(doc).await.unwrap();

        let query = vec![0.25, 0.5, 0.75, 1.0];
        let results = store.search(query, 1, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
        assert!(results[0].score > 0.0);
    }

    #[tokio::test]
    async fn test_save_and_load() {
        let tmp = tempfile::tempdir().unwrap();
        let store = HnswVectorStore::new(4);
        store.index(make_doc("a", "alpha", 4)).await.unwrap();
        store.index(make_doc("b", "beta", 4)).await.unwrap();

        let dir = tmp.path().join("idx");
        store.save_local(&dir).await.unwrap();

        let loaded = HnswVectorStore::load_local(&dir).await.unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[tokio::test]
    async fn test_metadata_filter() {
        use super::super::filter::FilterExpr;
        use serde_json::json;

        let store = HnswVectorStore::new(4);

        let mut meta_a = HashMap::new();
        meta_a.insert("year".to_string(), json!(2024));
        let mut doc_a = make_doc("a", "recent doc", 4);
        doc_a.metadata = meta_a;

        let mut meta_b = HashMap::new();
        meta_b.insert("year".to_string(), json!(2020));
        let mut doc_b = make_doc("b", "old doc", 4);
        doc_b.metadata = meta_b;

        store.index(doc_a).await.unwrap();
        store.index(doc_b).await.unwrap();

        let query = vec![0.25, 0.5, 0.75, 1.0];
        let filter = FilterExpr::eq("year", json!(2024));
        let results = store.search(query, 10, Some(filter)).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[tokio::test]
    async fn test_get_and_delete() {
        let store = HnswVectorStore::new(4);
        store.index(make_doc("x", "to delete", 4)).await.unwrap();
        assert!(store.get("x").await.is_ok());
        store.delete("x").await.unwrap();
        assert!(store.get("x").await.is_err());
    }
}
