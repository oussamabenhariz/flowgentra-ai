//! Document Store Trait & Implementations
//!
//! Mirrors LangChain's `BaseStore` / `InMemoryStore` / `LocalFileStore` pattern.
//! Used by [`ParentDocumentRetriever`] and [`MultiVectorRetriever`] to persist
//! parent documents separately from their vector embeddings.
//!
//! ## Implementations
//!
//! | Type | Storage | Feature |
//! |---|---|---|
//! | [`InMemoryDocStore`] | `DashMap` in process | always |
//! | [`LocalFileDocStore`] | One JSON file per document | always |
//! | [`RedisDocStore`] | Redis hashes | `redis-store` |
//! | [`MongoDocStore`] | MongoDB collection | `mongodb-store` |

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum DocStoreError {
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("IO error: {0}")]
    Io(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Backend error: {0}")]
    Backend(String),
}

pub type DocStoreResult<T> = std::result::Result<T, DocStoreError>;

// ── Document ──────────────────────────────────────────────────────────────────

/// A stored document with id, text content, and arbitrary metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredDocument {
    pub id: String,
    pub text: String,
    pub metadata: HashMap<String, Value>,
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Key-value store for documents. Implementations must be `Send + Sync`.
#[async_trait]
pub trait DocStore: Send + Sync {
    /// Store a document.
    async fn mset(&self, docs: Vec<StoredDocument>) -> DocStoreResult<()>;

    /// Retrieve documents by ids. Missing ids yield `None`.
    async fn mget(&self, ids: &[&str]) -> DocStoreResult<Vec<Option<StoredDocument>>>;

    /// Delete documents by ids.
    async fn mdelete(&self, ids: &[&str]) -> DocStoreResult<()>;

    /// List all stored document ids.
    async fn yield_keys(&self) -> DocStoreResult<Vec<String>>;

    /// Convenience: store a single document.
    async fn set(&self, doc: StoredDocument) -> DocStoreResult<()> {
        self.mset(vec![doc]).await
    }

    /// Convenience: retrieve a single document.
    async fn get(&self, id: &str) -> DocStoreResult<Option<StoredDocument>> {
        Ok(self.mget(&[id]).await?.into_iter().next().flatten())
    }

    /// Convenience: delete a single document.
    async fn delete(&self, id: &str) -> DocStoreResult<()> {
        self.mdelete(&[id]).await
    }
}

// ── InMemoryDocStore ─────────────────────────────────────────────────────────

/// Thread-safe in-memory document store using `DashMap`.
pub struct InMemoryDocStore {
    store: Arc<DashMap<String, StoredDocument>>,
}

impl InMemoryDocStore {
    pub fn new() -> Self {
        Self {
            store: Arc::new(DashMap::new()),
        }
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}

impl Default for InMemoryDocStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocStore for InMemoryDocStore {
    async fn mset(&self, docs: Vec<StoredDocument>) -> DocStoreResult<()> {
        for doc in docs {
            self.store.insert(doc.id.clone(), doc);
        }
        Ok(())
    }

    async fn mget(&self, ids: &[&str]) -> DocStoreResult<Vec<Option<StoredDocument>>> {
        Ok(ids
            .iter()
            .map(|id| self.store.get(*id).map(|r| r.clone()))
            .collect())
    }

    async fn mdelete(&self, ids: &[&str]) -> DocStoreResult<()> {
        for id in ids {
            self.store.remove(*id);
        }
        Ok(())
    }

    async fn yield_keys(&self) -> DocStoreResult<Vec<String>> {
        Ok(self.store.iter().map(|r| r.key().clone()).collect())
    }
}

// ── LocalFileDocStore ─────────────────────────────────────────────────────────

/// Stores one JSON file per document in a local directory.
///
/// File path: `{dir}/{id}.json`. The directory is created if it doesn't exist.
pub struct LocalFileDocStore {
    dir: String,
}

impl LocalFileDocStore {
    pub fn new(dir: impl Into<String>) -> Self {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).ok();
        Self { dir }
    }

    fn path(&self, id: &str) -> String {
        let safe_id = id.replace(['/', '\\', ':'], "_");
        format!("{}/{safe_id}.json", self.dir)
    }
}

#[async_trait]
impl DocStore for LocalFileDocStore {
    async fn mset(&self, docs: Vec<StoredDocument>) -> DocStoreResult<()> {
        for doc in docs {
            let json = serde_json::to_string_pretty(&doc)
                .map_err(|e| DocStoreError::Serialization(e.to_string()))?;
            std::fs::write(self.path(&doc.id), json)
                .map_err(|e| DocStoreError::Io(e.to_string()))?;
        }
        Ok(())
    }

    async fn mget(&self, ids: &[&str]) -> DocStoreResult<Vec<Option<StoredDocument>>> {
        let mut result = Vec::new();
        for id in ids {
            let path = self.path(id);
            match std::fs::read_to_string(&path) {
                Ok(json) => {
                    let doc: StoredDocument = serde_json::from_str(&json)
                        .map_err(|e| DocStoreError::Serialization(e.to_string()))?;
                    result.push(Some(doc));
                }
                Err(_) => result.push(None),
            }
        }
        Ok(result)
    }

    async fn mdelete(&self, ids: &[&str]) -> DocStoreResult<()> {
        for id in ids {
            std::fs::remove_file(self.path(id)).ok();
        }
        Ok(())
    }

    async fn yield_keys(&self) -> DocStoreResult<Vec<String>> {
        let dir = std::fs::read_dir(&self.dir).map_err(|e| DocStoreError::Io(e.to_string()))?;
        let keys = dir
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                name.strip_suffix(".json").map(str::to_string)
            })
            .collect();
        Ok(keys)
    }
}

// ── RedisDocStore ─────────────────────────────────────────────────────────────

#[cfg(feature = "redis-store")]
pub mod redis_doc_store {
    use super::*;
    use redis::aio::ConnectionManager;
    use redis::AsyncCommands;

    /// Stores documents as JSON strings in Redis (key: `doc:{id}`).
    pub struct RedisDocStore {
        manager: ConnectionManager,
        key_prefix: String,
        ttl_secs: Option<u64>,
    }

    impl RedisDocStore {
        pub async fn new(url: &str, key_prefix: impl Into<String>) -> DocStoreResult<Self> {
            let client = redis::Client::open(url)
                .map_err(|e| DocStoreError::Backend(e.to_string()))?;
            let manager = ConnectionManager::new(client)
                .await
                .map_err(|e| DocStoreError::Backend(e.to_string()))?;
            Ok(Self {
                manager,
                key_prefix: key_prefix.into(),
                ttl_secs: None,
            })
        }

        pub fn with_ttl(mut self, ttl_secs: u64) -> Self {
            self.ttl_secs = Some(ttl_secs);
            self
        }

        fn key(&self, id: &str) -> String {
            format!("{}:{id}", self.key_prefix)
        }

        fn index_key(&self) -> String {
            format!("{}:__index__", self.key_prefix)
        }
    }

    #[async_trait]
    impl DocStore for RedisDocStore {
        async fn mset(&self, docs: Vec<StoredDocument>) -> DocStoreResult<()> {
            for doc in docs {
                let json = serde_json::to_string(&doc)
                    .map_err(|e| DocStoreError::Serialization(e.to_string()))?;
                let key = self.key(&doc.id);
                let mut mgr = self.manager.clone();
                if let Some(ttl) = self.ttl_secs {
                    let _: () = mgr.set_ex(&key, &json, ttl).await
                        .map_err(|e| DocStoreError::Backend(e.to_string()))?;
                } else {
                    let _: () = mgr.set(&key, &json).await
                        .map_err(|e| DocStoreError::Backend(e.to_string()))?;
                }
                let _: () = mgr.sadd(self.index_key(), &doc.id).await
                    .map_err(|e| DocStoreError::Backend(e.to_string()))?;
            }
            Ok(())
        }

        async fn mget(&self, ids: &[&str]) -> DocStoreResult<Vec<Option<StoredDocument>>> {
            let mut result = Vec::new();
            for id in ids {
                let mut mgr = self.manager.clone();
                let json: Option<String> = mgr.get(self.key(id)).await
                    .map_err(|e| DocStoreError::Backend(e.to_string()))?;
                match json {
                    None => result.push(None),
                    Some(j) => {
                        let doc: StoredDocument = serde_json::from_str(&j)
                            .map_err(|e| DocStoreError::Serialization(e.to_string()))?;
                        result.push(Some(doc));
                    }
                }
            }
            Ok(result)
        }

        async fn mdelete(&self, ids: &[&str]) -> DocStoreResult<()> {
            for id in ids {
                let mut mgr = self.manager.clone();
                let _: () = mgr.del(self.key(id)).await
                    .map_err(|e| DocStoreError::Backend(e.to_string()))?;
                let _: () = mgr.srem(self.index_key(), id).await
                    .map_err(|e| DocStoreError::Backend(e.to_string()))?;
            }
            Ok(())
        }

        async fn yield_keys(&self) -> DocStoreResult<Vec<String>> {
            let mut mgr = self.manager.clone();
            let keys: Vec<String> = mgr.smembers(self.index_key()).await
                .map_err(|e| DocStoreError::Backend(e.to_string()))?;
            Ok(keys)
        }
    }
}

// ── MongoDocStore ─────────────────────────────────────────────────────────────

#[cfg(feature = "mongodb-store")]
pub mod mongo_doc_store {
    use super::*;
    use mongodb::{bson::doc, Client};
    use futures::TryStreamExt;

    /// Stores documents as MongoDB documents.
    pub struct MongoDocStore {
        collection: mongodb::Collection<mongodb::bson::Document>,
    }

    impl MongoDocStore {
        pub async fn new(
            url: &str,
            db: &str,
            collection: &str,
        ) -> DocStoreResult<Self> {
            let client = Client::with_uri_str(url)
                .await
                .map_err(|e| DocStoreError::Backend(e.to_string()))?;
            let col = client.database(db).collection(collection);
            Ok(Self { collection: col })
        }
    }

    #[async_trait]
    impl DocStore for MongoDocStore {
        async fn mset(&self, docs: Vec<StoredDocument>) -> DocStoreResult<()> {
            for doc_item in docs {
                let json = serde_json::to_string(&doc_item)
                    .map_err(|e| DocStoreError::Serialization(e.to_string()))?;
                let bson_doc: mongodb::bson::Document = mongodb::bson::from_slice(json.as_bytes())
                    .map_err(|e| DocStoreError::Serialization(e.to_string()))?;
                self.collection
                    .replace_one(doc! { "id": &doc_item.id }, bson_doc, {
                        let mut opts = mongodb::options::ReplaceOptions::default();
                        opts.upsert = Some(true);
                        opts
                    })
                    .await
                    .map_err(|e| DocStoreError::Backend(e.to_string()))?;
            }
            Ok(())
        }

        async fn mget(&self, ids: &[&str]) -> DocStoreResult<Vec<Option<StoredDocument>>> {
            let mut result = Vec::new();
            for id in ids {
                let found = self
                    .collection
                    .find_one(doc! { "id": id }, None)
                    .await
                    .map_err(|e| DocStoreError::Backend(e.to_string()))?;
                match found {
                    None => result.push(None),
                    Some(bson_doc) => {
                        let json = serde_json::to_string(&bson_doc)
                            .map_err(|e| DocStoreError::Serialization(e.to_string()))?;
                        let doc: StoredDocument = serde_json::from_str(&json)
                            .map_err(|e| DocStoreError::Serialization(e.to_string()))?;
                        result.push(Some(doc));
                    }
                }
            }
            Ok(result)
        }

        async fn mdelete(&self, ids: &[&str]) -> DocStoreResult<()> {
            for id in ids {
                self.collection
                    .delete_one(doc! { "id": id }, None)
                    .await
                    .map_err(|e| DocStoreError::Backend(e.to_string()))?;
            }
            Ok(())
        }

        async fn yield_keys(&self) -> DocStoreResult<Vec<String>> {
            let ids = self
                .collection
                .distinct("id", None, None)
                .await
                .map_err(|e| DocStoreError::Backend(e.to_string()))?;
            Ok(ids
                .into_iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect())
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_in_memory_doc_store() {
        let store = InMemoryDocStore::new();
        let doc = StoredDocument {
            id: "d1".to_string(),
            text: "hello world".to_string(),
            metadata: HashMap::new(),
        };
        store.set(doc.clone()).await.unwrap();
        assert_eq!(store.len(), 1);

        let got = store.get("d1").await.unwrap();
        assert!(got.is_some());
        assert_eq!(got.unwrap().text, "hello world");

        let keys = store.yield_keys().await.unwrap();
        assert!(keys.contains(&"d1".to_string()));

        store.delete("d1").await.unwrap();
        assert_eq!(store.len(), 0);
    }

    #[tokio::test]
    async fn test_local_file_doc_store() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LocalFileDocStore::new(tmp.path().to_str().unwrap());
        let doc = StoredDocument {
            id: "file_doc".to_string(),
            text: "test content".to_string(),
            metadata: HashMap::new(),
        };
        store.set(doc).await.unwrap();
        let got = store.get("file_doc").await.unwrap();
        assert!(got.is_some());
        assert_eq!(got.unwrap().text, "test content");
    }
}
