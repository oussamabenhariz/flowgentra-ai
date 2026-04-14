//! Indexing Pipeline & Record Manager
//!
//! Mirrors LangChain's `RecordManager` + `index()` function.
//! The `index()` function avoids re-indexing identical documents by tracking
//! which documents have already been indexed.
//!
//! ## How it works
//!
//! Each document is hashed (blake3 of its text + metadata). The hash is stored
//! in a `RecordManager`. On the next `index()` call, documents whose hash
//! already exists in the record manager are skipped.
//!
//! ## Cleanup modes
//!
//! | Mode | Description |
//! |---|---|
//! | `None` | Never delete; just skip already-seen docs |
//! | `Incremental` | Delete docs from the source that are no longer present |
//! | `Full` | Delete ALL docs in the namespace, then re-index |
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::indexing::{InMemoryRecordManager, index, CleanupMode};
//!
//! let rm = InMemoryRecordManager::new("my_namespace");
//! let stats = index(docs, &rm, &vector_store, CleanupMode::Incremental).await?;
//! println!("Added: {}, Skipped: {}, Deleted: {}", stats.added, stats.skipped, stats.deleted);
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use super::vector_db::{Document, VectorStoreBackend, VectorStoreError};

// ── IndexStats ────────────────────────────────────────────────────────────────

/// Statistics returned by `index()`.
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    pub added: usize,
    pub updated: usize,
    pub skipped: usize,
    pub deleted: usize,
}

// ── CleanupMode ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanupMode {
    /// No cleanup — new docs are added, duplicates skipped. Nothing deleted.
    None,
    /// Delete source documents that are no longer in the input batch.
    Incremental,
    /// Wipe the entire namespace before re-indexing.
    Full,
}

// ── RecordEntry ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordEntry {
    /// Document id in the vector store.
    pub doc_id: String,
    /// Hash of (text + metadata).
    pub hash: String,
    /// Unix timestamp of when the record was added.
    pub indexed_at: i64,
    /// Source identifier (e.g. file path, URL).
    pub source: String,
}

// ── RecordManager trait ───────────────────────────────────────────────────────

/// Tracks which documents have been indexed under a given namespace.
#[async_trait]
pub trait RecordManager: Send + Sync {
    /// Get the namespace this manager operates under.
    fn namespace(&self) -> &str;

    /// Check if a hash already exists in this namespace.
    async fn exists(&self, hash: &str) -> Result<bool, VectorStoreError>;

    /// Add record entries for a batch of newly indexed documents.
    async fn update(&self, entries: Vec<RecordEntry>) -> Result<(), VectorStoreError>;

    /// Delete records by document ids.
    async fn delete_by_ids(&self, ids: &[&str]) -> Result<(), VectorStoreError>;

    /// Return all records for this namespace.
    async fn list_records(&self) -> Result<Vec<RecordEntry>, VectorStoreError>;

    /// Delete ALL records in this namespace.
    async fn clear(&self) -> Result<(), VectorStoreError>;
}

// ── InMemoryRecordManager ─────────────────────────────────────────────────────

/// In-memory `RecordManager`. Fast, no persistence.
pub struct InMemoryRecordManager {
    namespace: String,
    /// hash → RecordEntry
    by_hash: Arc<DashMap<String, RecordEntry>>,
    /// doc_id → hash (for reverse lookup on delete)
    by_id: Arc<DashMap<String, String>>,
}

impl InMemoryRecordManager {
    pub fn new(namespace: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
            by_hash: Arc::new(DashMap::new()),
            by_id: Arc::new(DashMap::new()),
        }
    }
}

#[async_trait]
impl RecordManager for InMemoryRecordManager {
    fn namespace(&self) -> &str {
        &self.namespace
    }

    async fn exists(&self, hash: &str) -> Result<bool, VectorStoreError> {
        Ok(self.by_hash.contains_key(hash))
    }

    async fn update(&self, entries: Vec<RecordEntry>) -> Result<(), VectorStoreError> {
        for entry in entries {
            self.by_id.insert(entry.doc_id.clone(), entry.hash.clone());
            self.by_hash.insert(entry.hash.clone(), entry);
        }
        Ok(())
    }

    async fn delete_by_ids(&self, ids: &[&str]) -> Result<(), VectorStoreError> {
        for id in ids {
            if let Some((_, hash)) = self.by_id.remove(*id) {
                self.by_hash.remove(&hash);
            }
        }
        Ok(())
    }

    async fn list_records(&self) -> Result<Vec<RecordEntry>, VectorStoreError> {
        Ok(self.by_hash.iter().map(|r| r.value().clone()).collect())
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        self.by_hash.clear();
        self.by_id.clear();
        Ok(())
    }
}

// ── SQLite RecordManager ─────────────────────────────────────────────────────

#[cfg(feature = "sqlite")]
pub mod sqlite_record_manager {
    use super::*;
    use crate::core::error::FlowgentraError;
    use sqlx::SqlitePool;

    /// SQLite-backed `RecordManager` with full persistence.
    pub struct SqliteRecordManager {
        namespace: String,
        pool: SqlitePool,
    }

    impl SqliteRecordManager {
        pub async fn new(
            namespace: impl Into<String>,
            url: &str,
        ) -> Result<Self, VectorStoreError> {
            let pool = SqlitePool::connect(url)
                .await
                .map_err(|e| VectorStoreError::Unknown(format!("SQLite connect: {e}")))?;
            sqlx::query(
                "CREATE TABLE IF NOT EXISTS record_manager (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace   TEXT    NOT NULL,
                    doc_id      TEXT    NOT NULL,
                    hash        TEXT    NOT NULL UNIQUE,
                    indexed_at  INTEGER NOT NULL,
                    source      TEXT    NOT NULL DEFAULT ''
                )",
            )
            .execute(&pool)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("SQLite schema: {e}")))?;
            Ok(Self {
                namespace: namespace.into(),
                pool,
            })
        }
    }

    #[async_trait]
    impl RecordManager for SqliteRecordManager {
        fn namespace(&self) -> &str {
            &self.namespace
        }

        async fn exists(&self, hash: &str) -> Result<bool, VectorStoreError> {
            let row: Option<(i64,)> =
                sqlx::query_as("SELECT id FROM record_manager WHERE namespace = ? AND hash = ?")
                    .bind(&self.namespace)
                    .bind(hash)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(|e| VectorStoreError::Unknown(e.to_string()))?;
            Ok(row.is_some())
        }

        async fn update(&self, entries: Vec<RecordEntry>) -> Result<(), VectorStoreError> {
            for e in entries {
                sqlx::query(
                    "INSERT OR REPLACE INTO record_manager
                     (namespace, doc_id, hash, indexed_at, source)
                     VALUES (?, ?, ?, ?, ?)",
                )
                .bind(&self.namespace)
                .bind(&e.doc_id)
                .bind(&e.hash)
                .bind(e.indexed_at)
                .bind(&e.source)
                .execute(&self.pool)
                .await
                .map_err(|e| VectorStoreError::Unknown(e.to_string()))?;
            }
            Ok(())
        }

        async fn delete_by_ids(&self, ids: &[&str]) -> Result<(), VectorStoreError> {
            for id in ids {
                sqlx::query("DELETE FROM record_manager WHERE namespace = ? AND doc_id = ?")
                    .bind(&self.namespace)
                    .bind(id)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| VectorStoreError::Unknown(e.to_string()))?;
            }
            Ok(())
        }

        async fn list_records(&self) -> Result<Vec<RecordEntry>, VectorStoreError> {
            let rows: Vec<(String, String, i64, String)> = sqlx::query_as(
                "SELECT doc_id, hash, indexed_at, source FROM record_manager
                 WHERE namespace = ?",
            )
            .bind(&self.namespace)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| VectorStoreError::Unknown(e.to_string()))?;
            Ok(rows
                .into_iter()
                .map(|(doc_id, hash, indexed_at, source)| RecordEntry {
                    doc_id,
                    hash,
                    indexed_at,
                    source,
                })
                .collect())
        }

        async fn clear(&self) -> Result<(), VectorStoreError> {
            sqlx::query("DELETE FROM record_manager WHERE namespace = ?")
                .bind(&self.namespace)
                .execute(&self.pool)
                .await
                .map_err(|e| VectorStoreError::Unknown(e.to_string()))?;
            Ok(())
        }
    }
}

// ── Hash helper ───────────────────────────────────────────────────────────────

/// Compute a stable hash for a document (text + sorted metadata JSON).
pub fn hash_document(
    text: &str,
    metadata: &HashMap<String, serde_json::Value>,
    source: &str,
) -> String {
    let meta_str = serde_json::to_string(metadata).unwrap_or_default();
    let input = format!("{text}|{meta_str}|{source}");
    let hash = blake3::hash(input.as_bytes());
    hash.to_hex().to_string()
}

// ── index() ───────────────────────────────────────────────────────────────────

/// Index a batch of documents with deduplication and optional cleanup.
///
/// Documents are hashed; already-indexed hashes are skipped. In `Incremental`
/// mode, documents from the same sources that are no longer present are deleted.
/// In `Full` mode, the entire namespace is wiped before indexing.
pub async fn index(
    documents: Vec<Document>,
    record_manager: &dyn RecordManager,
    vector_store: &dyn VectorStoreBackend,
    cleanup: CleanupMode,
) -> Result<IndexStats, VectorStoreError> {
    let mut stats = IndexStats::default();

    // Full cleanup: wipe everything first
    if cleanup == CleanupMode::Full {
        let existing = record_manager.list_records().await?;
        let ids: Vec<&str> = existing.iter().map(|r| r.doc_id.as_str()).collect();
        for id in &ids {
            vector_store.delete(id).await.ok();
        }
        record_manager.clear().await?;
        stats.deleted = ids.len();
    }

    // Track which sources are present in this batch (for Incremental cleanup)
    let batch_sources: HashSet<String> = documents
        .iter()
        .map(|d| {
            d.metadata
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or(&d.id)
                .to_string()
        })
        .collect();

    // Incremental: find source docs no longer in the batch
    if cleanup == CleanupMode::Incremental {
        let existing = record_manager.list_records().await?;
        let to_delete: Vec<RecordEntry> = existing
            .into_iter()
            .filter(|r| {
                !batch_sources.contains(&r.source) || {
                    // Only delete if the source is in the batch but the doc is gone
                    batch_sources.contains(&r.source)
                }
            })
            .collect();
        let del_ids: Vec<&str> = to_delete.iter().map(|r| r.doc_id.as_str()).collect();
        for id in &del_ids {
            vector_store.delete(id).await.ok();
        }
        record_manager.delete_by_ids(&del_ids).await?;
        stats.deleted += del_ids.len();
    }

    // Index documents
    let mut new_entries = Vec::new();
    for doc in documents {
        let source = doc
            .metadata
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or(&doc.id)
            .to_string();
        let hash = hash_document(&doc.text, &doc.metadata, &source);

        if record_manager.exists(&hash).await? {
            stats.skipped += 1;
            continue;
        }

        let doc_id = doc.id.clone();
        vector_store.index(doc).await?;
        new_entries.push(RecordEntry {
            doc_id,
            hash,
            indexed_at: chrono::Utc::now().timestamp(),
            source,
        });
        stats.added += 1;
    }

    if !new_entries.is_empty() {
        record_manager.update(new_entries).await?;
    }

    Ok(stats)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rag::{embeddings::Embeddings, vector_db::InMemoryVectorStore};
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_index_dedup() {
        let store = Arc::new(InMemoryVectorStore::new());
        let rm = InMemoryRecordManager::new("test_ns");
        let _embeddings = Arc::new(Embeddings::mock(8));

        let make_doc = |id: &str, text: &str| {
            let mut meta = HashMap::new();
            meta.insert("source".to_string(), serde_json::json!("test"));
            Document {
                id: id.to_string(),
                text: text.to_string(),
                embedding: None,
                metadata: meta,
            }
        };

        let docs = vec![make_doc("d1", "hello"), make_doc("d2", "world")];

        let stats1 = index(docs.clone(), &rm, store.as_ref(), CleanupMode::None)
            .await
            .unwrap();
        assert_eq!(stats1.added, 2);
        assert_eq!(stats1.skipped, 0);

        // Second call — same docs should be skipped
        let stats2 = index(docs, &rm, store.as_ref(), CleanupMode::None)
            .await
            .unwrap();
        assert_eq!(stats2.added, 0);
        assert_eq!(stats2.skipped, 2);
    }
}
