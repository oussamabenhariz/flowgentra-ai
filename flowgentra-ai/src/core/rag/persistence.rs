//! InMemoryVectorStore Persistence — save/load to disk as JSON
//!
//! Allows the in-memory store to survive process restarts by serializing
//! all documents (with embeddings) to a JSON file.

use std::path::Path;

use super::vector_db::{Document, InMemoryVectorStore, VectorStoreError};

impl InMemoryVectorStore {
    /// Save all documents to a JSON file on disk
    pub async fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), VectorStoreError> {
        let docs: Vec<Document> = self
            .documents
            .iter()
            .map(|entry| -> Document { entry.value().clone() })
            .collect();

        let path = path.as_ref().to_path_buf();
        tokio::task::spawn_blocking(move || {
            let json = serde_json::to_string_pretty(&docs).map_err(|e| {
                VectorStoreError::SerializationError(format!("Failed to serialize: {}", e))
            })?;
            std::fs::write(&path, json).map_err(|e| {
                VectorStoreError::Unknown(format!("Failed to write file: {}", e))
            })?;
            Ok(())
        })
        .await
        .map_err(|e| VectorStoreError::Unknown(format!("Task join error: {}", e)))?
    }

    /// Load documents from a JSON file on disk
    ///
    /// Replaces any existing documents in the store.
    pub async fn load_from_file(&self, path: impl AsRef<Path>) -> Result<usize, VectorStoreError> {
        let path = path.as_ref().to_path_buf();
        let docs: Vec<Document> = tokio::task::spawn_blocking(move || {
            let json = std::fs::read_to_string(&path).map_err(|e| {
                VectorStoreError::Unknown(format!("Failed to read file: {}", e))
            })?;
            let docs: Vec<Document> = serde_json::from_str(&json).map_err(|e| {
                VectorStoreError::SerializationError(format!("Failed to deserialize: {}", e))
            })?;
            Ok::<_, VectorStoreError>(docs)
        })
        .await
        .map_err(|e| VectorStoreError::Unknown(format!("Task join error: {}", e)))??;

        self.documents.clear();
        let count = docs.len();
        for doc in docs {
            self.documents.insert(doc.id.clone(), doc);
        }
        Ok(count)
    }

    /// Number of documents currently stored
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Whether the store is empty
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_save_and_load() {
        let store = InMemoryVectorStore::new();

        let mut doc = Document::new("test-1", "hello world");
        doc.embedding = Some(vec![1.0, 2.0, 3.0]);
        doc.metadata
            .insert("source".to_string(), serde_json::json!("test.pdf"));
        store.documents.insert(doc.id.clone(), doc);

        let mut doc2 = Document::new("test-2", "goodbye world");
        doc2.embedding = Some(vec![4.0, 5.0, 6.0]);
        store.documents.insert(doc2.id.clone(), doc2);

        let tmp = std::env::temp_dir().join("flowgentra_test_store.json");

        store.save_to_file(&tmp).await.unwrap();

        // Load into a fresh store
        let store2 = InMemoryVectorStore::new();
        let count = store2.load_from_file(&tmp).await.unwrap();
        assert_eq!(count, 2);
        assert_eq!(store2.len(), 2);

        let loaded = store2.documents.get("test-1").unwrap();
        assert_eq!(loaded.text, "hello world");
        assert_eq!(loaded.embedding, Some(vec![1.0, 2.0, 3.0]));

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_len_and_is_empty() {
        let store = InMemoryVectorStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store
            .documents
            .insert("a".to_string(), Document::new("a", "text"));
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }
}
