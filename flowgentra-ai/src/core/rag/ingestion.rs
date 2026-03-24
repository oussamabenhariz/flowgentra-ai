//! Streaming Ingestion — process large document sets without loading all into memory
//!
//! Provides an iterator-based pipeline: chunk → embed → index, one batch at a time.
//! This is ideal for ingesting hundreds of PDFs or very large text files.

use std::sync::Arc;

use super::embeddings::{EmbeddingError, Embeddings};
use super::vector_db::{Document, VectorStore, VectorStoreError};

/// Ingestion statistics
#[derive(Debug, Clone, Default)]
pub struct IngestionStats {
    pub documents_processed: usize,
    pub chunks_indexed: usize,
    pub errors: Vec<String>,
}

/// Streaming ingestion pipeline.
///
/// Processes documents in batches to bound memory usage.
pub struct IngestionPipeline {
    store: Arc<VectorStore>,
    embeddings: Arc<Embeddings>,
    batch_size: usize,
}

impl IngestionPipeline {
    pub fn new(store: Arc<VectorStore>, embeddings: Arc<Embeddings>, batch_size: usize) -> Self {
        Self {
            store,
            embeddings,
            batch_size: batch_size.max(1),
        }
    }

    /// Ingest a stream of (id, text) pairs.
    ///
    /// Chunks are embedded and indexed in batches. Errors on individual
    /// documents are collected but don't stop the pipeline.
    pub async fn ingest(
        &self,
        documents: Vec<(String, String)>,
    ) -> Result<IngestionStats, VectorStoreError> {
        let mut stats = IngestionStats::default();

        for batch in documents.chunks(self.batch_size) {
            let texts: Vec<&str> = batch.iter().map(|(_, text)| text.as_str()).collect();

            match self.embeddings.embed_batch(texts).await {
                Ok(embeddings) => {
                    for ((id, text), embedding) in batch.iter().zip(embeddings) {
                        let mut doc = Document::new(id.as_str(), text.as_str());
                        doc.embedding = Some(embedding);

                        match self
                            .store
                            .index_document(&doc.id, &doc.text, serde_json::json!({}))
                            .await
                        {
                            Ok(_) => {
                                stats.chunks_indexed += 1;
                            }
                            Err(e) => {
                                stats
                                    .errors
                                    .push(format!("Index error for '{}': {}", id, e));
                            }
                        }
                    }
                }
                Err(e) => {
                    let ids: Vec<&str> = batch.iter().map(|(id, _)| id.as_str()).collect();
                    stats.errors.push(format!(
                        "Embedding batch error for [{}]: {}",
                        ids.join(", "),
                        e
                    ));
                }
            }

            stats.documents_processed += batch.len();
        }

        Ok(stats)
    }

    /// Ingest pre-chunked documents with metadata.
    ///
    /// Each item is `(id, text, metadata)`. Useful when chunks already have
    /// source/page info attached.
    pub async fn ingest_with_metadata(
        &self,
        documents: Vec<(String, String, serde_json::Value)>,
    ) -> Result<IngestionStats, VectorStoreError> {
        let mut stats = IngestionStats::default();

        for batch in documents.chunks(self.batch_size) {
            let texts: Vec<&str> = batch.iter().map(|(_, text, _)| text.as_str()).collect();

            match self.embeddings.embed_batch(texts).await {
                Ok(embeddings) => {
                    for ((id, text, metadata), embedding) in batch.iter().zip(embeddings) {
                        let mut doc = Document::new(id.as_str(), text.as_str());
                        doc.embedding = Some(embedding);

                        if let serde_json::Value::Object(map) = metadata {
                            doc.metadata =
                                map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                        }

                        match self
                            .store
                            .index_document(&doc.id, &doc.text, metadata.clone())
                            .await
                        {
                            Ok(_) => stats.chunks_indexed += 1,
                            Err(e) => stats
                                .errors
                                .push(format!("Index error for '{}': {}", id, e)),
                        }
                    }
                }
                Err(EmbeddingError::RateLimited { retry_after_ms }) => {
                    let wait = retry_after_ms.unwrap_or(1000);
                    tokio::time::sleep(std::time::Duration::from_millis(wait)).await;
                    // Re-try this batch
                    let texts: Vec<&str> = batch.iter().map(|(_, text, _)| text.as_str()).collect();
                    if let Ok(embeddings) = self.embeddings.embed_batch(texts).await {
                        for ((id, text, metadata), embedding) in batch.iter().zip(embeddings) {
                            let mut doc = Document::new(id.as_str(), text.as_str());
                            doc.embedding = Some(embedding);
                            if let serde_json::Value::Object(map) = metadata {
                                doc.metadata =
                                    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                            }
                            match self
                                .store
                                .index_document(&doc.id, &doc.text, metadata.clone())
                                .await
                            {
                                Ok(_) => stats.chunks_indexed += 1,
                                Err(e) => stats
                                    .errors
                                    .push(format!("Index error for '{}': {}", id, e)),
                            }
                        }
                    }
                }
                Err(e) => {
                    let ids: Vec<&str> = batch.iter().map(|(id, _, _)| id.as_str()).collect();
                    stats.errors.push(format!(
                        "Embedding batch error for [{}]: {}",
                        ids.join(", "),
                        e
                    ));
                }
            }

            stats.documents_processed += batch.len();
        }

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingestion_stats_default() {
        let stats = IngestionStats::default();
        assert_eq!(stats.documents_processed, 0);
        assert_eq!(stats.chunks_indexed, 0);
        assert!(stats.errors.is_empty());
    }
}
