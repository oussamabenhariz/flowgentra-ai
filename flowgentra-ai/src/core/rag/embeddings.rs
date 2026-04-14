//! Embedding Generation and Management
//!
//! Handles creation and management of vector embeddings from text.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Structured error type for embedding operations
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Network error: {0}")]
    Network(String),

    #[error("Rate limited{}", match .retry_after_ms {
        Some(ms) => format!(" (retry after {}ms)", ms),
        None => String::new(),
    })]
    RateLimited { retry_after_ms: Option<u64> },

    #[error("Authentication failed: {0}")]
    AuthError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Provider error ({status}): {message}")]
    ApiError { status: u16, message: String },

    #[error("Empty response from provider")]
    EmptyResponse,
}

/// Type of embedding model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EmbeddingModel {
    #[serde(rename = "openai")]
    OpenAI { model: String },
    #[serde(rename = "cohere")]
    Cohere { model: String },
    #[serde(rename = "huggingface")]
    HuggingFace { model: String },
    #[serde(rename = "local")]
    Local { model_path: String },
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        EmbeddingModel::OpenAI {
            model: "text-embedding-3-small".to_string(),
        }
    }
}

impl EmbeddingModel {
    pub fn openai(model: &str) -> Self {
        Self::OpenAI {
            model: model.to_string(),
        }
    }

    pub fn cohere(model: &str) -> Self {
        Self::Cohere {
            model: model.to_string(),
        }
    }

    pub fn huggingface(model: &str) -> Self {
        Self::HuggingFace {
            model: model.to_string(),
        }
    }

    pub fn get_dimension(&self) -> usize {
        match self {
            EmbeddingModel::OpenAI { model } => match model.as_str() {
                "text-embedding-3-large" => 3072,
                "text-embedding-3-small" => 1536,
                "text-embedding-ada-002" => 1536,
                _ => 1536,
            },
            EmbeddingModel::Cohere { .. } => 4096,
            EmbeddingModel::HuggingFace { .. } => 768,
            EmbeddingModel::Local { .. } => 768,
        }
    }
}

/// Embeddings provider trait
#[async_trait]
pub trait EmbeddingsProvider: Send + Sync {
    /// Generate embedding for a single text
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Generate embeddings for multiple texts (batch)
    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Get embedding dimension
    fn get_dimension(&self) -> usize;
}

/// Mock embeddings for testing — uses per-dimension hashing for realistic separation
pub struct MockEmbeddings {
    dimension: usize,
}

impl MockEmbeddings {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn hash_with_seed(s: &str, seed: u64) -> f32 {
        // Use blake3 for stable, reproducible mock embeddings across Rust versions.
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed.to_le_bytes());
        hasher.update(s.as_bytes());
        let hash = hasher.finalize();
        let raw = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        (raw % 10000) as f32 / 10000.0
    }
}

#[async_trait]
impl EmbeddingsProvider for MockEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let embedding: Vec<f32> = (0..self.dimension)
            .map(|i| Self::hash_with_seed(text, i as u64))
            .collect();

        // Normalize to unit vector for meaningful cosine similarity
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return Ok(vec![0.0; self.dimension]);
        }
        Ok(embedding.into_iter().map(|x| x / norm).collect())
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }
}

/// Embeddings manager
pub struct Embeddings {
    provider: Arc<dyn EmbeddingsProvider>,
}

impl Embeddings {
    pub fn new(provider: Arc<dyn EmbeddingsProvider>) -> Self {
        Self { provider }
    }

    pub fn mock(dimension: usize) -> Self {
        Self {
            provider: Arc::new(MockEmbeddings::new(dimension)),
        }
    }

    /// Generate embedding for text
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        self.provider.embed(text).await
    }

    /// Generate embeddings for multiple texts
    pub async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        self.provider.embed_batch(texts).await
    }

    /// Get embedding dimension
    pub fn get_dimension(&self) -> usize {
        self.provider.get_dimension()
    }

    /// Pre-generate embeddings for a collection of documents (batched)
    ///
    /// Sends texts in batches to reduce API calls.
    pub async fn embed_documents(
        &self,
        documents: Vec<(String, String)>,
        batch_size: usize,
    ) -> Result<Vec<(String, String, Vec<f32>)>, EmbeddingError> {
        let batch_size = batch_size.max(1);
        let mut results = Vec::with_capacity(documents.len());

        for batch in documents.chunks(batch_size) {
            let texts: Vec<&str> = batch.iter().map(|(_, text)| text.as_str()).collect();
            let embeddings = self.embed_batch(texts).await?;

            for ((id, text), emb) in batch.iter().zip(embeddings) {
                results.push((id.clone(), text.clone(), emb));
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dimensions() {
        assert_eq!(
            EmbeddingModel::OpenAI {
                model: "text-embedding-3-large".to_string()
            }
            .get_dimension(),
            3072
        );

        assert_eq!(
            EmbeddingModel::OpenAI {
                model: "text-embedding-3-small".to_string()
            }
            .get_dimension(),
            1536
        );
    }

    #[tokio::test]
    async fn test_mock_embeddings() {
        let embeddings = Embeddings::mock(128);
        let embedding = embeddings.embed("test text").await.unwrap();

        assert_eq!(embedding.len(), 128);
        assert!(embedding.iter().all(|&v| v >= -1.0 && v <= 1.0));
    }

    #[tokio::test]
    async fn test_mock_embeddings_different_texts_differ() {
        let embeddings = Embeddings::mock(128);
        let e1 = embeddings.embed("Rust programming").await.unwrap();
        let e2 = embeddings.embed("Python cooking recipe").await.unwrap();

        // Cosine similarity should not be near 1.0
        let dot: f32 = e1.iter().zip(e2.iter()).map(|(a, b)| a * b).sum();
        assert!(
            dot < 0.95,
            "Different texts should have different embeddings, got cosine={dot}"
        );
    }

    #[tokio::test]
    async fn test_batch_embeddings() {
        let embeddings = Embeddings::mock(768);
        let texts = vec!["text 1", "text 2", "text 3"];
        let result = embeddings.embed_batch(texts).await.unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|e| e.len() == 768));
    }

    #[tokio::test]
    async fn test_embed_documents_batched() {
        let embeddings = Embeddings::mock(64);
        let docs = vec![
            ("a".into(), "doc a".into()),
            ("b".into(), "doc b".into()),
            ("c".into(), "doc c".into()),
        ];

        let results = embeddings.embed_documents(docs, 2).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, "a");
        assert_eq!(results[2].0, "c");
    }
}
