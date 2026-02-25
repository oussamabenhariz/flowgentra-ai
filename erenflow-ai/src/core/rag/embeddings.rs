//! Embedding Generation and Management
//!
//! Handles creation and management of vector embeddings from text.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
            EmbeddingModel::HuggingFace { .. } => 768, // Common size, varies by model
            EmbeddingModel::Local { .. } => 768,
        }
    }
}

/// Embeddings provider trait
#[async_trait]
pub trait EmbeddingsProvider: Send + Sync {
    /// Generate embedding for a single text
    async fn embed(&self, text: &str) -> Result<Vec<f32>, String>;

    /// Generate embeddings for multiple texts (batch)
    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, String>;

    /// Get embedding dimension
    fn get_dimension(&self) -> usize;
}

/// Mock embeddings for testing (creates dummy vectors)
pub struct MockEmbeddings {
    dimension: usize,
}

impl MockEmbeddings {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    fn hash_string(s: &str) -> f32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        let hash = hasher.finish();
        ((hash % 1000) as f32) / 1000.0
    }
}

#[async_trait]
impl EmbeddingsProvider for MockEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let base = Self::hash_string(text);
        let mut embedding = vec![base; self.dimension];

        // Make it vary slightly for different texts
        for (i, item) in embedding.iter_mut().enumerate() {
            *item = (*item + (i as f32) * 0.001).min(1.0);
        }

        Ok(embedding)
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, String> {
        let mut results = Vec::new();
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
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        self.provider.embed(text).await
    }

    /// Generate embeddings for multiple texts
    pub async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, String> {
        self.provider.embed_batch(texts).await
    }

    /// Get embedding dimension
    pub fn get_dimension(&self) -> usize {
        self.provider.get_dimension()
    }

    /// Pre-generate embeddings for a collection of documents
    pub async fn embed_documents(
        &self,
        documents: Vec<(String, String)>,
    ) -> Result<Vec<(String, String, Vec<f32>)>, String> {
        let mut results = Vec::new();

        for (id, text) in documents {
            let embedding = self.embed(&text).await?;
            results.push((id, text, embedding));
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
        let embeddings = Embeddings::mock(1536);
        let embedding = embeddings.embed("test text").await.unwrap();

        assert_eq!(embedding.len(), 1536);
        assert!(embedding.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    #[tokio::test]
    async fn test_batch_embeddings() {
        let embeddings = Embeddings::mock(768);
        let texts = vec!["text 1", "text 2", "text 3"];
        let result = embeddings.embed_batch(texts).await.unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|e| e.len() == 768));
    }
}
