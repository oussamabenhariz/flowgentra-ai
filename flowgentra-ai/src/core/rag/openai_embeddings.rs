//! OpenAI Embeddings Provider
//!
//! Generates real embeddings via the OpenAI API.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::embeddings::{EmbeddingError, EmbeddingsProvider};

/// OpenAI embeddings provider — calls the OpenAI embeddings API
pub struct OpenAIEmbeddings {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dimension: usize,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl OpenAIEmbeddings {
    /// Create a new OpenAI embeddings provider
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let model = model.into();
        let dimension = match model.as_str() {
            "text-embedding-3-large" => 3072,
            "text-embedding-3-small" => 1536,
            "text-embedding-ada-002" => 1536,
            _ => 1536,
        };

        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model,
            dimension,
        }
    }

    /// Create with a specific dimension override
    pub fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = dim;
        self
    }
}

#[async_trait]
impl EmbeddingsProvider for OpenAIEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let results = self.embed_batch(vec![text]).await?;
        results.into_iter().next().ok_or(EmbeddingError::EmptyResponse)
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let body = EmbeddingRequest {
            model: self.model.clone(),
            input: texts.iter().map(|t| t.to_string()).collect(),
        };

        let resp = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbeddingError::Network(format!("OpenAI request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();

            return Err(match status {
                401 => EmbeddingError::AuthError(text),
                429 => EmbeddingError::RateLimited { retry_after_ms: None },
                _ => EmbeddingError::ApiError { status, message: text },
            });
        }

        let result: EmbeddingResponse = resp.json().await.map_err(|e| {
            EmbeddingError::ApiError {
                status: 0,
                message: format!("Failed to parse response: {}", e),
            }
        })?;

        Ok(result.data.into_iter().map(|d| d.embedding).collect())
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_embeddings_dimensions() {
        let emb = OpenAIEmbeddings::new("test-key", "text-embedding-3-small");
        assert_eq!(emb.get_dimension(), 1536);

        let emb = OpenAIEmbeddings::new("test-key", "text-embedding-3-large");
        assert_eq!(emb.get_dimension(), 3072);
    }

    #[test]
    fn test_openai_embeddings_custom_dimension() {
        let emb =
            OpenAIEmbeddings::new("test-key", "text-embedding-3-small").with_dimension(512);
        assert_eq!(emb.get_dimension(), 512);
    }
}
