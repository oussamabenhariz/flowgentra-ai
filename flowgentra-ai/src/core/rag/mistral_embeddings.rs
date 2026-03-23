//! Mistral Embeddings Provider
//!
//! Generates embeddings via the Mistral API using the `mistral-embed` model.
//! Uses the same API key as the LLM provider — no extra setup needed.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::embeddings::{EmbeddingError, EmbeddingsProvider};

/// Mistral embeddings provider — calls the Mistral embeddings API
pub struct MistralEmbeddings {
    client: reqwest::Client,
    api_key: String,
    model: String,
    dimension: usize,
}

#[derive(Debug, Serialize)]
struct MistralEmbedRequest {
    model: String,
    input: Vec<String>,
    encoding_format: String,
}

#[derive(Debug, Deserialize)]
struct MistralEmbedResponse {
    data: Vec<MistralEmbedData>,
}

#[derive(Debug, Deserialize)]
struct MistralEmbedData {
    embedding: Vec<f32>,
}

impl MistralEmbeddings {
    /// Create a new Mistral embeddings provider
    pub fn new(api_key: impl Into<String>, model: Option<String>) -> Self {
        let model = model.unwrap_or_else(|| "mistral-embed".to_string());
        let dimension = match model.as_str() {
            "mistral-embed" => 1024,
            _ => 1024,
        };

        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model,
            dimension,
        }
    }
}

#[async_trait]
impl EmbeddingsProvider for MistralEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let results = self.embed_batch(vec![text]).await?;
        results.into_iter().next().ok_or(EmbeddingError::EmptyResponse)
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let body = MistralEmbedRequest {
            model: self.model.clone(),
            input: texts.iter().map(|t| t.to_string()).collect(),
            encoding_format: "float".to_string(),
        };

        let resp = self
            .client
            .post("https://api.mistral.ai/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbeddingError::Network(format!("Mistral request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();

            return Err(match status {
                401 => EmbeddingError::AuthError(text),
                429 => EmbeddingError::RateLimited { retry_after_ms: None },
                _ => EmbeddingError::ApiError { status, message: text },
            });
        }

        let result: MistralEmbedResponse = resp.json().await.map_err(|e| {
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
    fn test_mistral_embeddings_defaults() {
        let emb = MistralEmbeddings::new("test-key", None);
        assert_eq!(emb.model, "mistral-embed");
        assert_eq!(emb.get_dimension(), 1024);
    }
}
