//! Ollama Embeddings Provider
//!
//! Generates embeddings locally via the Ollama API (free, no API key needed).
//! Requires Ollama running locally: https://ollama.com
//!
//! Recommended models:
//!   - nomic-embed-text (768 dims, good quality)
//!   - all-minilm (384 dims, fast)
//!   - mxbai-embed-large (1024 dims, best quality)

use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};

use super::embeddings::{EmbeddingError, EmbeddingsProvider};

/// Ollama embeddings provider — calls the local Ollama API (free, no API key)
pub struct OllamaEmbeddings {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dimension: usize,
}

#[derive(Debug, Serialize)]
struct OllamaEmbedRequest {
    model: String,
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

impl OllamaEmbeddings {
    /// Create a new Ollama embeddings provider
    ///
    /// # Arguments
    /// * `model` - Model name (e.g. "nomic-embed-text", "all-minilm", "mxbai-embed-large")
    /// * `base_url` - Ollama API base URL (default: "http://localhost:11434")
    pub fn new(model: impl Into<String>, base_url: Option<String>) -> Self {
        let model = model.into();
        let dimension = match model.as_str() {
            "nomic-embed-text" => 768,
            "all-minilm" => 384,
            "mxbai-embed-large" => 1024,
            "snowflake-arctic-embed" => 1024,
            _ => 768,
        };

        Self {
            client: reqwest::Client::new(),
            base_url: base_url.unwrap_or_else(|| "http://localhost:11434".to_string()),
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
impl EmbeddingsProvider for OllamaEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let body = OllamaEmbedRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };

        let url = format!("{}/api/embeddings", self.base_url);

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                EmbeddingError::Network(format!(
                    "Ollama request failed (is Ollama running?): {}",
                    e
                ))
            })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();

            if status == 429 {
                return Err(EmbeddingError::RateLimited {
                    retry_after_ms: None,
                });
            }

            return Err(EmbeddingError::ApiError {
                status,
                message: format!("{} — try: ollama pull {}", text, self.model),
            });
        }

        let result: OllamaEmbedResponse = resp.json().await.map_err(|e| {
            EmbeddingError::ApiError {
                status: 0,
                message: format!("Failed to parse response: {}", e),
            }
        })?;

        Ok(result.embedding)
    }

    /// Ollama has no batch endpoint — use bounded concurrency (max 4 parallel)
    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let owned: Vec<String> = texts.into_iter().map(|t| t.to_string()).collect();

        let results: Vec<Result<Vec<f32>, EmbeddingError>> = stream::iter(owned)
            .map(|text| async move { self.embed(&text).await })
            .buffer_unordered(4)
            .collect()
            .await;

        results.into_iter().collect()
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ollama_embeddings_dimensions() {
        let emb = OllamaEmbeddings::new("nomic-embed-text", None);
        assert_eq!(emb.get_dimension(), 768);

        let emb = OllamaEmbeddings::new("all-minilm", None);
        assert_eq!(emb.get_dimension(), 384);

        let emb = OllamaEmbeddings::new("mxbai-embed-large", None);
        assert_eq!(emb.get_dimension(), 1024);
    }

    #[test]
    fn test_ollama_custom_url() {
        let emb =
            OllamaEmbeddings::new("nomic-embed-text", Some("http://myhost:11434".to_string()));
        assert_eq!(emb.base_url, "http://myhost:11434");
    }
}
