//! HuggingFace Embeddings — generate embeddings via HuggingFace Inference API
//!
//! Supports both the HuggingFace cloud Inference API and self-hosted
//! Text Embeddings Inference (TEI) servers.

use async_trait::async_trait;

use super::embeddings::{EmbeddingError, EmbeddingsProvider};

/// HuggingFace embedding provider.
///
/// # Example
/// ```ignore
/// let embeddings = HuggingFaceEmbeddings::new(
///     "sentence-transformers/all-MiniLM-L6-v2",
///     "hf_...",
/// );
/// let vector = embeddings.embed("Hello world").await?;
/// ```
pub struct HuggingFaceEmbeddings {
    #[allow(dead_code)]
    model: String,
    api_key: String,
    endpoint: String,
    dimension: usize,
    client: reqwest::Client,
}

impl HuggingFaceEmbeddings {
    /// Create a new HuggingFace embeddings provider using the Inference API.
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        let model = model.into();
        let endpoint = format!(
            "https://api-inference.huggingface.co/pipeline/feature-extraction/{}",
            model
        );
        let dimension = Self::guess_dimension(&model);
        Self {
            model,
            api_key: api_key.into(),
            endpoint,
            dimension,
            client: reqwest::Client::new(),
        }
    }

    /// Use a custom endpoint (e.g., self-hosted TEI server).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Override the embedding dimension (if auto-detection doesn't match).
    pub fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = dim;
        self
    }

    fn guess_dimension(model: &str) -> usize {
        let m = model.to_lowercase();
        if m.contains("minilm-l6") || m.contains("minilm-l12") {
            384
        } else if m.contains("bge-large") || m.contains("e5-large") {
            1024
        } else {
            768 // Default for mpnet, bge-base, bge-small, e5-base, e5-small, and most sentence-transformers
        }
    }

    async fn call_api(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let payload = serde_json::json!({
            "inputs": texts,
            "options": {"wait_for_model": true}
        });

        let response = self
            .client
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&payload)
            .send()
            .await
            .map_err(|e| EmbeddingError::Network(e.to_string()))?;

        let status = response.status().as_u16();

        if status == 401 || status == 403 {
            return Err(EmbeddingError::AuthError(
                "Invalid HuggingFace API key".to_string(),
            ));
        }

        if status == 429 {
            return Err(EmbeddingError::RateLimited {
                retry_after_ms: None,
            });
        }

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::ApiError {
                status,
                message: body,
            });
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| EmbeddingError::Network(format!("JSON parse error: {}", e)))?;

        // HuggingFace returns either:
        // - Array of arrays (batch): [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        // - Single array (single input): [0.1, 0.2, ...]
        // - Array of arrays of arrays (token-level): [[[0.1, ...], ...]]
        self.parse_embeddings(&body, texts.len())
    }

    fn parse_embeddings(
        &self,
        value: &serde_json::Value,
        expected_count: usize,
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if let Some(outer) = value.as_array() {
            if outer.is_empty() {
                return Err(EmbeddingError::EmptyResponse);
            }

            // Check if it's a batch of embeddings
            if let Some(first) = outer.first() {
                if let Some(inner) = first.as_array() {
                    // Check if it's token-level (array of arrays of arrays)
                    if inner.first().and_then(|v| v.as_array()).is_some() {
                        // Token-level: mean-pool each sentence
                        return outer
                            .iter()
                            .map(|sentence| self.mean_pool(sentence))
                            .collect();
                    }

                    // Batch of embeddings
                    if outer.len() == expected_count {
                        return outer
                            .iter()
                            .map(|emb| self.parse_float_array(emb))
                            .collect();
                    }
                }

                // Single embedding (non-batch)
                if first.is_f64() || first.is_i64() {
                    let emb = self.parse_float_array(value)?;
                    return Ok(vec![emb]);
                }
            }
        }

        Err(EmbeddingError::ApiError {
            status: 0,
            message: "Unexpected response format from HuggingFace".to_string(),
        })
    }

    fn parse_float_array(&self, value: &serde_json::Value) -> Result<Vec<f32>, EmbeddingError> {
        value
            .as_array()
            .ok_or_else(|| EmbeddingError::ApiError {
                status: 0,
                message: "Expected array of floats".to_string(),
            })?
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| EmbeddingError::ApiError {
                        status: 0,
                        message: "Expected float value".to_string(),
                    })
            })
            .collect()
    }

    fn mean_pool(&self, token_embeddings: &serde_json::Value) -> Result<Vec<f32>, EmbeddingError> {
        let tokens = token_embeddings
            .as_array()
            .ok_or(EmbeddingError::EmptyResponse)?;

        if tokens.is_empty() {
            return Err(EmbeddingError::EmptyResponse);
        }

        let dim = tokens
            .first()
            .and_then(|t| t.as_array())
            .map(|a| a.len())
            .unwrap_or(0);

        let mut pooled = vec![0.0f32; dim];
        let count = tokens.len() as f32;

        for token in tokens {
            if let Some(values) = token.as_array() {
                for (i, v) in values.iter().enumerate() {
                    if i < dim {
                        pooled[i] += v.as_f64().unwrap_or(0.0) as f32;
                    }
                }
            }
        }

        for v in &mut pooled {
            *v /= count;
        }

        // Normalize to unit vector
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut pooled {
                *v /= norm;
            }
        }

        Ok(pooled)
    }
}

#[async_trait]
impl EmbeddingsProvider for HuggingFaceEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let results = self.call_api(&[text]).await?;
        results
            .into_iter()
            .next()
            .ok_or(EmbeddingError::EmptyResponse)
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Process in batches of 32 to avoid payload limits
        let batch_size = 32;
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for batch in texts.chunks(batch_size) {
            let embeddings = self.call_api(batch).await?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_guessing() {
        assert_eq!(
            HuggingFaceEmbeddings::guess_dimension("sentence-transformers/all-MiniLM-L6-v2"),
            384
        );
        assert_eq!(
            HuggingFaceEmbeddings::guess_dimension("BAAI/bge-large-en"),
            1024
        );
        assert_eq!(
            HuggingFaceEmbeddings::guess_dimension("sentence-transformers/all-mpnet-base-v2"),
            768
        );
        assert_eq!(HuggingFaceEmbeddings::guess_dimension("unknown-model"), 768);
    }

    #[test]
    fn test_creation() {
        let hf = HuggingFaceEmbeddings::new("all-MiniLM-L6-v2", "hf_test_key").with_dimension(512);
        assert_eq!(hf.dimension, 512);
        assert_eq!(hf.model, "all-MiniLM-L6-v2");
    }

    #[test]
    fn test_custom_endpoint() {
        let hf =
            HuggingFaceEmbeddings::new("model", "key").with_endpoint("http://localhost:8080/embed");
        assert_eq!(hf.endpoint, "http://localhost:8080/embed");
    }
}
