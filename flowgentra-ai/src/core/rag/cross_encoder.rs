//! CrossEncoder Reranker — reranks results using a cross-encoder model via HTTP API
//!
//! Cross-encoders jointly encode (query, document) pairs and produce a relevance
//! score. They are more accurate than bi-encoders for reranking but slower.
//!
//! Supports any API that accepts `{query, texts}` and returns `{scores}`.

use async_trait::async_trait;

use super::reranker::Reranker;
use super::vector_db::SearchResult;

/// Cross-encoder reranker that calls an external scoring API.
///
/// Compatible with:
/// - HuggingFace Inference API (cross-encoder models)
/// - Custom hosted cross-encoder endpoints
/// - Cohere Rerank API
///
/// # Example
/// ```ignore
/// let reranker = CrossEncoderReranker::new(
///     "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-6-v2",
///     Some("hf_...".to_string()),
/// );
/// let reranked = reranker.rerank("query", results).await?;
/// ```
pub struct CrossEncoderReranker {
    endpoint: String,
    api_key: Option<String>,
    client: reqwest::Client,
    top_k: Option<usize>,
}

impl CrossEncoderReranker {
    pub fn new(endpoint: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key,
            client: reqwest::Client::new(),
            top_k: None,
        }
    }

    /// Only return top-k results after reranking
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }

    /// Score a batch of (query, document) pairs via the API.
    /// Returns scores in the same order as the input documents.
    async fn score_pairs(&self, query: &str, texts: &[String]) -> Result<Vec<f32>, String> {
        // Build request payload (HuggingFace cross-encoder format)
        let payload = serde_json::json!({
            "inputs": texts.iter().map(|t| {
                serde_json::json!({"text": query, "text_pair": t})
            }).collect::<Vec<_>>()
        });

        let mut request = self.client.post(&self.endpoint).json(&payload);

        if let Some(key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {}", key));
        }

        let response = request
            .send()
            .await
            .map_err(|e| format!("CrossEncoder HTTP error: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("CrossEncoder API error ({}): {}", status, body));
        }

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("CrossEncoder parse error: {}", e))?;

        // Parse scores from response — handle both array-of-arrays and flat array formats
        let scores = if let Some(arr) = body.as_array() {
            arr.iter()
                .map(|v| {
                    // HuggingFace returns [[{"label":"LABEL_0","score":0.5},{"label":"LABEL_1","score":0.5}]]
                    if let Some(inner) = v.as_array() {
                        // Find the positive/LABEL_1 score
                        inner
                            .iter()
                            .find(|item| {
                                item.get("label")
                                    .and_then(|l| l.as_str())
                                    .map(|l| l == "LABEL_1" || l == "entailment")
                                    .unwrap_or(false)
                            })
                            .and_then(|item| item.get("score").and_then(|s| s.as_f64()))
                            .unwrap_or_else(|| {
                                // Fallback: use first score
                                inner
                                    .first()
                                    .and_then(|item| item.get("score").and_then(|s| s.as_f64()))
                                    .unwrap_or(0.0)
                            }) as f32
                    } else if let Some(score) = v.as_f64() {
                        // Simple flat array of scores
                        score as f32
                    } else if let Some(score) = v.get("score").and_then(|s| s.as_f64()) {
                        score as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            return Err("CrossEncoder: unexpected response format".to_string());
        };

        Ok(scores)
    }
}

#[async_trait]
impl Reranker for CrossEncoderReranker {
    async fn rerank(
        &self,
        query: &str,
        mut results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, String> {
        if results.is_empty() {
            return Ok(results);
        }

        let texts: Vec<String> = results.iter().map(|r| r.text.clone()).collect();
        let scores = self.score_pairs(query, &texts).await?;

        // Update scores
        for (result, score) in results.iter_mut().zip(scores) {
            result.score = score;
        }

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate if top_k set
        if let Some(k) = self.top_k {
            results.truncate(k);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_result(id: &str, text: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            text: text.to_string(),
            score,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_cross_encoder_creation() {
        let ce =
            CrossEncoderReranker::new("http://localhost:8080/rerank", Some("test-key".to_string()))
                .with_top_k(5);

        assert_eq!(ce.endpoint, "http://localhost:8080/rerank");
        assert_eq!(ce.top_k, Some(5));
    }
}
