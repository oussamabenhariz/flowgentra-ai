//! Re-ranking for RAG retrieval
//!
//! After initial vector recall (high recall, noisy), a re-ranker scores
//! each candidate more precisely. Supports LLM-based and score-based strategies.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::vector_db::SearchResult;

/// Re-ranking strategy
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RerankStrategy {
    /// No re-ranking (pass-through)
    #[serde(rename = "none")]
    #[default]
    None,

    /// Re-rank using an LLM to score query-document relevance
    #[serde(rename = "llm")]
    LLM,

    /// Re-rank using a cross-encoder model (via API)
    #[serde(rename = "cross_encoder")]
    CrossEncoder { model: String, endpoint: String },

    /// Reciprocal Rank Fusion — combine multiple result lists
    #[serde(rename = "rrf")]
    ReciprocalRankFusion { k: usize },
}

/// Trait for re-ranking search results
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Re-rank results given a query. Returns results in new order with updated scores.
    async fn rerank(
        &self,
        query: &str,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, String>;
}

/// Pass-through reranker — returns results unchanged
pub struct NoopReranker;

#[async_trait]
impl Reranker for NoopReranker {
    async fn rerank(
        &self,
        _query: &str,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, String> {
        Ok(results)
    }
}

/// LLM-based reranker — uses an LLM to score query-document relevance.
///
/// Sends each (query, document) pair to the LLM and asks it to rate relevance 0-10.
/// Requires a scoring function that wraps your LLM.
pub struct LLMReranker<F>
where
    F: Fn(
            String,
            String,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<f32, String>> + Send>>
        + Send
        + Sync,
{
    score_fn: F,
}

impl<F> LLMReranker<F>
where
    F: Fn(
            String,
            String,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<f32, String>> + Send>>
        + Send
        + Sync,
{
    /// Create a new LLM reranker with a scoring function.
    ///
    /// The scoring function takes (query, document_text) and returns a relevance score 0.0-1.0.
    ///
    /// # Example
    /// ```ignore
    /// let reranker = LLMReranker::new(|query, doc| {
    ///     Box::pin(async move {
    ///         let prompt = format!(
    ///             "Rate relevance 0-10.\nQuery: {query}\nDocument: {doc}\nScore:"
    ///         );
    ///         let resp = llm.chat(vec![Message::user(&prompt)]).await?;
    ///         let score: f32 = resp.content.trim().parse().unwrap_or(5.0) / 10.0;
    ///         Ok(score)
    ///     })
    /// });
    /// ```
    pub fn new(score_fn: F) -> Self {
        Self { score_fn }
    }
}

#[async_trait]
impl<F> Reranker for LLMReranker<F>
where
    F: Fn(
            String,
            String,
        )
            -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<f32, String>> + Send>>
        + Send
        + Sync,
{
    async fn rerank(
        &self,
        query: &str,
        mut results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, String> {
        for result in &mut results {
            let score = (self.score_fn)(query.to_string(), result.text.clone()).await?;
            result.score = score;
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }
}

/// Reciprocal Rank Fusion — merges multiple ranked result lists.
///
/// Useful when combining results from vector search + keyword search.
/// Score = sum(1 / (k + rank)) across all lists.
pub struct RRFReranker {
    k: usize,
}

impl RRFReranker {
    pub fn new(k: usize) -> Self {
        Self { k: k.max(1) }
    }

    /// Fuse multiple result lists into one re-ranked list
    pub fn fuse(&self, result_lists: Vec<Vec<SearchResult>>) -> Vec<SearchResult> {
        use std::collections::HashMap;

        let mut scores: HashMap<String, (f32, SearchResult)> = HashMap::new();

        for list in &result_lists {
            for (rank, result) in list.iter().enumerate() {
                let rrf_score = 1.0 / (self.k as f32 + rank as f32 + 1.0);
                let entry = scores
                    .entry(result.id.clone())
                    .or_insert((0.0, result.clone()));
                entry.0 += rrf_score;
            }
        }

        let mut fused: Vec<SearchResult> = scores
            .into_values()
            .map(|(score, mut r)| {
                r.score = score;
                r
            })
            .collect();

        fused.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        fused
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_result(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            text: format!("text for {id}"),
            score,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_noop_reranker() {
        let reranker = NoopReranker;
        let results = vec![make_result("a", 0.9), make_result("b", 0.8)];
        let reranked = reranker.rerank("query", results).await.unwrap();
        assert_eq!(reranked[0].id, "a");
        assert_eq!(reranked[1].id, "b");
    }

    #[test]
    fn test_rrf_fusion() {
        let rrf = RRFReranker::new(60);
        let list1 = vec![make_result("a", 0.9), make_result("b", 0.8)];
        let list2 = vec![make_result("b", 0.95), make_result("c", 0.7)];

        let fused = rrf.fuse(vec![list1, list2]);

        // "b" appears in both lists, should score highest
        assert_eq!(fused[0].id, "b");
        assert!(fused.len() == 3); // a, b, c
    }

    #[tokio::test]
    async fn test_llm_reranker() {
        let reranker = LLMReranker::new(|_query, doc| {
            Box::pin(async move {
                // Simulate: docs containing "rust" get high scores
                if doc.contains("rust") {
                    Ok(0.9)
                } else {
                    Ok(0.1)
                }
            })
        });

        let results = vec![make_result("python", 0.8), make_result("rust", 0.7)];

        let mut results_with_text = results;
        results_with_text[0].text = "python is great".to_string();
        results_with_text[1].text = "rust is fast".to_string();

        let reranked = reranker.rerank("query", results_with_text).await.unwrap();
        assert_eq!(reranked[0].id, "rust");
    }
}
