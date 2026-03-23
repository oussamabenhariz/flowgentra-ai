//! Retrieval Strategies and Configuration
//!
//! Defines different strategies for retrieving relevant documents.

use serde::{Deserialize, Serialize};

/// Retrieval strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub strategy: RetrieverStrategy,
    pub top_k: usize,
    pub similarity_threshold: f32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            strategy: RetrieverStrategy::SemanticSearch,
            top_k: 5,
            similarity_threshold: 0.7,
        }
    }
}

impl RetrievalConfig {
    pub fn new(strategy: RetrieverStrategy) -> Self {
        Self {
            strategy,
            ..Default::default()
        }
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }
}

/// Different retrieval strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum RetrieverStrategy {
    /// Pure semantic similarity search
    #[serde(rename = "semantic")]
    SemanticSearch,

    /// Hybrid: combine semantic + keyword search
    #[serde(rename = "hybrid")]
    Hybrid { keyword_weight: f32 },

    /// Multi-query: expand query via an LLM and retrieve for each variant
    #[serde(rename = "multiquery")]
    MultiQuery { query_variants: usize },

    /// Query decomposition: break complex queries into sub-queries
    #[serde(rename = "decomposed")]
    Decomposed { decomposition_depth: usize },

    /// Custom chain strategy
    #[serde(rename = "custom")]
    Custom { chain_name: String },
}

impl RetrieverStrategy {
    pub fn semantic() -> Self {
        Self::SemanticSearch
    }

    pub fn hybrid(keyword_weight: f32) -> Self {
        Self::Hybrid { keyword_weight }
    }

    pub fn multiquery(variants: usize) -> Self {
        Self::MultiQuery {
            query_variants: variants,
        }
    }

    pub fn decomposed(depth: usize) -> Self {
        Self::Decomposed {
            decomposition_depth: depth,
        }
    }
}

/// Query expansion for multi-query retrieval.
///
/// For production use, pass an LLM-based expand function via `expand_with_fn`.
/// The built-in `decompose_query` splits on conjunctions, which is useful for
/// structured queries like "X and Y".
pub struct QueryExpander;

impl QueryExpander {
    /// Expand a query using a caller-provided async function (typically an LLM call).
    ///
    /// # Example
    /// ```ignore
    /// let variants = QueryExpander::expand_with_fn("Rust ownership", 3, |q| {
    ///     Box::pin(async move {
    ///         let prompt = format!("Generate 3 search queries for: {q}");
    ///         let response = llm.chat(vec![Message::user(&prompt)]).await?;
    ///         Ok(response.content.lines().map(String::from).collect())
    ///     })
    /// }).await?;
    /// ```
    pub async fn expand_with_fn<F, Fut>(
        query: &str,
        max_variants: usize,
        expand_fn: F,
    ) -> Result<Vec<String>, String>
    where
        F: FnOnce(String) -> Fut,
        Fut: std::future::Future<Output = Result<Vec<String>, String>>,
    {
        let mut variants = vec![query.to_string()];

        match expand_fn(query.to_string()).await {
            Ok(expanded) => {
                for v in expanded {
                    let trimmed = v.trim().to_string();
                    if !trimmed.is_empty() && !variants.contains(&trimmed) {
                        variants.push(trimmed);
                    }
                    if variants.len() >= max_variants {
                        break;
                    }
                }
            }
            Err(_) => {
                // Fallback: at least return the original query
            }
        }

        variants.truncate(max_variants);
        Ok(variants)
    }

    /// Decompose a complex query into sub-queries by splitting on conjunctions.
    ///
    /// This is a deterministic heuristic — no LLM needed.
    /// "Rust ownership and error handling" → ["Rust ownership and error handling", "Rust ownership", "error handling"]
    pub fn decompose_query(query: &str, max_depth: usize) -> Vec<String> {
        let mut queries = vec![query.to_string()];

        if max_depth > 1 {
            for sep in &[" and ", " or ", "; "] {
                let parts: Vec<&str> = query.split(sep).collect();
                if parts.len() > 1 {
                    queries.extend(parts.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()));
                }
            }
        }

        queries.dedup();
        queries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_decomposition() {
        let query = "machine learning and deep learning";
        let decomposed = QueryExpander::decompose_query(query, 2);

        assert!(decomposed.len() >= 2);
        assert_eq!(decomposed[0], query);
        assert!(decomposed.iter().any(|q| q == "machine learning"));
        assert!(decomposed.iter().any(|q| q == "deep learning"));
    }

    #[test]
    fn test_decomposition_no_split() {
        let query = "simple query";
        let decomposed = QueryExpander::decompose_query(query, 2);
        assert_eq!(decomposed.len(), 1);
        assert_eq!(decomposed[0], "simple query");
    }

    #[test]
    fn test_retrieval_config_builder() {
        let config = RetrievalConfig::new(RetrieverStrategy::semantic())
            .with_top_k(10)
            .with_threshold(0.5);

        assert_eq!(config.top_k, 10);
        assert_eq!(config.similarity_threshold, 0.5);
    }

    #[tokio::test]
    async fn test_expand_with_fn() {
        let result = QueryExpander::expand_with_fn("test query", 3, |_q| async {
            Ok(vec!["variant 1".to_string(), "variant 2".to_string()])
        })
        .await
        .unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "test query");
        assert_eq!(result[1], "variant 1");
    }
}
