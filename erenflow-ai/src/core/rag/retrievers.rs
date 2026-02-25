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
    pub enable_reranking: bool,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            strategy: RetrieverStrategy::SemanticSearch,
            top_k: 5,
            similarity_threshold: 0.7,
            enable_reranking: false,
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

    pub fn with_reranking(mut self, enable: bool) -> Self {
        self.enable_reranking = enable;
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

    /// Multi-query: expand query and retrieve for each variant
    #[serde(rename = "multiquery")]
    MultiQuery { query_variants: usize },

    /// SVM (Support Vector Machine) based retrieval
    #[serde(rename = "svm")]
    SVM { kernel: String },

    /// Query decomposition: break into sub-queries
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

/// Query expansion for better retrieval
pub struct QueryExpander;

impl QueryExpander {
    /// Generate query variants for multi-query retrieval
    pub fn expand_query(query: &str, num_variants: usize) -> Vec<String> {
        let mut variants = vec![query.to_string()];

        // Simple expansion strategies
        if num_variants > 1 {
            // Variant 1: Add synonyms (simplified)
            variants.push(format!("{}?", query));
        }

        if num_variants > 2 {
            // Variant 2: Rephrase
            variants.push(format!("What about {}", query));
        }

        if num_variants > 3 {
            // Variant 3: Question format
            variants.push(format!("Tell me about {}", query));
        }

        variants.truncate(num_variants);
        variants
    }

    /// Decompose complex query into sub-queries
    pub fn decompose_query(query: &str, depth: usize) -> Vec<String> {
        let mut queries = vec![query.to_string()];

        if depth > 1 {
            // Split by common connectives
            let parts: Vec<&str> = query.split(" and ").collect();
            if parts.len() > 1 {
                queries.extend(parts.into_iter().map(|s| s.to_string()));
            }
        }

        if depth > 2 {
            // Add more decomposition strategies
            let parts: Vec<&str> = query.split(" or ").collect();
            if parts.len() > 1 {
                queries.extend(parts.into_iter().map(|s| s.to_string()));
            }
        }

        queries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_expansion() {
        let query = "machine learning algorithms";
        let variants = QueryExpander::expand_query(query, 3);

        assert_eq!(variants.len(), 3);
        assert!(variants[0].contains("machine learning"));
    }

    #[test]
    fn test_query_decomposition() {
        let query = "machine learning and deep learning";
        let decomposed = QueryExpander::decompose_query(query, 2);

        assert!(decomposed.len() >= 1);
        assert!(decomposed.iter().any(|q| q.contains("machine learning")));
    }

    #[test]
    fn test_retrieval_config_builder() {
        let config = RetrievalConfig::new(RetrieverStrategy::semantic())
            .with_top_k(10)
            .with_threshold(0.5)
            .with_reranking(true);

        assert_eq!(config.top_k, 10);
        assert_eq!(config.similarity_threshold, 0.5);
        assert!(config.enable_reranking);
    }
}
