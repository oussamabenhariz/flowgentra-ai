//! Multi-Query Retriever
//!
//! Mirrors LangChain's `MultiQueryRetriever`. Generates several paraphrased
//! variants of the user's query using an LLM, runs each through the base
//! retriever, and merges the results (deduplicated by document id).
//!
//! ## Why multiple queries?
//!
//! A single query formulation may miss relevant documents phrased differently.
//! Generating N variants captures different semantic angles, improving recall
//! at the cost of additional retrieval + LLM calls.
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::multi_query_retriever::{MultiQueryRetriever, MultiQueryConfig};
//!
//! let retriever = MultiQueryRetriever::new(base_retriever, config);
//! let results = retriever.retrieve("What is Rust ownership?").await?;
//! ```

use std::collections::HashMap;

use async_trait::async_trait;

use super::ensemble_retriever::AsyncRetriever;
use super::vector_db::{SearchResult, VectorStoreError};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MultiQueryConfig {
    /// LLM API URL (OpenAI-compatible, e.g. `https://api.openai.com/v1/chat/completions`).
    pub api_url: String,
    pub api_key: String,
    pub model: String,
    /// Number of alternative queries to generate (default 3).
    pub num_queries: usize,
    /// Max docs returned after deduplication.
    pub top_k: usize,
}

impl MultiQueryConfig {
    pub fn new(
        api_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            api_url: api_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            num_queries: 3,
            top_k: 10,
        }
    }
}

// ── MultiQueryRetriever ───────────────────────────────────────────────────────

pub struct MultiQueryRetriever {
    base: Box<dyn AsyncRetriever>,
    config: MultiQueryConfig,
}

impl MultiQueryRetriever {
    pub fn new(base: impl AsyncRetriever + 'static, config: MultiQueryConfig) -> Self {
        Self {
            base: Box::new(base),
            config,
        }
    }

    /// Call the LLM to generate `n` alternative phrasings of `query`.
    async fn generate_queries(
        &self,
        query: &str,
    ) -> Result<Vec<String>, VectorStoreError> {
        let prompt = format!(
            "You are a helpful assistant that generates multiple search queries based on \
             a single input query. Your task is to generate {} different versions of the \
             given query to improve document retrieval. Provide these alternative queries \
             separated by newlines, with no numbering or bullets.\n\nOriginal query: {}",
            self.config.num_queries, query
        );

        let client = reqwest::Client::new();
        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.7,
        });

        let resp = client
            .post(&self.config.api_url)
            .bearer_auth(&self.config.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("MultiQuery HTTP: {e}")))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("MultiQuery JSON: {e}")))?;

        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let mut queries: Vec<String> = content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .take(self.config.num_queries)
            .collect();

        // Always include the original query
        queries.push(query.to_string());
        Ok(queries)
    }
}

#[async_trait]
impl AsyncRetriever for MultiQueryRetriever {
    async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, VectorStoreError> {
        let queries = match self.generate_queries(query).await {
            Ok(qs) => qs,
            Err(_) => vec![query.to_string()], // fallback to original on LLM error
        };

        // Run all variant queries and collect unique results by id
        let mut seen: HashMap<String, SearchResult> = HashMap::new();
        for q in &queries {
            if let Ok(results) = self.base.retrieve(q).await {
                for result in results {
                    seen.entry(result.id.clone())
                        .and_modify(|existing| {
                            // Keep the higher score
                            if result.score > existing.score {
                                *existing = result.clone();
                            }
                        })
                        .or_insert(result);
                }
            }
        }

        let mut merged: Vec<SearchResult> = seen.into_values().collect();
        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(self.config.top_k);
        Ok(merged)
    }
}
