//! Contextual Compression Retriever
//!
//! Mirrors LangChain's `ContextualCompressionRetriever`. After a base retriever
//! fetches candidates, a **compressor** filters or shortens each document so
//! that only the parts relevant to the query are passed to the LLM.
//!
//! ## Compressor implementations
//!
//! | Type | Description |
//! |---|---|
//! | [`LLMCompressor`] | Sends each doc to an LLM with a compression prompt |
//! | [`EmbeddingsFilter`] | Keeps docs whose embedding similarity ≥ threshold |
//! | [`DocumentCompressorPipeline`] | Chains multiple compressors in sequence |
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{
//!     compression_retriever::{ContextualCompressionRetriever, EmbeddingsFilter},
//!     ensemble_retriever::VectorRetriever,
//! };
//!
//! let base = VectorRetriever::new(store, embeddings.clone(), 20);
//! let compressor = EmbeddingsFilter::new(embeddings, 0.75);
//! let retriever = ContextualCompressionRetriever::new(base, compressor, 5);
//!
//! let docs = retriever.retrieve("Rust memory model").await?;
//! ```

use std::sync::Arc;

use async_trait::async_trait;

use super::embeddings::Embeddings;
use super::ensemble_retriever::AsyncRetriever;
use super::vector_db::{SearchResult, VectorStoreError};

// ── DocumentCompressor trait ──────────────────────────────────────────────────

/// Compresses / filters a list of documents given the original query.
#[async_trait]
pub trait DocumentCompressor: Send + Sync {
    /// Given the query and candidate documents, return a (possibly smaller
    /// or shorter) list. Implementations may filter low-relevance docs,
    /// extract relevant spans, or summarize.
    async fn compress(
        &self,
        query: &str,
        documents: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, VectorStoreError>;
}

// ── EmbeddingsFilter ─────────────────────────────────────────────────────────

/// Keeps only documents whose cosine similarity to the query is ≥ `threshold`.
pub struct EmbeddingsFilter {
    embeddings: Arc<Embeddings>,
    threshold: f32,
}

impl EmbeddingsFilter {
    pub fn new(embeddings: Arc<Embeddings>, threshold: f32) -> Self {
        Self { embeddings, threshold }
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
    }
}

#[async_trait]
impl DocumentCompressor for EmbeddingsFilter {
    async fn compress(
        &self,
        query: &str,
        documents: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let query_emb = self
            .embeddings
            .embed(query)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Embed error: {e}")))?;

        let mut filtered = Vec::new();
        for doc in documents {
            let doc_emb = self
                .embeddings
                .embed(&doc.text)
                .await
                .map_err(|e| VectorStoreError::Unknown(format!("Embed error: {e}")))?;
            let sim = Self::cosine(&query_emb, &doc_emb);
            if sim >= self.threshold {
                filtered.push(SearchResult { score: sim, ..doc });
            }
        }
        Ok(filtered)
    }
}

// ── LLMCompressor ─────────────────────────────────────────────────────────────

/// Uses an HTTP LLM endpoint to extract / summarize the relevant portion of
/// each document.
///
/// The compressor sends a prompt like:
/// > Given the question `{query}`, extract the relevant parts from the following
/// > document. If nothing is relevant, return "NOT_RELEVANT".
///
/// Documents returning "NOT_RELEVANT" are dropped from the result.
pub struct LLMCompressor {
    api_url: String,
    api_key: String,
    model: String,
    max_tokens: u32,
}

impl LLMCompressor {
    pub fn new(
        api_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        max_tokens: u32,
    ) -> Self {
        Self {
            api_url: api_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            max_tokens,
        }
    }

    async fn call_llm(&self, prompt: &str) -> Result<String, VectorStoreError> {
        let client = reqwest::Client::new();
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        });
        let resp = client
            .post(&self.api_url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("LLM HTTP: {e}")))?;
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("LLM JSON: {e}")))?;
        Ok(json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }
}

#[async_trait]
impl DocumentCompressor for LLMCompressor {
    async fn compress(
        &self,
        query: &str,
        documents: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let mut compressed = Vec::new();
        for doc in documents {
            let prompt = format!(
                "Given the question: \"{query}\"\n\nExtract only the parts of the following \
                 document that are relevant to answering the question. If nothing is relevant, \
                 respond with exactly \"NOT_RELEVANT\".\n\nDocument:\n{}",
                doc.text
            );
            let response = self.call_llm(&prompt).await?;
            if response.trim() != "NOT_RELEVANT" {
                compressed.push(SearchResult {
                    text: response,
                    ..doc
                });
            }
        }
        Ok(compressed)
    }
}

// ── DocumentCompressorPipeline ────────────────────────────────────────────────

/// Chains multiple compressors: output of each becomes input of the next.
pub struct DocumentCompressorPipeline {
    compressors: Vec<Box<dyn DocumentCompressor>>,
}

impl DocumentCompressorPipeline {
    pub fn new(compressors: Vec<Box<dyn DocumentCompressor>>) -> Self {
        Self { compressors }
    }
}

#[async_trait]
impl DocumentCompressor for DocumentCompressorPipeline {
    async fn compress(
        &self,
        query: &str,
        mut documents: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        for compressor in &self.compressors {
            documents = compressor.compress(query, documents).await?;
            if documents.is_empty() {
                break;
            }
        }
        Ok(documents)
    }
}

// ── ContextualCompressionRetriever ────────────────────────────────────────────

/// Retriever that applies a compressor to the results of a base retriever.
pub struct ContextualCompressionRetriever {
    base_retriever: Box<dyn AsyncRetriever>,
    compressor: Box<dyn DocumentCompressor>,
    top_k: usize,
}

impl ContextualCompressionRetriever {
    pub fn new(
        base_retriever: impl AsyncRetriever + 'static,
        compressor: impl DocumentCompressor + 'static,
        top_k: usize,
    ) -> Self {
        Self {
            base_retriever: Box::new(base_retriever),
            compressor: Box::new(compressor),
            top_k,
        }
    }
}

#[async_trait]
impl AsyncRetriever for ContextualCompressionRetriever {
    async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, VectorStoreError> {
        let candidates = self.base_retriever.retrieve(query).await?;
        let mut compressed = self.compressor.compress(query, candidates).await?;
        compressed.truncate(self.top_k);
        Ok(compressed)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rag::{embeddings::Embeddings, vector_db::InMemoryVectorStore};
    use crate::core::rag::ensemble_retriever::VectorRetriever;
    use crate::core::rag::vector_db::Document;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_embeddings_filter() {
        let embeddings = Arc::new(Embeddings::mock(16));
        let store = Arc::new(InMemoryVectorStore::new());

        for (id, text) in [("r1", "Rust memory safety"), ("r2", "Python pandas")] {
            let emb = embeddings.embed(text).await.unwrap();
            store
                .index(Document {
                    id: id.to_string(),
                    text: text.to_string(),
                    embedding: Some(emb),
                    metadata: HashMap::new(),
                })
                .await
                .unwrap();
        }

        let base = VectorRetriever::new(store, embeddings.clone(), 10);
        let filter = EmbeddingsFilter::new(embeddings, 0.0); // 0.0 keeps all
        let retriever = ContextualCompressionRetriever::new(base, filter, 5);

        let results = retriever.retrieve("memory safety").await.unwrap();
        assert!(!results.is_empty());
    }
}
