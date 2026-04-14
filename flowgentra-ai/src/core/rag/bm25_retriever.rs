//! Standalone BM25 Retriever
//!
//! Mirrors LangChain's `BM25Retriever`. Pure keyword-based retrieval with no
//! vector store or embeddings required.
//!
//! Uses the same BM25 implementation as the hybrid search module but exposes
//! it as a stand-alone retriever that can be used directly or composed into
//! an `EnsembleRetriever`.
//!
//! ## When to use
//!
//! - Exact term matching (product IDs, proper nouns, codes)
//! - Baseline for evaluation before adding semantic search
//! - As one arm of an `EnsembleRetriever` alongside a vector-based retriever
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{Bm25Retriever, LoadedDocument};
//!
//! let docs = vec![
//!     ("doc1", "Rust memory safety and ownership"),
//!     ("doc2", "Python machine learning with pandas"),
//! ];
//! let retriever = Bm25Retriever::from_texts(docs, Bm25Config::default());
//! let results = retriever.retrieve("rust ownership", 3);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::hybrid::bm25_score;
use super::vector_db::SearchResult;

/// Configuration for the BM25 retriever.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Config {
    /// Minimum BM25 score (after normalisation) to include a document.
    pub score_threshold: f32,
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Preprocess text before indexing/querying (lowercase, strip punctuation).
    pub preprocess: bool,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            score_threshold: 0.0,
            top_k: 5,
            preprocess: true,
        }
    }
}

/// An indexed document in the BM25 corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Document {
    pub id: String,
    pub text: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Pure keyword retriever based on BM25 scoring.
///
/// Documents are stored in memory; the corpus is rebuilt when documents are added.
/// For large corpora, consider sharding or using a dedicated search engine.
pub struct Bm25Retriever {
    corpus: Vec<Bm25Document>,
    config: Bm25Config,
}

impl Bm25Retriever {
    /// Create a new empty retriever.
    pub fn new(config: Bm25Config) -> Self {
        Self {
            corpus: Vec::new(),
            config,
        }
    }

    /// Create a retriever from a slice of `(id, text)` pairs.
    pub fn from_texts<I, S1, S2>(texts: I, config: Bm25Config) -> Self
    where
        I: IntoIterator<Item = (S1, S2)>,
        S1: Into<String>,
        S2: Into<String>,
    {
        let corpus = texts
            .into_iter()
            .map(|(id, text)| Bm25Document {
                id: id.into(),
                text: text.into(),
                metadata: HashMap::new(),
            })
            .collect();
        Self { corpus, config }
    }

    /// Create a retriever from a list of full `Bm25Document` values.
    pub fn from_documents(docs: Vec<Bm25Document>, config: Bm25Config) -> Self {
        Self {
            corpus: docs,
            config,
        }
    }

    /// Add documents to the corpus.
    pub fn add_documents(&mut self, docs: Vec<Bm25Document>) {
        self.corpus.extend(docs);
    }

    /// Retrieve the top-k documents by BM25 score.
    ///
    /// This is a synchronous call (no embeddings needed).
    pub fn retrieve(&self, query: &str) -> Vec<SearchResult> {
        if self.corpus.is_empty() {
            return vec![];
        }

        let texts: Vec<&str> = self.corpus.iter().map(|d| d.text.as_str()).collect();
        let query_text = if self.config.preprocess {
            preprocess(query)
        } else {
            query.to_string()
        };
        let processed_texts: Vec<String>;
        let texts_to_score: Vec<&str> = if self.config.preprocess {
            processed_texts = texts.iter().map(|t| preprocess(t)).collect();
            processed_texts.iter().map(|s| s.as_str()).collect()
        } else {
            texts.clone()
        };

        let mut scores = bm25_score(&query_text, &texts_to_score);

        // Normalise to 0-1
        let max_score = scores.iter().copied().fold(0.0_f32, f32::max);
        if max_score > 0.0 {
            for s in &mut scores {
                *s /= max_score;
            }
        }

        let mut results: Vec<SearchResult> = self
            .corpus
            .iter()
            .zip(scores)
            .filter(|(_, score)| *score >= self.config.score_threshold)
            .map(|(doc, score)| SearchResult {
                id: doc.id.clone(),
                text: doc.text.clone(),
                score,
                metadata: doc.metadata.clone(),
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(self.config.top_k);
        results
    }

    /// Number of documents in the corpus.
    pub fn len(&self) -> usize {
        self.corpus.len()
    }

    pub fn is_empty(&self) -> bool {
        self.corpus.is_empty()
    }
}

/// Lowercase and strip non-alphanumeric characters.
fn preprocess(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_retriever_basic() {
        let retriever = Bm25Retriever::from_texts(
            vec![
                ("d1", "Rust programming language memory safety"),
                ("d2", "Python machine learning data science"),
                ("d3", "Rust ownership borrow checker"),
                ("d4", "JavaScript web development front end"),
            ],
            Bm25Config::default(),
        );

        let results = retriever.retrieve("rust memory");
        assert!(!results.is_empty());
        // Rust docs should score higher
        assert!(results[0].id == "d1" || results[0].id == "d3");
    }

    #[test]
    fn test_bm25_top_k() {
        let retriever = Bm25Retriever::from_texts(
            vec![
                ("a", "rust"),
                ("b", "rust programming"),
                ("c", "rust language"),
                ("d", "python"),
                ("e", "java"),
            ],
            Bm25Config {
                top_k: 2,
                ..Default::default()
            },
        );
        let results = retriever.retrieve("rust");
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_empty_corpus() {
        let retriever = Bm25Retriever::new(Bm25Config::default());
        let results = retriever.retrieve("anything");
        assert!(results.is_empty());
    }

    #[test]
    fn test_add_documents() {
        let mut retriever = Bm25Retriever::new(Bm25Config::default());
        assert!(retriever.is_empty());
        retriever.add_documents(vec![Bm25Document {
            id: "x".into(),
            text: "test document".into(),
            metadata: HashMap::new(),
        }]);
        assert_eq!(retriever.len(), 1);
    }
}
