//! Hybrid Search — combines vector similarity with BM25 keyword scoring
//!
//! Useful when semantic search alone misses exact-match terms (e.g. product IDs,
//! proper nouns). The final score is a weighted blend:
//!
//! ```text
//! score = (1 - keyword_weight) * vector_score + keyword_weight * bm25_score
//! ```

use std::collections::HashMap;

use super::vector_db::SearchResult;

/// BM25 parameters
const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;

/// Simple tokenizer — lowercase + split on non-alphanumeric
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 1)
        .map(String::from)
        .collect()
}

/// Compute BM25 scores for a query against a set of documents.
///
/// Returns a map of document index → BM25 score.
pub fn bm25_score(query: &str, documents: &[&str]) -> Vec<f32> {
    let query_tokens = tokenize(query);
    let doc_tokens: Vec<Vec<String>> = documents.iter().map(|d| tokenize(d)).collect();

    let n = documents.len() as f32;
    let avg_dl: f32 = if documents.is_empty() {
        1.0
    } else {
        doc_tokens.iter().map(|d| d.len() as f32).sum::<f32>() / n
    };

    // Document frequency for each query term
    let mut df: HashMap<&str, f32> = HashMap::new();
    for qt in &query_tokens {
        let count = doc_tokens
            .iter()
            .filter(|doc| doc.iter().any(|t| t == qt))
            .count() as f32;
        df.insert(qt.as_str(), count);
    }

    doc_tokens
        .iter()
        .map(|doc| {
            let dl = doc.len() as f32;
            let mut score = 0.0f32;

            for qt in &query_tokens {
                let tf = doc.iter().filter(|t| *t == qt).count() as f32;
                let doc_freq = df.get(qt.as_str()).copied().unwrap_or(0.0);

                // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                let idf = ((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();

                // TF component
                let tf_component =
                    (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avg_dl));

                score += idf * tf_component;
            }

            score
        })
        .collect()
}

/// Normalize scores to 0.0–1.0 range using min-max normalization
fn normalize_scores(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;

    if range == 0.0 {
        for s in scores.iter_mut() {
            *s = 1.0;
        }
    } else {
        for s in scores.iter_mut() {
            *s = (*s - min) / range;
        }
    }
}

/// Merge vector search results with BM25 keyword scores.
///
/// `keyword_weight` controls the blend (0.0 = pure vector, 1.0 = pure keyword).
/// Both score streams are min-max normalized before blending.
pub fn hybrid_merge(
    results: Vec<SearchResult>,
    query: &str,
    keyword_weight: f32,
) -> Vec<SearchResult> {
    if results.is_empty() {
        return results;
    }

    let keyword_weight = keyword_weight.clamp(0.0, 1.0);
    let vector_weight = 1.0 - keyword_weight;

    let texts: Vec<&str> = results.iter().map(|r| r.text.as_str()).collect();
    let mut keyword_scores = bm25_score(query, &texts);
    normalize_scores(&mut keyword_scores);

    let mut vector_scores: Vec<f32> = results.iter().map(|r| r.score).collect();
    normalize_scores(&mut vector_scores);

    let mut merged: Vec<SearchResult> = results
        .into_iter()
        .enumerate()
        .map(|(i, mut r)| {
            r.score = vector_weight * vector_scores[i] + keyword_weight * keyword_scores[i];
            r
        })
        .collect();

    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Single-char tokens are filtered
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_bm25_basic() {
        let docs = vec![
            "rust programming language",
            "python scripting language",
            "rust is fast and safe",
        ];
        let scores = bm25_score("rust programming", &docs);
        assert_eq!(scores.len(), 3);
        // Doc 0 has both terms, should score highest
        assert!(scores[0] > scores[1]);
    }

    #[test]
    fn test_hybrid_merge() {
        let results = vec![
            SearchResult {
                id: "a".to_string(),
                text: "rust programming language systems".to_string(),
                score: 0.9,
                metadata: HashMap::new(),
            },
            SearchResult {
                id: "b".to_string(),
                text: "python scripting language".to_string(),
                score: 0.85,
                metadata: HashMap::new(),
            },
            SearchResult {
                id: "c".to_string(),
                text: "rust ownership borrow checker".to_string(),
                score: 0.7,
                metadata: HashMap::new(),
            },
        ];

        let merged = hybrid_merge(results, "rust programming", 0.5);
        assert_eq!(merged.len(), 3);
        // "a" should still be top (high vector + has both keywords)
        assert_eq!(merged[0].id, "a");
    }

    #[test]
    fn test_hybrid_pure_vector() {
        let results = vec![
            SearchResult {
                id: "a".to_string(),
                text: "unrelated text".to_string(),
                score: 0.9,
                metadata: HashMap::new(),
            },
            SearchResult {
                id: "b".to_string(),
                text: "exact match query".to_string(),
                score: 0.5,
                metadata: HashMap::new(),
            },
        ];

        let merged = hybrid_merge(results, "exact match query", 0.0);
        // Pure vector: original order preserved
        assert_eq!(merged[0].id, "a");
    }
}
