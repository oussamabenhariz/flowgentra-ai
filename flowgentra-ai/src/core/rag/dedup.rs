//! Search Result Deduplication
//!
//! Removes near-duplicate results from retrieval output. Two strategies:
//! - **Exact**: remove results with identical document IDs
//! - **Similarity**: remove results whose text overlaps above a threshold (Jaccard)

use std::collections::HashSet;

use super::vector_db::SearchResult;

/// Deduplicate search results by document ID (keeps highest-scoring)
pub fn dedup_by_id(results: Vec<SearchResult>) -> Vec<SearchResult> {
    let mut seen = HashSet::new();
    results
        .into_iter()
        .filter(|r| seen.insert(r.id.clone()))
        .collect()
}

/// Jaccard similarity between two token sets
fn jaccard_similarity(a: &str, b: &str) -> f32 {
    let set_a: HashSet<&str> = a.split_whitespace().collect();
    let set_b: HashSet<&str> = b.split_whitespace().collect();

    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }

    let intersection = set_a.intersection(&set_b).count() as f32;
    let union = set_a.union(&set_b).count() as f32;

    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Deduplicate results by text similarity.
///
/// If two results have Jaccard similarity >= `threshold`, the lower-scoring
/// one is removed. `threshold` of 0.85 is a reasonable default.
pub fn dedup_by_similarity(results: Vec<SearchResult>, threshold: f32) -> Vec<SearchResult> {
    let mut kept: Vec<SearchResult> = Vec::new();

    for result in results {
        let is_dup = kept
            .iter()
            .any(|existing| jaccard_similarity(&existing.text, &result.text) >= threshold);

        if !is_dup {
            kept.push(result);
        }
    }

    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make(id: &str, text: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            text: text.to_string(),
            score,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_dedup_by_id() {
        let results = vec![
            make("a", "text 1", 0.9),
            make("a", "text 1 copy", 0.8),
            make("b", "text 2", 0.7),
        ];
        let deduped = dedup_by_id(results);
        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[0].id, "a");
        assert_eq!(deduped[0].score, 0.9); // kept first (highest)
    }

    #[test]
    fn test_dedup_by_similarity() {
        let results = vec![
            make("a", "the quick brown fox jumps over the lazy dog", 0.9),
            make("b", "the quick brown fox jumps over the lazy cat", 0.85),
            make("c", "completely different text about rust programming", 0.7),
        ];
        let deduped = dedup_by_similarity(results, 0.7);
        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[0].id, "a");
        assert_eq!(deduped[1].id, "c");
    }

    #[test]
    fn test_no_duplicates() {
        let results = vec![
            make("a", "unique text one", 0.9),
            make("b", "unique text two", 0.8),
        ];
        let deduped = dedup_by_similarity(results, 0.85);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_jaccard() {
        let sim = jaccard_similarity("a b c d", "a b c d");
        assert!((sim - 1.0).abs() < f32::EPSILON);

        let sim = jaccard_similarity("a b c", "d e f");
        assert!((sim - 0.0).abs() < f32::EPSILON);
    }
}
