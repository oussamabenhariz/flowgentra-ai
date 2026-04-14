//! Long-Context Reorder
//!
//! Mirrors LangChain's `LongContextReorder`. Addresses the **"lost in the middle"**
//! problem described in the paper *"Lost in the Middle: How Language Models Use
//! Long Contexts"* (Liu et al., 2023).
//!
//! LLMs tend to attend most to content at the **beginning and end** of the
//! context window and underweight content in the **middle**. This reorder places
//! the most-relevant documents at the start and end of the list, and the
//! least-relevant documents in the middle.
//!
//! ## Algorithm
//!
//! Given documents ranked [1st, 2nd, 3rd, 4th, 5th] by score:
//!
//! 1. Split into two halves: high-ranked and low-ranked.
//! 2. Interleave them so the result is: [1st, 3rd, 5th, 4th, 2nd]
//!    → most relevant at start, second most-relevant at end.
//!
//! ## Example
//!
//! ```rust
//! use flowgentra_ai::core::rag::reorder::{reorder_for_long_context, ReorderStrategy};
//!
//! // results already sorted by score descending
//! let reordered = reorder_for_long_context(results, ReorderStrategy::LostInTheMiddle);
//! ```

use super::vector_db::SearchResult;

/// Strategy used when reordering retrieved documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReorderStrategy {
    /// Place best docs at start and end; worst in middle.
    /// This is the "lost in the middle" mitigation from Liu et al. 2023.
    #[default]
    LostInTheMiddle,

    /// Leave documents in their original score order (no-op).
    None,

    /// Reverse the order (worst first — useful for some chain-of-thought prompts).
    Reverse,
}

/// Reorder a list of search results according to `strategy`.
///
/// Input `results` should be **sorted by score descending** (best first).
/// Most retrieval pipelines already return them in this order.
pub fn reorder_for_long_context(
    results: Vec<SearchResult>,
    strategy: ReorderStrategy,
) -> Vec<SearchResult> {
    match strategy {
        ReorderStrategy::None => results,
        ReorderStrategy::Reverse => {
            let mut r = results;
            r.reverse();
            r
        }
        ReorderStrategy::LostInTheMiddle => lost_in_the_middle(results),
    }
}

/// Interleave best and worst so the most relevant appear at the edges.
///
/// Example for 6 docs ranked 0–5 (0 = best):
/// Output order: [0, 2, 4, 5, 3, 1]
///               ↑             ↑
///              start         end
fn lost_in_the_middle(results: Vec<SearchResult>) -> Vec<SearchResult> {
    if results.len() <= 2 {
        return results;
    }

    let n = results.len();
    let mut reordered = vec![None::<SearchResult>; n];

    let mut left = 0;      // fills from start
    let mut right = n - 1; // fills from end
    let mut i = 0;         // index into original sorted list

    // Alternate: even indices go to start, odd indices go to end
    while i < n {
        if i % 2 == 0 {
            reordered[left] = Some(results[i].clone());
            left += 1;
        } else {
            reordered[right] = Some(results[i].clone());
            if right > 0 {
                right -= 1;
            }
        }
        i += 1;
    }

    reordered.into_iter().flatten().collect()
}

/// Convenience wrapper: reorder using the default `LostInTheMiddle` strategy.
pub fn reorder(results: Vec<SearchResult>) -> Vec<SearchResult> {
    lost_in_the_middle(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_results(n: usize) -> Vec<SearchResult> {
        (0..n)
            .map(|i| SearchResult {
                id: format!("doc{}", i),
                text: format!("text {}", i),
                score: 1.0 - (i as f32 * 0.1),
                metadata: HashMap::new(),
            })
            .collect()
    }

    #[test]
    fn test_reorder_none() {
        let results = make_results(4);
        let reordered = reorder_for_long_context(results.clone(), ReorderStrategy::None);
        assert_eq!(reordered[0].id, "doc0");
        assert_eq!(reordered[3].id, "doc3");
    }

    #[test]
    fn test_reorder_reverse() {
        let results = make_results(4);
        let reordered = reorder_for_long_context(results, ReorderStrategy::Reverse);
        assert_eq!(reordered[0].id, "doc3");
        assert_eq!(reordered[3].id, "doc0");
    }

    #[test]
    fn test_lost_in_the_middle_best_at_edges() {
        let results = make_results(6); // best is doc0 (score 1.0), worst is doc5 (score 0.5)
        let reordered = reorder_for_long_context(results, ReorderStrategy::LostInTheMiddle);

        assert_eq!(reordered.len(), 6);

        // Best doc (doc0) must be at position 0 (start)
        assert_eq!(reordered[0].id, "doc0");

        // Second-best (doc1) must be at the end
        assert_eq!(reordered[5].id, "doc1");
    }

    #[test]
    fn test_two_docs_unchanged() {
        let results = make_results(2);
        let reordered = reorder_for_long_context(results.clone(), ReorderStrategy::LostInTheMiddle);
        assert_eq!(reordered.len(), 2);
        assert_eq!(reordered[0].id, results[0].id);
        assert_eq!(reordered[1].id, results[1].id);
    }

    #[test]
    fn test_single_doc() {
        let results = make_results(1);
        let reordered = reorder(results);
        assert_eq!(reordered.len(), 1);
    }

    #[test]
    fn test_empty() {
        let results = vec![];
        let reordered = reorder(results);
        assert!(reordered.is_empty());
    }
}
