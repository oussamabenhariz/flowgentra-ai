//! RAG Evaluation Metrics
//!
//! Standard information retrieval metrics for measuring RAG quality:
//! - **Hit Rate (Recall@K)**: fraction of queries where at least one relevant doc is in top-K
//! - **MRR (Mean Reciprocal Rank)**: average of 1/rank of first relevant doc
//! - **nDCG (Normalized Discounted Cumulative Gain)**: measures ranking quality

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A single evaluation query with known relevant document IDs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalQuery {
    pub query: String,
    pub relevant_doc_ids: Vec<String>,
}

/// Evaluation results for a set of queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResults {
    pub hit_rate: f64,
    pub mrr: f64,
    pub ndcg: f64,
    pub num_queries: usize,
    pub per_query: Vec<QueryResult>,
}

/// Per-query evaluation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub query: String,
    pub hit: bool,
    pub reciprocal_rank: f64,
    pub ndcg: f64,
    pub retrieved_ids: Vec<String>,
}

/// Compute hit rate: fraction of queries where at least one relevant doc appears in results
pub fn hit_rate(queries: &[EvalQuery], retrieved: &[Vec<String>]) -> f64 {
    if queries.is_empty() {
        return 0.0;
    }

    let hits = queries
        .iter()
        .zip(retrieved)
        .filter(|(q, results)| {
            let relevant: HashSet<&str> = q.relevant_doc_ids.iter().map(|s| s.as_str()).collect();
            results.iter().any(|id| relevant.contains(id.as_str()))
        })
        .count();

    hits as f64 / queries.len() as f64
}

/// Compute Mean Reciprocal Rank
pub fn mrr(queries: &[EvalQuery], retrieved: &[Vec<String>]) -> f64 {
    if queries.is_empty() {
        return 0.0;
    }

    let sum: f64 = queries
        .iter()
        .zip(retrieved)
        .map(|(q, results)| {
            let relevant: HashSet<&str> = q.relevant_doc_ids.iter().map(|s| s.as_str()).collect();
            for (rank, id) in results.iter().enumerate() {
                if relevant.contains(id.as_str()) {
                    return 1.0 / (rank as f64 + 1.0);
                }
            }
            0.0
        })
        .sum();

    sum / queries.len() as f64
}

/// Compute nDCG for a single query
fn ndcg_single(relevant: &HashSet<&str>, retrieved: &[String]) -> f64 {
    if relevant.is_empty() || retrieved.is_empty() {
        return 0.0;
    }

    // DCG
    let dcg: f64 = retrieved
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let rel = if relevant.contains(id.as_str()) {
                1.0
            } else {
                0.0
            };
            rel / (i as f64 + 2.0).log2()
        })
        .sum();

    // Ideal DCG: all relevant docs at top positions
    let ideal_count = relevant.len().min(retrieved.len());
    let idcg: f64 = (0..ideal_count)
        .map(|i| 1.0 / (i as f64 + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Compute mean nDCG across all queries
pub fn mean_ndcg(queries: &[EvalQuery], retrieved: &[Vec<String>]) -> f64 {
    if queries.is_empty() {
        return 0.0;
    }

    let sum: f64 = queries
        .iter()
        .zip(retrieved)
        .map(|(q, results)| {
            let relevant: HashSet<&str> = q.relevant_doc_ids.iter().map(|s| s.as_str()).collect();
            ndcg_single(&relevant, results)
        })
        .sum();

    sum / queries.len() as f64
}

/// Run a full evaluation and return aggregate + per-query results
pub fn evaluate(queries: &[EvalQuery], retrieved: &[Vec<String>]) -> EvalResults {
    let per_query: Vec<QueryResult> = queries
        .iter()
        .zip(retrieved)
        .map(|(q, results)| {
            let relevant: HashSet<&str> = q.relevant_doc_ids.iter().map(|s| s.as_str()).collect();

            let hit = results.iter().any(|id| relevant.contains(id.as_str()));

            let reciprocal_rank = results
                .iter()
                .enumerate()
                .find(|(_, id)| relevant.contains(id.as_str()))
                .map(|(rank, _)| 1.0 / (rank as f64 + 1.0))
                .unwrap_or(0.0);

            let ndcg = ndcg_single(&relevant, results);

            QueryResult {
                query: q.query.clone(),
                hit,
                reciprocal_rank,
                ndcg,
                retrieved_ids: results.clone(),
            }
        })
        .collect();

    let num_queries = queries.len();
    let hit_rate_val = if num_queries > 0 {
        per_query.iter().filter(|q| q.hit).count() as f64 / num_queries as f64
    } else {
        0.0
    };
    let mrr_val = if num_queries > 0 {
        per_query.iter().map(|q| q.reciprocal_rank).sum::<f64>() / num_queries as f64
    } else {
        0.0
    };
    let ndcg_val = if num_queries > 0 {
        per_query.iter().map(|q| q.ndcg).sum::<f64>() / num_queries as f64
    } else {
        0.0
    };

    EvalResults {
        hit_rate: hit_rate_val,
        mrr: mrr_val,
        ndcg: ndcg_val,
        num_queries,
        per_query,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hit_rate_all_hit() {
        let queries = vec![
            EvalQuery {
                query: "q1".into(),
                relevant_doc_ids: vec!["a".into()],
            },
            EvalQuery {
                query: "q2".into(),
                relevant_doc_ids: vec!["b".into()],
            },
        ];
        let retrieved = vec![vec!["x".into(), "a".into()], vec!["b".into(), "y".into()]];
        assert!((hit_rate(&queries, &retrieved) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hit_rate_partial() {
        let queries = vec![
            EvalQuery {
                query: "q1".into(),
                relevant_doc_ids: vec!["a".into()],
            },
            EvalQuery {
                query: "q2".into(),
                relevant_doc_ids: vec!["z".into()],
            },
        ];
        let retrieved = vec![vec!["a".into(), "b".into()], vec!["c".into(), "d".into()]];
        assert!((hit_rate(&queries, &retrieved) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mrr() {
        let queries = vec![
            EvalQuery {
                query: "q1".into(),
                relevant_doc_ids: vec!["a".into()],
            },
            EvalQuery {
                query: "q2".into(),
                relevant_doc_ids: vec!["b".into()],
            },
        ];
        // "a" is at rank 2 (index 1) → RR = 1/2
        // "b" is at rank 1 (index 0) → RR = 1/1
        let retrieved = vec![vec!["x".into(), "a".into()], vec!["b".into(), "y".into()]];
        let result = mrr(&queries, &retrieved);
        assert!((result - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ndcg_perfect() {
        let queries = vec![EvalQuery {
            query: "q1".into(),
            relevant_doc_ids: vec!["a".into(), "b".into()],
        }];
        // Perfect ranking: both relevant docs at top
        let retrieved = vec![vec!["a".into(), "b".into(), "c".into()]];
        let result = mean_ndcg(&queries, &retrieved);
        assert!((result - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_evaluate_full() {
        let queries = vec![EvalQuery {
            query: "test".into(),
            relevant_doc_ids: vec!["a".into()],
        }];
        let retrieved = vec![vec!["a".into(), "b".into()]];

        let results = evaluate(&queries, &retrieved);
        assert_eq!(results.num_queries, 1);
        assert!((results.hit_rate - 1.0).abs() < f64::EPSILON);
        assert!((results.mrr - 1.0).abs() < f64::EPSILON);
        assert!(results.per_query[0].hit);
    }
}
