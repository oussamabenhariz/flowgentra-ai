//! # Node Output Scoring
//!
//! Quantitative assessment of node outputs based on configurable criteria.
//!
//! The scoring system evaluates outputs on multiple dimensions:
//! - **Completeness**: How complete is the output?
//! - **Validity**: Is the output in expected format/type?
//! - **Usefulness**: Is the output actionable for downstream nodes?
//! - **Consistency**: Does it align with previous outputs in the state?

use crate::core::state::DynState;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Criteria used for scoring output
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoringCriteria {
    /// Should check output is not null/empty
    #[serde(default = "default_true")]
    pub check_empty: bool,

    /// Should verify JSON structure validity
    #[serde(default = "default_true")]
    pub check_validity: bool,

    /// Should assess output usefulness (non-trivial response)
    #[serde(default = "default_true")]
    pub check_usefulness: bool,

    /// Should check consistency with state history
    #[serde(default = "default_true")]
    pub check_consistency: bool,

    /// Minimum acceptable length for string outputs (0 = no minimum)
    #[serde(default)]
    pub min_length: usize,

    /// Maximum acceptable length for string outputs (0 = no maximum)
    #[serde(default)]
    pub max_length: usize,
}

fn default_true() -> bool {
    true
}

impl Default for ScoringCriteria {
    fn default() -> Self {
        ScoringCriteria {
            check_empty: true,
            check_validity: true,
            check_usefulness: true,
            check_consistency: true,
            min_length: 1,
            max_length: 0, // 0 = unlimited
        }
    }
}

/// Individual component scores
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeScore {
    /// Overall score (0.0-1.0), weighted average of components
    pub overall: f64,

    /// Emptiness check (1.0 if not empty, 0.0 if empty)
    pub completeness: f64,

    /// Validity check (1.0 if valid structure, 0.0 if not)
    pub validity: f64,

    /// Usefulness assessment (1.0 if useful/actionable)
    pub usefulness: f64,

    /// Consistency check (1.0 if consistent with history)
    pub consistency: f64,

    /// Detailed explanation of scoring
    pub explanation: String,
}

/// Scores node outputs quantitatively
pub struct NodeScorer;

impl NodeScorer {
    /// Score an output value
    pub fn score(
        output: &Value,
        criteria: &ScoringCriteria,
        state: &DynState,
        node_name: &str,
    ) -> NodeScore {
        let mut scores = Vec::new();
        let mut explanation = String::new();

        // Check completeness (not empty)
        let completeness = if criteria.check_empty {
            let score = Self::score_completeness(output);
            explanation.push_str(&format!("Completeness: {:.2} - ", score));
            scores.push(score);
            score
        } else {
            1.0
        };

        // Check validity
        let validity = if criteria.check_validity {
            let score = Self::score_validity(output);
            explanation.push_str(&format!("Validity: {:.2} - ", score));
            scores.push(score);
            score
        } else {
            1.0
        };

        // Check usefulness
        let usefulness = if criteria.check_usefulness {
            let score = Self::score_usefulness(output, state, node_name);
            explanation.push_str(&format!("Usefulness: {:.2} - ", score));
            scores.push(score);
            score
        } else {
            1.0
        };

        // Check consistency
        let consistency = if criteria.check_consistency {
            let score = Self::score_consistency(output, state, node_name);
            explanation.push_str(&format!("Consistency: {:.2}", score));
            scores.push(score);
            score
        } else {
            1.0
        };

        // Calculate overall score
        let overall = if scores.is_empty() {
            1.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        NodeScore {
            overall,
            completeness,
            validity,
            usefulness,
            consistency,
            explanation,
        }
    }

    /// Check if output is complete (not null, empty, etc.)
    fn score_completeness(output: &Value) -> f64 {
        match output {
            Value::Null => 0.0,
            Value::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.5
                }
            }
            Value::Number(n) => {
                if n.is_f64() && n.as_f64() == Some(0.0) {
                    0.5
                } else {
                    1.0
                }
            }
            Value::String(s) => {
                if s.is_empty() {
                    0.0
                } else if s.len() < 3 {
                    0.5
                } else {
                    1.0
                }
            }
            Value::Array(a) => {
                if a.is_empty() {
                    0.0
                } else {
                    1.0
                }
            }
            Value::Object(o) => {
                if o.is_empty() {
                    0.0
                } else {
                    1.0
                }
            }
        }
    }

    /// Check if value has valid structure
    fn score_validity(output: &Value) -> f64 {
        // All JSON values are technically valid if they parse
        // This checks for logical validity
        match output {
            Value::Null => 0.5, // Null is valid but suspicious
            Value::Object(o) => {
                // Prefer objects with content
                if o.is_empty() {
                    0.7
                } else {
                    1.0
                }
            }
            Value::Array(_) => 1.0,
            _ => 1.0,
        }
    }

    /// Check if output is useful/actionable
    fn score_usefulness(output: &Value, _state: &DynState, _node_name: &str) -> f64 {
        // Useful outputs have meaningful content
        match output {
            Value::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    0.0
                } else if trimmed.len() < 5 {
                    0.5
                } else if is_trivial_response(trimmed) {
                    0.6
                } else {
                    1.0
                }
            }
            Value::Object(o) => {
                if o.is_empty() {
                    0.3
                } else if has_meaningful_keys(o) {
                    1.0
                } else {
                    0.7
                }
            }
            Value::Array(a) => {
                if a.is_empty() {
                    0.2
                } else if a.len() > 1 {
                    1.0
                } else {
                    0.8
                }
            }
            _ => 0.8,
        }
    }

    /// Check consistency with prior outputs
    fn score_consistency(output: &Value, state: &DynState, node_name: &str) -> f64 {
        // Check if this node has output history
        let history_key = format!("__node_output_history__{}", node_name);
        if let Some(Value::Array(histories)) = state.get(&history_key) {
            if let Some(last) = histories.last() {
                // Output is consistent if it's similar to or better than previous
                let last_type = Self::value_type(last);
                let current_type = Self::value_type(output);

                if last_type == current_type {
                    1.0 // Same type is highly consistent
                } else {
                    0.6 // Different type reduces consistency
                }
            } else {
                1.0
            }
        } else {
            1.0 // No history, assume consistent
        }
    }

    /// Get the type of a JSON value as a string
    fn value_type(value: &Value) -> &str {
        match value {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }
}

/// Check if a string is a trivial response (yes/no, ok, etc.)
fn is_trivial_response(s: &str) -> bool {
    matches!(
        s.to_lowercase().trim(),
        "yes" | "no" | "ok" | "ok." | "true" | "false" | "1" | "0" | "none" | "null"
    )
}

/// Check if object has meaningful (non-metadata) keys
fn has_meaningful_keys(obj: &serde_json::Map<String, Value>) -> bool {
    obj.keys()
        .any(|k| !k.starts_with('_') && !k.starts_with("__"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_score_completeness() {
        assert_eq!(NodeScorer::score_completeness(&Value::Null), 0.0);
        assert!(NodeScorer::score_completeness(&json!("hello")) > 0.9);
        assert!(NodeScorer::score_completeness(&json!([])) < 0.5);
        assert!(NodeScorer::score_completeness(&json!({})) < 0.5);
    }

    #[test]
    fn test_score_validity() {
        assert!(NodeScorer::score_validity(&json!({"key": "value"})) > 0.9);
        assert!(NodeScorer::score_validity(&json!({})) < 0.9);
    }

    #[test]
    fn test_node_scorer_full() {
        let criteria = ScoringCriteria::default();
        let state = DynState::new();

        let output = json!({"result": "success", "count": 42});
        let score = NodeScorer::score(&output, &criteria, &state, "test_node");

        assert!(score.overall > 0.7);
        assert!(score.completeness > 0.7);
    }

    #[test]
    fn test_trivial_response_detection() {
        assert!(is_trivial_response("yes"));
        assert!(is_trivial_response("NO"));
        assert!(!is_trivial_response("the quick brown fox jumps"));
    }
}
