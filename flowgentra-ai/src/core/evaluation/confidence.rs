//! # Confidence Scoring  
//!
//! Estimates confidence/trust in node outputs.
//!
//! Confidence is computed from multiple signals:
//! - LLM confidence indicators
//! - Output consistency
//! - Information completeness
//! - Entropy/uncertainty measures

use crate::core::state::State;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Configuration for confidence scoring
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfidenceConfig {
    /// Weight for clarity score
    #[serde(default = "default_clarity_weight")]
    pub clarity_weight: f64,

    /// Weight for relevance score
    #[serde(default = "default_relevance_weight")]
    pub relevance_weight: f64,

    /// Weight for completeness score
    #[serde(default = "default_completeness_weight")]
    pub completeness_weight: f64,

    /// Threshold below which confidence is considered low
    #[serde(default = "default_low_confidence_threshold")]
    pub low_threshold: f64,

    /// Threshold above which confidence is considered high
    #[serde(default = "default_high_confidence_threshold")]
    pub high_threshold: f64,
}

fn default_clarity_weight() -> f64 {
    0.3
}

fn default_relevance_weight() -> f64 {
    0.4
}

fn default_completeness_weight() -> f64 {
    0.3
}

fn default_low_confidence_threshold() -> f64 {
    0.5
}

fn default_high_confidence_threshold() -> f64 {
    0.8
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        ConfidenceConfig {
            clarity_weight: 0.3,
            relevance_weight: 0.4,
            completeness_weight: 0.3,
            low_threshold: 0.5,
            high_threshold: 0.8,
        }
    }
}

/// Multi-dimensional confidence score
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfidenceScore {
    /// Overall confidence (0.0-1.0), weighted combination
    pub overall: f64,

    /// Clarity of the output (0.0-1.0)
    pub clarity: f64,

    /// Relevance to task (0.0-1.0)
    pub relevance: f64,

    /// Completeness of response (0.0-1.0)
    pub completeness: f64,

    /// Confidence level category
    pub level: ConfidenceLevel,

    /// Indicators that affected the score
    pub indicators: Vec<String>,
}

/// Categorization of confidence levels
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ConfidenceLevel {
    /// Score < 0.5: Very uncertain, likely needs retry
    VeryLow,
    /// Score 0.5-0.6: Uncertain, consider retry
    Low,
    /// Score 0.6-0.8: Reasonable confidence
    Medium,
    /// Score 0.8-0.95: High confidence
    High,
    /// Score >= 0.95: Very high confidence
    VeryHigh,
}

/// Scores confidence in node outputs
pub struct ConfidenceScorer;

impl ConfidenceScorer {
    /// Score confidence in an output
        pub fn score<T: State>(
            output: &Value,
            task: Option<&str>,
            state: &T,
        node_name: &str,
        config: &ConfidenceConfig,
    ) -> ConfidenceScore {
        let mut indicators = Vec::new();

        // Score clarity
        let clarity = Self::score_clarity(output, &mut indicators);

        // Score relevance
        let relevance = Self::score_relevance(output, task, state, node_name, &mut indicators);

        // Score completeness
        let completeness = Self::score_completeness(output, state, node_name, &mut indicators);

        // Calculate weighted overall score
        let overall = (clarity * config.clarity_weight
            + relevance * config.relevance_weight
            + completeness * config.completeness_weight)
            / (config.clarity_weight + config.relevance_weight + config.completeness_weight);

        // Determine confidence level
        let level = if overall >= config.high_threshold {
            if overall >= 0.95 {
                ConfidenceLevel::VeryHigh
            } else {
                ConfidenceLevel::High
            }
        } else if overall >= 0.65 {
            ConfidenceLevel::Medium
        } else if overall >= config.low_threshold {
            ConfidenceLevel::Low
        } else {
            ConfidenceLevel::VeryLow
        };

        ConfidenceScore {
            overall,
            clarity,
            relevance,
            completeness,
            level,
            indicators,
        }
    }

    /// Score clarity of output (is it clear, unambiguous?)
    fn score_clarity(output: &Value, indicators: &mut Vec<String>) -> f64 {
        match output {
            Value::String(s) => {
                let trimmed = s.trim();

                if trimmed.is_empty() {
                    indicators.push("Empty response".into());
                    return 0.1;
                }

                let mut score = 0.8; // Higher base score so perfect text can exceed 0.8 threshold

                // Penalize very short responses
                if trimmed.len() < 10 {
                    indicators.push("Very short response".into());
                    score -= 0.2;
                }

                // Penalize responses with too many uncertainties
                let uncertainty_words = ["maybe", "perhaps", "possibly", "unclear"];
                let uncertainty_count = uncertainty_words
                    .iter()
                    .filter(|w| trimmed.to_lowercase().contains(**w))
                    .count();

                if uncertainty_count > 2 {
                    indicators.push("High uncertainty language".into());
                    score -= 0.15 * uncertainty_count as f64;
                }

                // Reward well-structured responses
                if trimmed.contains('\n') && trimmed.lines().count() > 1 {
                    indicators.push("Well-structured response".into());
                    score += 0.2; // Higher reward
                }

                score.clamp(0.0, 1.0)
            }
            Value::Object(o) => {
                if o.is_empty() {
                    indicators.push("Empty object".into());
                    return 0.3;
                }
                
                // Try to find a string value to evaluate its text
                // Usually the main content is the longest string in the object
                let mut longest_str = "";
                for v in o.values() {
                    if let Value::String(s) = v {
                        if s.len() > longest_str.len() {
                            longest_str = s;
                        }
                    }
                }
                
                if !longest_str.is_empty() {
                    // Evaluate the text found inside the object
                    Self::score_clarity(&Value::String(longest_str.to_string()), indicators)
                } else if has_descriptive_keys(o) {
                    indicators.push("Descriptive object structure".into());
                    0.9
                } else {
                    0.6
                }
            }
            Value::Array(a) => {
                if a.is_empty() {
                    indicators.push("Empty array".into());
                    0.2
                } else if a.len() > 10 {
                    indicators.push("Large comprehensive result set".into());
                    0.85
                } else {
                    0.7
                }
            }
            _ => 0.6,
        }
    }

    /// Score relevance to task  
        fn score_relevance<T: State>(
            output: &Value,
            task: Option<&str>,
            _state: &T,
        _node_name: &str,
        indicators: &mut Vec<String>,
    ) -> f64 {
        // If we have task description, can score relevance
        if let Some(task_desc) = task {
            // Check if output mentions key concepts from task
            // Extract string representation for keyword matching
            let output_str = match output {
                Value::Object(o) => {
                    let mut longest_str = String::new();
                    for v in o.values() {
                        if let Value::String(s) = v {
                            if s.len() > longest_str.len() {
                                longest_str = s.clone();
                            }
                        }
                    }
                    if longest_str.is_empty() {
                        output.to_string()
                    } else {
                        longest_str
                    }
                }
                _ => output.to_string()
            }.to_lowercase();
            
            let task_str = task_desc.to_lowercase();

            // Simple keyword matching
            let key_words: Vec<&str> = task_str.split_whitespace().collect();
            let matched = key_words
                .iter()
                .filter(|&&w| output_str.contains(w))
                .count();

            let relevance = if key_words.is_empty() {
                0.7
            } else {
                (matched as f64) / (key_words.len() as f64)
            };

            if matched > key_words.len() / 2 {
                indicators.push("Relevant to task".into());
            }

            relevance.clamp(0.4, 1.0)
        } else {
            // Without task description, assume highly relevant
            0.9
        }
    }

    /// Score completeness of output
        fn score_completeness<T: State>(
            output: &Value,
            _state: &T,
        _node_name: &str,
        indicators: &mut Vec<String>,
    ) -> f64 {
        // Check if output addresses expected fields
        let mut score: f64 = 0.7;

        match output {
            Value::Object(o) => {
                if o.is_empty() {
                    return 0.3;
                }
                
                // Track if it has common expected API response fields
                let common_fields = ["result", "output", "response", "data", "items", "status"];
                let has_common = common_fields.iter().any(|f: &&str| o.contains_key(*f));

                // Find the longest string to evaluate text completeness
                let mut longest_str = "";
                for v in o.values() {
                    if let Value::String(s) = v {
                        if s.len() > longest_str.len() {
                            longest_str = s;
                        }
                    }
                }
                
                if !longest_str.is_empty() {
                    // Score the actual text content found
                    let mut text_score = Self::score_completeness(&Value::String(longest_str.to_string()), _state, _node_name, indicators);
                    if has_common {
                        text_score += 0.1; // Bonus for good structure
                    }
                    text_score.clamp(0.0, 1.0)
                } else {
                    // Objects should have multiple fields if no text content is found
                    let field_count = o.len();
                    if field_count >= 3 {
                        indicators.push("Multiple output fields".into());
                        score += 0.2;
                    }
                    if has_common {
                        indicators.push("Contains expected fields".into());
                        score += 0.1;
                    }
                    score.clamp(0.0, 1.0)
                }
            }
            Value::String(s) => {
                let words = s.split_whitespace().count();
                if words >= 50 {
                    indicators.push("Very comprehensive response".into());
                    score = 1.0;
                } else if words >= 20 {
                    indicators.push("Comprehensive response".into());
                    score = 0.9;
                } else if words >= 10 {
                    indicators.push("Adequate response length".into());
                    score = 0.75;
                } else {
                    indicators.push("Brief response".into());
                    score = 0.5;
                }
                score
            }
            Value::Array(a) => {
                if a.len() >= 5 {
                    indicators.push("Multiple results provided".into());
                    score = 0.85;
                } else if a.len() > 1 {
                    indicators.push("Multiple items".into());
                    score = 0.75;
                }
                score
            }
            _ => 0.6,
        }
    }
}

/// Check if object has descriptive (non-metadata) keys
fn has_descriptive_keys(obj: &serde_json::Map<String, Value>) -> bool {
    obj.keys().filter(|k| !k.starts_with('_')).count() >= 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::SharedState;
    use serde_json::json;

    #[test]
    fn test_confidence_score_string() {
        let config = ConfidenceConfig::default();
        let state = SharedState::new(Default::default());

        let output = json!("This is a comprehensive response with details");
        let score = ConfidenceScorer::score(&output, None, &state, "test", &config);

        assert!(score.overall > 0.0);
        assert_eq!(score.level, ConfidenceLevel::Medium);
    }

    #[test]
    fn test_confidence_score_object() {
        let config = ConfidenceConfig::default();
        let state = SharedState::new(Default::default());

        let output = json!({"result": "ok", "status": "success", "data": [1,2,3]});
        let score = ConfidenceScorer::score(&output, None, &state, "test", &config);

        assert!(score.overall > 0.7);
    }

    #[test]
    fn test_confidence_level_categorization() {
        let config = ConfidenceConfig::default();
        let state = SharedState::new(Default::default());

        let low_output = json!("");
        let low_score = ConfidenceScorer::score(&low_output, None, &state, "test", &config);
        assert_eq!(low_score.level, ConfidenceLevel::Low);
    }

    #[test]
    fn test_clarity_score() {
        let mut indicators = Vec::new();
        let score = ConfidenceScorer::score_clarity(
            &json!("Well structured\nWith multiple\nLines of text"),
            &mut indicators,
        );
        assert!(score > 0.7);
    }
}
