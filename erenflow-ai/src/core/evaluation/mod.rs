//! # Auto-Evaluation Module
//!
//! Provides automatic evaluation and quality assessment capabilities for agent nodes.
//!
//! This module enables:
//! - **Node Scoring** - Quantitative assessment of node outputs
//! - **LLM Grading** - AI-powered evaluation using LLMs
//! - **Confidence Scoring** - Automatic confidence estimation for outputs
//! - **Automatic Retry** - Intelligent retry with self-correction
//! - **Self-Correcting Agents** - Agents that learn and improve autonomously
//!
//! ## Architecture
//!
//! The evaluation system consists of four main components:
//!
//! ```text
//! Node Execution
//!      ↓
//! Output Scoring (0.0-1.0)
//!      ↓
//! LLM Grading (Quality assessment)
//!      ↓
//! Confidence Scoring (Trust level)
//!      ↓
//! Retry Decision (If confidence < threshold)
//! ```
//!
//! ## Example
//!
//! ```ignore
//! use erenflow_ai::core::evaluation::{
//!     EvaluationPolicy, NodeScorer, ConfidenceConfig, RetryPolicy
//! };
//!
//! // Create evaluation policy
//! let policy = EvaluationPolicy {
//!     enable_scoring: true,
//!     enable_grading: true,
//!     confidence_threshold: 0.7,
//!     max_retries: 3,
//! };
//!
//! // Use in runtime
//! // runtime.set_evaluation_policy(policy);
//! ```

pub mod confidence;
pub mod grader;
pub mod middleware;
pub mod retry;
pub mod scorer;

pub use confidence::{ConfidenceConfig, ConfidenceLevel, ConfidenceScore, ConfidenceScorer};
pub use grader::{GradeResult, LLMGrader};
pub use middleware::AutoEvaluationMiddleware;
pub use retry::{RetryConfig, RetryPolicy, RetryResult};
pub use scorer::{NodeScore, NodeScorer, ScoringCriteria};

use crate::core::state::State;
use serde::{Deserialize, Serialize};

/// Unified evaluation policy for agent execution
///
/// Controls whether scoring, grading, and retries are enabled.
/// This is the primary interface for configuring evaluation behavior.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationPolicy {
    /// Enable node output scoring
    #[serde(default = "default_true")]
    pub enable_scoring: bool,

    /// Enable LLM-based output grading
    #[serde(default = "default_true")]
    pub enable_grading: bool,

    /// Enable confidence scoring
    #[serde(default = "default_true")]
    pub enable_confidence_scoring: bool,

    /// Minimum confidence score to accept output (0.0-1.0)
    /// If output confidence is below this, retry will be triggered
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f64,

    /// Maximum number of retry attempts
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Enable self-correction mode
    /// When enabled, agents receive feedback on low-confidence outputs
    #[serde(default = "default_true")]
    pub enable_self_correction: bool,

    /// Store evaluation history in state
    #[serde(default = "default_true")]
    pub store_evaluation_history: bool,
}

fn default_true() -> bool {
    true
}

fn default_confidence_threshold() -> f64 {
    0.7
}

fn default_max_retries() -> u32 {
    3
}

impl Default for EvaluationPolicy {
    fn default() -> Self {
        EvaluationPolicy {
            enable_scoring: true,
            enable_grading: true,
            enable_confidence_scoring: true,
            confidence_threshold: 0.7,
            max_retries: 3,
            enable_self_correction: true,
            store_evaluation_history: true,
        }
    }
}

/// Accumulated evaluation data for a single output
///
/// Contains all scores and assessments for tracking output quality throughout execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Node output score (0.0-1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<NodeScore>,

    /// LLM grading result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grade: Option<GradeResult>,

    /// Confidence assessment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<ConfidenceScore>,

    /// Retry information if applicable
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_info: Option<RetryResult>,
}

impl EvaluationResult {
    /// Create an empty evaluation result
    pub fn new() -> Self {
        EvaluationResult {
            score: None,
            grade: None,
            confidence: None,
            retry_info: None,
        }
    }

    /// Check if overall evaluation passed
    /// Passes if confidence >= threshold (or no confidence data)
    pub fn passed(&self, threshold: f64) -> bool {
        match &self.confidence {
            Some(conf) => conf.overall >= threshold,
            None => true, // Default to pass if no confidence data
        }
    }

    /// Get combined quality score (0.0-1.0)
    pub fn quality_score(&self) -> f64 {
        let mut scores = Vec::new();

        if let Some(score) = &self.score {
            scores.push(score.overall);
        }
        if let Some(grade) = &self.grade {
            scores.push(grade.score);
        }
        if let Some(conf) = &self.confidence {
            scores.push(conf.overall);
        }

        if scores.is_empty() {
            0.5 // Neutral score if no evaluations
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        }
    }
}

impl Default for EvaluationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating evaluation results programmatically
pub struct EvaluationResultBuilder {
    result: EvaluationResult,
}

impl EvaluationResultBuilder {
    /// Create a new evaluation result builder
    pub fn new() -> Self {
        EvaluationResultBuilder {
            result: EvaluationResult::new(),
        }
    }

    /// Add a scoring result
    pub fn with_score(mut self, score: NodeScore) -> Self {
        self.result.score = Some(score);
        self
    }

    /// Add a grading result
    pub fn with_grade(mut self, grade: GradeResult) -> Self {
        self.result.grade = Some(grade);
        self
    }

    /// Add a confidence score
    pub fn with_confidence(mut self, confidence: ConfidenceScore) -> Self {
        self.result.confidence = Some(confidence);
        self
    }

    /// Add retry information
    pub fn with_retry_info(mut self, retry_info: RetryResult) -> Self {
        self.result.retry_info = Some(retry_info);
        self
    }

    /// Build the evaluation result
    pub fn build(self) -> EvaluationResult {
        self.result
    }
}

impl Default for EvaluationResultBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to store evaluation data in state
impl State {
    /// Store evaluation result in state
    pub fn set_evaluation(&mut self, node_name: &str, result: EvaluationResult) {
        let key = format!("__evaluation__{}", node_name);
        self.set(
            &key,
            serde_json::to_value(result).unwrap_or(serde_json::json!({})),
        );
    }

    /// Retrieve evaluation result from state
    pub fn get_evaluation(&self, node_name: &str) -> Option<EvaluationResult> {
        let key = format!("__evaluation__{}", node_name);
        self.get(&key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Get entire evaluation history
    pub fn get_evaluation_history(&self) -> Vec<(String, EvaluationResult)> {
        self.keys()
            .filter(|k| k.starts_with("__evaluation__"))
            .filter_map(|k| {
                let node_name = k.strip_prefix("__evaluation__")?.to_string();
                let result = self
                    .get(k)
                    .and_then(|v| serde_json::from_value(v.clone()).ok())?;
                Some((node_name, result))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_result_quality_score() {
        let result = EvaluationResult::new();
        assert_eq!(result.quality_score(), 0.5);
    }

    #[test]
    fn test_evaluation_policy_defaults() {
        let policy = EvaluationPolicy::default();
        assert!(policy.enable_scoring);
        assert_eq!(policy.confidence_threshold, 0.7);
        assert_eq!(policy.max_retries, 3);
    }

    #[test]
    fn test_evaluation_result_builder() {
        let result = EvaluationResultBuilder::new()
            .with_confidence(ConfidenceScore {
                overall: 0.85,
                clarity: 0.90,
                relevance: 0.80,
                completeness: 0.85,
                level: ConfidenceLevel::High,
                indicators: vec!["clear_reasoning".to_string()],
            })
            .build();

        assert!(result.confidence.is_some());
        assert!(result.passed(0.7));
        assert!(!result.passed(0.9));
    }
}
