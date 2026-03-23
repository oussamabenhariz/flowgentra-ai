//! # Auto-Evaluation Module (MIGRATED TO core::node::evaluation)
//!
//! This module has been integrated into the node system.
//! For new code, use: use flowgentra_ai::core::node::evaluation_node::*;
//! 
//! Legacy imports are maintained for backward compatibility.

// Re-export from the consolidated evaluation_node module
pub use crate::core::node::evaluation_node::{
    EvaluationNode, EvaluationNodeConfig,
    Attempt, ExitReason, EvaluationResult,
};

pub mod confidence;
pub mod grader;
pub mod middleware;
pub mod retry;
pub mod scorer;
pub mod reporting;
pub mod smart_fallback;

pub use confidence::{ConfidenceConfig, ConfidenceLevel, ConfidenceScore, ConfidenceScorer};
pub use grader::{GradeResult, LLMGrader};
pub use middleware::AutoEvaluationMiddleware;
pub use retry::{RetryConfig, RetryPolicy, RetryResult};
pub use scorer::{NodeScore, NodeScorer, ScoringCriteria};
pub use reporting::EvaluationReport;
pub use smart_fallback::{FallbackLevel, SmartFallback};

// Legacy types for backward compatibility
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LegacyEvaluationResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<NodeScore>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grade: Option<GradeResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<ConfidenceScore>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_info: Option<RetryResult>,
}

impl LegacyEvaluationResult {
    pub fn new() -> Self {
        LegacyEvaluationResult {
            score: None,
            grade: None,
            confidence: None,
            retry_info: None,
        }
    }

    pub fn passed(&self, threshold: f64) -> bool {
        match &self.confidence {
            Some(conf) => conf.overall >= threshold,
            None => false,
        }
    }
}

impl Default for LegacyEvaluationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Legacy evaluation policy configuration
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct LegacyEvaluationPolicy {
    pub enable_scoring: bool,
    pub enable_grading: bool,
    pub enable_confidence_scoring: bool,
    pub confidence_threshold: f64,
    pub max_retries: u32,
    pub enable_self_correction: bool,
    pub store_evaluation_history: bool,
}

/// Builder for evaluation results
#[derive(Clone, Debug, Default)]
pub struct LegacyEvaluationResultBuilder {
    score: Option<NodeScore>,
    grade: Option<GradeResult>,
    confidence: Option<ConfidenceScore>,
    retry_info: Option<RetryResult>,
}

impl LegacyEvaluationResultBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn score(mut self, score: NodeScore) -> Self {
        self.score = Some(score);
        self
    }

    pub fn grade(mut self, grade: GradeResult) -> Self {
        self.grade = Some(grade);
        self
    }

    pub fn confidence(mut self, confidence: ConfidenceScore) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn retry_info(mut self, retry_info: RetryResult) -> Self {
        self.retry_info = Some(retry_info);
        self
    }

    // Alias methods for backward compatibility
    pub fn with_score(mut self, score: NodeScore) -> Self {
        self.score = Some(score);
        self
    }

    pub fn with_confidence(mut self, confidence: ConfidenceScore) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn build(self) -> LegacyEvaluationResult {
        LegacyEvaluationResult {
            score: self.score,
            grade: self.grade,
            confidence: self.confidence,
            retry_info: self.retry_info,
        }
    }
}
