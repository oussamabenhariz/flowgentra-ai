//! # Auto-Evaluation Middleware
//!
//! Middleware that automatically evaluates node outputs and triggers retries when needed.

use crate::core::evaluation::{
    ConfidenceConfig, ConfidenceScorer, EvaluationPolicy, EvaluationResultBuilder, NodeScorer,
    RetryConfig, RetryPolicy, ScoringCriteria,
};
use crate::core::middleware::{
    ExecutionContext as MiddlewareContext, Middleware, MiddlewareResult,
};

/// Auto-evaluation middleware that assesses and corrects node outputs
pub struct AutoEvaluationMiddleware {
    /// Evaluation policy
    policy: EvaluationPolicy,

    /// Scoring criteria
    scoring_criteria: ScoringCriteria,

    /// Confidence configuration
    confidence_config: ConfidenceConfig,

    /// Retry configuration
    retry_config: RetryConfig,

    /// Task description for context
    task_description: Option<String>,
}

impl AutoEvaluationMiddleware {
    /// Create a new auto-evaluation middleware
    pub fn new() -> Self {
        AutoEvaluationMiddleware {
            policy: EvaluationPolicy::default(),
            scoring_criteria: ScoringCriteria::default(),
            confidence_config: ConfidenceConfig::default(),
            retry_config: RetryConfig::default(),
            task_description: None,
        }
    }

    /// Set the evaluation policy
    pub fn with_policy(mut self, policy: EvaluationPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set task description for relevance evaluation
    pub fn with_task(mut self, task: String) -> Self {
        self.task_description = Some(task);
        self
    }

    /// Set scoring criteria
    pub fn with_scoring_criteria(mut self, criteria: ScoringCriteria) -> Self {
        self.scoring_criteria = criteria;
        self
    }

    /// Set confidence configuration
    pub fn with_confidence_config(mut self, config: ConfidenceConfig) -> Self {
        self.confidence_config = config;
        self
    }

    /// Set retry configuration
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }
}

impl Default for AutoEvaluationMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Middleware for AutoEvaluationMiddleware {
    /// Before node execution - prepare evaluation context
    async fn before_node(&self, ctx: &mut MiddlewareContext) -> MiddlewareResult {
        let node_name = &ctx.node_name;

        // Initialize evaluation history for this node if needed
        let history_key = format!("__node_output_history__{}", node_name);

        if !ctx.state.contains_key(&history_key) {
            ctx.state.set(&history_key, serde_json::json!([]));
        }

        MiddlewareResult::Continue
    }

    /// After node execution - evaluate and potentially retry
    async fn after_node(&self, ctx: &mut MiddlewareContext) -> MiddlewareResult {
        let node_name = &ctx.node_name;

        if !self.policy.enable_scoring
            && !self.policy.enable_grading
            && !self.policy.enable_confidence_scoring
        {
            return MiddlewareResult::Continue;
        }

        // Get the node output (last modified value in state)
        let output_key = format!("__node_output__{}", node_name);
        let node_output = if let Some(output) = ctx.state.get(&output_key) {
            output.clone()
        } else {
            serde_json::json!({})
        };

        let mut eval_result = EvaluationResultBuilder::new();

        // 1. Node Scoring
        if self.policy.enable_scoring {
            let score =
                NodeScorer::score(&node_output, &self.scoring_criteria, &ctx.state, node_name);

            tracing::debug!(
                node = node_name,
                overall_score = score.overall,
                explanation = &score.explanation,
                "Node output scored"
            );

            eval_result = eval_result.with_score(score);
        }

        // 3. Confidence Scoring
        if self.policy.enable_confidence_scoring {
            let confidence = ConfidenceScorer::score(
                &node_output,
                self.task_description.as_deref(),
                &ctx.state,
                node_name,
                &self.confidence_config,
            );

            tracing::debug!(
                node = node_name,
                overall_confidence = confidence.overall,
                level = ?confidence.level,
                "Confidence scored"
            );

            eval_result = eval_result.with_confidence(confidence);
        }

        let eval = eval_result.build();

        // 4. Store evaluation in state
        if self.policy.store_evaluation_history {
            ctx.state.set_evaluation(node_name, eval.clone());
        }

        // 5. Automatic Retry Decision
        if self.policy.enable_confidence_scoring {
            if let Some(conf) = &eval.confidence {
                let should_retry = RetryPolicy::should_retry(conf.overall, 0, &self.retry_config);

                if should_retry {
                    tracing::info!(
                        node = node_name,
                        confidence = conf.overall,
                        threshold = self.retry_config.confidence_threshold,
                        "Low confidence detected - marking for retry"
                    );

                    // Generate feedback prompt for self-correction
                    let feedback = format!(
                        "Your output has low confidence ({:.2}). Please revise to improve clarity, relevance, and completeness.",
                        conf.overall
                    );

                    // Store retry feedback in state for the node handler to use
                    let feedback_key = format!("__node_feedback__{}", node_name);
                    ctx.state.set(&feedback_key, serde_json::json!(feedback));

                    // Mark that retry is needed
                    let retry_key = format!("__node_retry_needed__{}", node_name);
                    ctx.state.set(&retry_key, serde_json::json!(true));

                    tracing::debug!(node = node_name, "Retry feedback stored in state");
                }
            }
        }

        MiddlewareResult::Continue
    }

    fn name(&self) -> &str {
        "AutoEvaluationMiddleware"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_middleware_creation() {
        let middleware = AutoEvaluationMiddleware::new()
            .with_task("Test task".to_string())
            .with_policy(EvaluationPolicy::default());

        assert!(middleware.policy.enable_scoring);
        assert_eq!(middleware.task_description, Some("Test task".to_string()));
    }
}
