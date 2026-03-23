//! # Evaluation Node
//!
//! Public-facing wrapper that encapsulates the runner.

use crate::core::error::Result;
use crate::core::state::SharedState;
use super::evaluation_node_runner::{EvaluationNodeRunner, EvaluationResult};

/// An evaluation node combines a handler with automatic retry logic.
///
/// The node:
/// 1. Executes the handler
/// 2. Evaluates the output (0.0-1.0)
/// 3. If confident enough, returns result
/// 4. If not confident and retries remain, adds feedback and retries
/// 5. After max retries, returns best attempt
///
/// # Example
///
/// ```ignore
/// use flowgentra_ai::core::evaluation::EvaluationNode;
///
/// let mut node = EvaluationNodeBuilder::new()
///     .name("generate")
///     .handler(my_handler)
///     .min_confidence(0.85)
///     .max_retries(3)
///     .build()?;
///
/// let result = node.execute(state).await?;
/// match result {
///     EvaluationResult { success: true, .. } => println!("✅ Success"),
///     EvaluationResult { exit_reason, .. } => println!("Retried: {:?}", exit_reason),
/// }
/// ```
pub struct EvaluationNode {
    runner: EvaluationNodeRunner,
}

impl EvaluationNode {
    /// Create a new evaluation node from a runner
    pub(crate) fn new(runner: EvaluationNodeRunner) -> Self {
        EvaluationNode { runner }
    }

    /// Execute the evaluation node with retry logic
    ///
    /// This runs the handler in a loop until:
    /// - Output reaches min_confidence threshold, or
    /// - Max retries exhausted (returns best attempt), or
    /// - Fatal error occurs
    ///
    /// # Arguments
    ///
    /// * `state` - Initial execution state
    ///
    /// # Returns
    ///
    /// EvaluationResult containing:
    /// - success flag
    /// - final attempt
    /// - all attempts (for history)
    /// - best attempt (highest score)
    /// - exit reason
    pub async fn execute(&mut self, state: SharedState) -> Result<EvaluationResult> {
        self.runner.execute(state).await
    }

    /// Get a detailed execution trace for debugging
    ///
    /// # Returns
    ///
    /// Formatted string showing:
    /// - Node name and state
    /// - All attempts with scores and times
    /// - Best attempt marked with ⭐
    ///
    /// # Example Output
    ///
    /// ```text
    /// ╔════════════════════════════════════════════════════════╗
    /// ║ Node: generate_content                                ║
    /// ║ State: Complete                                       ║
    /// ║ Attempts: 3/3                                         ║
    /// ╠════════════════════════════════════════════════════════╣
    /// ║ Attempt History:                                       ║
    /// ├────────────────────────────────────────────────────────┤
    /// ║   #1: Score 0.65 | Duration: 2145ms   ║
    /// ║   #2: Score 0.78 | Duration: 2312ms   ║
    /// ║ ⭐ #3: Score 0.88 | Duration: 2087ms   ║
    /// ├────────────────────────────────────────────────────────┤
    /// ║ ✨ Best Attempt: #3 (Score: 0.88)                    ║
    /// ╚════════════════════════════════════════════════════════╝
    /// ```
    pub fn get_execution_trace(&self) -> String {
        self.runner.get_execution_trace()
    }

    /// Get the node name
    pub fn name(&self) -> &str {
        &self.runner.node_name
    }

    /// Get current policy
    pub fn policy(&self) -> &super::evaluation_node_runner::EvaluationPolicy {
        &self.runner.policy
    }

    /// Get current execution state (for debugging)
    pub fn current_state(&self) -> String {
        self.runner.current_state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_name() {
        let runner = EvaluationNodeRunner::new("test".to_string(), Default::default());
        let node = EvaluationNode::new(runner);
        assert_eq!(node.name(), "test");
    }
}
