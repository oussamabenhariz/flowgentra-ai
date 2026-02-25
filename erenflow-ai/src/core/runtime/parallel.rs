//! # Parallel Execution Engine
//!
//! Manages concurrent execution of multiple branches with advanced synchronization.
//!
//! ## Features
//!
//! - **Parallel Execution** - Run multiple branches concurrently with tokio
//! - **Join Strategies** - WaitAll, WaitAny, WaitCount, WaitTimeout
//! - **Error Handling** - Configurable error handling for failed branches
//! - **State Merging** - Multiple merge strategies for branch results
//! - **Timeout Support** - Per-branch and overall timeout management
//!
//! ## Example
//!
//! ```ignore
//! use erenflow_ai::core::parallel::{ParallelExecutor, JoinStrategy};
//! use erenflow_ai::core::state::State;
//!
//! let mut executor = ParallelExecutor::new()
//!     .with_join_strategy(JoinStrategy::WaitAll);
//! ```

use crate::core::advanced_nodes::{JoinType, MergeStrategy};
use crate::core::error::{ErenFlowError, Result};
use crate::core::state::State;
use serde_json::{json, Value};
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio::time::timeout;

/// Result from a single branch execution
#[derive(Debug, Clone)]
pub struct BranchResult {
    /// Name of the branch
    pub branch_name: String,

    /// Resulting state from branch execution
    pub state: State,

    /// Duration of branch execution
    pub duration: Duration,

    /// Whether the branch succeeded
    pub success: bool,

    /// Error if branch failed
    pub error: Option<String>,
}

/// Manager for parallel branch execution
pub struct ParallelExecutor {
    /// How to join the branches
    join_type: JoinType,

    /// Overall timeout for all branches
    timeout: Option<Duration>,

    /// Whether to continue on branch error
    continue_on_error: bool,

    /// Merge strategy for results
    merge_strategy: MergeStrategy,
}

impl ParallelExecutor {
    /// Create a new parallel executor
    pub fn new() -> Self {
        ParallelExecutor {
            join_type: JoinType::WaitAll,
            timeout: None,
            continue_on_error: false,
            merge_strategy: MergeStrategy::Combine,
        }
    }

    /// Set join type
    pub fn with_join_type(mut self, join_type: JoinType) -> Self {
        self.join_type = join_type;
        self
    }

    /// Set overall timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set whether to continue on error
    pub fn with_continue_on_error(mut self, continue_on_error: bool) -> Self {
        self.continue_on_error = continue_on_error;
        self
    }

    /// Set merge strategy
    pub fn with_merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }

    /// Execute branches in parallel
    ///
    /// # Returns
    /// - State with merged results from all branches
    /// - Results from individual branches
    pub async fn execute<F>(
        &self,
        initial_state: State,
        branches: Vec<(&str, F)>,
    ) -> Result<(State, Vec<BranchResult>)>
    where
        F: Fn(State) -> futures::future::BoxFuture<'static, Result<State>> + Send + Sync + 'static,
    {
        let _num_branches = branches.len();
        let mut handles: Vec<JoinHandle<BranchResult>> = Vec::new();

        // Spawn all branches as tokio tasks
        for (branch_name, branch_fn) in branches {
            let state = initial_state.clone();
            let branch_name = branch_name.to_string();

            let handle = tokio::spawn(async move {
                let start = std::time::Instant::now();

                match branch_fn(state).await {
                    Ok(result_state) => BranchResult {
                        branch_name,
                        state: result_state,
                        duration: start.elapsed(),
                        success: true,
                        error: None,
                    },
                    Err(e) => BranchResult {
                        branch_name,
                        state: State::new(),
                        duration: start.elapsed(),
                        success: false,
                        error: Some(e.to_string()),
                    },
                }
            });

            handles.push(handle);
        }

        // Wait for branches based on join strategy
        let results = self.collect_results(handles).await?;

        // Check for errors if configured
        if !self.continue_on_error {
            for result in &results {
                if !result.success {
                    return Err(ErenFlowError::ParallelExecutionError(format!(
                        "Branch '{}' failed: {}",
                        result.branch_name,
                        result
                            .error
                            .as_ref()
                            .unwrap_or(&"unknown error".to_string())
                    )));
                }
            }
        }

        // Merge results using configured strategy
        let merged_state = self.merge_results(&initial_state, &results)?;

        Ok((merged_state, results))
    }

    /// Collect results from all branches based on join strategy
    async fn collect_results(
        &self,
        mut handles: Vec<JoinHandle<BranchResult>>,
    ) -> Result<Vec<BranchResult>> {
        let timeout_duration = self.timeout;

        match self.join_type {
            JoinType::WaitAll => {
                // Wait for all branches to complete
                let futures = futures::future::join_all(handles);

                let results = if let Some(dur) = timeout_duration {
                    match timeout(dur, futures).await {
                        Ok(results) => results,
                        Err(_) => {
                            return Err(ErenFlowError::ExecutionTimeout(
                                "Parallel execution timeout waiting for all branches".to_string(),
                            ))
                        }
                    }
                } else {
                    futures.await
                };

                Ok(results.into_iter().filter_map(|r| r.ok()).collect())
            }

            JoinType::WaitAny => {
                // Continue as soon as first branch completes
                let mut results = Vec::new();

                loop {
                    if handles.is_empty() {
                        break;
                    }

                    let (result, _, remaining) = futures::future::select_all(handles).await;
                    handles = remaining;

                    if let Ok(branch_result) = result {
                        results.push(branch_result);
                        break; // Exit on first result
                    }
                }

                Ok(results)
            }

            JoinType::WaitCount(count) => {
                // Wait for specified number of branches
                let mut results = Vec::new();

                loop {
                    if handles.is_empty() || results.len() >= count {
                        break;
                    }

                    let (result, _, remaining) = futures::future::select_all(handles).await;
                    handles = remaining;

                    if let Ok(branch_result) = result {
                        results.push(branch_result);
                    }
                }

                Ok(results)
            }

            JoinType::WaitTimeout => {
                // Use timeout or default
                let dur = timeout_duration.unwrap_or(Duration::from_secs(30));
                let futures = futures::future::join_all(handles);

                match timeout(dur, futures).await {
                    Ok(results) => Ok(results.into_iter().filter_map(|r| r.ok()).collect()),
                    Err(_) => Err(ErenFlowError::ExecutionTimeout(
                        "Parallel execution timeout".to_string(),
                    )),
                }
            }
        }
    }

    /// Merge results from branches into a single state
    fn merge_results(&self, initial_state: &State, results: &[BranchResult]) -> Result<State> {
        let mut merged = initial_state.clone();

        match self.merge_strategy {
            MergeStrategy::Combine => {
                // Create array of all successful results
                let mut combined = Vec::new();
                for result in results {
                    if result.success {
                        combined.push(result.state.to_value());
                    }
                }
                merged.set("_parallel_results", json!(combined));
            }

            MergeStrategy::First => {
                // Take first successful result
                if let Some(first) = results.iter().find(|r| r.success) {
                    merged.merge(first.state.clone());
                }
            }

            MergeStrategy::Last => {
                // Take last successful result
                if let Some(last) = results.iter().rfind(|r| r.success) {
                    merged.merge(last.state.clone());
                }
            }

            MergeStrategy::ByBranch => {
                // Create object with branch names as keys
                let mut by_branch = serde_json::Map::new();
                for result in results {
                    by_branch.insert(result.branch_name.clone(), result.state.to_value());
                }
                merged.set("_parallel_branches", Value::Object(by_branch));
            }

            MergeStrategy::Custom => {
                // Custom merge - just combine all results
                for result in results {
                    if result.success {
                        merged.merge(result.state.clone());
                    }
                }
            }
        }

        Ok(merged)
    }
}

impl Default for ParallelExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Synchronization point for parallel branches
pub struct BranchSync {
    /// Name of this sync point
    pub name: String,

    /// Results collected so far
    pub results: Vec<BranchResult>,

    /// Whether all branches completed
    pub all_complete: bool,

    /// Time when sync started
    pub created_at: std::time::Instant,
}

impl BranchSync {
    /// Create a new branch sync point
    pub fn new(name: impl Into<String>) -> Self {
        BranchSync {
            name: name.into(),
            results: Vec::new(),
            all_complete: false,
            created_at: std::time::Instant::now(),
        }
    }

    /// Add a branch result
    pub fn add_result(&mut self, result: BranchResult) {
        self.results.push(result);
    }

    /// Mark as complete
    pub fn mark_complete(&mut self) {
        self.all_complete = true;
    }

    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get the number of successful branches
    pub fn successful_count(&self) -> usize {
        self.results.iter().filter(|r| r.success).count()
    }

    /// Get the number of failed branches
    pub fn failed_count(&self) -> usize {
        self.results.iter().filter(|r| !r.success).count()
    }

    /// Merge all results into a single state
    pub fn merge_all(&self) -> State {
        let mut merged = State::new();

        for result in &self.results {
            merged.merge(result.state.clone());
        }

        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parallel_executor_creation() {
        let executor = ParallelExecutor::new()
            .with_join_type(JoinType::WaitAll)
            .with_merge_strategy(MergeStrategy::Combine);

        assert_eq!(executor.join_type, JoinType::WaitAll);
        assert_eq!(executor.merge_strategy, MergeStrategy::Combine);
    }

    #[tokio::test]
    async fn test_branch_sync() {
        let mut sync = BranchSync::new("test_sync");

        let result = BranchResult {
            branch_name: "branch_1".to_string(),
            state: State::new(),
            duration: Duration::from_millis(100),
            success: true,
            error: None,
        };

        sync.add_result(result);
        assert_eq!(sync.successful_count(), 1);
    }
}
