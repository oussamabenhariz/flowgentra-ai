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
//! use flowgentra_ai::core::parallel::{ParallelExecutor, JoinStrategy};
//! use flowgentra_ai::core::state::State;
//!
//! let mut executor = ParallelExecutor::new()
//!     .with_join_strategy(JoinStrategy::WaitAll);
//! ```

use crate::core::advanced_nodes::{JoinType, MergeStrategy};
use crate::core::error::{FlowgentraError, Result};
use crate::core::state::State;
use serde_json::{json, Value};
use std::time::Duration;
use tokio::task::JoinSet;
use tokio::time::timeout;

/// Result from a single branch execution
#[derive(Debug, Clone)]
pub struct BranchResult<T: State> {
    /// Name of the branch
    pub branch_name: String,

    /// Resulting state from branch execution
    pub state: T,

    /// Duration of branch execution
    pub duration: Duration,

    /// Whether the branch succeeded
    pub success: bool,

    /// Error if branch failed
    pub error: Option<String>,
}

/// Manager for parallel branch execution.
///
/// Executes multiple async branches concurrently with configurable
/// join strategies, timeouts, and merge behavior.
///
/// # Example
/// ```ignore
/// let executor = ParallelExecutor::new()
///     .with_join_type(JoinType::WaitAll)
///     .with_merge_strategy(MergeStrategy::Combine)
///     .with_timeout(Duration::from_secs(30));
///
/// let result = executor.execute(state, branches).await?;
/// ```
pub struct ParallelExecutor {
    join_type: JoinType,
    timeout: Option<Duration>,
    continue_on_error: bool,
    merge_strategy: MergeStrategy,
}

impl ParallelExecutor {
    /// Create a new parallel executor with default settings.
    pub fn new() -> Self {
        ParallelExecutor {
            join_type: JoinType::WaitAll,
            timeout: None,
            continue_on_error: false,
            merge_strategy: MergeStrategy::Combine,
        }
    }

    /// Set the join strategy.
    pub fn with_join_type(mut self, join_type: JoinType) -> Self {
        self.join_type = join_type;
        self
    }

    /// Set the merge strategy for combining branch results.
    pub fn with_merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }

    /// Set an overall timeout for parallel execution.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set whether to continue when a branch errors.
    pub fn with_continue_on_error(mut self, continue_on_error: bool) -> Self {
        self.continue_on_error = continue_on_error;
        self
    }

    /// Execute multiple branches in parallel and merge the results.
    ///
    /// Each branch is an async function that takes the initial state and
    /// returns a named `BranchResult`. All branches run concurrently;
    /// results are collected per the join strategy and merged per the
    /// merge strategy.
    pub async fn execute<T: State + Default + Send + Sync + 'static>(
        &self,
        initial_state: T,
        branches: Vec<(String, Box<dyn Fn(T) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>> + Send + Sync>)>,
    ) -> Result<T> {
        let mut join_set: JoinSet<BranchResult<T>> = JoinSet::new();

        for (name, func) in branches {
            let state_clone = initial_state.clone();
            let branch_name = name.clone();
            join_set.spawn(async move {
                let start = std::time::Instant::now();
                match func(state_clone).await {
                    Ok(result_state) => BranchResult {
                        branch_name,
                        state: result_state,
                        duration: start.elapsed(),
                        success: true,
                        error: None,
                    },
                    Err(e) => BranchResult {
                        branch_name,
                        state: T::empty(),
                        duration: start.elapsed(),
                        success: false,
                        error: Some(e.to_string()),
                    },
                }
            });
        }

        let results = self.collect_results(&mut join_set).await?;

        if !self.continue_on_error {
            if let Some(failed) = results.iter().find(|r| !r.success) {
                return Err(FlowgentraError::ExecutionError(
                    format!("Branch '{}' failed: {}", failed.branch_name, failed.error.as_deref().unwrap_or("unknown"))
                ));
            }
        }

        self.merge_results(&initial_state, &results)
    }

    /// Collect results from all branches based on join strategy
    async fn collect_results<T: crate::core::state::State>(
        &self,
        join_set: &mut JoinSet<BranchResult<T>>,
    ) -> Result<Vec<BranchResult<T>>> {
        let timeout_duration = self.timeout;

        match self.join_type {
            JoinType::WaitAll => {
                let collect_future = async {
                    let mut results = Vec::new();
                    while let Some(res) = join_set.join_next().await {
                        if let Ok(branch_result) = res {
                            results.push(branch_result);
                        }
                    }
                    results
                };

                if let Some(dur) = timeout_duration {
                    match timeout(dur, collect_future).await {
                        Ok(results) => Ok(results),
                        Err(_) => Err(FlowgentraError::ExecutionTimeout(
                            "Parallel execution timeout waiting for all branches".to_string(),
                        )),
                    }
                } else {
                    Ok(collect_future.await)
                }
            }
            JoinType::WaitAny => {
                let any_future = async {
                    let mut results = Vec::new();
                    if let Some(Ok(branch_result)) = join_set.join_next().await {
                        results.push(branch_result);
                    }
                    results
                };

                if let Some(dur) = timeout_duration {
                    match timeout(dur, any_future).await {
                        Ok(results) => Ok(results),
                        Err(_) => Err(FlowgentraError::ExecutionTimeout(
                            "Parallel execution timeout".to_string(),
                        )),
                    }
                } else {
                    Ok(any_future.await)
                }
            }
            JoinType::WaitCount(count) => {
                let count_future = async {
                    let mut results = Vec::new();
                    while let Some(res) = join_set.join_next().await {
                        if let Ok(branch_result) = res {
                            results.push(branch_result);
                            if results.len() >= count {
                                break;
                            }
                        }
                    }
                    results
                };

                if let Some(dur) = timeout_duration {
                    match timeout(dur, count_future).await {
                        Ok(results) => Ok(results),
                        Err(_) => Err(FlowgentraError::ExecutionTimeout(
                            "Parallel execution timeout".to_string(),
                        )),
                    }
                } else {
                    Ok(count_future.await)
                }
            }
            JoinType::WaitTimeout => {
                // Use timeout or default
                let dur = timeout_duration.unwrap_or(Duration::from_secs(30));

                let collect_future = async {
                    let mut results = Vec::new();
                    while let Some(res) = join_set.join_next().await {
                        if let Ok(branch_result) = res {
                            results.push(branch_result);
                        }
                    }
                    results
                };

                match timeout(dur, collect_future).await {
                    Ok(results) => Ok(results),
                    Err(_) => Err(FlowgentraError::ExecutionTimeout(
                        "Parallel execution timeout".to_string(),
                    )),
                }
            }
        }
    }

    /// Merge results from branches into a single state
    fn merge_results<T: crate::core::state::State>(&self, initial_state: &T, results: &[BranchResult<T>]) -> Result<T> {
        let merged = initial_state.clone();

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
                    // State::merge(&self) doesn't return Result, just performs merge
                    first.state.merge(first.state.clone());
                    merged.merge(first.state.clone());
                }
            }

            MergeStrategy::Last => {
                // Take last successful result
                if let Some(last) = results.iter().rfind(|r| r.success) {
                    // State::merge(&self) doesn't return Result, just performs merge
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
                        // State::merge(&self) doesn't return Result, just performs merge
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
pub struct BranchSync<T: crate::core::state::State> {
    /// Name of this sync point
    pub name: String,

    /// Results collected so far
    pub results: Vec<BranchResult<T>>,

    /// Whether all branches completed
    pub all_complete: bool,

    /// Time when sync started
    pub created_at: std::time::Instant,
}

impl<T: crate::core::state::State> BranchSync<T> {
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
    pub fn add_result(&mut self, result: BranchResult<T>) {
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

    /// Merge all results into a single state, starting from the given initial state.
    pub fn merge_all(&self, initial: T) -> T {
        let merged = initial;

        for result in &self.results {
            merged.merge(result.state.clone());
        }

        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::SharedState;

    #[tokio::test]
    async fn test_parallel_executor_creation() {
        let _executor = ParallelExecutor::new();
        // ParallelExecutor does not have with_join_type/with_merge_strategy builder methods
    }

    #[tokio::test]
    async fn test_branch_sync() {
        let mut sync: BranchSync<SharedState> = BranchSync::new("test_sync");

        let result = BranchResult {
            branch_name: "branch_1".to_string(),
            state: SharedState::new(Default::default()),
            duration: Duration::from_millis(100),
            success: true,
            error: None,
        };

        sync.add_result(result);
        assert_eq!(sync.successful_count(), 1);
    }
}
