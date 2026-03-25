//! # Advanced Node Types for Loops and Branching
//!
//! Provides specialized node types for:
//! - Loop management (with iteration limits and break conditions)
//! - Parallel branching (concurrent execution of multiple branches)
//! - Subgraph composition (executing graphs within graphs)
//!
//! ## Architecture
//!
//! These node types extend the basic Node model with additional semantics:
//!
//! - **LoopNode** - Manages repeated execution of a subgraph
//! - **ParallelNode** - Executes multiple branches concurrently
//! - **SubgraphNode** - Invokes another graph as a node
//! - **JoinNode** - Synchronizes multiple parallel branches
//!
//! ## Examples
//!
//! ```ignore
//! use flowgentra_ai::core::advanced_nodes::*;
//!
//! // Create a loop node
//! let loop_node = LoopConfig::new("retry_handler")
//!     .with_max_iterations(3)
//!     .with_break_condition("is_successful");
//!
//! // Create a parallel node
//! let parallel = ParallelNodeConfig::new("process_multiple")
//!     .add_branch("branch_a")
//!     .add_branch("branch_b")
//!     .with_join_type(JoinType::WaitAll);
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for a loop node that repeats execution
///
/// Allows a node to be executed multiple times with:
/// - Maximum iteration limit
/// - Break condition (checked each iteration)
/// - State accumulation across iterations
///
/// # Example
///
/// ```yaml
/// nodes:
///   - name: retry_handler
///     type: loop
///     handler: attempt_handler
///     loop_config:
///       max_iterations: 3
///       break_condition: "is_successful"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopNodeConfig {
    /// Handler function to repeat
    pub handler: String,

    /// Maximum number of iterations
    pub max_iterations: usize,

    /// Optional break condition expression
    /// If this evaluates to true, the loop exits early
    pub break_condition: Option<String>,

    /// Whether to accumulate results or use last result
    #[serde(default)]
    pub accumulate_results: bool,

    /// Additional configuration for the handler
    #[serde(default)]
    pub config: std::collections::HashMap<String, serde_json::Value>,
}

impl LoopNodeConfig {
    /// Build a LoopNodeConfig from a NodeConfig (YAML deserialization target).
    /// Fields are read from `node.config` map; `handler` from `node.handler`.
    pub fn from_node_config(
        node: &crate::core::node::NodeConfig,
    ) -> crate::core::error::Result<Self> {
        let max_iterations = node
            .config
            .get("max_iterations")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let break_condition = node
            .config
            .get("break_condition")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let accumulate_results = node
            .config
            .get("accumulate_results")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Ok(LoopNodeConfig {
            handler: node.handler.clone(),
            max_iterations,
            break_condition,
            accumulate_results,
            config: node.config.clone(),
        })
    }

    /// Create a new loop node configuration
    pub fn new(handler: impl Into<String>) -> Self {
        LoopNodeConfig {
            handler: handler.into(),
            max_iterations: 1,
            break_condition: None,
            accumulate_results: false,
            config: std::collections::HashMap::new(),
        }
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set break condition
    pub fn with_break_condition(mut self, condition: impl Into<String>) -> Self {
        self.break_condition = Some(condition.into());
        self
    }

    /// Enable result accumulation
    pub fn with_accumulation(mut self, accumulate: bool) -> Self {
        self.accumulate_results = accumulate;
        self
    }

    /// Add handler configuration
    pub fn with_config(
        mut self,
        config: std::collections::HashMap<String, serde_json::Value>,
    ) -> Self {
        self.config = config;
        self
    }
}

/// Configuration for parallel branch execution
///
/// Allows a node to spawn multiple concurrent branches that:
/// - Execute in parallel
/// - Join at a synchronization point
/// - Share or merge state
///
/// # Example
///
/// ```yaml
/// nodes:
///   - name: parallel_processor
///     type: parallel
///     branches:
///       - name: branch_a
///         handler: process_a
///       - name: branch_b
///         handler: process_b
///     join_type: wait_all
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelNodeConfig {
    /// Name of the parallel node
    pub name: String,

    /// Branches to execute in parallel
    pub branches: Vec<BranchConfig>,

    /// How to wait for branches to complete
    pub join_type: JoinType,

    /// Timeout for parallel execution (ms)
    #[serde(default)]
    pub timeout_ms: Option<u64>,

    /// Whether to continue on branch failure
    #[serde(default)]
    pub continue_on_error: bool,
}

/// Configuration for a single branch in parallel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchConfig {
    /// Branch identifier
    pub name: String,

    /// Handler to execute in this branch
    pub handler: String,

    /// MCP tools available to this branch
    #[serde(default)]
    pub mcps: Vec<String>,

    /// Branch-specific configuration
    #[serde(default)]
    pub config: std::collections::HashMap<String, serde_json::Value>,
}

impl BranchConfig {
    /// Create a new branch configuration
    pub fn new(name: impl Into<String>, handler: impl Into<String>) -> Self {
        BranchConfig {
            name: name.into(),
            handler: handler.into(),
            mcps: Vec::new(),
            config: std::collections::HashMap::new(),
        }
    }

    /// Add MCP tools to this branch
    pub fn with_mcps(mut self, mcps: Vec<String>) -> Self {
        self.mcps = mcps;
        self
    }

    /// Add configuration
    pub fn with_config(
        mut self,
        config: std::collections::HashMap<String, serde_json::Value>,
    ) -> Self {
        self.config = config;
        self
    }
}

/// How to join parallel branches
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum JoinType {
    /// Wait for all branches to complete
    WaitAll,

    /// Continue as soon as any branch completes
    WaitAny,

    /// Wait for a specific number of branches
    WaitCount(usize),

    /// Timeout-based join
    WaitTimeout,
}

impl Default for ParallelNodeConfig {
    fn default() -> Self {
        Self::new("_default")
    }
}

impl ParallelNodeConfig {
    /// Create a new parallel node configuration
    pub fn new(name: impl Into<String>) -> Self {
        ParallelNodeConfig {
            name: name.into(),
            branches: Vec::new(),
            join_type: JoinType::WaitAll,
            timeout_ms: None,
            continue_on_error: false,
        }
    }

    /// Add a branch to execute
    pub fn add_branch(mut self, handler: impl Into<String>) -> Self {
        let name = format!("branch_{}", self.branches.len());
        self.branches.push(BranchConfig::new(&name, handler));
        self
    }

    /// Add a named branch
    pub fn add_named_branch(mut self, name: impl Into<String>, handler: impl Into<String>) -> Self {
        self.branches.push(BranchConfig::new(name, handler));
        self
    }

    /// Set join type
    pub fn with_join_type(mut self, join_type: JoinType) -> Self {
        self.join_type = join_type;
        self
    }

    /// Set timeout for parallel execution
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }

    /// Set whether to continue on error
    pub fn with_continue_on_error(mut self, continue_on_error: bool) -> Self {
        self.continue_on_error = continue_on_error;
        self
    }
}

/// Configuration for subgraph nodes
///
/// Allows invoking another graph as a node in the current graph.
///
/// # Example
///
/// ```yaml
/// nodes:
///   - name: validation_step
///     type: subgraph
///     subgraph_path: validation.yaml
///     parameters:
///       strict_mode: true
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgraphNodeConfig {
    /// Name of this subgraph node
    pub name: String,

    /// Path to the subgraph configuration file
    pub subgraph_path: String,

    /// Parameters to pass to the subgraph
    #[serde(default)]
    pub parameters: std::collections::HashMap<String, serde_json::Value>,

    /// Whether to inherit parent state
    #[serde(default = "default_inherit_state")]
    pub inherit_state: bool,

    /// State keys to pass from parent
    #[serde(default)]
    pub map_input_keys: Vec<String>,

    /// State keys to extract from subgraph result
    #[serde(default)]
    pub map_output_keys: Vec<String>,

    /// Timeout for subgraph execution (ms)
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

fn default_inherit_state() -> bool {
    true
}

impl SubgraphNodeConfig {
    /// Create a new subgraph node configuration
    pub fn new(name: impl Into<String>, subgraph_path: impl Into<String>) -> Self {
        SubgraphNodeConfig {
            name: name.into(),
            subgraph_path: subgraph_path.into(),
            parameters: std::collections::HashMap::new(),
            inherit_state: true,
            map_input_keys: Vec::new(),
            map_output_keys: Vec::new(),
            timeout_ms: None,
        }
    }

    /// Add a parameter for the subgraph
    pub fn with_parameter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.parameters.insert(key.into(), value);
        self
    }

    /// Set whether to inherit parent state
    pub fn with_inherit_state(mut self, inherit: bool) -> Self {
        self.inherit_state = inherit;
        self
    }

    /// Add input key mapping
    pub fn add_input_mapping(mut self, key: impl Into<String>) -> Self {
        self.map_input_keys.push(key.into());
        self
    }

    /// Add output key mapping
    pub fn add_output_mapping(mut self, key: impl Into<String>) -> Self {
        self.map_output_keys.push(key.into());
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
}

/// Configuration for join points in parallel execution
///
/// A join node synchronizes results from parallel branches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinNodeConfig {
    /// Name of this join node
    pub name: String,

    /// How to merge results from branches
    pub merge_strategy: MergeStrategy,

    /// Keys to merge into state
    #[serde(default)]
    pub merge_keys: Vec<String>,

    /// Whether to fail if any branch failed
    #[serde(default = "default_fail_on_error")]
    pub fail_on_error: bool,

    /// Custom merge function (if applicable)
    #[serde(default)]
    pub merge_function: Option<String>,
}

fn default_fail_on_error() -> bool {
    true
}

/// Strategy for merging parallel branch results
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Merge all results into a "results" array
    Combine,

    /// Take the first successful result
    First,

    /// Take the last result
    Last,

    /// Merge as object with branch names as keys
    ByBranch,

    /// Custom merge logic
    Custom,
}

impl JoinNodeConfig {
    /// Create a new join node configuration
    pub fn new(name: impl Into<String>) -> Self {
        JoinNodeConfig {
            name: name.into(),
            merge_strategy: MergeStrategy::Combine,
            merge_keys: Vec::new(),
            fail_on_error: true,
            merge_function: None,
        }
    }

    /// Set merge strategy
    pub fn with_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }

    /// Add a key to merge
    pub fn add_merge_key(mut self, key: impl Into<String>) -> Self {
        self.merge_keys.push(key.into());
        self
    }

    /// Set fail_on_error
    pub fn with_fail_on_error(mut self, fail: bool) -> Self {
        self.fail_on_error = fail;
        self
    }

    /// Set custom merge function
    pub fn with_merge_function(mut self, function: impl Into<String>) -> Self {
        self.merge_function = Some(function.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_loop_node_config() {
        let loop_config = LoopNodeConfig::new("handler")
            .with_max_iterations(3)
            .with_break_condition("is_done");

        assert_eq!(loop_config.max_iterations, 3);
        assert_eq!(loop_config.break_condition, Some("is_done".to_string()));
    }

    #[test]
    fn test_parallel_node_config() {
        let parallel = ParallelNodeConfig::new("parallel")
            .add_named_branch("branch_a", "handler_a")
            .add_named_branch("branch_b", "handler_b")
            .with_join_type(JoinType::WaitAll);

        assert_eq!(parallel.branches.len(), 2);
        assert_eq!(parallel.join_type, JoinType::WaitAll);
    }

    #[test]
    fn test_subgraph_node_config() {
        let subgraph =
            SubgraphNodeConfig::new("sub", "sub.yaml").with_parameter("mode", json!("strict"));

        assert_eq!(subgraph.name, "sub");
        assert!(subgraph.parameters.contains_key("mode"));
    }

    #[test]
    fn test_join_node_config() {
        let join = JoinNodeConfig::new("join").with_strategy(MergeStrategy::ByBranch);

        assert_eq!(join.merge_strategy, MergeStrategy::ByBranch);
    }
}
