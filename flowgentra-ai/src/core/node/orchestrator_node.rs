// # Supervisor Node
//
// Coordinates multiple sub-agents or subgraphs, enabling true multi-agent orchestration.
// The supervisor acts as the "brain" — it breaks down tasks, delegates to specialized
// sub-agents, monitors outputs, decides the next step, and synthesizes final results.
//
// ## Orchestration Strategies
//
// - **Sequential**: Children run one after another; each receives the output of the previous.
//   Best for pipelines where steps are dependent and order matters.
//
// - **Parallel**: All children run simultaneously on the same input; results are merged.
//   Best when tasks have no dependencies and speed is a priority.
//
// - **Autonomous**: Loop-based orchestration selecting agents dynamically based on missing
//   outputs, stopping when all required_outputs are present or max_iterations is reached.
//
// - **Dynamic (LLM-driven)**: The supervisor asks an LLM at runtime to decide which agents
//   to call, in what order, and with what inputs. Best for complex or unpredictable tasks.
//
// - **RoundRobin**: Tasks from a state array are distributed evenly across agents in rotation.
//   Best for load balancing when agents are interchangeable.
//
// - **Hierarchical**: The supervisor delegates to sub-supervisors, each managing their own
//   group of agents. Best for large multi-domain tasks broken into sub-systems.
//
// - **Broadcast (Fan-out)**: The same task is sent to all agents simultaneously, and the
//   supervisor picks the best result. Best for quality-critical tasks needing redundancy.
//
// - **MapReduce**: Input is split into chunks (map), each chunk processed by a separate
//   agent in parallel, then all results merged into a final output (reduce).
//
// - **ConditionalRouting**: The supervisor inspects the task's type or context and routes
//   it to the most appropriate specialized agent based on routing rules.
//
// - **RetryFallback**: Agents are tried in order; if one fails, the next is tried.
//   Best for building fault-tolerant, reliable systems.
//
// - **Debate**: Multiple agents generate responses, then critique each other's outputs,
//   with the supervisor synthesizing the final answer. Best for high-accuracy tasks.
//
// ## State written
// - `__supervisor_meta__<name>` — JSON with per-child results, errors, timing stats
//
// ## Example
// ```yaml
// - name: coordinator
//   type: supervisor
//   config:
//     strategy: parallel
//     children: [research_agent, writer_agent, critic_agent]
//     child_timeout_ms: 30000
//     merge_strategy: latest
//     collect_stats: true
// ```
//
// ## Notes
// - Children can be any node type: plain handlers, subgraph nodes, evaluation nodes, etc.
// - Use `type: orchestrator` as an alias for backwards compatibility.

use std::time::{Duration, Instant};
use tracing::{error, info, warn};

/// Evaluate a simple condition expression against a DynState.
/// Supported forms: "key", "key == value", "key != value", "key == null", "key != null"
fn evaluate_condition(condition: &str, state: &crate::core::state::DynState) -> bool {
    let condition = condition.trim();
    if condition.contains("!=") {
        let parts: Vec<&str> = condition.splitn(2, "!=").collect();
        let key = parts[0].trim();
        let rhs = parts[1].trim();
        let val = state.get(key);
        if rhs == "null" {
            return val.as_ref().map(|v| !v.is_null()).unwrap_or(false);
        }
        return val
            .map(|v| v.to_string().trim_matches('"') != rhs)
            .unwrap_or(true);
    }
    if condition.contains("==") {
        let parts: Vec<&str> = condition.splitn(2, "==").collect();
        let key = parts[0].trim();
        let rhs = parts[1].trim();
        let val = state.get(key);
        if rhs == "null" {
            return val.is_none() || val.as_ref().map(|v| v.is_null()).unwrap_or(true);
        }
        return val
            .map(|v| v.to_string().trim_matches('"') == rhs)
            .unwrap_or(false);
    }
    // Bare key: true if non-null and non-false
    state
        .get(condition)
        .map(|v| !v.is_null() && v.as_bool() != Some(false))
        .unwrap_or(false)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ParallelAggregation {
    FirstSuccess,
    All,
    Majority,
}

/// Strategy for merging state from parallel child executions
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub enum ParallelMergeStrategy {
    /// Use the first successful child's state
    #[default]
    FirstSuccess,
    /// Use the last successful child's state
    Latest,
    /// Deep merge all successful states (user-defined logic)
    DeepMerge,
    /// Custom merge strategy (placeholder for future plugins)
    Custom(String),
}

/// Per-child execution statistics collected when `collect_stats: true`
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChildExecutionStats {
    pub name: String,
    pub duration_ms: u128,
    pub success: bool,
    pub error: Option<String>,
    pub timeout: bool,
}

use crate::core::error::Result;
use crate::core::node::nodes_trait::{NodeOutput, PluggableNode};
use crate::core::state::DynState;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum OrchestrationStrategy {
    /// Children run one after another; state flows through each.
    #[default]
    Sequential,
    /// All children run simultaneously on the same input; results are merged.
    Parallel,
    /// Loop-based orchestration selecting agents based on missing required outputs.
    Autonomous,
    /// LLM decides at runtime which agents to call, in what order, and with what inputs.
    Dynamic,
    /// Tasks from a state array are distributed across agents in rotation.
    RoundRobin,
    /// Delegates to sub-supervisors, each managing their own agent group.
    Hierarchical,
    /// Same task sent to all agents; supervisor picks the best result.
    Broadcast,
    /// Input split into chunks, each processed in parallel, results merged.
    MapReduce,
    /// Routes to the most appropriate agent based on state-driven routing rules.
    ConditionalRouting,
    /// Agents tried in order; if one fails, the next is tried until success.
    RetryFallback,
    /// Multiple agents generate and critique each other's outputs across rounds.
    Debate,
    /// Placeholder for user-defined custom strategies.
    Custom(String),
}

/// Configuration for the Supervisor node.
/// Parsed from YAML via `from_node_config` or constructed programmatically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisorNodeConfig {
    pub name: String,
    /// Names of child nodes (handlers, subgraphs, or other built-in nodes) to orchestrate
    pub children: Vec<String>,
    #[serde(default)]
    pub strategy: OrchestrationStrategy,
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub parallel_aggregation: Option<ParallelAggregation>,
    /// Stop sequential execution on the first child error
    #[serde(default)]
    pub fail_fast: bool,
    /// Per-child timeout in milliseconds
    #[serde(default)]
    pub child_timeout_ms: Option<u64>,
    /// Global timeout for the entire supervisor execution
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// How to merge state from parallel executions
    #[serde(default)]
    pub merge_strategy: ParallelMergeStrategy,
    /// Collect per-child timing and error stats (written to `__supervisor_meta__`)
    #[serde(default)]
    pub collect_stats: bool,
    /// How many times to retry a failing child before giving up (0 = no retry)
    #[serde(default)]
    pub max_retries_per_child: usize,
    /// Max number of children running concurrently in parallel mode (None = unlimited)
    #[serde(default)]
    pub max_concurrent: Option<usize>,
    /// Skip a child when a state key matches a value.
    /// Key = child name, Value = "state_key == value" or just "state_key" (truthy check).
    /// Example: `{"analyst_agent": "analysis != null"}` skips analyst if analysis exists.
    #[serde(default)]
    pub skip_conditions: HashMap<String, String>,

    // ── Autonomous strategy fields ────────────────────────────────────────────
    /// Human-readable goal description (written to state for observability)
    #[serde(default)]
    pub goal: Option<String>,
    /// State keys that must all be non-null for the goal to be considered achieved.
    /// The supervisor loops until all are present or max_iterations is hit.
    #[serde(default)]
    pub required_outputs: Vec<String>,
    /// Which agent to call when a required output is missing.
    /// Key = required output state key, Value = child name responsible for producing it.
    /// Example: `{"raw_data": "researcher_agent", "analysis": "analyst_agent"}`
    #[serde(default)]
    pub output_owners: HashMap<String, String>,
    /// Maximum number of orchestration iterations in autonomous/dynamic/debate mode (default: 10)
    #[serde(default)]
    pub max_iterations: usize,

    // ── Dynamic (LLM-driven) strategy fields ─────────────────────────────────
    /// System prompt for the LLM that decides which agents to call.
    /// The LLM receives the current state summary, available agents, and this prompt.
    /// It must respond with a JSON array of agent names to call.
    #[serde(default)]
    pub selector_prompt: Option<String>,

    // ── RoundRobin strategy fields ───────────────────────────────────────────
    /// State key containing a JSON array of tasks to distribute across agents.
    /// Each task is set as `__current_task__` in state before the agent runs.
    #[serde(default)]
    pub tasks_key: Option<String>,

    // ── Broadcast strategy fields ────────────────────────────────────────────
    /// How to select the best result from broadcast agents.
    /// "first_success" (default), "highest_score", or "llm_judge"
    #[serde(default)]
    pub selection_criteria: Option<String>,
    /// State key where each agent writes a numeric confidence/quality score (for "highest_score").
    #[serde(default)]
    pub score_key: Option<String>,

    // ── MapReduce strategy fields ────────────────────────────────────────────
    /// State key containing a JSON array to split across agents (map phase).
    #[serde(default)]
    pub map_key: Option<String>,
    /// State key where the merged results are written (reduce phase).
    #[serde(default)]
    pub reduce_key: Option<String>,

    // ── ConditionalRouting strategy fields ────────────────────────────────────
    /// Routing rules mapping condition expressions to child agent names.
    /// Key = condition expression (e.g. "task_type == code"), Value = child name.
    /// The first matching rule wins. If none match, the first child is used as default.
    #[serde(default)]
    pub routing_rules: HashMap<String, String>,

    // ── RetryFallback strategy fields ────────────────────────────────────────
    /// Ordered list of agents to try. If empty, `children` order is used.
    /// The first agent that succeeds stops the chain.
    #[serde(default)]
    pub fallback_order: Vec<String>,

    // ── Debate strategy fields ───────────────────────────────────────────────
    /// Number of debate/critique rounds (default: 2).
    #[serde(default)]
    pub debate_rounds: usize,
    /// State key for the initial topic/question that agents debate about.
    #[serde(default)]
    pub debate_key: Option<String>,
}

/// Backwards-compatible alias — use `SupervisorNodeConfig` in new code.
pub type OrchestratorNodeConfig = SupervisorNodeConfig;

impl SupervisorNodeConfig {
    /// Build from a NodeConfig (YAML deserialization target).
    pub fn from_node_config(
        node: &crate::core::node::NodeConfig,
    ) -> crate::core::error::Result<Self> {
        let children: Vec<String> = node
            .config
            .get("children")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        if children.is_empty() {
            return Err(crate::core::error::FlowgentraError::ConfigError(format!(
                "Supervisor '{}': 'children' list must not be empty",
                node.name
            )));
        }

        let strategy = match node
            .config
            .get("strategy")
            .and_then(|v| v.as_str())
            .unwrap_or("sequential")
        {
            "parallel" => OrchestrationStrategy::Parallel,
            "sequential" => OrchestrationStrategy::Sequential,
            "autonomous" => OrchestrationStrategy::Autonomous,
            "dynamic" => OrchestrationStrategy::Dynamic,
            "round_robin" | "roundrobin" => OrchestrationStrategy::RoundRobin,
            "hierarchical" => OrchestrationStrategy::Hierarchical,
            "broadcast" | "fan_out" | "fanout" => OrchestrationStrategy::Broadcast,
            "map_reduce" | "mapreduce" => OrchestrationStrategy::MapReduce,
            "conditional_routing" | "conditional" | "routing" => {
                OrchestrationStrategy::ConditionalRouting
            }
            "retry_fallback" | "retry" | "fallback" => OrchestrationStrategy::RetryFallback,
            "debate" | "critique" => OrchestrationStrategy::Debate,
            other => OrchestrationStrategy::Custom(other.to_string()),
        };

        let parallel_aggregation = node
            .config
            .get("parallel_aggregation")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "all" => ParallelAggregation::All,
                "majority" => ParallelAggregation::Majority,
                _ => ParallelAggregation::FirstSuccess,
            });

        let merge_strategy = node
            .config
            .get("merge_strategy")
            .and_then(|v| v.as_str())
            .map(|s| match s {
                "latest" => ParallelMergeStrategy::Latest,
                "deep_merge" => ParallelMergeStrategy::DeepMerge,
                other if other.starts_with("custom:") => {
                    ParallelMergeStrategy::Custom(other.trim_start_matches("custom:").to_string())
                }
                _ => ParallelMergeStrategy::FirstSuccess,
            })
            .unwrap_or_default();

        let skip_conditions: HashMap<String, String> = node
            .config
            .get("skip_conditions")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        let goal = node
            .config
            .get("goal")
            .and_then(|v| v.as_str())
            .map(String::from);
        let required_outputs: Vec<String> = node
            .config
            .get("required_outputs")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let output_owners: HashMap<String, String> = node
            .config
            .get("output_owners")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        let max_iterations = node
            .config
            .get("max_iterations")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        let selector_prompt = node
            .config
            .get("selector_prompt")
            .and_then(|v| v.as_str())
            .map(String::from);
        let tasks_key = node
            .config
            .get("tasks_key")
            .and_then(|v| v.as_str())
            .map(String::from);
        let selection_criteria = node
            .config
            .get("selection_criteria")
            .and_then(|v| v.as_str())
            .map(String::from);
        let score_key = node
            .config
            .get("score_key")
            .and_then(|v| v.as_str())
            .map(String::from);
        let map_key = node
            .config
            .get("map_key")
            .and_then(|v| v.as_str())
            .map(String::from);
        let reduce_key = node
            .config
            .get("reduce_key")
            .and_then(|v| v.as_str())
            .map(String::from);
        let routing_rules: HashMap<String, String> = node
            .config
            .get("routing_rules")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        let fallback_order: Vec<String> = node
            .config
            .get("fallback_order")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let debate_rounds = node
            .config
            .get("debate_rounds")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let debate_key = node
            .config
            .get("debate_key")
            .and_then(|v| v.as_str())
            .map(String::from);

        Ok(SupervisorNodeConfig {
            name: node.name.clone(),
            children,
            strategy,
            config: node.config.clone(),
            parallel_aggregation,
            fail_fast: node
                .config
                .get("fail_fast")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            child_timeout_ms: node.config.get("child_timeout_ms").and_then(|v| v.as_u64()),
            timeout_ms: node.config.get("timeout_ms").and_then(|v| v.as_u64()),
            merge_strategy,
            collect_stats: node
                .config
                .get("collect_stats")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            max_retries_per_child: node
                .config
                .get("max_retries_per_child")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            max_concurrent: node
                .config
                .get("max_concurrent")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            skip_conditions,
            goal,
            required_outputs,
            output_owners,
            max_iterations,
            selector_prompt,
            tasks_key,
            selection_criteria,
            score_key,
            map_key,
            reduce_key,
            routing_rules,
            fallback_order,
            debate_rounds,
            debate_key,
        })
    }
}

/// PluggableNode-based implementation (used by `create_node_from_config`).
/// In practice the runtime uses the `Handler<DynState>` created by `create_supervisor_handler`.
pub struct SupervisorNode {
    pub config: SupervisorNodeConfig,
    pub children: Vec<Arc<dyn PluggableNode<DynState>>>,
}

/// Backwards-compatible alias — use `SupervisorNode` in new code.
pub type OrchestratorNode = SupervisorNode;

impl Clone for SupervisorNode {
    fn clone(&self) -> Self {
        SupervisorNode {
            config: self.config.clone(),
            children: self.children.clone(),
        }
    }
}

impl SupervisorNode {
    pub fn new(
        config: SupervisorNodeConfig,
        children: Vec<Arc<dyn PluggableNode<DynState>>>,
    ) -> crate::core::error::Result<Self> {
        Self::validate_config(&config, &children)?;
        Ok(SupervisorNode { config, children })
    }

    fn validate_config(
        config: &SupervisorNodeConfig,
        children: &[Arc<dyn PluggableNode<DynState>>],
    ) -> crate::core::error::Result<()> {
        if config.children.is_empty() {
            return Err(crate::core::error::FlowgentraError::ConfigError(
                "SupervisorNodeConfig: 'children' must not be empty".to_string(),
            ));
        }
        if children.is_empty() {
            return Err(crate::core::error::FlowgentraError::ConfigError(format!(
                "SupervisorNode '{}': resolved children list is empty. \
                 All configured children must exist in node_map.",
                config.name
            )));
        }
        if config.children.len() != children.len() {
            return Err(crate::core::error::FlowgentraError::ConfigError(format!(
                "SupervisorNode '{}': mismatch — configured {} children, resolved {}. \
                 Check that all child node names exist.",
                config.name,
                config.children.len(),
                children.len()
            )));
        }
        for child_name in &config.children {
            if child_name == &config.name {
                return Err(crate::core::error::FlowgentraError::ConfigError(format!(
                    "SupervisorNode '{}': cannot reference itself as a child",
                    config.name
                )));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl PluggableNode<DynState> for SupervisorNode {
    async fn run(&self, state: DynState) -> Result<NodeOutput<DynState>> {
        let start = Instant::now();
        info!(
            "SupervisorNode '{}' running with strategy {:?} over {} children",
            self.config.name,
            self.config.strategy,
            self.children.len()
        );

        match self.config.strategy {
            OrchestrationStrategy::Sequential => {
                let mut current_state = state;
                let mut orchestration_metadata = HashMap::new();
                let mut per_child_metadata: HashMap<String, HashMap<String, serde_json::Value>> =
                    HashMap::new();
                let mut child_stats: Vec<ChildExecutionStats> = Vec::new();
                let mut errors = Vec::new();

                for child in &self.children {
                    info!(
                        "Supervisor '{}': running child '{}'",
                        self.config.name,
                        child.name()
                    );
                    let child_start = Instant::now();
                    let child_name = child.name().to_string();

                    match child.run(current_state.clone()).await {
                        Ok(output) => {
                            let duration = child_start.elapsed().as_millis();
                            current_state = output.state;
                            per_child_metadata.insert(child_name.clone(), output.metadata.clone());
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name.clone(),
                                    duration_ms: duration,
                                    success: output.success,
                                    error: output.error.clone(),
                                    timeout: false,
                                });
                            }
                            if !output.success {
                                errors.push(format!(
                                    "Child '{}' failed: {:?}",
                                    child_name, output.error
                                ));
                                if self.config.fail_fast {
                                    warn!(
                                        "Supervisor '{}': fail_fast — stopping after '{}' failed",
                                        self.config.name, child_name
                                    );
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            let duration = child_start.elapsed().as_millis();
                            warn!(
                                "Supervisor '{}': child '{}' error: {:?}",
                                self.config.name, child_name, e
                            );
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name.clone(),
                                    duration_ms: duration,
                                    success: false,
                                    error: Some(format!("{:?}", e)),
                                    timeout: false,
                                });
                            }
                            errors.push(format!("Child '{}' error: {:?}", child_name, e));
                            if self.config.fail_fast {
                                break;
                            }
                        }
                    }
                }

                orchestration_metadata.insert(
                    "children".to_string(),
                    serde_json::to_value(&per_child_metadata).unwrap_or(serde_json::json!({})),
                );
                if self.config.collect_stats && !child_stats.is_empty() {
                    orchestration_metadata.insert(
                        "stats".to_string(),
                        serde_json::to_value(&child_stats).unwrap_or(serde_json::json!([])),
                    );
                }
                if !errors.is_empty() {
                    orchestration_metadata.insert(
                        "errors".to_string(),
                        serde_json::Value::Array(
                            errors
                                .iter()
                                .map(|e| serde_json::Value::String(e.clone()))
                                .collect(),
                        ),
                    );
                }

                let elapsed = start.elapsed().as_millis();
                Ok(NodeOutput {
                    state: current_state,
                    metadata: orchestration_metadata,
                    success: errors.is_empty(),
                    error: if errors.is_empty() {
                        None
                    } else {
                        Some(errors.join("; "))
                    },
                    execution_time_ms: elapsed,
                })
            }

            OrchestrationStrategy::Parallel => {
                use ParallelAggregation::*;
                let agg = self
                    .config
                    .parallel_aggregation
                    .clone()
                    .unwrap_or(FirstSuccess);

                let timeout_duration = self
                    .config
                    .child_timeout_ms
                    .map(Duration::from_millis)
                    .or_else(|| {
                        self.config
                            .timeout_ms
                            .map(|ms| Duration::from_millis(ms / self.children.len() as u64))
                    })
                    .unwrap_or(Duration::from_secs(300));

                let shared_state = Arc::new(state.clone());
                let child_start_times: Vec<_> =
                    self.children.iter().map(|_| Instant::now()).collect();

                let futures = self.children.iter().enumerate().map(|(i, child)| {
                    let timeout_dur = timeout_duration;
                    let state_copy = (*shared_state).clone();
                    (i, async move {
                        let result = tokio::time::timeout(timeout_dur, child.run(state_copy)).await;
                        (i, result)
                    })
                });

                let results: Vec<_> = futures::future::join_all(futures.map(|(_, fut)| fut)).await;

                let mut per_child_metadata: HashMap<String, HashMap<String, serde_json::Value>> =
                    HashMap::new();
                let mut child_stats: Vec<ChildExecutionStats> = Vec::new();
                let mut errors = Vec::new();
                let mut successes = Vec::new();

                for (i, result) in results {
                    let child_name = self.children[i].name().to_string();
                    let duration = child_start_times[i].elapsed().as_millis();

                    match result {
                        Ok(Ok(output)) => {
                            per_child_metadata.insert(child_name.clone(), output.metadata.clone());
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name.clone(),
                                    duration_ms: duration,
                                    success: output.success,
                                    error: output.error.clone(),
                                    timeout: false,
                                });
                            }
                            if output.success {
                                successes.push((child_name.clone(), output.clone()));
                            } else {
                                errors.push(format!(
                                    "Child '{}' failed: {:?}",
                                    child_name, output.error
                                ));
                            }
                        }
                        Ok(Err(e)) => {
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name.clone(),
                                    duration_ms: duration,
                                    success: false,
                                    error: Some(format!("{:?}", e)),
                                    timeout: false,
                                });
                            }
                            errors.push(format!("Child '{}' error: {:?}", child_name, e));
                        }
                        Err(_) => {
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name.clone(),
                                    duration_ms: duration,
                                    success: false,
                                    error: Some("Timeout".to_string()),
                                    timeout: true,
                                });
                            }
                            errors.push(format!(
                                "Child '{}' timed out after {:?}",
                                child_name, timeout_duration
                            ));
                        }
                    }
                }

                let mut orchestration_metadata = HashMap::new();
                orchestration_metadata.insert(
                    "children".to_string(),
                    serde_json::to_value(&per_child_metadata).unwrap_or(serde_json::json!({})),
                );
                if self.config.collect_stats && !child_stats.is_empty() {
                    orchestration_metadata.insert(
                        "stats".to_string(),
                        serde_json::to_value(&child_stats).unwrap_or(serde_json::json!([])),
                    );
                }
                if !errors.is_empty() {
                    orchestration_metadata.insert(
                        "errors".to_string(),
                        serde_json::Value::Array(
                            errors
                                .iter()
                                .map(|e| serde_json::Value::String(e.clone()))
                                .collect(),
                        ),
                    );
                }

                let elapsed = start.elapsed().as_millis();
                let final_state = match &self.config.merge_strategy {
                    ParallelMergeStrategy::Latest => successes
                        .last()
                        .map(|(_, o)| o.state.clone())
                        .unwrap_or_else(|| (*shared_state).clone()),
                    _ => successes
                        .first()
                        .map(|(_, o)| o.state.clone())
                        .unwrap_or_else(|| (*shared_state).clone()),
                };

                match agg {
                    FirstSuccess => {
                        if !successes.is_empty() {
                            Ok(NodeOutput {
                                state: final_state,
                                metadata: orchestration_metadata,
                                success: true,
                                error: None,
                                execution_time_ms: elapsed,
                            })
                        } else {
                            Ok(NodeOutput {
                                state: (*shared_state).clone(),
                                metadata: orchestration_metadata,
                                success: false,
                                error: Some(errors.join("; ")),
                                execution_time_ms: elapsed,
                            })
                        }
                    }
                    All => Ok(NodeOutput {
                        state: final_state,
                        metadata: orchestration_metadata,
                        success: errors.is_empty(),
                        error: if errors.is_empty() {
                            None
                        } else {
                            Some(errors.join("; "))
                        },
                        execution_time_ms: elapsed,
                    }),
                    Majority => {
                        let total = self.children.len();
                        let success_count = successes.len();
                        if success_count * 2 > total {
                            Ok(NodeOutput {
                                state: final_state,
                                metadata: orchestration_metadata,
                                success: true,
                                error: if errors.is_empty() {
                                    None
                                } else {
                                    Some(errors.join("; "))
                                },
                                execution_time_ms: elapsed,
                            })
                        } else {
                            Ok(NodeOutput {
                                state: (*shared_state).clone(),
                                metadata: orchestration_metadata,
                                success: false,
                                error: Some(format!(
                                    "Majority failed: {}/{} succeeded. Errors: {}",
                                    success_count,
                                    total,
                                    errors.join("; ")
                                )),
                                execution_time_ms: elapsed,
                            })
                        }
                    }
                }
            }

            OrchestrationStrategy::RoundRobin => {
                // Distribute tasks from state array across children in rotation
                let tasks_key = self.config.tasks_key.as_deref().unwrap_or("tasks");
                let tasks: Vec<serde_json::Value> = state
                    .get(tasks_key)
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default();

                if tasks.is_empty() {
                    warn!(
                        "SupervisorNode '{}': round_robin — no tasks found at key '{}'",
                        self.config.name, tasks_key
                    );
                    let elapsed = start.elapsed().as_millis();
                    return Ok(NodeOutput {
                        state,
                        metadata: HashMap::new(),
                        success: true,
                        error: None,
                        execution_time_ms: elapsed,
                    });
                }

                let mut current_state = state;
                let mut child_stats: Vec<ChildExecutionStats> = Vec::new();
                let mut all_results: Vec<serde_json::Value> = Vec::new();
                let num_children = self.children.len();

                for (i, task) in tasks.iter().enumerate() {
                    let child_idx = i % num_children;
                    let child = &self.children[child_idx];
                    let child_name = child.name().to_string();

                    current_state.set("__current_task__", task.clone());
                    current_state.set("__task_index__", serde_json::json!(i));

                    let child_start = Instant::now();
                    match child.run(current_state.clone()).await {
                        Ok(output) => {
                            let duration = child_start.elapsed().as_millis();
                            if let Some(result) = output.state.get("__task_result__") {
                                all_results.push(result);
                            }
                            current_state = output.state;
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name,
                                    duration_ms: duration,
                                    success: output.success,
                                    error: output.error,
                                    timeout: false,
                                });
                            }
                        }
                        Err(e) => {
                            let duration = child_start.elapsed().as_millis();
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name,
                                    duration_ms: duration,
                                    success: false,
                                    error: Some(format!("{:?}", e)),
                                    timeout: false,
                                });
                            }
                            if self.config.fail_fast {
                                break;
                            }
                        }
                    }
                }

                current_state.set("__round_robin_results__", serde_json::json!(all_results));
                let elapsed = start.elapsed().as_millis();
                let mut metadata = HashMap::new();
                if self.config.collect_stats {
                    metadata.insert(
                        "stats".to_string(),
                        serde_json::to_value(&child_stats).unwrap_or_default(),
                    );
                }
                Ok(NodeOutput {
                    state: current_state,
                    metadata,
                    success: true,
                    error: None,
                    execution_time_ms: elapsed,
                })
            }

            OrchestrationStrategy::Broadcast => {
                // Send same task to all children, pick best result
                let timeout_duration = self
                    .config
                    .child_timeout_ms
                    .map(Duration::from_millis)
                    .unwrap_or(Duration::from_secs(300));

                let shared_state = Arc::new(state.clone());
                let futures = self.children.iter().map(|child| {
                    let state_copy = (*shared_state).clone();
                    let timeout_dur = timeout_duration;
                    async move {
                        let result = tokio::time::timeout(timeout_dur, child.run(state_copy)).await;
                        (child.name().to_string(), result)
                    }
                });

                let results: Vec<_> = futures::future::join_all(futures).await;
                let score_key = self.config.score_key.as_deref().unwrap_or("__score__");
                let criteria = self
                    .config
                    .selection_criteria
                    .as_deref()
                    .unwrap_or("first_success");

                let mut successes: Vec<(String, NodeOutput<DynState>)> = Vec::new();
                let mut child_stats: Vec<ChildExecutionStats> = Vec::new();

                for (name, result) in results {
                    match result {
                        Ok(Ok(output)) if output.success => {
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: name.clone(),
                                    duration_ms: output.execution_time_ms,
                                    success: true,
                                    error: None,
                                    timeout: false,
                                });
                            }
                            successes.push((name, output));
                        }
                        Ok(Ok(output)) => {
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name,
                                    duration_ms: output.execution_time_ms,
                                    success: false,
                                    error: output.error,
                                    timeout: false,
                                });
                            }
                        }
                        Ok(Err(e)) => {
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name,
                                    duration_ms: 0,
                                    success: false,
                                    error: Some(format!("{:?}", e)),
                                    timeout: false,
                                });
                            }
                        }
                        Err(_) => {
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name,
                                    duration_ms: 0,
                                    success: false,
                                    error: Some("Timeout".to_string()),
                                    timeout: true,
                                });
                            }
                        }
                    }
                }

                let elapsed = start.elapsed().as_millis();
                let mut metadata = HashMap::new();
                if self.config.collect_stats {
                    metadata.insert(
                        "stats".to_string(),
                        serde_json::to_value(&child_stats).unwrap_or_default(),
                    );
                }

                if successes.is_empty() {
                    return Ok(NodeOutput {
                        state: (*shared_state).clone(),
                        metadata,
                        success: false,
                        error: Some("Broadcast: no agent succeeded".to_string()),
                        execution_time_ms: elapsed,
                    });
                }

                let winner = match criteria {
                    "highest_score" => successes.into_iter().max_by(|(_, a), (_, b)| {
                        let score_a = a
                            .state
                            .get(score_key)
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        let score_b = b
                            .state
                            .get(score_key)
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        score_a
                            .partial_cmp(&score_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }),
                    _ => successes.into_iter().next(),
                };

                match winner {
                    Some((name, output)) => {
                        metadata.insert("winner".to_string(), serde_json::json!(name));
                        Ok(NodeOutput {
                            state: output.state,
                            metadata,
                            success: true,
                            error: None,
                            execution_time_ms: elapsed,
                        })
                    }
                    None => Ok(NodeOutput {
                        state: (*shared_state).clone(),
                        metadata,
                        success: false,
                        error: Some("Broadcast: no winner selected".to_string()),
                        execution_time_ms: elapsed,
                    }),
                }
            }

            OrchestrationStrategy::ConditionalRouting => {
                // Route to the first child whose condition matches
                let mut selected_child: Option<&Arc<dyn PluggableNode<DynState>>> = None;
                let mut selected_name = String::new();

                for (condition, child_name) in &self.config.routing_rules {
                    // Find the child by name
                    let child_idx = self.config.children.iter().position(|c| c == child_name);
                    if let Some(idx) = child_idx {
                        if evaluate_condition(condition, &state) {
                            selected_child = Some(&self.children[idx]);
                            selected_name = child_name.clone();
                            break;
                        }
                    }
                }

                // Default to first child if no rule matched
                if selected_child.is_none() && !self.children.is_empty() {
                    selected_child = Some(&self.children[0]);
                    selected_name = self.config.children[0].clone();
                    info!(
                        "SupervisorNode '{}': no routing rule matched, using default '{}'",
                        self.config.name, selected_name
                    );
                }

                let elapsed = start.elapsed().as_millis();
                match selected_child {
                    Some(child) => {
                        let result = child.run(state).await?;
                        let mut metadata = result.metadata.clone();
                        metadata.insert("routed_to".to_string(), serde_json::json!(selected_name));
                        Ok(NodeOutput {
                            state: result.state,
                            metadata,
                            success: result.success,
                            error: result.error,
                            execution_time_ms: elapsed,
                        })
                    }
                    None => Ok(NodeOutput {
                        state,
                        metadata: HashMap::new(),
                        success: false,
                        error: Some("ConditionalRouting: no children available".to_string()),
                        execution_time_ms: elapsed,
                    }),
                }
            }

            OrchestrationStrategy::RetryFallback => {
                // Try agents in order until one succeeds
                let order: Vec<usize> = if !self.config.fallback_order.is_empty() {
                    self.config
                        .fallback_order
                        .iter()
                        .filter_map(|name| self.config.children.iter().position(|c| c == name))
                        .collect()
                } else {
                    (0..self.children.len()).collect()
                };

                let mut child_stats: Vec<ChildExecutionStats> = Vec::new();
                let mut last_error = String::new();

                for idx in order {
                    let child = &self.children[idx];
                    let child_name = child.name().to_string();
                    let child_start = Instant::now();

                    match child.run(state.clone()).await {
                        Ok(output) if output.success => {
                            let duration = child_start.elapsed().as_millis();
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name.clone(),
                                    duration_ms: duration,
                                    success: true,
                                    error: None,
                                    timeout: false,
                                });
                            }
                            let elapsed = start.elapsed().as_millis();
                            let mut metadata = output.metadata.clone();
                            metadata.insert(
                                "succeeded_agent".to_string(),
                                serde_json::json!(child_name),
                            );
                            if self.config.collect_stats {
                                metadata.insert(
                                    "stats".to_string(),
                                    serde_json::to_value(&child_stats).unwrap_or_default(),
                                );
                            }
                            return Ok(NodeOutput {
                                state: output.state,
                                metadata,
                                success: true,
                                error: None,
                                execution_time_ms: elapsed,
                            });
                        }
                        Ok(output) => {
                            let duration = child_start.elapsed().as_millis();
                            last_error = output
                                .error
                                .unwrap_or_else(|| "unknown failure".to_string());
                            warn!(
                                "SupervisorNode '{}': fallback '{}' failed: {}",
                                self.config.name, child_name, last_error
                            );
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name,
                                    duration_ms: duration,
                                    success: false,
                                    error: Some(last_error.clone()),
                                    timeout: false,
                                });
                            }
                        }
                        Err(e) => {
                            let duration = child_start.elapsed().as_millis();
                            last_error = format!("{:?}", e);
                            warn!(
                                "SupervisorNode '{}': fallback '{}' error: {}",
                                self.config.name, child_name, last_error
                            );
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name,
                                    duration_ms: duration,
                                    success: false,
                                    error: Some(last_error.clone()),
                                    timeout: false,
                                });
                            }
                        }
                    }
                }

                let elapsed = start.elapsed().as_millis();
                let mut metadata = HashMap::new();
                if self.config.collect_stats {
                    metadata.insert(
                        "stats".to_string(),
                        serde_json::to_value(&child_stats).unwrap_or_default(),
                    );
                }
                Ok(NodeOutput {
                    state,
                    metadata,
                    success: false,
                    error: Some(format!(
                        "RetryFallback: all agents failed. Last error: {}",
                        last_error
                    )),
                    execution_time_ms: elapsed,
                })
            }

            OrchestrationStrategy::MapReduce => {
                // Split input array, process in parallel, merge results
                let map_key = self.config.map_key.as_deref().unwrap_or("input_chunks");
                let reduce_key = self
                    .config
                    .reduce_key
                    .as_deref()
                    .unwrap_or("reduced_output");

                let chunks: Vec<serde_json::Value> = state
                    .get(map_key)
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default();

                if chunks.is_empty() {
                    warn!(
                        "SupervisorNode '{}': map_reduce — no data at key '{}'",
                        self.config.name, map_key
                    );
                    let elapsed = start.elapsed().as_millis();
                    return Ok(NodeOutput {
                        state,
                        metadata: HashMap::new(),
                        success: true,
                        error: None,
                        execution_time_ms: elapsed,
                    });
                }

                let num_children = self.children.len();
                let shared_state = Arc::new(state.clone());

                // Map phase: distribute chunks across children
                let futures = chunks.iter().enumerate().map(|(i, chunk)| {
                    let child_idx = i % num_children;
                    let child = self.children[child_idx].clone();
                    let chunk_state = (*shared_state).clone();
                    chunk_state.set("__map_chunk__", chunk.clone());
                    chunk_state.set("__chunk_index__", serde_json::json!(i));
                    async move {
                        let result = child.run(chunk_state).await;
                        (i, result)
                    }
                });

                let results: Vec<_> = futures::future::join_all(futures).await;

                // Reduce phase: collect all results
                let mut reduced: Vec<serde_json::Value> = Vec::new();
                let mut errors = Vec::new();
                let mut child_stats: Vec<ChildExecutionStats> = Vec::new();

                for (i, result) in results {
                    let child_idx = i % num_children;
                    let child_name = self.children[child_idx].name().to_string();
                    match result {
                        Ok(output) => {
                            if let Some(chunk_result) = output.state.get("__map_result__") {
                                reduced.push(chunk_result);
                            } else {
                                reduced.push(serde_json::json!(null));
                            }
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name,
                                    duration_ms: output.execution_time_ms,
                                    success: output.success,
                                    error: output.error,
                                    timeout: false,
                                });
                            }
                        }
                        Err(e) => {
                            reduced.push(serde_json::json!(null));
                            errors.push(format!("Chunk {}: {:?}", i, e));
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name,
                                    duration_ms: 0,
                                    success: false,
                                    error: Some(format!("{:?}", e)),
                                    timeout: false,
                                });
                            }
                        }
                    }
                }

                let final_state = (*shared_state).clone();
                final_state.set(reduce_key, serde_json::json!(reduced));
                let elapsed = start.elapsed().as_millis();
                let mut metadata = HashMap::new();
                metadata.insert("chunk_count".to_string(), serde_json::json!(chunks.len()));
                if self.config.collect_stats {
                    metadata.insert(
                        "stats".to_string(),
                        serde_json::to_value(&child_stats).unwrap_or_default(),
                    );
                }
                Ok(NodeOutput {
                    state: final_state,
                    metadata,
                    success: errors.is_empty(),
                    error: if errors.is_empty() {
                        None
                    } else {
                        Some(errors.join("; "))
                    },
                    execution_time_ms: elapsed,
                })
            }

            OrchestrationStrategy::Debate => {
                // Multiple agents generate responses, then critique each other
                let rounds = if self.config.debate_rounds == 0 {
                    2
                } else {
                    self.config.debate_rounds
                };
                let _debate_key = self.config.debate_key.as_deref().unwrap_or("debate_topic");

                let mut current_state = state;
                let mut debate_log: Vec<serde_json::Value> = Vec::new();

                for round in 0..rounds {
                    info!(
                        "SupervisorNode '{}': debate round {}/{}",
                        self.config.name,
                        round + 1,
                        rounds
                    );
                    let mut round_responses: Vec<serde_json::Value> = Vec::new();

                    // Each agent generates a response
                    for child in &self.children {
                        let child_name = child.name().to_string();

                        // Provide previous round's responses as context
                        if !debate_log.is_empty() {
                            current_state.set("__debate_history__", serde_json::json!(debate_log));
                        }
                        current_state.set("__debate_round__", serde_json::json!(round));

                        match child.run(current_state.clone()).await {
                            Ok(output) => {
                                let response = output
                                    .state
                                    .get("__debate_response__")
                                    .unwrap_or(serde_json::json!(null));
                                round_responses.push(serde_json::json!({
                                    "agent": child_name,
                                    "response": response,
                                }));
                                current_state = output.state;
                            }
                            Err(e) => {
                                round_responses.push(serde_json::json!({
                                    "agent": child_name,
                                    "error": format!("{:?}", e),
                                }));
                            }
                        }
                    }

                    // Store all responses for this round so next round can critique
                    current_state.set(
                        "__debate_current_responses__",
                        serde_json::json!(round_responses.clone()),
                    );
                    debate_log.push(serde_json::json!({
                        "round": round + 1,
                        "responses": round_responses,
                    }));
                }

                current_state.set("__debate_log__", serde_json::json!(debate_log));
                let elapsed = start.elapsed().as_millis();
                let mut metadata = HashMap::new();
                metadata.insert("debate_rounds".to_string(), serde_json::json!(rounds));
                metadata.insert(
                    "participants".to_string(),
                    serde_json::json!(self.config.children),
                );
                Ok(NodeOutput {
                    state: current_state,
                    metadata,
                    success: true,
                    error: None,
                    execution_time_ms: elapsed,
                })
            }

            OrchestrationStrategy::Hierarchical => {
                // Delegates to children which are themselves supervisors/subgraphs.
                // Runs them sequentially by default, merging states.
                let mut current_state = state;
                let mut child_stats: Vec<ChildExecutionStats> = Vec::new();
                let mut errors = Vec::new();

                for child in &self.children {
                    let child_name = child.name().to_string();
                    let child_start = Instant::now();
                    info!(
                        "SupervisorNode '{}': hierarchical delegation to '{}'",
                        self.config.name, child_name
                    );

                    match child.run(current_state.clone()).await {
                        Ok(output) => {
                            let duration = child_start.elapsed().as_millis();
                            current_state = output.state;
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name.clone(),
                                    duration_ms: duration,
                                    success: output.success,
                                    error: output.error.clone(),
                                    timeout: false,
                                });
                            }
                            if !output.success {
                                errors.push(format!(
                                    "Sub-supervisor '{}' failed: {:?}",
                                    child_name, output.error
                                ));
                                if self.config.fail_fast {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            let duration = child_start.elapsed().as_millis();
                            errors.push(format!("Sub-supervisor '{}' error: {:?}", child_name, e));
                            if self.config.collect_stats {
                                child_stats.push(ChildExecutionStats {
                                    name: child_name,
                                    duration_ms: duration,
                                    success: false,
                                    error: Some(format!("{:?}", e)),
                                    timeout: false,
                                });
                            }
                            if self.config.fail_fast {
                                break;
                            }
                        }
                    }
                }

                let elapsed = start.elapsed().as_millis();
                let mut metadata = HashMap::new();
                if self.config.collect_stats {
                    metadata.insert(
                        "stats".to_string(),
                        serde_json::to_value(&child_stats).unwrap_or_default(),
                    );
                }
                Ok(NodeOutput {
                    state: current_state,
                    metadata,
                    success: errors.is_empty(),
                    error: if errors.is_empty() {
                        None
                    } else {
                        Some(errors.join("; "))
                    },
                    execution_time_ms: elapsed,
                })
            }

            OrchestrationStrategy::Autonomous | OrchestrationStrategy::Dynamic => {
                // The PluggableNode path is rarely used directly for these strategies.
                // Full logic lives in create_supervisor_handler (agent/mod.rs).
                let strategy_name = match &self.config.strategy {
                    OrchestrationStrategy::Autonomous => "autonomous",
                    OrchestrationStrategy::Dynamic => "dynamic",
                    _ => unreachable!(),
                };
                let elapsed = start.elapsed().as_millis();
                warn!(
                    "SupervisorNode '{}': {} strategy called via PluggableNode::run; \
                     use from_config_path for full support.",
                    self.config.name, strategy_name
                );
                Ok(NodeOutput {
                    state,
                    metadata: {
                        let mut m = HashMap::new();
                        m.insert(
                            "warning".to_string(),
                            serde_json::json!(format!(
                                "{} strategy requires create_supervisor_handler from agent/mod.rs",
                                strategy_name
                            )),
                        );
                        m
                    },
                    success: false,
                    error: Some(format!(
                        "{} strategy not supported via PluggableNode path",
                        strategy_name
                    )),
                    execution_time_ms: elapsed,
                })
            }

            OrchestrationStrategy::Custom(ref name) => {
                let elapsed = start.elapsed().as_millis();
                error!(
                    "SupervisorNode '{}': custom strategy '{}' is not implemented.",
                    self.config.name, name
                );
                Ok(NodeOutput {
                    state,
                    metadata: {
                        let mut m = HashMap::new();
                        m.insert(
                            "error".to_string(),
                            serde_json::json!("Custom strategy not registered"),
                        );
                        m.insert("strategy_name".to_string(), serde_json::json!(name));
                        m
                    },
                    success: false,
                    error: Some(format!(
                        "SupervisorNode '{}': custom strategy '{}' not implemented.",
                        self.config.name, name
                    )),
                    execution_time_ms: elapsed,
                })
            }
        }
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn node_type(&self) -> &str {
        "supervisor"
    }

    fn config(&self) -> &HashMap<String, serde_json::Value> {
        &self.config.config
    }

    fn clone_box(&self) -> Box<dyn PluggableNode<DynState>> {
        Box::new(self.clone()) as Box<dyn PluggableNode<DynState>>
    }
}
