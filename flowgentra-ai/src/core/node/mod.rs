/// NodeId type alias for node identification
pub type NodeId = String;
use crate::core::error::{FlowgentraError, Result};
use crate::core::node::nodes_trait::PluggableNode;
use crate::core::state::{DynState, State};
use std::sync::Arc;

/// Factory function to create a node from config.
///
/// Supports: `supervisor` (`orchestrator`), `subgraph` (`agent`, `agent_or_graph`), `evaluation`.
///
/// # Arguments
/// * `config` - The node configuration (typically from YAML)
/// * `node_map` - A map of all available nodes, indexed by name
pub fn create_node_from_config(
    config: &serde_json::Value,
    node_map: &std::collections::HashMap<String, Box<dyn PluggableNode<DynState>>>,
) -> Result<Box<dyn PluggableNode<DynState>>> {
    let node_type = config.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match node_type {
        // ── Supervisor (alias: orchestrator) ─────────────────────────────────
        "supervisor" | "orchestrator" => {
            use crate::core::node::orchestrator_node::{SupervisorNode, SupervisorNodeConfig};
            let cfg: SupervisorNodeConfig =
                serde_json::from_value(config.clone()).map_err(|e| {
                    FlowgentraError::ConfigError(format!(
                        "Failed to parse supervisor node config: {}",
                        e
                    ))
                })?;
            let node_name = &cfg.name;

            let mut children: Vec<Arc<dyn PluggableNode<DynState>>> = Vec::new();
            let mut missing_children = Vec::new();

            for child_name in &cfg.children {
                match node_map.get(child_name) {
                    Some(node) => {
                        let cloned = node.clone_box();
                        let arc_node: Arc<dyn PluggableNode<DynState>> = Arc::from(cloned);
                        children.push(arc_node);
                    }
                    None => missing_children.push(child_name.clone()),
                }
            }

            if !missing_children.is_empty() {
                return Err(FlowgentraError::ConfigError(format!(
                    "SupervisorNode '{}': missing children: {:?}",
                    node_name, missing_children
                )));
            }

            Ok(Box::new(SupervisorNode::new(cfg, children)?))
        }

        // ── Subgraph (aliases: agent, agent_or_graph) ──────────────────────
        "subgraph" | "agent" | "agent_or_graph" => {
            use crate::core::node::agent_or_graph_node::{SubgraphNode, SubgraphNodeConfig};
            let cfg: SubgraphNodeConfig = serde_json::from_value(config.clone()).map_err(|e| {
                FlowgentraError::ConfigError(format!("Failed to parse subgraph node config: {}", e))
            })?;
            let target_name = &cfg.path;

            // For the PluggableNode path, look up an already-compiled inner node
            match node_map.get(target_name) {
                Some(node) => {
                    let inner = node.clone_box();
                    Ok(Box::new(SubgraphNode::new(cfg, inner)?))
                }
                None => Err(FlowgentraError::ConfigError(format!(
                    "SubgraphNode '{}': inner node '{}' not found in node_map. \
                     (In YAML config, set 'path' to the sub-agent YAML file.)",
                    cfg.name, target_name
                ))),
            }
        }

        // ── Evaluation ────────────────────────────────────────────────────────
        "evaluation" => {
            use crate::core::node::evaluation_node::{EvaluationNode, EvaluationNodeConfig};
            let eval_config: EvaluationNodeConfig = serde_json::from_value(config.clone())
                .map_err(|e| {
                    FlowgentraError::ConfigError(format!(
                        "Failed to parse evaluation node config: {}",
                        e
                    ))
                })?;
            let handler_name = &eval_config.handler;
            match node_map.get(handler_name) {
                Some(inner_node) => {
                    let inner = inner_node.clone_box();
                    Ok(Box::new(EvaluationNode::new(eval_config, inner)?))
                }
                None => Err(FlowgentraError::ConfigError(format!(
                    "EvaluationNode '{}': handler '{}' not found in node_map.",
                    eval_config.name, handler_name
                ))),
            }
        }

        _ => Err(FlowgentraError::ConfigError(format!(
            "Unknown node type: '{}'. Expected: supervisor, subgraph, evaluation \
             (aliases: orchestrator, agent, agent_or_graph)",
            node_type
        ))),
    }
}

pub mod agent_or_graph_node;
pub mod evaluation_node;
pub mod orchestrator_node;

// ── Re-exports ────────────────────────────────────────────────────────────────

// Supervisor (canonical) + backwards-compat aliases
pub use orchestrator_node::{
    ChildExecutionStats,
    // supporting types
    OrchestrationStrategy,
    OrchestratorNode,
    // aliases
    OrchestratorNodeConfig,
    ParallelAggregation,
    ParallelMergeStrategy,
    SupervisorNode,
    SupervisorNodeConfig,
};

// Subgraph (canonical) + backwards-compat aliases
pub use agent_or_graph_node::{
    AgentOrGraphNode,
    // aliases
    AgentOrGraphNodeConfig,
    // kept for existing imports
    AgentOrGraphType,
    SubgraphNode,
    SubgraphNodeConfig,
};

pub use evaluation_node::{
    default_scorer, scorer_combine, scorer_from_confidence, scorer_from_llm_grader,
    scorer_from_node_scorer, scorer_from_sync, Attempt, EvaluationNode, EvaluationNodeConfig,
    EvaluationResult, ExitReason, ScorerFn,
};
// # Node Configuration and Execution
//
// Nodes represent individual computational steps in your agent's workflow.
// Each node has a handler function that processes state and produces new state.
//
// ## Node Concepts
//
// **Handlers** - The actual computation logic executed by a node.
// Each handler receives the current state and returns updated state.
//
// **Conditions** - Decision points that determine which edges to take.
// Used for branching logic based on the current state.
//
// **MCPs** - External tools and services available to the node.
// Nodes can reference MCP clients to extend their capabilities.
//
// ## Example
//
// ```yaml
// # Standard node
// - name: process_data
//   type: handler
//   handler: my_handler
//   mcps: [web_search]
//
// # Orchestrator node
// - name: orchestrate_agents
//   type: orchestrator
//   strategy: parallel
//   children: [agent1, agent2, agent3]
//
// # Agent or Graph node
// - name: agent1
//   type: agent
//   target: some_agent
// - name: subgraph1
//   type: graph
//   target: subgraph.yaml
/// ```
use crate::core::routing::Condition;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

fn deserialize_to_one_or_many<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(String),
        Many(Vec<String>),
    }
    match OneOrMany::deserialize(deserializer)? {
        OneOrMany::One(s) => Ok(vec![s]),
        OneOrMany::Many(v) => Ok(v),
    }
}

fn serialize_to_one_or_many<S>(v: &[String], s: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::Serialize;
    if v.len() == 1 {
        v[0].serialize(s)
    } else {
        v.serialize(s)
    }
}

// =============================================================================
// Node Configuration (from YAML)
// =============================================================================

/// Configuration for a node as specified in the YAML config file.
///
/// This represents the node definition before it's compiled with actual handler code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Unique identifier for this node
    pub name: String,

    /// Name of the handler function to execute.
    /// Optional for node types that don't use a registered handler
    /// (e.g., "planner", "human_in_the_loop", "memory").
    #[serde(alias = "function", default)]
    pub handler: String,

    /// Optional node type (e.g., "standard", "evaluation", "orchestrator")
    #[serde(rename = "type", default)]
    pub node_type: Option<String>,

    /// Optional MCP tools available to this node
    #[serde(default)]
    pub mcps: Vec<String>,

    /// Optional execution timeout in milliseconds
    /// If the node takes longer than this, execution will be interrupted
    #[serde(default)]
    pub timeout_ms: Option<u64>,

    /// Optional node-specific configuration
    /// (Useful for passing parameters to the handler)
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

impl NodeConfig {
    /// Create a new node configuration with the given name and handler.
    pub fn new(name: impl Into<String>, handler: impl Into<String>) -> Self {
        NodeConfig {
            name: name.into(),
            handler: handler.into(),
            node_type: None,
            mcps: Vec::new(),
            timeout_ms: None,
            config: HashMap::new(),
        }
    }

    /// Attach MCP tools to this node.
    pub fn with_mcps(mut self, mcps: Vec<String>) -> Self {
        self.mcps = mcps;
        self
    }

    /// Set timeout for this node in milliseconds
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
    /// Add custom configuration for this node.
    pub fn with_config(mut self, config: HashMap<String, serde_json::Value>) -> Self {
        self.config = config;
        self
    }

    /// Validate the node configuration
    ///
    /// Checks:
    /// - Timeout is in a reasonable range (if specified)
    /// - Timeout is not exactly 0 (would timeout immediately)
    ///
    /// # Returns
    /// Error if validation fails
    pub fn validate(&self) -> crate::core::error::Result<()> {
        if let Some(timeout_ms) = self.timeout_ms {
            // Timeout of 0 doesn't make sense
            if timeout_ms == 0 {
                return Err(crate::core::error::FlowgentraError::ConfigError(
                    format!(
                        "Node '{}': timeout_ms cannot be 0 (would timeout immediately). Set it to a positive milliseconds value.",
                        self.name
                    ),
                ));
            }

            // Warn if timeout is extremely large (more than 1 hour = 3,600,000ms)
            const MAX_REASONABLE_TIMEOUT_MS: u64 = 3_600_000; // 1 hour
            if timeout_ms > MAX_REASONABLE_TIMEOUT_MS {
                eprintln!(
                    "⚠️  Warning: Node '{}' has timeout_ms = {} (more than 1 hour). Is this intentional?",
                    self.name, timeout_ms
                );
            }

            // Warn if timeout is very small (less than 100ms)
            const MIN_REASONABLE_TIMEOUT_MS: u64 = 100;
            if timeout_ms < MIN_REASONABLE_TIMEOUT_MS {
                eprintln!(
                    "⚠️  Warning: Node '{}' has timeout_ms = {} (less than 100ms). This might be too short for I/O operations.",
                    self.name, timeout_ms
                );
            }
        }
        Ok(())
    }
}

// =============================================================================
// Node Function Type
// =============================================================================

/// The function signature for node handlers.
///
/// A node handler is an async function that:
/// 1. Receives the current state
/// 2. Performs its computation (possibly using MCPs or LLM)
/// 3. Returns the updated state or an error
///
/// # Example Handler
/// ```no_run
/// use flowgentra_ai::core::node::NodeFunction;
/// use flowgentra_ai::core::state::DynState;
/// use serde_json::json;
///
/// fn my_handler() -> NodeFunction<DynState> {
///     Box::new(|state| {
///         Box::pin(async move {
///             // Your logic here
///             state.set("processed", json!(true));
///             Ok(state)
///         })
///     })
/// }
/// ```
pub type NodeFunction<T> = Box<
    dyn Fn(T) -> futures::future::BoxFuture<'static, crate::core::error::Result<T>> + Send + Sync,
>;

// =============================================================================
// Compiled Node
// =============================================================================

/// A compiled node ready for execution.
///
/// This is created internally by the runtime when configuration is loaded.
pub struct Node<T: State = DynState> {
    /// Node name (identifier)
    pub name: String,

    /// The actual handler function to execute
    pub function: NodeFunction<T>,

    /// MCP tools available to this node
    pub mcps: Vec<String>,

    /// Custom configuration
    pub config: HashMap<String, serde_json::Value>,

    /// Whether this node is a planner node (LLM-driven dynamic routing).
    pub(crate) is_planner: bool,

    /// Whether this node still has a placeholder function (not yet registered).
    /// If true, execution will return an error instead of silently passing through.
    pub(crate) is_placeholder: bool,
}

impl<T: State> Node<T> {
    /// Create a new compiled node.
    pub fn new(
        name: impl Into<String>,
        function: NodeFunction<T>,
        mcps: Vec<String>,
        config: HashMap<String, serde_json::Value>,
    ) -> Self {
        Node {
            name: name.into(),
            function,
            mcps,
            config,
            is_planner: false,
            is_placeholder: false,
        }
    }

    /// Create a placeholder node (identity function that will be replaced via `register_node`).
    pub fn new_placeholder(
        name: impl Into<String>,
        function: NodeFunction<T>,
        mcps: Vec<String>,
        config: HashMap<String, serde_json::Value>,
    ) -> Self {
        Node {
            name: name.into(),
            function,
            mcps,
            config,
            is_planner: false,
            is_placeholder: true,
        }
    }

    /// Create a planner node (LLM-driven dynamic next-node selection).
    pub fn new_planner(
        name: impl Into<String>,
        function: NodeFunction<T>,
        mcps: Vec<String>,
        config: HashMap<String, serde_json::Value>,
    ) -> Self {
        Node {
            name: name.into(),
            function,
            mcps,
            config,
            is_planner: true,
            is_placeholder: false,
        }
    }

    /// Execute this node with the given state.
    ///
    /// # Returns
    /// The updated state after node execution, or an error.
    /// Returns an error if the node handler was never registered (still a placeholder).
    pub async fn execute(&self, state: T) -> crate::core::error::Result<T> {
        if self.is_placeholder {
            return Err(FlowgentraError::ExecutionError(format!(
                "Node '{}' has no registered handler. Call register_node() before execution.",
                self.name
            )));
        }
        (self.function)(state).await
    }
}

// =============================================================================
// Edge Configuration (from YAML)
// =============================================================================

/// Configuration for an edge (connection between nodes) from the YAML config.
/// Supports `to` as a single string or list (parallel targets): `to: [a, b, c]`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConfig {
    /// Source node name
    pub from: String,

    /// Target node name, or list for parallel: `to: [analyze_logs, analyze_pcap]`
    #[serde(
        deserialize_with = "deserialize_to_one_or_many",
        serialize_with = "serialize_to_one_or_many"
    )]
    pub to: Vec<String>,

    /// Optional condition expression for conditional routing
    /// If specified, the edge is only taken if this condition evaluates to true.
    /// The condition name must correspond to a registered condition function.
    #[serde(default)]
    pub condition: Option<String>,
}

impl EdgeConfig {
    /// Create a new edge from source to target node.
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        EdgeConfig {
            from: from.into(),
            to: vec![to.into()],
            condition: None,
        }
    }

    /// Iterate over target nodes (one or many for parallel)
    pub fn to_targets(&self) -> impl Iterator<Item = &str> {
        self.to.iter().map(String::as_str)
    }

    /// Add a condition that must be satisfied for this edge to be taken.
    pub fn with_condition(mut self, condition: impl Into<String>) -> Self {
        self.condition = Some(condition.into());
        self
    }
}

// =============================================================================
// Edge Condition Type
// =============================================================================

/// A condition function that determines if an edge should be taken.
///
/// The condition function receives the current state and returns:
/// - `Some(node_name)` to explicitly jump to that node
/// - `None` to use the default target node
/// - An error if the condition evaluation fails
///
/// # Example
/// ```ignore
/// use flowgentra_ai::core::state::State;
/// use std::sync::Arc;
///
/// let is_complex = Arc::new(|state: &State| -> Result<Option<String>, _> {
///     if let Some(score) = state.get("complexity_score") {
///         if score.as_i64().unwrap_or(0) > 50 {
///             return Ok(Some("process_complex".to_string()));
///         }
///     }
///     Ok(Some("process_simple".to_string()))
/// } as Box<dyn Fn(&State) -> _>);
/// ```
pub type EdgeCondition<T> =
    std::sync::Arc<dyn Fn(&T) -> crate::core::error::Result<Option<String>> + Send + Sync>;

// =============================================================================
// Compiled Edge
// =============================================================================

/// A compiled edge (connection) between two nodes, with optional condition.
#[derive(Clone)]
pub struct Edge<T: State = DynState> {
    /// Source node name
    pub from: String,

    /// Target node name
    pub to: String,

    /// Optional condition function (legacy string-based conditions)
    pub condition: Option<EdgeCondition<T>>,

    /// The original condition name from config (for reference)
    pub condition_name: Option<String>,

    /// Type-safe routing condition (new DSL-based conditions)
    pub routing_condition: Option<Condition>,
}

impl<T: State> std::fmt::Debug for Edge<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Edge")
            .field("from", &self.from)
            .field("to", &self.to)
            .field("condition", &self.condition.is_some())
            .field("condition_name", &self.condition_name)
            .field("routing_condition", &self.routing_condition)
            .finish()
    }
}

impl<T: State> Edge<T> {
    /// Create a new edge from source to target.
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        condition: Option<EdgeCondition<T>>,
    ) -> Self {
        Edge {
            from: from.into(),
            to: to.into(),
            condition,
            condition_name: None,
            routing_condition: None,
        }
    }

    /// Attach the original condition name (for reference).
    pub fn with_condition_name(mut self, name: impl Into<String>) -> Self {
        self.condition_name = Some(name.into());
        self
    }

    /// Set the type-safe routing condition for this edge
    pub fn with_routing_condition(
        mut self,
        condition: crate::core::graph::routing::Condition,
    ) -> Self {
        self.routing_condition = Some(condition);
        self
    }

    /// Check if this edge should be taken given the current state.
    ///
    /// Evaluates both legacy conditions (EdgeCondition) and new conditions (Condition DSL).
    /// Returns true if the edge is unconditional or if the condition evaluates to true.
    pub async fn should_take(&self, state: &T) -> crate::core::error::Result<bool> {
        // First check legacy condition function if present
        if let Some(cond) = &self.condition {
            match cond(state)? {
                Some(_) => return Ok(true),
                None => return Ok(false),
            }
        }

        // Then check new routing condition if present (requires DynState)
        if let Some(routing_cond) = &self.routing_condition {
            // Serialize to Value, wrap in DynState for evaluation
            let value = serde_json::to_value(state).unwrap_or_default();
            if let Ok(temp_state) = DynState::from_json(value) {
                return Ok(routing_cond.evaluate(&temp_state));
            }
        }

        // No condition = always traversable
        Ok(true)
    }

    /// Get the next node if the condition specifies one.
    ///
    /// Returns the explicitly specified next node, or None to use the default target.
    pub async fn get_next_node(&self, state: &T) -> crate::core::error::Result<Option<String>> {
        match &self.condition {
            Some(cond) => cond(state),
            None => Ok(None),
        }
    }
}

// Sub-modules for node organization
pub mod advanced_nodes;
pub mod builtin_nodes;
pub mod memory_handlers;
pub mod nodes_trait;
pub mod planner;
