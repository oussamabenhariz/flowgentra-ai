//! # Node Configuration and Execution
//!
//! Nodes represent individual computational steps in your agent's workflow.
//! Each node has a handler function that processes state and produces new state.
//!
//! ## Node Concepts
//!
//! **Handlers** - The actual computation logic executed by a node.
//! Each handler receives the current state and returns updated state.
//!
//! **Conditions** - Decision points that determine which edges to take.
//! Used for branching logic based on the current state.
//!
//! **MCPs** - External tools and services available to the node.
//! Nodes can reference MCP clients to extend their capabilities.
//!
//! ## Example
//!
//! ```ignore
//! use erenflow_ai::core::node::{NodeConfig, EdgeConfig};
//! use serde_json::json;
//!
//! // Define a node in YAML (recommended)
//! // Or programmatically:
//! let node = NodeConfig::new("process_data", "my_handler")
//!     .with_mcps(vec!["web_search".to_string()]);
//!
//! // Define edges with conditions
//! let edge = EdgeConfig::new("process_data", "format_results")
//!     .with_condition("is_complex");
//! ```

use crate::core::routing::Condition;
use crate::core::state::State;
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

    /// Name of the handler function to execute
    /// (Can be specified as "handler" or "function" for backward compatibility)
    #[serde(alias = "function")]
    pub handler: String,

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
/// use erenflow_ai::core::node::NodeFunction;
/// use erenflow_ai::core::state::State;
/// use serde_json::json;
///
/// fn my_handler() -> NodeFunction {
///     Box::new(|mut state| {
///         Box::pin(async move {
///             // Your logic here
///             state.set("processed", json!(true));
///             Ok(state)
///         })
///     })
/// }
/// ```
pub type NodeFunction = Box<
    dyn Fn(State) -> futures::future::BoxFuture<'static, crate::core::error::Result<State>>
        + Send
        + Sync,
>;

// =============================================================================
// Compiled Node
// =============================================================================

/// A compiled node ready for execution.
///
/// This is created internally by the runtime when configuration is loaded.
pub struct Node {
    /// Node name (identifier)
    pub name: String,

    /// The actual handler function to execute
    pub function: NodeFunction,

    /// MCP tools available to this node
    pub mcps: Vec<String>,

    /// Custom configuration
    pub config: HashMap<String, serde_json::Value>,

    /// Whether this node is a planner node (LLM-driven dynamic routing).
    pub(crate) is_planner: bool,
}

impl Node {
    /// Create a new compiled node.
    pub fn new(
        name: impl Into<String>,
        function: NodeFunction,
        mcps: Vec<String>,
        config: HashMap<String, serde_json::Value>,
    ) -> Self {
        Node {
            name: name.into(),
            function,
            mcps,
            config,
            is_planner: false,
        }
    }

    /// Create a planner node (LLM-driven dynamic next-node selection).
    pub fn new_planner(
        name: impl Into<String>,
        function: NodeFunction,
        mcps: Vec<String>,
        config: HashMap<String, serde_json::Value>,
    ) -> Self {
        Node {
            name: name.into(),
            function,
            mcps,
            config,
            is_planner: true,
        }
    }

    /// Execute this node with the given state.
    ///
    /// # Returns
    /// The updated state after node execution, or an error.
    pub async fn execute(&self, state: State) -> crate::core::error::Result<State> {
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

    /// Type-safe Condition for code-based routing (set programmatically)
    #[serde(skip)]
    pub routing_condition: Option<Condition>,
}

impl EdgeConfig {
    /// Create a new edge from source to target node.
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        EdgeConfig {
            from: from.into(),
            to: vec![to.into()],
            condition: None,
            routing_condition: None,
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

    /// Add a type-safe routing condition for this edge
    ///
    /// This allows using the new Condition DSL instead of string-based conditions.
    pub fn with_routing_condition(mut self, condition: Condition) -> Self {
        self.routing_condition = Some(condition);
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
/// use erenflow_ai::core::state::State;
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
pub type EdgeCondition =
    std::sync::Arc<dyn Fn(&State) -> crate::core::error::Result<Option<String>> + Send + Sync>;

// =============================================================================
// Compiled Edge
// =============================================================================

/// A compiled edge (connection) between two nodes, with optional condition.
pub struct Edge {
    /// Source node name
    pub from: String,

    /// Target node name
    pub to: String,

    /// Optional condition function (legacy string-based conditions)
    pub condition: Option<EdgeCondition>,

    /// The original condition name from config (for reference)
    pub condition_name: Option<String>,

    /// Type-safe routing condition (new DSL-based conditions)
    pub routing_condition: Option<Condition>,
}

impl Edge {
    /// Create a new edge from source to target.
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        condition: Option<EdgeCondition>,
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
    pub fn with_routing_condition(mut self, condition: Condition) -> Self {
        self.routing_condition = Some(condition);
        self
    }

    /// Check if this edge should be taken given the current state.
    ///
    /// Evaluates both legacy conditions (EdgeCondition) and new conditions (Condition DSL).
    /// Returns true if the edge is unconditional or if the condition evaluates to true.
    pub async fn should_take(&self, state: &State) -> crate::core::error::Result<bool> {
        // First check legacy condition function if present
        if let Some(cond) = &self.condition {
            match cond(state)? {
                Some(_) => return Ok(true),
                None => return Ok(false),
            }
        }

        // Then check new routing condition if present
        if let Some(routing_cond) = &self.routing_condition {
            return Ok(routing_cond.evaluate(state));
        }

        // No condition = always traversable
        Ok(true)
    }

    /// Get the next node if the condition specifies one.
    ///
    /// Returns the explicitly specified next node, or None to use the default target.
    pub async fn get_next_node(&self, state: &State) -> crate::core::error::Result<Option<String>> {
        match &self.condition {
            Some(cond) => cond(state),
            None => Ok(None),
        }
    }
}

// Sub-modules for node organization
pub mod advanced_nodes;
pub mod builtin_nodes;
pub mod nodes_trait;
pub mod planner;
