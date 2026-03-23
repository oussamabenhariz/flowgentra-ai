// # SubgraphNode
//
// A node that IS itself an agent — it loads and executes a complete agent graph
// (defined in a separate YAML file) as a single step in the parent graph.
//
// This enables true hierarchical multi-agent systems:
// - The parent graph orchestrates high-level flow
// - Each subgraph node runs an independent agent with its own nodes, edges, and handlers
// - State flows transparently between parent and subgraph
//
// ## Features
// - Loads a nested agent config from a YAML file path
// - Runs the subgraph with the parent's current state
// - Returns the subgraph's final state back to the parent graph
// - Handlers are auto-discovered from the same inventory (shared #[register_handler] pool)
//
// ## Example
// ```yaml
// nodes:
//   - name: research_agent
//     type: subgraph
//     config:
//       path: agents/research_agent.yaml
//
//   - name: writer_agent
//     type: subgraph
//     config:
//       path: agents/writer_agent.yaml
//
//   - name: coordinate
//     type: supervisor
//     config:
//       strategy: sequential
//       children: [research_agent, writer_agent]
// ```
//
// ## Notes
// - `path` is resolved relative to the working directory at runtime
// - The subgraph uses the same handler registry (all #[register_handler] functions are visible)
// - For backwards compatibility, `type: agent_or_graph` is also accepted

use crate::core::error::{FlowgentraError, Result};
use crate::core::node::nodes_trait::{NodeOutput, PluggableNode};
use crate::core::state::State;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for a subgraph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgraphNodeConfig {
    /// Node name (used as the key in the parent graph)
    pub name: String,
    /// Path to the sub-agent YAML config file
    pub path: String,
    /// Optional extra config passed through (available to the subgraph via state)
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

/// Backwards-compatible alias — use `SubgraphNodeConfig` in new code.
pub type AgentOrGraphNodeConfig = SubgraphNodeConfig;

impl SubgraphNodeConfig {
    /// Build from a NodeConfig (YAML deserialization target).
    ///
    /// Expects `config.path` to be set to the sub-agent YAML file.
    pub fn from_node_config(
        node: &crate::core::node::NodeConfig,
    ) -> crate::core::error::Result<Self> {
        let path = node
            .config
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                FlowgentraError::ConfigError(format!(
                    "SubgraphNode '{}': 'path' is required in config \
                     (e.g. config:\\n  path: agents/my_agent.yaml)",
                    node.name
                ))
            })?
            .to_string();

        Ok(SubgraphNodeConfig {
            name: node.name.clone(),
            path,
            config: node.config.clone(),
        })
    }
}

// ── PluggableNode impl ────────────────────────────────────────────────────────

/// PluggableNode adapter for a subgraph.
/// Used by `create_node_from_config`; the runtime path goes through `create_subgraph_handler`.
pub struct SubgraphNode<T: State> {
    pub config: SubgraphNodeConfig,
    /// The inner node (pre-loaded subgraph compiled as a PluggableNode)
    pub inner: Box<dyn PluggableNode<T>>,
}

/// Backwards-compatible alias — use `SubgraphNode` in new code.
pub type AgentOrGraphNode<T> = SubgraphNode<T>;

impl<T: State> SubgraphNode<T> {
    pub fn new(
        config: SubgraphNodeConfig,
        inner: Box<dyn PluggableNode<T>>,
    ) -> crate::core::error::Result<Self> {
        Ok(SubgraphNode { config, inner })
    }
}

impl<T: State> Clone for SubgraphNode<T> {
    fn clone(&self) -> Self {
        SubgraphNode {
            config: self.config.clone(),
            inner: self.inner.clone_box(),
        }
    }
}

#[async_trait]
impl<T: State> PluggableNode<T> for SubgraphNode<T> {
    async fn run(&self, state: T) -> Result<NodeOutput<T>> {
        info!(
            "SubgraphNode '{}' executing subgraph from '{}'",
            self.config.name, self.config.path
        );
        match self.inner.run(state).await {
            Ok(output) => {
                info!(
                    "SubgraphNode '{}' completed in {}ms",
                    self.config.name, output.execution_time_ms
                );
                Ok(output)
            }
            Err(e) => {
                warn!("SubgraphNode '{}' failed: {:?}", self.config.name, e);
                Err(e)
            }
        }
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn node_type(&self) -> &str {
        "subgraph"
    }

    fn config(&self) -> &HashMap<String, serde_json::Value> {
        &self.config.config
    }

    fn clone_box(&self) -> Box<dyn PluggableNode<T>> {
        Box::new(self.clone())
    }
}

// Keep old enum for any existing code that imported it
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentOrGraphType {
    Agent,
    Graph,
}
