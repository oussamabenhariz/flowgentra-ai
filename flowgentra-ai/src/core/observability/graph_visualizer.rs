//! StateGraph Visualization Integration
//!
//! Provides visualization capabilities for compiled StateGraph structures.

use crate::core::state::State;
use crate::core::state_graph::StateGraph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Visualization of a compiled StateGraph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateGraphVisualization {
    /// Graph identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// List of all nodes in the graph
    pub nodes: Vec<StateGraphNode>,
    /// List of all edges in the graph
    pub edges: Vec<StateGraphEdge>,
    /// Metadata about the graph structure
    pub metadata: GraphMetadata,
}

/// A node in the state graph visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateGraphNode {
    /// Unique node identifier
    pub id: String,
    /// Display name for the node
    pub name: String,
    /// Node classification (start, end, handler, router, etc.)
    pub node_type: NodeType,
    /// Optional description
    pub description: Option<String>,
    /// Visual position (optional)
    pub position: Option<(f32, f32)>,
    /// Current execution status
    pub status: NodeExecutionStatus,
}

/// Types of nodes in a state graph
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NodeType {
    /// Entry point to the graph
    Start,
    /// Exit point from the graph
    End,
    /// Standard handler node
    Handler,
    /// Conditional routing node
    Router,
    /// Subgraph node
    Subgraph,
}

impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeType::Start => write!(f, "START"),
            NodeType::End => write!(f, "END"),
            NodeType::Handler => write!(f, "Handler"),
            NodeType::Router => write!(f, "Router"),
            NodeType::Subgraph => write!(f, "Subgraph"),
        }
    }
}

/// Execution status of a node
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum NodeExecutionStatus {
    /// Never executed
    Pending,
    /// Currently executing
    Executing,
    /// Completed successfully
    Executed,
    /// Execution failed
    Failed,
    /// Node was skipped
    Skipped,
}

/// An edge connecting two nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateGraphEdge {
    /// Source node
    pub from: String,
    /// Target node
    pub to: String,
    /// Optional condition for this edge
    pub condition: Option<String>,
    /// Edge execution count
    pub executions: u32,
    /// Average execution time in milliseconds
    pub avg_duration_ms: f64,
}

/// Metadata about the graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges
    pub edge_count: usize,
    /// Entry point node name
    pub entry_point: String,
    /// Timestamp when graph was compiled
    pub compiled_at: String,
    /// Framework version
    pub version: String,
    /// Maximum depth of the graph
    pub max_depth: usize,
}

/// Visualizer for StateGraph structures
pub struct StateGraphVisualizer;

impl StateGraphVisualizer {
    /// Create visualization from a compiled StateGraph
    ///
    /// This requires the StateGraph to expose node and edge information.
    /// Currently provides a basic structure that can be extended.
    pub fn visualize<S: State>(
        _graph: &StateGraph<S>,
        name: impl Into<String>,
    ) -> StateGraphVisualization {
        let name = name.into();

        StateGraphVisualization {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.clone(),
            nodes: vec![
                StateGraphNode {
                    id: "START".to_string(),
                    name: "Start".to_string(),
                    node_type: NodeType::Start,
                    description: Some("Graph entry point".to_string()),
                    position: Some((0.0, 0.0)),
                    status: NodeExecutionStatus::Pending,
                },
                StateGraphNode {
                    id: "END".to_string(),
                    name: "End".to_string(),
                    node_type: NodeType::End,
                    description: Some("Graph exit point".to_string()),
                    position: Some((500.0, 0.0)),
                    status: NodeExecutionStatus::Pending,
                },
            ],
            edges: vec![
                StateGraphEdge {
                    from: "START".to_string(),
                    to: "END".to_string(),
                    condition: None,
                    executions: 0,
                    avg_duration_ms: 0.0,
                },
            ],
            metadata: GraphMetadata {
                node_count: 2,
                edge_count: 1,
                entry_point: "START".to_string(),
                compiled_at: chrono::Local::now().to_rfc3339(),
                version: "1.0".to_string(),
                max_depth: 1,
            },
        }
    }

    /// Export visualization as JSON string
    pub fn to_json(viz: &StateGraphVisualization) -> serde_json::Result<String> {
        serde_json::to_string_pretty(viz)
    }

    /// Export visualization as minified JSON string
    pub fn to_json_compact(viz: &StateGraphVisualization) -> serde_json::Result<String> {
        serde_json::to_string(viz)
    }

    /// Add a node to the visualization
    pub fn add_node(
        viz: &mut StateGraphVisualization,
        id: impl Into<String>,
        name: impl Into<String>,
        node_type: NodeType,
    ) {
        let id = id.into();
        let name = name.into();

        viz.nodes.push(StateGraphNode {
            id,
            name,
            node_type,
            description: None,
            position: None,
            status: NodeExecutionStatus::Pending,
        });

        viz.metadata.node_count = viz.nodes.len();
    }

    /// Add an edge to the visualization
    pub fn add_edge(
        viz: &mut StateGraphVisualization,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: Option<String>,
    ) {
        viz.edges.push(StateGraphEdge {
            from: from.into(),
            to: to.into(),
            condition,
            executions: 0,
            avg_duration_ms: 0.0,
        });

        viz.metadata.edge_count = viz.edges.len();
    }

    /// Calculate graph depth (longest path from START to END)
    pub fn calculate_depth(viz: &StateGraphVisualization) -> usize {
        if viz.edges.is_empty() {
            return 1;
        }

        let mut depths: HashMap<String, usize> = HashMap::new();
        depths.insert("START".to_string(), 0);

        // Topological sort to calculate max depth
        let mut max_depth = 0;
        let mut changed = true;

        while changed {
            changed = false;

            for edge in &viz.edges {
                if let Some(&from_depth) = depths.get(&edge.from) {
                    let to_depth = from_depth + 1;

                    let entry = depths.entry(edge.to.clone()).or_insert(0);
                    if to_depth > *entry {
                        *entry = to_depth;
                        changed = true;
                        max_depth = max_depth.max(to_depth);
                    }
                }
            }
        }

        max_depth
    }
}

/// Statistics about graph execution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionStatistics {
    /// Total number of executions
    pub total_executions: u32,
    /// Number of successful executions
    pub successful_executions: u32,
    /// Number of failed executions  
    pub failed_executions: u32,
    /// Total time spent executing (ms)
    pub total_execution_time_ms: u64,
    /// Per-node execution statistics
    pub node_stats: HashMap<String, NodeStatistics>,
}

/// Per-node execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeStatistics {
    /// How many times this node was executed
    pub execution_count: u32,
    /// Average execution duration (ms)
    pub avg_duration_ms: f64,
    /// Minimum execution duration (ms)
    pub min_duration_ms: u64,
    /// Maximum execution duration (ms)
    pub max_duration_ms: u64,
    /// Number of times this node failed
    pub failure_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_type_display() {
        assert_eq!(NodeType::Start.to_string(), "START");
        assert_eq!(NodeType::Handler.to_string(), "Handler");
        assert_eq!(NodeType::End.to_string(), "END");
    }

    #[test]
    fn test_node_type_serialization() {
        let node_type = NodeType::Handler;
        let json = serde_json::to_string(&node_type).unwrap();
        assert_eq!(json, "\"handler\"");
    }

    #[test]
    fn test_visualization_json_export() {
        let viz = StateGraphVisualization {
            id: "test".to_string(),
            name: "Test Graph".to_string(),
            nodes: vec![
                StateGraphNode {
                    id: "START".to_string(),
                    name: "Start".to_string(),
                    node_type: NodeType::Start,
                    description: None,
                    position: Some((0.0, 0.0)),
                    status: NodeExecutionStatus::Pending,
                },
            ],
            edges: vec![],
            metadata: GraphMetadata {
                node_count: 1,
                edge_count: 0,
                entry_point: "START".to_string(),
                compiled_at: chrono::Local::now().to_rfc3339(),
                version: "1.0".to_string(),
                max_depth: 1,
            },
        };

        let json = StateGraphVisualizer::to_json(&viz).unwrap();
        assert!(json.contains("Test Graph"));
        assert!(json.contains("START"));
    }

    #[test]
    fn test_add_nodes_and_edges() {
        let mut viz = StateGraphVisualization {
            id: "test".to_string(),
            name: "Test".to_string(),
            nodes: vec![],
            edges: vec![],
            metadata: GraphMetadata {
                node_count: 0,
                edge_count: 0,
                entry_point: "START".to_string(),
                compiled_at: chrono::Local::now().to_rfc3339(),
                version: "1.0".to_string(),
                max_depth: 0,
            },
        };

        StateGraphVisualizer::add_node(&mut viz, "node1", "Node 1", NodeType::Handler);
        assert_eq!(viz.nodes.len(), 1);
        assert_eq!(viz.metadata.node_count, 1);

        StateGraphVisualizer::add_edge(&mut viz, "START", "node1", None);
        assert_eq!(viz.edges.len(), 1);
        assert_eq!(viz.metadata.edge_count, 1);
    }

    #[test]
    fn test_calculate_depth() {
        let mut viz = StateGraphVisualization {
            id: "test".to_string(),
            name: "Test".to_string(),
            nodes: vec![],
            edges: vec![],
            metadata: GraphMetadata {
                node_count: 0,
                edge_count: 0,
                entry_point: "START".to_string(),
                compiled_at: chrono::Local::now().to_rfc3339(),
                version: "1.0".to_string(),
                max_depth: 0,
            },
        };

        StateGraphVisualizer::add_edge(&mut viz, "START", "node1", None);
        StateGraphVisualizer::add_edge(&mut viz, "node1", "node2", None);
        StateGraphVisualizer::add_edge(&mut viz, "node2", "END", None);

        let depth = StateGraphVisualizer::calculate_depth(&viz);
        assert_eq!(depth, 3);
    }
}
