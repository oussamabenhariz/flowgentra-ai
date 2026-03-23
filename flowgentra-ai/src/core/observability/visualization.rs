//! # Graph Visualization Module
//!
//! Provides serialization and visualization capabilities for agent execution graphs.
//!
//! ## Features
//!
//! - **Graph Serialization**: Convert graph structures to JSON format
//! - **Visualization Data**: Generate node and edge data for external tools
//! - **Layout Algorithms**: Automatic and customizable graph layout support
//! - **Performance Metrics**: Collect execution statistics and profiling data
//!
//! ## Example
//!
//! ```ignore
//!
//! // Create a visualizer from a compiled graph
//! let viz = GraphVisualizer::from_compiled_graph(&compiled_graph);
//! let json = viz.to_json()?;
//!
//! // Create a tracer for execution monitoring
//! let tracer = ExecutionTracer::new();
//! tracer.trace_node_start("node_name");
//! tracer.trace_node_end("node_name", Duration::from_secs(1), true);
//! ```

use crate::core::graph::Graph;
use crate::core::state::State;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Represents a node as it appears in the visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationNode {
    /// Unique node identifier
    pub id: String,

    /// Display name for the node
    pub name: String,

    /// Node type (handler, conditional router, etc.)
    pub node_type: String,

    /// Visual position (for layout algorithms)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub position: Option<Position>,

    /// Metadata about the node
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,

    /// Node status during execution
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<NodeStatus>,
}

/// Represents an edge in the visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationEdge {
    /// Source node ID
    pub from: String,

    /// Target node ID
    pub to: String,

    /// Edge condition description (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,

    /// Visual routing style
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing: Option<RoutingStyle>,

    /// Edge execution metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<EdgeMetrics>,
}

/// Visual positioning for nodes in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// X coordinate
    pub x: f32,

    /// Y coordinate
    pub y: f32,
}

/// Describes how an edge is visually routed
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RoutingStyle {
    /// Straight line between nodes
    Straight,

    /// Curved path between nodes
    Curved,

    /// Right-angle routing
    Orthogonal,
}

/// Status of a node during execution
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Waiting to be executed
    Pending,

    /// Currently executing
    Running,

    /// Successfully completed
    Success,

    /// Execution failed with error
    Failed,

    /// Execution was skipped
    Skipped,
}

/// Metrics for edge traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMetrics {
    /// Number of times this edge was traversed
    pub traversal_count: u32,

    /// Total time spent traversing this edge (including target node execution)
    pub total_duration_ms: u64,

    /// Average traversal time
    pub avg_duration_ms: f64,

    /// Condition evaluation time (if applicable)
    pub condition_eval_ms: Option<u64>,
}

/// Metrics for node execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetrics {
    /// Number of times the node was executed
    pub execution_count: u32,

    /// Total execution time
    pub total_duration_ms: u64,

    /// Average execution time
    pub avg_duration_ms: f64,

    /// Success count
    pub success_count: u32,

    /// Failure count
    pub failure_count: u32,

    /// Last execution status
    pub last_status: Option<String>,

    /// Last execution error (if any)
    pub last_error: Option<String>,
}

/// Complete graph visualization with layout and metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphVisualization {
    /// Graph metadata
    pub name: String,

    /// All nodes in the graph
    pub nodes: Vec<VisualizationNode>,

    /// All edges in the graph
    pub edges: Vec<VisualizationEdge>,

    /// Graph-level metrics
    pub metrics: GraphMetrics,

    /// Execution status
    pub status: GraphExecutionStatus,

    /// Layout algorithm used for positioning
    pub layout: LayoutAlgorithm,
}

/// Metrics for the entire graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    /// Total number of node executions
    pub total_executions: u32,

    /// Total execution time
    pub total_duration_ms: u64,

    /// Success count
    pub success_count: u32,

    /// Failure count
    pub failure_count: u32,

    /// Node-specific metrics
    pub node_metrics: HashMap<String, NodeMetrics>,

    /// Edge-specific metrics
    pub edge_metrics: HashMap<(String, String), EdgeMetrics>,
}

/// Status of graph execution
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GraphExecutionStatus {
    /// Graph has not been executed yet
    Idle,

    /// Graph execution is in progress
    Running,

    /// Graph execution completed successfully
    Completed,

    /// Graph execution failed
    Failed,
}

/// Layout algorithm for positioning nodes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LayoutAlgorithm {
    /// Hierarchical/layered layout
    Hierarchical,

    /// Force-directed layout
    ForceDirected,

    /// Grid layout
    Grid,

    /// Best-fit (automatic selection)
    Auto,
}

/// Converts a graph into a visualization structure
pub struct GraphVisualizer<'a, T: State> {
    graph: &'a Graph<T>,
    layout: LayoutAlgorithm,
}

impl<'a, T: State> GraphVisualizer<'a, T> {
    /// Create a visualizer for the given graph
    pub fn new(graph: &'a Graph<T>, layout: LayoutAlgorithm) -> Self {
        GraphVisualizer { graph, layout }
    }

    /// Create a visualizer with automatic layout selection
    pub fn with_auto_layout(graph: &'a Graph<T>) -> Self {
        GraphVisualizer {
            graph,
            layout: LayoutAlgorithm::Auto,
        }
    }

    /// Generate a complete visualization
    pub fn visualize(&self) -> GraphVisualization {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Add nodes
        for name in self.graph.nodes.keys() {
            nodes.push(VisualizationNode {
                id: name.clone(), // Add explicit type annotation if needed
                name: name.clone(),
                node_type: "handler".to_string(), // Could be enhanced to distinguish types
                position: None,
                metadata: None,
                status: Some(NodeStatus::Pending),
            });
        }

        // Add edges
        for edge in &self.graph.edges {
            edges.push(VisualizationEdge {
                from: edge.from.clone(),
                to: edge.to.clone(),
                condition: edge.condition_name.clone(),
                routing: Some(RoutingStyle::Curved),
                metrics: None,
            });
        }

        // Apply layout if specified
        if matches!(self.layout, LayoutAlgorithm::Hierarchical) {
            self.apply_hierarchical_layout(&mut nodes);
        }

        GraphVisualization {
            name: "Agent Graph".to_string(),
            nodes,
            edges,
            metrics: GraphMetrics {
                total_executions: 0,
                total_duration_ms: 0,
                success_count: 0,
                failure_count: 0,
                node_metrics: HashMap::new(),
                edge_metrics: HashMap::new(),
            },
            status: GraphExecutionStatus::Idle,
            layout: self.layout,
        }
    }

    /// Apply hierarchical layout to position nodes
    fn apply_hierarchical_layout(&self, nodes: &mut [VisualizationNode]) {
        // Simple hierarchical layout: arrange in columns by depth
        let mut positions: HashMap<String, (i32, i32)> = HashMap::new();
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        // Start from START nodes
        queue.push_back(("START".to_string(), 0, 0));

        while let Some((node_id, depth, index)) = queue.pop_front() {
            if visited.insert(node_id.clone()) {
                positions.insert(node_id.clone(), (depth, index));

                // Find next nodes
                let mut next_index = 0;
                for edge in &self.graph.edges {
                    if edge.from == node_id {
                        queue.push_back((edge.to.clone(), depth + 1, next_index));
                        next_index += 1;
                    }
                }
            }
        }

        // Apply positions to nodes
        for node in nodes {
            if let Some((depth, index)) = positions.get(&node.id) {
                node.position = Some(Position {
                    x: (depth * 200) as f32,
                    y: (index * 100) as f32,
                });
            }
        }
    }

    /// Serialize visualization to JSON
    pub fn to_json(&self) -> serde_json::Result<String> {
        let viz = self.visualize();
        serde_json::to_string_pretty(&viz)
    }
}

/// Tracks execution events for real-time visualization
pub struct ExecutionTracer {
    events: Arc<Mutex<Vec<ExecutionEvent>>>,
    start_time: SystemTime,
}

/// An event during graph execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEvent {
    /// Event timestamp (milliseconds since execution start)
    pub timestamp_ms: u64,

    /// Type of event
    pub event_type: ExecutionEventType,

    /// Associated node name (if applicable)
    pub node_id: Option<String>,

    /// Event details
    pub details: Option<String>,

    /// Duration of the event (if applicable)
    pub duration_ms: Option<u64>,
}

/// Types of execution events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionEventType {
    /// Node execution started
    NodeStarted,

    /// Node execution completed successfully
    NodeCompleted,

    /// Node execution failed
    NodeFailed,

    /// Edge was traversed
    EdgeTraversed,

    /// Condition was evaluated
    ConditionEvaluated,

    /// Graph execution started
    GraphStarted,

    /// Graph execution completed
    GraphCompleted,

    /// State was updated
    StateUpdated,

    /// Custom event (user-defined)
    Custom(String),
}

impl ExecutionTracer {
    /// Create a new execution tracer
    pub fn new() -> Self {
        ExecutionTracer {
            events: Arc::new(Mutex::new(Vec::new())),
            start_time: SystemTime::now(),
        }
    }

    /// Record the start of node execution
    pub fn trace_node_start(&self, node_id: &str) {
        self.add_event(ExecutionEvent {
            timestamp_ms: self.elapsed_ms(),
            event_type: ExecutionEventType::NodeStarted,
            node_id: Some(node_id.to_string()),
            details: None,
            duration_ms: None,
        });
    }

    /// Record the completion of node execution
    pub fn trace_node_end(&self, node_id: &str, duration: Duration, success: bool) {
        let event_type = if success {
            ExecutionEventType::NodeCompleted
        } else {
            ExecutionEventType::NodeFailed
        };

        self.add_event(ExecutionEvent {
            timestamp_ms: self.elapsed_ms(),
            event_type,
            node_id: Some(node_id.to_string()),
            details: None,
            duration_ms: Some(duration.as_millis() as u64),
        });
    }

    /// Record edge traversal
    pub fn trace_edge_traversal(&self, from: &str, to: &str, condition_met: bool) {
        self.add_event(ExecutionEvent {
            timestamp_ms: self.elapsed_ms(),
            event_type: ExecutionEventType::EdgeTraversed,
            node_id: Some(format!("{}→{}", from, to)),
            details: Some(format!("condition_met={}", condition_met)),
            duration_ms: None,
        });
    }

    /// Record state update
    pub fn trace_state_update(&self, key: &str, value: &str) {
        self.add_event(ExecutionEvent {
            timestamp_ms: self.elapsed_ms(),
            event_type: ExecutionEventType::StateUpdated,
            node_id: Some(key.to_string()),
            details: Some(value.to_string()),
            duration_ms: None,
        });
    }

    /// Record a custom event
    pub fn trace_custom(&self, event_name: &str, details: Option<&str>) {
        self.add_event(ExecutionEvent {
            timestamp_ms: self.elapsed_ms(),
            event_type: ExecutionEventType::Custom(event_name.to_string()),
            node_id: None,
            details: details.map(|s| s.to_string()),
            duration_ms: None,
        });
    }

    /// Add an event to the trace
    fn add_event(&self, event: ExecutionEvent) {
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }
    }

    /// Get all recorded events
    pub fn get_events(&self) -> Vec<ExecutionEvent> {
        self.events
            .lock()
            .ok()
            .map(|e| e.clone())
            .unwrap_or_default()
    }

    /// Get events as JSON
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(&self.get_events())
    }

    /// Calculate elapsed time since tracing started
    fn elapsed_ms(&self) -> u64 {
        self.start_time.elapsed().unwrap_or_default().as_millis() as u64
    }

    /// Clear all recorded events
    pub fn clear(&self) {
        if let Ok(mut events) = self.events.lock() {
            events.clear();
        }
    }
}

impl Default for ExecutionTracer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_tracer() {
        let tracer = ExecutionTracer::new();

        tracer.trace_node_start("node1");
        std::thread::sleep(Duration::from_millis(10));
        tracer.trace_node_end("node1", Duration::from_millis(10), true);

        let events = tracer.get_events();
        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0].event_type,
            ExecutionEventType::NodeStarted
        ));
        assert!(matches!(
            events[1].event_type,
            ExecutionEventType::NodeCompleted
        ));
    }

    #[test]
    fn test_visualization_serialization() {
        let tracer = ExecutionTracer::new();
        tracer.trace_node_start("test");

        let json = tracer.to_json().expect("Should serialize");
        assert!(json.contains("NodeStarted"));
    }
}
