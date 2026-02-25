//! # Graph Data Structure and Operations
//!
//! The Graph represents your agent's workflow as a Directed Acyclic Graph (DAG).
//!
//! ## Key Concepts
//!
//! **Nodes** - Computational steps in your workflow
//! **Edges** - Connections between nodes
//! **Conditions** - Optional logic to decide which edges to take
//!
//! ## Validation
//!
//! The graph is automatically validated for:
//! - No cycles (must be acyclic)
//! - All referenced nodes exist
//! - START and END nodes are properly connected
//! - No orphaned nodes
//!
//! Example graph:
//! ```text
//! START
//!  ├─→ extract_info
//!  └─→ analyze_content
//!      └─→ decide_routing
//!         ├─→ complex_path (if complexity > 50)
//!         └─→ simple_path
//!      └─→ format_results
//!          └─→ END
//! ```

use crate::core::error::Result;
use crate::core::node::{Edge, Node};
use std::collections::{HashMap, VecDeque};

/// Unique identifier for a node in the graph
pub type NodeId = String;

/// The execution graph for the agent
///
/// A graph represents the complete workflow structure:
/// - Which nodes exist and what they do
/// - How nodes are connected (edges)
/// - Which nodes are entry and exit points
/// - Optional: Loop configurations for repeated execution
pub struct Graph {
    /// All nodes in the graph, indexed by name
    pub(crate) nodes: HashMap<NodeId, Node>,

    /// All edges (connections) between nodes
    pub(crate) edges: Vec<Edge>,

    /// Entry point nodes (connected from START)
    start_nodes: Vec<NodeId>,

    /// Exit point nodes (connected to END)
    end_nodes: Vec<NodeId>,

    /// Whether to allow cycles (loops) in the graph
    /// When true, graph validation will not enforce DAG property
    allow_cycles: bool,

    /// Maximum iterations for any loop in the graph
    /// Prevents infinite loops
    max_loop_iterations: usize,
}

impl Graph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
            edges: Vec::new(),
            start_nodes: Vec::new(),
            end_nodes: Vec::new(),
            allow_cycles: false,
            max_loop_iterations: 10,
        }
    }

    /// Create a new graph that allows loops
    pub fn with_loops() -> Self {
        Graph {
            nodes: HashMap::new(),
            edges: Vec::new(),
            start_nodes: Vec::new(),
            end_nodes: Vec::new(),
            allow_cycles: true,
            max_loop_iterations: 10,
        }
    }

    // =========================================================================
    // Graph Construction
    // =========================================================================

    /// Add a node to the graph
    ///
    /// Nodes should be added before edges are created that reference them.
    pub fn add_node(&mut self, node: Node) {
        self.nodes.insert(node.name.clone(), node);
    }

    /// Add an edge to the graph
    ///
    /// Both the source and target nodes should already exist in the graph.
    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    /// Set the entry nodes (those connected from START)
    pub fn set_start_nodes(&mut self, nodes: Vec<NodeId>) {
        self.start_nodes = nodes;
    }

    /// Set the exit nodes (those connected to END)
    pub fn set_end_nodes(&mut self, nodes: Vec<NodeId>) {
        self.end_nodes = nodes;
    }

    /// Enable loop support (allow cycles in the graph)
    ///
    /// When enabled, the graph validation will not enforce DAG property.
    /// You should configure max_iterations to prevent infinite loops.
    pub fn allow_loops(&mut self, allow: bool) -> &mut Self {
        self.allow_cycles = allow;
        self
    }

    /// Check if loops are allowed
    pub fn loops_allowed(&self) -> bool {
        self.allow_cycles
    }

    /// Set maximum iterations for loops
    ///
    /// This prevents infinite loops by limiting the number of
    /// iterations the runtime will execute for any loop structure.
    pub fn set_max_loop_iterations(&mut self, max: usize) -> &mut Self {
        self.max_loop_iterations = max;
        self
    }

    /// Get maximum iterations for loops
    pub fn max_loop_iterations(&self) -> usize {
        self.max_loop_iterations
    }

    // =========================================================================
    // Graph Queries
    // =========================================================================

    /// Get a specific node by ID
    pub fn get_node(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> &HashMap<NodeId, Node> {
        &self.nodes
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Get the entry nodes
    pub fn start_nodes(&self) -> &[NodeId] {
        &self.start_nodes
    }

    /// Get the exit nodes
    pub fn end_nodes(&self) -> &[NodeId] {
        &self.end_nodes
    }

    // =========================================================================
    // Graph Operations
    // =========================================================================

    /// Get all outgoing edges from a node
    pub fn get_next_nodes(&self, node_id: &str) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.from == node_id).collect()
    }

    /// Get the IDs of nodes reachable in one step from the given node.
    /// Returns unique target node names from outgoing edges (includes "END" if present).
    pub fn get_reachable_node_ids(&self, from: &str) -> Vec<String> {
        use std::collections::HashSet;
        self.edges
            .iter()
            .filter(|e| e.from == from)
            .map(|e| e.to.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }

    /// Get all incoming edges to a node
    pub fn get_prev_nodes(&self, node_id: &str) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.to == node_id).collect()
    }

    /// Validate the graph structure
    ///
    /// Checks:
    /// - No cycles exist (DAG property) - unless loops are explicitly allowed
    /// - All referenced nodes exist
    /// - START and END nodes are properly connected
    /// - No orphaned nodes
    pub fn validate(&self) -> Result<()> {
        // Check for cycles using DFS (unless loops are allowed)
        if !self.allow_cycles && self.has_cycle() {
            return Err(crate::core::error::ErenFlowError::CycleDetected);
        }

        // Check all nodes referenced in edges exist
        for edge in &self.edges {
            if edge.from != "START" && !self.nodes.contains_key(&edge.from) {
                return Err(crate::core::error::ErenFlowError::GraphError(format!(
                    "Edge references non-existent node: {}",
                    edge.from
                )));
            }
            if edge.to != "END" && !self.nodes.contains_key(&edge.to) {
                return Err(crate::core::error::ErenFlowError::GraphError(format!(
                    "Edge references non-existent node: {}",
                    edge.to
                )));
            }
        }

        // Check start and end nodes exist
        for start in &self.start_nodes {
            if !self.nodes.contains_key(start) {
                return Err(crate::core::error::ErenFlowError::GraphError(format!(
                    "Start node does not exist: {}",
                    start
                )));
            }
        }

        for end in &self.end_nodes {
            if !self.nodes.contains_key(end) {
                return Err(crate::core::error::ErenFlowError::GraphError(format!(
                    "End node does not exist: {}",
                    end
                )));
            }
        }

        Ok(())
    }

    // =========================================================================
    // Cycle Detection (Internal)
    // =========================================================================

    /// Check if the graph contains a cycle (detects non-DAG property)
    fn has_cycle(&self) -> bool {
        let mut visited = std::collections::HashSet::new();
        let mut rec_stack = std::collections::HashSet::new();

        for node_id in self.nodes.keys() {
            if !visited.contains(node_id)
                && self.has_cycle_util(node_id, &mut visited, &mut rec_stack)
            {
                return true;
            }
        }

        false
    }

    /// DFS helper for cycle detection
    fn has_cycle_util(
        &self,
        node: &str,
        visited: &mut std::collections::HashSet<String>,
        rec_stack: &mut std::collections::HashSet<String>,
    ) -> bool {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        // Get all neighbors
        let neighbors: Vec<String> = self
            .edges
            .iter()
            .filter(|e| e.from == node)
            .map(|e| e.to.clone())
            .collect();

        for neighbor in neighbors {
            if neighbor == "END" {
                continue; // Skip END node
            }

            if !visited.contains(&neighbor) {
                if self.has_cycle_util(&neighbor, visited, rec_stack) {
                    return true;
                }
            } else if rec_stack.contains(&neighbor) {
                return true;
            }
        }

        rec_stack.remove(node);
        false
    }

    // =========================================================================
    // Execution Helpers
    // =========================================================================

    /// Topological sort of the graph
    ///
    /// Returns nodes in execution order (respecting dependencies).
    /// Useful for analysis and optimization.
    pub fn topological_sort(&self) -> Result<Vec<NodeId>> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();

        // Initialize in-degrees
        for node in self.nodes.keys() {
            in_degree.insert(node.clone(), 0);
        }

        for edge in &self.edges {
            if edge.to != "END" {
                *in_degree.get_mut(&edge.to).unwrap() += 1;
            }
        }

        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(id, _)| id.clone())
            .collect();

        let mut result = Vec::new();

        while let Some(node) = queue.pop_front() {
            result.push(node.clone());

            for edge in self.get_next_nodes(&node) {
                if edge.to != "END" {
                    let degree = in_degree.get_mut(&edge.to).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(edge.to.clone());
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(crate::core::error::ErenFlowError::CycleDetected);
        }

        Ok(result)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

// Sub-modules
pub mod compiler;
pub mod routing;

pub use compiler::{
    CompilationError, CompilationStats, CompiledGraph, CompilerOptions, ExecutionStage,
    GraphCompiler,
};
pub use routing::*;
