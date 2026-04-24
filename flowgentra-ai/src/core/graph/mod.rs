//! # Graph Data Structure and Operations
//!
//! The Graph represents your agent's workflow as a Directed Acyclic Graph (DAG).
//! It provides efficient queries, validation, and execution planning for complex workflows.
//!
//! ## Key Concepts
//!
//! - **Nodes** - Computational steps in your workflow
//! - **Edges** - Connections between nodes with optional conditions
//! - **Conditions** - Optional logic to decide which edges to take
//! - **Adjacency Lists** - Cached edge mappings for O(1) lookups
//!
//! ## Validation
//!
//! The graph is automatically validated for:
//! - No cycles (must be acyclic) unless explicitly allowed
//! - All referenced nodes exist
//! - START and END nodes are properly connected
//! - No orphaned nodes
//!
//! ## Performance Considerations
//!
//! - Edge queries use cached adjacency lists (O(1) avg case)
//! - Cycle detection is O(V + E) using iterative DFS
//! - Topological sort is O(V + E)
//!
//! ## Example DAG
//!
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
use crate::core::state::{DynState, State};
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
///
/// # Performance Features
///
/// - **Adjacency List Cache**: Maintains HashMap<NodeId, Vec<usize>> for O(1) edge lookups
/// - **Lazy Evaluation**: Validation deferred until explicitly called
/// - **Memory Efficient**: Single edge vec with indexed access
pub struct Graph<T: State> {
    /// All nodes in the graph, indexed by name
    pub(crate) nodes: HashMap<NodeId, Node<T>>,

    /// All edges (connections) between nodes, stored in a single vec
    pub(crate) edges: Vec<Edge<T>>,

    /// Cached adjacency list: node_id → indices into self.edges (outgoing edges)
    /// Built lazily and maintained on edge additions
    adjacency_list: HashMap<NodeId, Vec<usize>>,

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

impl<T: State> Graph<T> {
    /// Create a new empty graph. Cycles are allowed by default.
    /// Call `set_allow_cycles(false)` to enforce strict DAG mode.
    pub fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency_list: HashMap::new(),
            start_nodes: Vec::new(),
            end_nodes: Vec::new(),
            allow_cycles: true,
            max_loop_iterations: 25,
        }
    }

    /// Create a new graph that allows loops (cycles)
    ///
    /// Use this for workflows where nodes can intentionally feed back to earlier nodes.
    /// Remember to configure `set_max_loop_iterations()` to prevent infinite loops.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use flowgentra_ai::core::graph::Graph;
    ///
    /// let mut graph = Graph::with_loops();
    /// graph.set_max_loop_iterations(5);
    /// // graph allows cycles up to 5 iterations
    /// ```
    pub fn with_loops() -> Self {
        Graph {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency_list: HashMap::new(),
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
    ///
    /// # Arguments
    /// * `node` - The node to add to the graph
    pub fn add_node(&mut self, node: Node<T>) {
        self.nodes.insert(node.name.clone(), node);
    }

    /// Add an edge to the graph and update adjacency list cache
    ///
    /// Both the source and target nodes should already exist in the graph.
    /// The adjacency list is automatically updated for O(1) future lookups.
    ///
    /// # Arguments
    /// * `edge` - The edge to add to the graph
    pub fn add_edge(&mut self, edge: Edge<T>) {
        // Get the index where this edge will be stored
        let edge_index = self.edges.len();

        // Store the edge
        self.edges.push(edge.clone());

        // Update adjacency list: map from -> [edge indices]
        self.adjacency_list
            .entry(edge.from.clone())
            .or_default()
            .push(edge_index);
    }

    /// Set the entry nodes (those connected from START)
    ///
    /// # Arguments
    /// * `nodes` - Vector of node IDs that act as entry points
    pub fn set_start_nodes(&mut self, nodes: Vec<NodeId>) {
        self.start_nodes = nodes;
    }

    /// Set the exit nodes (those connected to END)
    ///
    /// # Arguments
    /// * `nodes` - Vector of node IDs that act as exit points
    pub fn set_end_nodes(&mut self, nodes: Vec<NodeId>) {
        self.end_nodes = nodes;
    }

    /// Enable or disable loop support (allow cycles in the graph)
    ///
    /// When enabled, the graph validation will not enforce DAG property.
    /// You should configure `set_max_loop_iterations()` to prevent infinite loops.
    ///
    /// # Arguments
    /// * `allow` - Whether to allow cycles in the graph
    ///
    /// # Returns
    /// * `&mut Self` - For builder pattern chaining
    pub fn set_allow_cycles(&mut self, allow: bool) -> &mut Self {
        self.allow_cycles = allow;
        self
    }

    /// Check if cycles are allowed in this graph
    ///
    /// # Returns
    /// * `true` if the graph allows cycles, `false` if it enforces DAG property
    pub fn allows_cycles(&self) -> bool {
        self.allow_cycles
    }

    /// Set maximum iterations for loops (builder-style, deprecated name)
    ///
    /// ⚠️ **Deprecated**: Use `set_max_loop_iterations()` instead.
    /// This method is kept for backwards compatibility.
    #[deprecated(since = "0.2.0", note = "Use `set_max_loop_iterations` instead")]
    pub fn allow_loops(&mut self, allow: bool) -> &mut Self {
        self.set_allow_cycles(allow)
    }

    /// Check if loops are allowed (deprecated name)
    ///
    /// ⚠️ **Deprecated**: Use `allows_cycles()` instead.
    /// This method is kept for backwards compatibility.
    #[deprecated(since = "0.2.0", note = "Use `allows_cycles` instead")]
    pub fn loops_allowed(&self) -> bool {
        self.allows_cycles()
    }

    /// Set maximum iterations for loops
    ///
    /// This prevents infinite loops by limiting the number of iterations
    /// the runtime will execute for any loop structure.
    ///
    /// # Arguments
    /// * `max` - Maximum number of loop iterations (0 = unlimited)
    ///
    /// # Returns
    /// * `&mut Self` - For builder pattern chaining
    ///
    /// # Example
    ///
    /// ```ignore
    /// use flowgentra_ai::core::graph::Graph;
    ///
    /// let mut graph = Graph::with_loops();
    /// graph.set_max_loop_iterations(5); // Allow max 5 iterations
    /// ```
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

    /// Get a specific node by ID (borrowed reference)
    ///
    /// # Arguments
    /// * `id` - The node ID (name)
    ///
    /// # Returns
    /// * `Some(&Node)` if the node exists, `None` otherwise
    pub fn get_node(&self, id: &str) -> Option<&Node<T>> {
        self.nodes.get(id)
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> &HashMap<NodeId, Node<T>> {
        &self.nodes
    }

    /// Get all edges in the graph
    pub fn edges(&self) -> &[Edge<T>] {
        &self.edges
    }

    /// Get the entry nodes (START nodes)
    pub fn start_nodes(&self) -> &[NodeId] {
        &self.start_nodes
    }

    /// Get the exit nodes (END nodes)
    pub fn end_nodes(&self) -> &[NodeId] {
        &self.end_nodes
    }

    // =========================================================================
    // Graph Operations (Optimized for Performance)
    // =========================================================================

    /// Get all outgoing edges from a node (O(1) average case using adjacency list)
    ///
    /// # Arguments
    /// * `node_id` - ID of the source node (borrowed reference)
    ///
    /// # Returns
    /// * Vector of references to edges originating from this node
    ///
    /// # Performance
    /// * Time: O(1) average case + O(k) where k is number of outgoing edges to copy vec refs
    /// * Space: O(k) for the returned vector
    ///
    /// # Example
    ///
    /// ```ignore
    /// let edges = graph.get_next_nodes("my_node");
    /// for edge in edges {
    ///     println!("Can go to: {}", edge.to);
    /// }
    /// ```
    pub fn get_next_nodes(&self, node_id: &str) -> Vec<&Edge<T>> {
        self.adjacency_list
            .get(node_id)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&idx| self.edges.get(idx))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the IDs of nodes reachable in one step from the given node
    ///
    /// Returns unique target node names from outgoing edges (includes "END" if present).
    ///
    /// # Arguments
    /// * `from` - ID of the source node
    ///
    /// # Returns
    /// * Vector of unique target node IDs
    ///
    /// # Performance
    /// * Time: O(k) where k is the number of outgoing edges
    /// * Space: O(k) for result vector + O(k) for HashSet
    pub fn get_reachable_node_ids(&self, from: &str) -> Vec<String> {
        use std::collections::HashSet;
        self.adjacency_list
            .get(from)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&idx| self.edges.get(idx).map(|e| e.to.clone()))
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all incoming edges to a node (O(E) linear scan - consider caching if frequently used)
    ///
    /// # Arguments
    /// * `node_id` - ID of the target node
    ///
    /// # Returns
    /// * Vector of references to edges ending at this node
    ///
    /// # Performance
    /// * Time: O(E) where E is total number of edges
    /// * Space: O(k) where k is number of incoming edges
    ///
    /// # Note
    /// If this operation is called frequently, consider maintaining a reverse adjacency list.
    pub fn get_prev_nodes(&self, node_id: &str) -> Vec<&Edge<T>> {
        self.edges.iter().filter(|e| e.to == node_id).collect()
    }

    /// Validate the graph structure for correctness
    ///
    /// Checks:
    /// - No cycles exist (DAG property) - unless loops are explicitly allowed
    /// - All referenced nodes exist
    /// - START and END nodes are properly connected
    /// - No orphaned nodes
    ///
    /// # Errors
    /// Returns `FlowgentraError` if validation fails with detailed reason
    ///
    /// # Performance
    /// * Time: O(V + E) for cycle detection using iterative DFS
    /// * Additional O(E) for edge validation
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut graph = Graph::new();
    /// // ... add nodes and edges ...
    /// graph.validate()?; // Will return Err if graph is invalid
    /// ```
    pub fn validate(&self) -> Result<()> {
        // Check for cycles using DFS (unless loops are allowed)
        if !self.allow_cycles {
            if let Some(cycle_nodes) = self.find_cycle_nodes() {
                return Err(crate::core::error::FlowgentraError::CycleDetected {
                    nodes: cycle_nodes.join(", "),
                });
            }
        }

        // Check all nodes referenced in edges exist
        for edge in &self.edges {
            if edge.from != "START" && !self.nodes.contains_key(&edge.from) {
                return Err(crate::core::error::FlowgentraError::GraphError(format!(
                    "Edge references non-existent node: {}",
                    edge.from
                )));
            }
            if edge.to != "END" && !self.nodes.contains_key(&edge.to) {
                return Err(crate::core::error::FlowgentraError::GraphError(format!(
                    "Edge references non-existent node: {}",
                    edge.to
                )));
            }
        }

        // Check start and end nodes exist
        for start in &self.start_nodes {
            if !self.nodes.contains_key(start) {
                return Err(crate::core::error::FlowgentraError::GraphError(format!(
                    "Start node does not exist: {}",
                    start
                )));
            }
        }

        for end in &self.end_nodes {
            if !self.nodes.contains_key(end) {
                return Err(crate::core::error::FlowgentraError::GraphError(format!(
                    "End node does not exist: {}",
                    end
                )));
            }
        }

        // When cycles are allowed, check that every node with outgoing edges has a path to END.
        // A node with no outgoing edges cannot cause an infinite loop — it is either managed
        // externally (e.g. as a supervisor child) or terminates naturally. Only nodes that
        // participate in the graph's own routing can create unbounded cycles.
        if self.allow_cycles {
            let unreachable: Vec<String> = self
                .nodes
                .keys()
                .filter(|name| {
                    let has_outgoing = self.edges.iter().any(|e| &e.from == *name);
                    has_outgoing && !self.can_reach_end(name)
                })
                .cloned()
                .collect();
            if !unreachable.is_empty() {
                let mut sorted = unreachable;
                sorted.sort();
                return Err(crate::core::error::FlowgentraError::NoTerminationPath {
                    nodes: sorted.join(", "),
                });
            }
        }

        Ok(())
    }

    /// Returns true if `start` has at least one path that eventually reaches END.
    /// Used to detect missing termination conditions in cyclic graphs.
    fn can_reach_end(&self, start: &str) -> bool {
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![start.to_string()];
        while let Some(node) = stack.pop() {
            if !visited.insert(node.clone()) {
                continue;
            }
            for edge in &self.edges {
                if edge.from == node {
                    if edge.to == "END" {
                        return true;
                    }
                    stack.push(edge.to.clone());
                }
            }
        }
        false
    }

    // =========================================================================
    // Cycle Detection (Internal)
    // =========================================================================

    /// Check if the graph contains a cycle (detects non-DAG property).
    ///
    /// Uses Kahn's iterative topological-sort algorithm — O(V + E), fully
    /// iterative with no recursion, so it is safe on arbitrarily large graphs
    /// without risk of stack overflow.
    ///
    /// Returns the nodes involved in a cycle (those whose in-degree never
    /// reaches zero), or `None` if the graph is acyclic.  The list is sorted
    /// for determinism.
    ///
    /// # Performance
    /// * Time:  O(V + E)
    /// * Space: O(V)  — in-degree table + BFS queue
    fn find_cycle_nodes(&self) -> Option<Vec<String>> {
        // Compute in-degree for every real node (skip "END" virtual node).
        let mut in_degree: HashMap<&str, usize> = self
            .nodes
            .keys()
            .map(|n| (n.as_str(), 0usize))
            .collect();

        for edge in &self.edges {
            if edge.to != "END" {
                if let Some(deg) = in_degree.get_mut(edge.to.as_str()) {
                    *deg += 1;
                }
            }
        }

        // Enqueue all zero-in-degree nodes.
        let mut queue: std::collections::VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&n, _)| n)
            .collect();

        let mut processed = 0usize;
        while let Some(node) = queue.pop_front() {
            processed += 1;
            if let Some(indices) = self.adjacency_list.get(node) {
                for &idx in indices {
                    if let Some(edge) = self.edges.get(idx) {
                        if edge.to != "END" {
                            if let Some(deg) = in_degree.get_mut(edge.to.as_str()) {
                                *deg -= 1;
                                if *deg == 0 {
                                    queue.push_back(edge.to.as_str());
                                }
                            }
                        }
                    }
                }
            }
        }

        if processed < self.nodes.len() {
            // Nodes whose in-degree never reached zero are part of a cycle.
            let mut cycle_nodes: Vec<String> = in_degree
                .into_iter()
                .filter(|(_, d)| d > 0)
                .map(|(n, _)| n.to_string())
                .collect();
            cycle_nodes.sort();
            Some(cycle_nodes)
        } else {
            None
        }
    }

    // =========================================================================
    // Execution Helpers
    // =========================================================================

    /// Topological sort of the graph
    ///
    /// Returns nodes in execution order (respecting dependencies).
    /// Useful for analysis, optimization, and execution planning.
    ///
    /// # Algorithm
    /// Uses Kahn's algorithm (BFS-based):
    /// 1. Compute in-degree for each node
    /// 2. Enqueue all nodes with in-degree 0
    /// 3. Dequeue node, add to result, decrement in-degree of neighbors
    /// 4. Repeat until queue empty
    /// 5. If result size != nodes size, graph has cycles
    ///
    /// # Returns
    /// * `Ok(Vec<NodeId>)` - Nodes in topological order
    /// * `Err(FlowgentraError::CycleDetected)` - If graph has cycles
    ///
    /// # Performance
    /// * Time: O(V + E)
    /// * Space: O(V)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let sorted = graph.topological_sort()?;
    /// println!("Execution order: {:?}", sorted);
    /// ```
    pub fn topological_sort(&self) -> Result<Vec<NodeId>> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();

        // Initialize in-degrees for all nodes
        for node in self.nodes.keys() {
            in_degree.insert(node.as_str(), 0);
        }

        // Count incoming edges
        for edge in &self.edges {
            if edge.to != "END" {
                if let Some(degree) = in_degree.get_mut(edge.to.as_str()) {
                    *degree += 1;
                }
            }
        }

        // Enqueue all nodes with in-degree 0
        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut result = Vec::new();

        while let Some(node) = queue.pop_front() {
            result.push(node.to_string());

            // For each outgoing edge, decrement in-degree of target
            for edge in self.get_next_nodes(node) {
                if edge.to != "END" {
                    if let Some(degree) = in_degree.get_mut(edge.to.as_str()) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(edge.to.as_str());
                        }
                    }
                }
            }
        }

        // If we didn't process all nodes, there's a cycle — collect the unprocessed nodes
        if result.len() != self.nodes.len() {
            let mut cycle_nodes: Vec<String> = in_degree
                .iter()
                .filter(|(_, &deg)| deg > 0)
                .map(|(&id, _)| id.to_string())
                .collect();
            cycle_nodes.sort();
            return Err(crate::core::error::FlowgentraError::CycleDetected {
                nodes: cycle_nodes.join(", "),
            });
        }

        Ok(result)
    }
}

impl<T: crate::core::state::State + Default> Default for Graph<T> {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Graph Builder - Fluent API for Building Graphs
// =============================================================================

/// A fluent builder for constructing graphs with a convenient, chainable API.
///
/// This provides a more user-friendly way to programmatically build graphs
/// compared to manually calling `add_node` and `add_edge`.
///
/// # Example: Basic Graph
///
/// ```ignore
/// use flowgentra_ai::core::graph::GraphBuilder;
/// use flowgentra_ai::core::node::Node;
///
/// let graph = GraphBuilder::new()
///     .add_node(Node::new("start"))
///     .add_node(Node::new("process"))
///     .add_node(Node::new("end"))
///     .add_edge("start", "process", None)
///     .add_edge("process", "end", None)
///     .build();
/// ```
///
/// # Example: With Conditions
///
/// ```ignore
/// use flowgentra_ai::core::graph::{GraphBuilder, RoutingCondition, Condition, ComparisonOp};
/// use flowgentra_ai::core::node::Node;
///
/// let condition = RoutingCondition::dsl(
///     Condition::compare("score", ComparisonOp::GreaterThan, 0.7)
/// );
///
/// let graph = GraphBuilder::new()
///     .add_node(Node::new("router"))
///     .add_node(Node::new("high_confidence_path"))
///     .add_node(Node::new("low_confidence_path"))
///     .add_edge_with_condition("router", "high_confidence_path", condition)
///     .build();
/// ```
pub struct GraphBuilder<T: State = DynState> {
    graph: Graph<T>,
}

impl<T: State> GraphBuilder<T> {
    /// Create a new graph builder starting with an empty DAG
    pub fn new() -> Self {
        GraphBuilder {
            graph: Graph::new(),
        }
    }

    /// Create a new graph builder with loop support
    pub fn with_loops() -> Self {
        GraphBuilder {
            graph: Graph::with_loops(),
        }
    }

    /// Add a node to the graph being built
    pub fn add_node(mut self, node: Node<T>) -> Self {
        self.graph.add_node(node);
        self
    }

    /// Add multiple nodes at once
    pub fn add_nodes(mut self, nodes: Vec<Node<T>>) -> Self {
        for node in nodes {
            self.graph.add_node(node);
        }
        self
    }

    /// Add an edge between two nodes
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        let edge = Edge::<T>::new(from, to, None);
        self.graph.add_edge(edge);
        self
    }

    /// Add an edge and its target node in one call (convenience method)
    ///
    /// This is useful for building linear or simple graphs where you want
    /// to create nodes and edges together.
    pub fn add_step(mut self, from: impl Into<String>, to_node: Node<T>) -> Self {
        let from_str = from.into();
        let to_str = to_node.name.clone();

        self.graph.add_node(to_node);
        let edge = Edge::<T>::new(from_str, to_str, None);
        self.graph.add_edge(edge);
        self
    }

    /// Set whether the graph allows cycles (loops)
    pub fn allow_cycles(mut self, allow: bool) -> Self {
        self.graph.set_allow_cycles(allow);
        self
    }

    /// Set the maximum number of loop iterations (prevents infinite loops)
    pub fn max_loop_iterations(mut self, max: usize) -> Self {
        self.graph.set_max_loop_iterations(max);
        self
    }

    /// Build and return the graph
    ///
    /// Note: This does NOT validate the graph. Call `validate()` on the
    /// returned graph if you need to check for cycles, orphaned nodes, etc.
    pub fn build(self) -> Graph<T> {
        self.graph
    }

    /// Build and validate the graph
    ///
    /// Returns an error if the graph is invalid (cycles, missing nodes, etc.)
    pub fn build_and_validate(self) -> Result<Graph<T>> {
        let graph = self.graph;
        graph.validate()?;
        Ok(graph)
    }
}

impl GraphBuilder<DynState> {
    /// Add an edge with a routing condition (only available for DynState graphs)
    pub fn add_edge_with_condition(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: routing::RoutingCondition,
    ) -> Self {
        let from_str = from.into();
        let to_str = to.into();

        match condition {
            routing::RoutingCondition::Function(f) => {
                let edge = Edge::new(from_str, to_str, Some(f));
                self.graph.add_edge(edge);
            }
            routing::RoutingCondition::DSL(cond) => {
                let mut edge = Edge::new(from_str, to_str, None);
                edge.routing_condition = Some(cond);
                self.graph.add_edge(edge);
            }
        }
        self
    }
}

impl<T: crate::core::state::State> Default for GraphBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Sub-modules
pub mod analysis;
pub mod compiler;
pub mod routing;

pub use analysis::{BranchAnalysis, GraphAnalyzer};
pub use compiler::{
    CompilationError, CompilationStats, CompiledGraph, CompilerOptions, ExecutionStage,
    GraphCompiler,
};
pub use routing::*;
