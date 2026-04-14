//! # Graph Compiler
//!
//! Transforms declarative graph definitions (YAML) into optimized execution plans.
//!
//! Instead of interpreting the graph at runtime, the compiler:
//! - **Parses Once** - Single compilation pass from YAML
//! - **Validates Structure** - Detects cycles, missing nodes, orphaned paths
//! - **Optimizes Parallelism** - Identifies independent nodes that run concurrently
//! - **Generates Plan** - Produces execution stages for efficient execution
//!
//! ## Example
//!
//! ```ignore
//! use flowgentra_ai::core::graph::compiler::GraphCompiler;
//! use flowgentra_ai::core::config::GraphConfig;
//!
//! let graph_config = GraphConfig { ... };  // from YAML
//! let compiler = GraphCompiler::new(&graph_config);
//! let compiled = compiler.compile()?;
//!
//! // Now you have:
//! // - compiled.execution_stages: parallel execution groups
//! // - compiled.topological_order: sequential node order
//! // - compiled.critical_path: longest path through graph
//! // - compiled.parallelizable_pairs: which nodes can run together
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

/// Represents a node in the compiled execution plan
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompiledNode {
    pub name: String,
    pub depth: usize,               // Distance from START
    pub is_parallel_eligible: bool, // Can run in parallel with others
}

/// A single stage of parallel execution
///
/// All nodes in a stage can execute concurrently since they have no
/// data dependencies between them.
#[derive(Debug, Clone)]
pub struct ExecutionStage {
    pub stage_number: usize,
    pub nodes: Vec<String>,
    pub is_critical: bool, // On the longest path through the graph
    pub incoming_stages: Vec<usize>,
    pub outgoing_stages: Vec<usize>,
}

impl ExecutionStage {
    pub fn new(stage_number: usize, nodes: Vec<String>) -> Self {
        ExecutionStage {
            stage_number,
            nodes,
            is_critical: false,
            incoming_stages: Vec::new(),
            outgoing_stages: Vec::new(),
        }
    }
}

/// Compilation statistics and metadata
#[derive(Debug, Clone)]
pub struct CompilationStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub max_depth: usize,
    pub stages: usize,
    pub critical_path_length: usize,
    pub parallelization_factor: f64, // avg nodes per stage
    pub has_cycles: bool,
    pub validation_warnings: Vec<String>,
}

/// The output of graph compilation
///
/// Contains everything needed to execute the graph efficiently:
/// - Execution stages (what can run in parallel)
/// - Topological order (valid sequential execution)
/// - Validation results
/// - Optimization metadata
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    pub execution_stages: Vec<ExecutionStage>,
    pub topological_order: Vec<String>,
    pub node_depths: HashMap<String, usize>,
    pub critical_path: Vec<String>,
    pub parallelizable_pairs: Vec<(String, String)>,
    pub stats: CompilationStats,
    /// Direct adjacency map (node → outgoing neighbors) for correct graph traversal.
    /// Private to avoid confusion with the public `predecessors`/`successors` methods.
    adjacency: HashMap<String, Vec<String>>,
}

impl CompiledGraph {
    /// Get the maximum number of nodes that could execute in parallel
    pub fn max_parallelism(&self) -> usize {
        self.execution_stages
            .iter()
            .map(|stage| stage.nodes.len())
            .max()
            .unwrap_or(0)
    }

    /// Get estimated speedup from parallelization
    pub fn parallelization_speedup(&self) -> f64 {
        if self.stats.total_nodes == 0 {
            return 1.0;
        }
        self.stats.total_nodes as f64 / self.stats.stages.max(1) as f64
    }

    /// Get nodes that have a direct edge **into** the given node (direct predecessors only).
    pub fn predecessors(&self, node: &str) -> Vec<String> {
        self.adjacency
            .iter()
            .filter(|(_, neighbors)| neighbors.iter().any(|n| n == node))
            .map(|(from, _)| from.clone())
            .collect()
    }

    /// Get nodes that the given node has a direct edge **to** (direct successors only).
    pub fn successors(&self, node: &str) -> Vec<String> {
        self.adjacency.get(node).cloned().unwrap_or_default()
    }

    /// Check if two nodes can execute in parallel
    pub fn can_run_parallel(&self, node1: &str, node2: &str) -> bool {
        self.parallelizable_pairs
            .contains(&(node1.to_string(), node2.to_string()))
            || self
                .parallelizable_pairs
                .contains(&(node2.to_string(), node1.to_string()))
    }

    /// Visualize the graph structure
    pub fn visualize(&self) -> String {
        let mut output = String::new();
        output.push_str("=== EXECUTION STAGES ===\n");

        for stage in &self.execution_stages {
            let critical_marker = if stage.is_critical { " [CRITICAL]" } else { "" };
            output.push_str(&format!(
                "Stage {}{}\n",
                stage.stage_number, critical_marker
            ));
            for node in &stage.nodes {
                let depth = self
                    .node_depths
                    .get(node)
                    .map(|d| d.to_string())
                    .unwrap_or_default();
                output.push_str(&format!("  ├─ {} (depth: {})\n", node, depth));
            }
        }

        output.push_str("\n=== CRITICAL PATH ===\n");
        for (idx, node) in self.critical_path.iter().enumerate() {
            output.push_str(&format!("  {}. {}\n", idx + 1, node));
        }

        output.push_str("\n=== STATISTICS ===\n");
        output.push_str(&format!("Total Nodes: {}\n", self.stats.total_nodes));
        output.push_str(&format!("Total Edges: {}\n", self.stats.total_edges));
        output.push_str(&format!("Execution Stages: {}\n", self.stats.stages));
        output.push_str(&format!(
            "Max Parallelism: {} nodes\n",
            self.max_parallelism()
        ));
        output.push_str(&format!(
            "Speedup Factor: {:.2}x\n",
            self.parallelization_speedup()
        ));
        output.push_str(&format!(
            "Parallelizable Pairs: {}\n",
            self.parallelizable_pairs.len()
        ));

        output
    }
}

/// Compiler configuration options
#[derive(Debug, Clone)]
pub struct CompilerOptions {
    pub detect_cycles: bool,
    pub optimize_parallel: bool,
    pub validate_all_edges: bool,
    pub compute_critical_path: bool,
    pub max_nodes: Option<usize>,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        CompilerOptions {
            detect_cycles: true,
            optimize_parallel: true,
            validate_all_edges: true,
            compute_critical_path: true,
            max_nodes: None,
        }
    }
}

/// Compilation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilationError {
    InfiniteLoop(Vec<String>),           // Cycle detected: path
    MissingNode(String),                 // Edge references non-existent node
    NodeNotReachable(String),            // Node not reachable from START
    StartNotFound,                       // No START node
    EndNotFound,                         // No END node
    OrphanedNode(String),                // Node not connected to workflow
    DuplicateNode(String),               // Duplicate node definition
    AmbiguousEdges(String, Vec<String>), // Node with multiple edges, no conditions
    GraphTooLarge,                       // Exceeds max_nodes limit
    EmptyGraph,                          // No nodes in graph
}

impl fmt::Display for CompilationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilationError::InfiniteLoop(path) => {
                write!(f, "Cycle detected: {}", path.join(" -> "))
            }
            CompilationError::MissingNode(name) => {
                write!(f, "Edge references non-existent node: {}", name)
            }
            CompilationError::NodeNotReachable(name) => {
                write!(f, "Node not reachable from START: {}", name)
            }
            CompilationError::StartNotFound => {
                write!(f, "Graph must have a START node")
            }
            CompilationError::EndNotFound => {
                write!(f, "Graph must have an END node")
            }
            CompilationError::OrphanedNode(name) => {
                write!(f, "Orphaned node (no incoming or outgoing edges): {}", name)
            }
            CompilationError::DuplicateNode(name) => {
                write!(f, "Duplicate node definition: {}", name)
            }
            CompilationError::AmbiguousEdges(from, tos) => {
                write!(
                    f,
                    "Node '{}' has multiple edges to {} with no conditions",
                    from,
                    tos.join(", ")
                )
            }
            CompilationError::GraphTooLarge => {
                write!(f, "Graph exceeds maximum node limit")
            }
            CompilationError::EmptyGraph => {
                write!(f, "Cannot compile empty graph")
            }
        }
    }
}

/// The main graph compiler
///
/// Transforms declarative graph config into optimized execution plans.
pub struct GraphCompiler {
    nodes: HashSet<String>,
    adjacency: HashMap<String, Vec<String>>, // node -> outgoing edges
    reverse_adjacency: HashMap<String, Vec<String>>, // node -> incoming edges
    options: CompilerOptions,
}

impl GraphCompiler {
    /// Create a new compiler from node and edge lists
    pub fn new(nodes: Vec<String>, edges: Vec<(String, String)>, options: CompilerOptions) -> Self {
        let mut nodes_set = HashSet::new();
        nodes_set.insert("START".to_string());
        nodes_set.insert("END".to_string());

        for node in nodes {
            nodes_set.insert(node);
        }

        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        let mut reverse_adjacency: HashMap<String, Vec<String>> = HashMap::new();

        // Initialize all nodes
        for node in &nodes_set {
            adjacency.entry(node.clone()).or_default();
            reverse_adjacency.entry(node.clone()).or_default();
        }

        // Add edges
        for (from, to) in edges {
            adjacency.entry(from.clone()).or_default().push(to.clone());
            reverse_adjacency.entry(to).or_default().push(from);
        }

        GraphCompiler {
            nodes: nodes_set,
            adjacency,
            reverse_adjacency,
            options,
        }
    }

    /// Compile the graph into an optimized execution plan
    pub fn compile(&self) -> Result<CompiledGraph, CompilationError> {
        // Validation phase
        self.validate()?;

        // Cycle detection first — fail fast before expensive BFS/topo computations
        if self.options.detect_cycles {
            self.detect_cycles()?;
        }

        // Compute depths (distance from START) — safe to run after cycle check
        let node_depths = self.compute_depths()?;

        // Compute topological order
        let topological_order = self.topological_sort()?;

        // Build execution stages
        let mut execution_stages = self.build_execution_stages(&node_depths)?;

        // Find critical path
        let critical_path = self.find_critical_path(&topological_order, &node_depths);

        // Find parallelizable pairs
        let parallelizable_pairs = if self.options.optimize_parallel {
            self.find_parallelizable_pairs(&node_depths)
        } else {
            Vec::new()
        };

        // Mark critical stages
        for node in &critical_path {
            for stage in &mut execution_stages {
                if stage.nodes.contains(node) {
                    stage.is_critical = true;
                    break;
                }
            }
        }

        // Compute stage dependencies
        self.compute_stage_dependencies(&mut execution_stages, &topological_order)?;

        let stats = CompilationStats {
            total_nodes: self.nodes.len().saturating_sub(2), // Exclude START, END
            total_edges: self.adjacency.values().map(|v| v.len()).sum(),
            max_depth: node_depths.values().max().copied().unwrap_or(0),
            stages: execution_stages.len(),
            critical_path_length: critical_path.len(),
            parallelization_factor: if !execution_stages.is_empty() {
                (self.nodes.len() as f64) / execution_stages.len() as f64
            } else {
                1.0
            },
            has_cycles: false, // Set to true if cycles detected and allowed
            validation_warnings: Vec::new(),
        };

        Ok(CompiledGraph {
            execution_stages,
            topological_order,
            node_depths,
            critical_path,
            parallelizable_pairs,
            stats,
            adjacency: self.adjacency.clone(),
        })
    }

    // =========================================================================
    // Validation
    // =========================================================================

    fn validate(&self) -> Result<(), CompilationError> {
        // Check START and END exist
        if !self.nodes.contains("START") {
            return Err(CompilationError::StartNotFound);
        }
        if !self.nodes.contains("END") {
            return Err(CompilationError::EndNotFound);
        }

        // Check graph is not empty
        if self.nodes.len() <= 2 {
            return Err(CompilationError::EmptyGraph);
        }

        // Check all edges reference existing nodes
        if self.options.validate_all_edges {
            for (from, neighbors) in &self.adjacency {
                if !self.nodes.contains(from) {
                    return Err(CompilationError::MissingNode(from.clone()));
                }
                for to in neighbors {
                    if !self.nodes.contains(to) {
                        return Err(CompilationError::MissingNode(to.clone()));
                    }
                }
            }
        }

        // Check for orphaned nodes
        for node in &self.nodes {
            if node == "START" || node == "END" {
                continue;
            }

            let has_incoming = self
                .reverse_adjacency
                .get(node)
                .map(|v| !v.is_empty())
                .unwrap_or(false);

            let has_outgoing = self
                .adjacency
                .get(node)
                .map(|v| !v.is_empty())
                .unwrap_or(false);

            if !has_incoming && !has_outgoing {
                return Err(CompilationError::OrphanedNode(node.clone()));
            }
        }

        Ok(())
    }

    // =========================================================================
    // Depth Computation
    // =========================================================================

    fn compute_depths(&self) -> Result<HashMap<String, usize>, CompilationError> {
        let mut depths: HashMap<String, usize> = HashMap::new();
        depths.insert("START".to_string(), 0);

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back("START".to_string());

        while let Some(node) = queue.pop_front() {
            if visited.contains(&node) {
                continue;
            }
            visited.insert(node.clone());

            let current_depth = *depths.get(&node).unwrap_or(&0);

            if let Some(neighbors) = self.adjacency.get(&node) {
                for neighbor in neighbors {
                    let new_depth = current_depth + 1;
                    let existing = depths.get(neighbor).copied().unwrap_or(0);

                    if new_depth > existing {
                        depths.insert(neighbor.clone(), new_depth);
                    }

                    if !visited.contains(neighbor) {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        Ok(depths)
    }

    // =========================================================================
    // Cycle Detection
    // =========================================================================

    fn detect_cycles(&self) -> Result<(), CompilationError> {
        let mut visited = HashSet::new();
        let mut rec_stack: Vec<String> = Vec::new();
        let mut rec_set: HashSet<String> = HashSet::new();

        for node in &self.nodes {
            if !visited.contains(node) {
                if let Some(cycle) =
                    self.dfs_find_cycle(node, &mut visited, &mut rec_stack, &mut rec_set)
                {
                    return Err(CompilationError::InfiniteLoop(cycle));
                }
            }
        }

        Ok(())
    }

    /// DFS that returns the actual cycle path when one is found.
    ///
    /// Uses both a `Vec` (for ordered reconstruction) and a `HashSet` (for O(1) lookup)
    /// to track the current recursion stack.
    fn dfs_find_cycle(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut Vec<String>,
        rec_set: &mut HashSet<String>,
    ) -> Option<Vec<String>> {
        visited.insert(node.to_string());
        rec_stack.push(node.to_string());
        rec_set.insert(node.to_string());

        if let Some(neighbors) = self.adjacency.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    if let Some(cycle) = self.dfs_find_cycle(neighbor, visited, rec_stack, rec_set)
                    {
                        return Some(cycle);
                    }
                } else if rec_set.contains(neighbor) {
                    // Back-edge found — reconstruct the cycle from the stack
                    let mut cycle: Vec<String> = rec_stack
                        .iter()
                        .skip_while(|n| n.as_str() != neighbor)
                        .cloned()
                        .collect();
                    cycle.push(neighbor.to_string()); // close the loop
                    return Some(cycle);
                }
            }
        }

        rec_stack.pop();
        rec_set.remove(node);
        None
    }

    // =========================================================================
    // Topological Sort
    // =========================================================================

    fn topological_sort(&self) -> Result<Vec<String>, CompilationError> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();

        for node in &self.nodes {
            in_degree.insert(node.clone(), 0);
        }

        for neighbors in self.adjacency.values() {
            for neighbor in neighbors {
                *in_degree.entry(neighbor.clone()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(node, _)| node.clone())
            .collect();

        let mut result = Vec::new();

        while let Some(node) = queue.pop_front() {
            result.push(node.clone());

            if let Some(neighbors) = self.adjacency.get(&node) {
                for neighbor in neighbors {
                    let degree = in_degree
                        .get_mut(neighbor)
                        .expect("Neighbor must exist in in_degree map (invalid graph structure)");
                    *degree -= 1;

                    if *degree == 0 {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            return Err(CompilationError::InfiniteLoop(vec![
                "Cycle detected in graph".to_string(),
            ]));
        }

        Ok(result)
    }

    // =========================================================================
    // Execution Stage Building
    // =========================================================================

    fn build_execution_stages(
        &self,
        node_depths: &HashMap<String, usize>,
    ) -> Result<Vec<ExecutionStage>, CompilationError> {
        let mut stages: HashMap<usize, Vec<String>> = HashMap::new();

        for (node, depth) in node_depths {
            stages.entry(*depth).or_default().push(node.clone());
        }

        let mut result = Vec::new();
        for stage_num in 0..=*node_depths.values().max().unwrap_or(&0) {
            if let Some(nodes) = stages.remove(&stage_num) {
                result.push(ExecutionStage::new(stage_num, nodes));
            }
        }

        Ok(result)
    }

    // =========================================================================
    // Critical Path Analysis
    // =========================================================================

    fn find_critical_path(
        &self,
        topological_order: &[String],
        node_depths: &HashMap<String, usize>,
    ) -> Vec<String> {
        if !self.options.compute_critical_path {
            return Vec::new();
        }

        // Critical path: longest path from START to END
        let mut path = Vec::new();
        let mut current_depth = 0;

        for node in topological_order {
            if let Some(depth) = node_depths.get(node) {
                if *depth == current_depth {
                    path.push(node.clone());
                    current_depth += 1;
                }
            }
        }

        path
    }

    // =========================================================================
    // Parallelization Analysis
    // =========================================================================

    fn find_parallelizable_pairs(
        &self,
        node_depths: &HashMap<String, usize>,
    ) -> Vec<(String, String)> {
        let mut pairs = Vec::new();

        let nodes: Vec<_> = self
            .nodes
            .iter()
            .filter(|n| *n != "START" && *n != "END")
            .collect();

        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                if self.can_run_parallel(nodes[i], nodes[j], node_depths) {
                    pairs.push((nodes[i].to_string(), nodes[j].to_string()));
                }
            }
        }

        pairs
    }

    fn can_run_parallel(
        &self,
        node1: &str,
        node2: &str,
        node_depths: &HashMap<String, usize>,
    ) -> bool {
        // Two nodes can run in parallel if:
        // 1. They have the same depth (no dependencies between them based on depth)
        // 2. Neither depends on the other directly

        let d1 = node_depths.get(node1).copied().unwrap_or(0);
        let d2 = node_depths.get(node2).copied().unwrap_or(0);

        if d1 != d2 {
            return false;
        }

        // Check if there's a direct edge between them
        if let Some(neighbors) = self.adjacency.get(node1) {
            if neighbors.contains(&node2.to_string()) {
                return false;
            }
        }

        if let Some(neighbors) = self.adjacency.get(node2) {
            if neighbors.contains(&node1.to_string()) {
                return false;
            }
        }

        true
    }

    // =========================================================================
    // Stage Dependency Computation
    // =========================================================================

    fn compute_stage_dependencies(
        &self,
        stages: &mut [ExecutionStage],
        _topological_order: &[String],
    ) -> Result<(), CompilationError> {
        // Build a map of node -> stage number
        let mut node_to_stage: HashMap<String, usize> = HashMap::new();
        for stage in stages.iter() {
            for node in &stage.nodes {
                node_to_stage.insert(node.clone(), stage.stage_number);
            }
        }

        // For each stage, find incoming and outgoing stages
        for stage in stages.iter_mut() {
            let mut incoming = HashSet::new();
            let mut outgoing = HashSet::new();

            for node in &stage.nodes {
                // Find predecessors in reverse adjacency
                if let Some(predecessors) = self.reverse_adjacency.get(node) {
                    for pred in predecessors {
                        if let Some(pred_stage) = node_to_stage.get(pred) {
                            if *pred_stage != stage.stage_number {
                                incoming.insert(*pred_stage);
                            }
                        }
                    }
                }

                // Find successors in adjacency
                if let Some(successors) = self.adjacency.get(node) {
                    for succ in successors {
                        if let Some(succ_stage) = node_to_stage.get(succ) {
                            if *succ_stage != stage.stage_number {
                                outgoing.insert(*succ_stage);
                            }
                        }
                    }
                }
            }

            stage.incoming_stages = incoming.into_iter().collect();
            stage.outgoing_stages = outgoing.into_iter().collect();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_linear_graph() {
        let nodes = vec![
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ];
        let edges = vec![
            ("START".to_string(), "node1".to_string()),
            ("node1".to_string(), "node2".to_string()),
            ("node2".to_string(), "node3".to_string()),
            ("node3".to_string(), "END".to_string()),
        ];

        let compiler = GraphCompiler::new(nodes, edges, CompilerOptions::default());
        let compiled = compiler.compile().unwrap();

        assert_eq!(compiled.stats.total_nodes, 3);
        assert_eq!(compiled.stats.stages, 5);
        assert_eq!(compiled.topological_order.len(), 5); // START + 3 nodes + END
    }

    #[test]
    fn test_parallel_paths() {
        let nodes = vec![
            "task_a".to_string(),
            "task_b".to_string(),
            "join".to_string(),
        ];
        let edges = vec![
            ("START".to_string(), "task_a".to_string()),
            ("START".to_string(), "task_b".to_string()),
            ("task_a".to_string(), "join".to_string()),
            ("task_b".to_string(), "join".to_string()),
            ("join".to_string(), "END".to_string()),
        ];

        let compiler = GraphCompiler::new(nodes, edges, CompilerOptions::default());
        let compiled = compiler.compile().unwrap();

        // task_a and task_b should be parallelizable
        assert!(compiled.can_run_parallel("task_a", "task_b"));
        assert_eq!(compiled.max_parallelism(), 2);
    }

    #[test]
    fn test_predecessors_and_successors() {
        let nodes = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let edges = vec![
            ("START".to_string(), "a".to_string()),
            ("a".to_string(), "b".to_string()),
            ("a".to_string(), "c".to_string()),
            ("b".to_string(), "END".to_string()),
            ("c".to_string(), "END".to_string()),
        ];
        let compiler = GraphCompiler::new(nodes, edges, CompilerOptions::default());
        let compiled = compiler.compile().unwrap();

        // "a" has only START as predecessor
        assert_eq!(compiled.predecessors("a"), vec!["START".to_string()]);

        // "b" and "c" have "a" as predecessor
        let mut pred_b = compiled.predecessors("b");
        pred_b.sort();
        assert_eq!(pred_b, vec!["a".to_string()]);

        // "a" reaches "b" and "c"
        let mut succ_a = compiled.successors("a");
        succ_a.sort();
        assert_eq!(succ_a, vec!["b".to_string(), "c".to_string()]);

        // "START" is not a successor of any node
        assert!(!compiled.successors("END").iter().any(|n| n == "START"));
    }

    #[test]
    fn test_cycle_error_includes_nodes() {
        let nodes = vec!["node1".to_string(), "node2".to_string()];
        let edges = vec![
            ("START".to_string(), "node1".to_string()),
            ("node1".to_string(), "node2".to_string()),
            ("node2".to_string(), "node1".to_string()), // cycle
            ("node2".to_string(), "END".to_string()),
        ];
        let compiler = GraphCompiler::new(nodes, edges, CompilerOptions::default());
        match compiler.compile() {
            Err(CompilationError::InfiniteLoop(path)) => {
                assert!(!path.is_empty(), "Cycle path must not be empty");
                assert_ne!(path[0], "<cycle>", "Must report actual node names");
                assert!(
                    path.iter().any(|n| n == "node1" || n == "node2"),
                    "Cycle path should contain the cycling nodes"
                );
            }
            other => panic!("Expected InfiniteLoop, got {:?}", other),
        }
    }

    #[test]
    fn test_cycle_detection() {
        let nodes = vec!["node1".to_string(), "node2".to_string()];
        let edges = vec![
            ("START".to_string(), "node1".to_string()),
            ("node1".to_string(), "node2".to_string()),
            ("node2".to_string(), "node1".to_string()), // cycle
            ("node2".to_string(), "END".to_string()),
        ];

        let compiler = GraphCompiler::new(nodes, edges, CompilerOptions::default());
        let result = compiler.compile();

        assert!(result.is_err());
        match result {
            Err(CompilationError::InfiniteLoop(_)) => (),
            _ => panic!("Expected InfiniteLoop error"),
        }
    }

    #[test]
    fn test_missing_node() {
        let nodes = vec!["node1".to_string()];
        let edges = vec![
            ("START".to_string(), "node1".to_string()),
            ("node1".to_string(), "missing".to_string()), // references non-existent node
            ("missing".to_string(), "END".to_string()),
        ];

        let compiler = GraphCompiler::new(nodes, edges, CompilerOptions::default());
        let result = compiler.compile();

        assert!(result.is_err());
    }

    #[test]
    fn test_visualization() {
        let nodes = vec!["step1".to_string(), "step2".to_string()];
        let edges = vec![
            ("START".to_string(), "step1".to_string()),
            ("step1".to_string(), "step2".to_string()),
            ("step2".to_string(), "END".to_string()),
        ];

        let compiler = GraphCompiler::new(nodes, edges, CompilerOptions::default());
        let compiled = compiler.compile().unwrap();

        let viz = compiled.visualize();
        assert!(viz.contains("EXECUTION STAGES"));
        assert!(viz.contains("CRITICAL PATH"));
        assert!(viz.contains("STATISTICS"));
    }
}
