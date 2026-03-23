//! # Graph Analysis for Automatic Parallelization
//!
//! Analyzes graph structure to automatically detect independent branches that can run in parallel.
//!
//! ## Algorithm
//!
//! 1. **Find Branch Points**: Nodes with multiple outgoing edges
//! 2. **Trace Branch Paths**: Follow each branch to find all nodes it contains
//! 3. **Detect Dependencies**: Check if branches depend on each other
//! 4. **Identify Independence**: Branches with no shared dependencies can run in parallel
//! 5. **Return Parallel Sets**: Groups of branches that can execute concurrently
//!
//! ## Example
//!
//! Graph structure:
//! ```text
//! START → router (branch point)
//!         ├─→ path_a → process_a → merge
//!         └─→ path_b → process_b → merge
//!             └─→ END
//! ```
//!
//! Analysis result: [["path_a", "process_a"], ["path_b", "process_b"]] (2 parallel branches)

use crate::core::error::Result;
use crate::core::graph::NodeId;
use std::collections::{HashSet, VecDeque};

/// Result of analyzing branches in a graph
#[derive(Debug, Clone)]
pub struct BranchAnalysis {
    /// Sets of nodes that can run in parallel
    /// Each inner Vec contains node IDs that form one independent branch
    pub parallel_branches: Vec<Vec<NodeId>>,

    /// Nodes that must run sequentially before any branches
    pub sequential_head: Vec<NodeId>,

    /// Nodes that must run sequentially after all parallel branches
    pub sequential_tail: Vec<NodeId>,

    /// Total number of independent parallel paths discovered
    pub parallelism_level: usize,

    /// Estimated speedup if branches were truly parallelized
    /// (rough estimate based on critical path)
    pub estimated_speedup: f32,
}

/// Analyzes a graph to find parallelizable branches
pub struct GraphAnalyzer;

impl GraphAnalyzer {
    /// Analyze graph for automatic parallelization opportunities
    ///
    /// # Returns
    /// `BranchAnalysis` containing detected parallel branches and sequential regions
    ///
    /// # Example
    /// ```ignore
    /// let analysis = GraphAnalyzer::analyze(&graph)?;
    /// println!("Found {} parallel branches", analysis.parallel_branches.len());
    /// for branch in &analysis.parallel_branches {
    ///     println!("Branch path: {:?}", branch);
    /// }
    /// ```
    pub fn analyze<T: crate::core::state::State>(graph: &crate::core::graph::Graph<T>) -> Result<BranchAnalysis> {
        // Step 1: Find the sequential head (linear path until first branch point)
        let sequential_head = Self::find_sequential_head(graph)?;

        // Step 2: Find branch points (nodes with multiple outgoing edges)
        let branch_points = Self::find_branch_points(graph);

        // Step 3: If no branch points, entire graph is sequential
        if branch_points.is_empty() {
            return Ok(BranchAnalysis {
                parallel_branches: vec![],
                sequential_head: graph.nodes().keys().cloned().collect(),
                sequential_tail: vec![],
                parallelism_level: 1,
                estimated_speedup: 1.0,
            });
        }

        // Step 4: For the first branch point, trace all branches
        let first_branch_point = &branch_points[0];
        let branches = Self::trace_branches(graph, first_branch_point)?;

        // Step 5: Find sequential tail (join point onwards)
        let sequential_tail = Self::find_sequential_tail(graph, &branches)?;

        // Step 6: Check if branches are independent
        let parallelism_level = Self::verify_independence(&branches)?;

        // Step 7: Estimate speedup
        let max_branch_length = branches.iter().map(|b| b.len()).max().unwrap_or(1) as f32;
        let total_length = graph.nodes().len() as f32;
        let estimated_speedup = total_length / (max_branch_length + sequential_head.len() as f32);

        Ok(BranchAnalysis {
            parallel_branches: branches,
            sequential_head,
            sequential_tail,
            parallelism_level,
            estimated_speedup,
        })
    }

    /// Find the linear sequential path at the start of the graph
    fn find_sequential_head<T: crate::core::state::State>(graph: &crate::core::graph::Graph<T>) -> Result<Vec<NodeId>> {
        let mut sequential = vec![];
        let mut current = Some("START".to_string());

        while let Some(node_id) = current {
            // Check outgoing edges from current node
            let outgoing = graph.get_next_nodes(&node_id);

            // If more than one outgoing edge, we've hit a branch point
            if outgoing.len() > 1 {
                break;
            }

            // If exactly one outgoing edge, continue
            if let Some(next_edge) = outgoing.first() {
                if next_edge.to != "END" {
                    sequential.push(next_edge.to.clone());
                    current = Some(next_edge.to.clone());
                } else {
                    break;
                }
            } else {
                // No outgoing edges
                break;
            }
        }

        Ok(sequential)
    }

    /// Find all nodes with multiple outgoing edges (branch points)
    fn find_branch_points<T: crate::core::state::State>(graph: &crate::core::graph::Graph<T>) -> Vec<NodeId> {
        let mut branch_points = vec![];
        for node_id in graph.nodes().keys() {
            let outgoing = graph.get_next_nodes(node_id);
            if outgoing.len() > 1 {
                branch_points.push(node_id.to_string());
            }
        }
        branch_points
    }

    /// Trace all branches emanating from a branch point
    fn trace_branches<T: crate::core::state::State>(graph: &crate::core::graph::Graph<T>, branch_point: &str) -> Result<Vec<Vec<NodeId>>> {
        let outgoing = graph.get_next_nodes(branch_point);
        let mut branches = vec![];

        for edge in outgoing {
            let mut branch = vec![];
            let mut current = Some(edge.to.clone());
            let mut visited = HashSet::new();

            // Trace this branch until we hit a join or END node
            while let Some(node_id) = current {
                if node_id == "END" {
                    break;
                }

                // Prevent infinite loops
                if visited.contains(&node_id) {
                    break;
                }
                visited.insert(node_id.clone());

                branch.push(node_id.clone());

                // Move to next node
                let outgoing_from_node = graph.get_next_nodes(&node_id);

                if outgoing_from_node.len() == 1 {
                    // Single path, continue
                    current = Some(outgoing_from_node[0].to.clone());
                } else if outgoing_from_node.is_empty() {
                    // End of branch
                    break;
                } else {
                    // Another branch point - stop here for now
                    break;
                }
            }

            if !branch.is_empty() {
                branches.push(branch);
            }
        }

        Ok(branches)
    }

    /// Find the sequential tail (nodes after all branches converge)
    fn find_sequential_tail<T: crate::core::state::State>(graph: &crate::core::graph::Graph<T>, branches: &[Vec<NodeId>]) -> Result<Vec<NodeId>> {
        let mut sequential = vec![];

        // Find the common join point (node reachable from all branches)
        let join_point = Self::find_join_point(graph, branches)?;

        if let Some(join) = join_point {
            // Trace sequentially from join point to END
            let mut current = Some(join);

            while let Some(node_id) = current {
                let outgoing = graph.get_next_nodes(&node_id);

                if outgoing.len() == 1 && outgoing[0].to != "END" {
                    sequential.push(outgoing[0].to.clone());
                    current = Some(outgoing[0].to.clone());
                } else {
                    break;
                }
            }
        }

        Ok(sequential)
    }

    /// Find the join point where branches converge
    fn find_join_point<T: crate::core::state::State>(graph: &crate::core::graph::Graph<T>, branches: &[Vec<NodeId>]) -> Result<Option<NodeId>> {
        if branches.is_empty() {
            return Ok(None);
        }

        // For each node in the graph, check if it's reachable from all branches
        for node_id in graph.nodes().keys() {
            if node_id == "START" || node_id == "END" {
                continue;
            }

            let mut reachable_from_all = true;

            for branch in branches {
                if !branch.contains(node_id)
                    && !Self::is_reachable_from(graph, branch.last(), node_id)?
                {
                    reachable_from_all = false;
                    break;
                }
            }

            if reachable_from_all {
                return Ok(Some(node_id.clone()));
            }
        }

        Ok(None)
    }

    /// Check if target is reachable from start node
    fn is_reachable_from<T: crate::core::state::State>(graph: &crate::core::graph::Graph<T>, start: Option<&NodeId>, target: &str) -> Result<bool> {
        let Some(start_id) = start else {
            return Ok(false);
        };

        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();

        queue.push_back(start_id.clone());
        visited.insert(start_id.clone());

        while let Some(current) = queue.pop_front() {
            if current == target {
                return Ok(true);
            }

            let outgoing = graph.get_next_nodes(&current);
            for edge in outgoing {
                if !visited.contains(&edge.to) {
                    visited.insert(edge.to.clone());
                    queue.push_back(edge.to.clone());
                }
            }
        }

        Ok(false)
    }

    /// Verify that branches are independent (no shared dependencies)
    fn verify_independence(branches: &[Vec<NodeId>]) -> Result<usize> {
        if branches.len() <= 1 {
            return Ok(1);
        }

        // Create sets for each branch
        let branch_sets: Vec<HashSet<_>> = branches
            .iter()
            .map(|b| b.iter().cloned().collect::<HashSet<_>>())
            .collect();

        // Check for intersections (shared nodes)
        for i in 0..branch_sets.len() {
            for j in (i + 1)..branch_sets.len() {
                let intersection: HashSet<_> = branch_sets[i]
                    .intersection(&branch_sets[j])
                    .cloned()
                    .collect();

                if !intersection.is_empty() {
                    // Branches share nodes - they're not fully independent
                    // But they can still run in parallel with synchronization
                    return Ok(1); // Conservative estimate
                }
            }
        }

        // All branches are independent
        Ok(branches.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::Graph;

    #[test]
    fn test_empty_graph() -> Result<()> {
        let graph: Graph<crate::core::state::SharedState> = Graph::new();
        let analysis = GraphAnalyzer::analyze(&graph)?;
        assert_eq!(analysis.parallelism_level, 1);
        Ok(())
    }
}
