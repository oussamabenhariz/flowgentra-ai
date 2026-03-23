//! State graph builder and execution engine

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

use crate::core::state::State;
use crate::core::middleware::{ExecutionContext, Middleware, MiddlewarePipeline};
use crate::core::plugins::PluginRegistry;
use super::error::{Result, StateGraphError};
use super::node::{Node, RouterFn};
use super::edge::{Edge, FixedEdge, START, END};
use super::checkpoint::{Checkpoint, Checkpointer, InMemoryCheckpointer};

/// State graph builder - fluent API for constructing graphs
pub struct StateGraphBuilder<S: State> {
    nodes: HashMap<String, Arc<dyn Node<S>>>,
    edges: Vec<Edge<S>>,
    entry_point: Option<String>,
    checkpointer: Option<Arc<dyn Checkpointer<S>>>,
    max_steps: usize,
    interrupt_before: HashSet<String>,
    interrupt_after: HashSet<String>,
    middleware: MiddlewarePipeline<S>,
    plugins: Arc<PluginRegistry>,
}

impl<S: State + Send + Sync + 'static> StateGraphBuilder<S> {
    /// Create a new graph builder
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_point: None,
            checkpointer: None,
            max_steps: 1000,
            interrupt_before: HashSet::new(),
            interrupt_after: HashSet::new(),
            middleware: MiddlewarePipeline::new(),
            plugins: Arc::new(PluginRegistry::new()),
        }
    }

    /// Add a node to the graph
    pub fn add_node(mut self, name: impl Into<String>, node: Arc<dyn Node<S>>) -> Self {
        self.nodes.insert(name.into(), node);
        self
    }

    /// Add a fixed edge (A → B)
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push(Edge::Fixed(FixedEdge::new(from, to)));
        self
    }

    /// Add a conditional edge (A → router → B or C or ...)
    pub fn add_conditional_edge(
        mut self,
        from: impl Into<String>,
        router: RouterFn<S>,
    ) -> Self {
        self.edges.push(Edge::Conditional {
            from: from.into(),
            router,
        });
        self
    }

    /// Set the entry point (node after START)
    pub fn set_entry_point(mut self, node_name: impl Into<String>) -> Self {
        self.entry_point = Some(node_name.into());
        self
    }

    /// Set a checkpointer for fault tolerance
    pub fn set_checkpointer(mut self, checkpointer: Arc<dyn Checkpointer<S>>) -> Self {
        self.checkpointer = Some(checkpointer);
        self
    }

    /// Set maximum number of steps before timeout
    pub fn set_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Mark a node to interrupt before execution
    pub fn interrupt_before(mut self, node_name: impl Into<String>) -> Self {
        self.interrupt_before.insert(node_name.into());
        self
    }

    /// Mark a node to interrupt after execution
    pub fn interrupt_after(mut self, node_name: impl Into<String>) -> Self {
        self.interrupt_after.insert(node_name.into());
        self
    }

    /// Add middleware to the graph execution pipeline
    pub fn use_middleware(mut self, mw: Arc<dyn Middleware<S>>) -> Self {
        self.middleware.use_middleware(mw);
        self
    }

    /// Set the plugin registry for the graph
    pub fn set_plugins(mut self, plugins: Arc<PluginRegistry>) -> Self {
        self.plugins = plugins;
        self
    }

    /// Build and validate the graph
    pub fn compile(self) -> Result<StateGraph<S>> {
        // Validate that entry point is set
        let entry_point = self
            .entry_point
            .ok_or_else(|| StateGraphError::InvalidGraph("No entry point set".to_string()))?;

        // Validate that entry point exists as a node
        if !self.nodes.contains_key(&entry_point) {
            return Err(StateGraphError::InvalidGraph(format!(
                "Entry point '{}' is not a registered node",
                entry_point
            )));
        }

        // Validate all edges reference valid nodes
        for edge in &self.edges {
            match edge {
                Edge::Fixed(fixed_edge) => {
                    if fixed_edge.from != START && !self.nodes.contains_key(&fixed_edge.from) {
                        return Err(StateGraphError::NodeNotFound(fixed_edge.from.clone()));
                    }
                    if fixed_edge.to != END && !self.nodes.contains_key(&fixed_edge.to) {
                        return Err(StateGraphError::NodeNotFound(fixed_edge.to.clone()));
                    }
                }
                Edge::Conditional { from, .. } => {
                    if from != START && !self.nodes.contains_key(from) {
                        return Err(StateGraphError::NodeNotFound(from.clone()));
                    }
                }
            }
        }

        let checkpointer = self.checkpointer.unwrap_or_else(|| {
            Arc::new(InMemoryCheckpointer::new())
        });

        Ok(StateGraph {
            nodes: self.nodes,
            edges: self.edges,
            entry_point,
            checkpointer,
            max_steps: self.max_steps,
            interrupt_before: self.interrupt_before,
            interrupt_after: self.interrupt_after,
            middleware: self.middleware,
            plugins: self.plugins,
        })
    }
}

impl<S: State> Default for StateGraphBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compiled state graph ready for execution
pub struct StateGraph<S: State> {
    nodes: HashMap<String, Arc<dyn Node<S>>>,
    edges: Vec<Edge<S>>,
    entry_point: String,
    checkpointer: Arc<dyn Checkpointer<S>>,
    max_steps: usize,
    interrupt_before: HashSet<String>,
    interrupt_after: HashSet<String>,
    middleware: MiddlewarePipeline<S>,
    plugins: Arc<PluginRegistry>,
}

impl<S: State + Send + Sync + 'static> StateGraph<S> {
    /// Build a new state graph using the builder pattern
    pub fn builder() -> StateGraphBuilder<S> {
        StateGraphBuilder::new()
    }

    /// Execute the graph synchronously
    pub async fn invoke(&self, initial_state: S) -> Result<S> {
        let thread_id = Uuid::new_v4().to_string();
        self.invoke_with_id(thread_id, initial_state).await
    }

    /// Execute the graph with a specific thread ID (for resuming from checkpoints)
    pub async fn invoke_with_id(&self, thread_id: String, initial_state: S) -> Result<S> {
        // Try to resume from latest checkpoint
        let state = if let Some(checkpoint) = self.checkpointer.load_latest(&thread_id).await? {
            checkpoint.state
        } else {
            initial_state
        };

        let mut current_state = state;
        let mut current_node = self.entry_point.clone();
        let mut step = 0;

        loop {
            // Check for max steps
            if step >= self.max_steps {
                return Err(StateGraphError::Timeout(
                    "Max steps exceeded".to_string(),
                ));
            }

            // Check for interruption before
            if self.interrupt_before.contains(&current_node) {
                return Err(StateGraphError::InterruptedAtBreakpoint {
                    node: current_node,
                });
            }

            // Create execution context for middleware
            let mut middleware_ctx = ExecutionContext::new(&current_node, current_state.clone());

            // Execute before_node hooks
            self.middleware.execute_before_node(&mut middleware_ctx).await
                .map_err(|e| StateGraphError::ExecutionError {
                    node: current_node.clone(),
                    reason: format!("Middleware error: {}", e),
                })?;
            
            // Apply any state modifications from middleware
            current_state = middleware_ctx.state;

            // Create plugin context for this execution
            let plugin_ctx = crate::core::plugins::PluginContext::new();
            
            // Execute plugin hooks for node start
            let _ = self.plugins.on_node_execute(&plugin_ctx, &current_node).await;

            // Execute current node
            tracing::info!(node = %current_node, step = %step, "Executing node");
            
            let node = self.nodes.get(&current_node)
                .ok_or_else(|| StateGraphError::NodeNotFound(current_node.clone()))?;
            
            match node.execute(&current_state).await {
                Ok(new_state) => {
                    current_state = new_state;
                    
                    // Update context for after_node hooks
                    middleware_ctx.state = current_state.clone();
                    
                    // Execute after_node hooks
                    self.middleware.execute_after_node(&mut middleware_ctx).await
                        .map_err(|e| StateGraphError::ExecutionError {
                            node: current_node.clone(),
                            reason: format!("Middleware error: {}", e),
                        })?;
                    
                    // Apply any state modifications from middleware
                    current_state = middleware_ctx.state;
                    
                    // Execute plugin hooks for node completion
                    let _ = self.plugins.on_node_complete(&plugin_ctx, &current_node, true).await;
                    
                    // CRITICAL PERFORMANCE: Checkpoint State Cloning
                    // – Issue: state.clone() called every step (N steps × clone cost)
                    // – Impact: 50-200ms overhead with large state (>500KB)
                    // – Why needed: Checkpoint must own state for persistence
                    // – Fix: Use SharedState (cheap clone) not PlainState (expensive clone)
                    //   SharedState.clone() = ~1μs; PlainState.clone() = 5-20ms
                    let checkpoint = Checkpoint::new(
                        thread_id.clone(),
                        step,
                        current_node.clone(),
                        current_state.clone(),  // Clone only when saving
                    );
                    self.checkpointer.save(&checkpoint).await?;

                    // Check for interruption after
                    if self.interrupt_after.contains(&current_node) {
                        return Err(StateGraphError::InterruptedAtBreakpoint {
                            node: current_node,
                        });
                    }

                    // Determine next node
                    let next_node = self.get_next_node(&current_node, &current_state).await?;
                    
                    if next_node == END {
                        // Execute on_complete hooks
                        self.middleware.execute_on_complete(&current_state).await;
                        return Ok(current_state);
                    }

                    current_node = next_node;
                }
                Err(e) => {
                    tracing::error!(node = %current_node, error = %e, "Node execution failed");
                    
                    // Execute plugin hooks for node failure
                    let _ = self.plugins.on_node_complete(&plugin_ctx, &current_node, false).await;
                    
                    // Execute error hooks with a fresh context
                    let error_ctx = ExecutionContext::new(&current_node, current_state.clone());
                    let err_as_flowgentra = crate::core::error::FlowgentraError::ExecutionError(e.to_string());
                    self.middleware.execute_on_error(&current_node, &err_as_flowgentra, &error_ctx).await;
                    
                    return Err(e);
                }
            }

            step += 1;
        }
    }

    /// Get the next node(s) from the current node
    async fn get_next_node(&self, current_node: &str, state: &S) -> Result<String> {
        for edge in &self.edges {
            if edge.from() == current_node {
                return edge.next_node(state).await;
            }
        }
        
        // Default: if no outgoing edge, go to END
        Ok(END.to_string())
    }

    /// Resume execution from a checkpoint
    pub async fn resume(
        &self,
        thread_id: &str,
    ) -> Result<S> {
        let checkpoint = self.checkpointer
            .load_latest(thread_id)
            .await?
            .ok_or_else(|| StateGraphError::ResumeFailed(
                "No checkpoint found".to_string(),
            ))?;

        self.invoke_with_id(thread_id.to_string(), checkpoint.state).await
    }

    /// Get execution history for a thread
    pub async fn history(&self, thread_id: &str) -> Result<Vec<(usize, String)>> {
        let checkpoints = self.checkpointer.list_checkpoints(thread_id).await?;
        Ok(checkpoints
            .into_iter()
            .map(|(step, _)| (step, "checkpoint".to_string()))
            .collect())
    }

    /// Clear all checkpoints for a thread
    pub async fn clear_history(&self, thread_id: &str) -> Result<()> {
        self.checkpointer.delete_thread(thread_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::PlainState;
    use serde_json::json;

    #[tokio::test]
    async fn test_graph_builder() {
        let node1 = Arc::new(super::super::node::FunctionNode::new("node1", |state: &PlainState| {
            let cloned = state.clone();
            Box::pin(async move {
                let mut new_state = cloned;
                PlainState::set(&mut new_state, "counter", json!(1));
                Ok(new_state)
            })
        }));

        let graph = StateGraph::builder()
            .add_node("node1", node1)
            .set_entry_point("node1")
            .add_edge("node1", END)
            .compile()
            .expect("Failed to compile graph");

        let state = PlainState::new();
        let result = graph.invoke(state).await.expect("Failed to invoke graph");
        assert_eq!(result.get("counter"), Some(&json!(1)));
    }
}
