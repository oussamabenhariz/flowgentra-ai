//! ## Advanced Patterns
//!
//! - Support custom middleware for logging, metrics, etc.
//! - Allow users to hook into state transitions, node execution, and errors.
//! ## Observability & Debugging
//!
//! - Use structured logging (e.g., `tracing`) throughout runtime and handlers.
//! - Integrate distributed tracing for workflow execution.
//! - Provide health check endpoints for production deployments.

/// Example: health check stub
///
/// ```ignore
/// // Example: log a state transition
/// // tracing::info!("Transitioned from {:?} to {:?}", old_state, new_state);
/// ```
pub fn health_check() -> bool {
    // Implement real health logic (e.g., DB, LLM, etc.)
    true
}
// # Runtime for typed state engine
//
// ## Node & Handler Patterns
//
// - Register all handlers with macros for auto-discovery.
// - Each handler should only mutate the state fields it owns.
// - Document handler contracts (fields read/written).
// - Use clear error types and never panic in handlers or runtime.
// - Compose handlers for complex logic, don't overload one function.
/// Runtime for typed state engine
use crate::core::state::DynState;

use crate::core::config::AgentConfig;
use crate::core::error::{FlowgentraError, Result};
use crate::core::graph::Graph;
use crate::core::llm::{create_llm_client, LLMClient};
use crate::core::mcp::{DefaultMCPClient, MCPClient};
use crate::core::memory::{
    CheckpointMetadata, Checkpointer, GenericCheckpointer, MemoryCheckpointer,
};
use crate::core::middleware::ExecutionContext as MiddlewareContext;
use crate::core::middleware::MiddlewarePipeline;
use crate::core::node::{Edge, EdgeCondition, Node, NodeFunction};
use crate::core::observability::ObservabilityMiddleware;
use crate::core::tracing::TimerGuard;
use futures::future::try_join_all;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

// =============================================================================
// Execution Context
// =============================================================================

pub struct ExecutionContext {
    /// Name of the node being executed
    pub node_name: String,

    /// Current state being processed
    pub state: DynState,

    /// LLM client for AI operations
    pub llm_client: Arc<dyn LLMClient>,

    /// Available MCP clients for tool access
    pub mcp_clients: HashMap<String, Arc<dyn MCPClient>>,
}

// =============================================================================
// Runtime
// =============================================================================

/// The core execution engine for agent graphs
///
/// Responsible for:
/// - Managing the graph structure
/// - Executing nodes in order
/// - Applying edge conditions
/// - Maintaining service clients
/// - Executing middleware hooks
pub struct AgentRuntime {
    /// Agent configuration
    config: AgentConfig,

    /// The execution graph
    graph: Graph<DynState>,

    /// LLM client shared across all nodes
    llm_client: Arc<dyn LLMClient>,

    /// MCP clients for tool access
    mcp_clients: HashMap<String, Arc<dyn MCPClient>>,

    /// Middleware pipeline for cross-cutting concerns
    middleware_pipeline: MiddlewarePipeline<DynState>,

    /// Optional checkpointer for thread-scoped state persistence (resume, multi-turn).
    checkpointer: Option<Arc<MemoryCheckpointer>>,
}

impl AgentRuntime {
    // =========================================================================
    // Initialization
    // =========================================================================

    /// Create a new runtime from a configuration
    ///
    /// This performs:
    /// - Configuration validation
    /// - LLM client initialization
    /// - MCP client setup
    /// - Graph construction
    pub fn from_config(config: AgentConfig) -> Result<Self> {
        // Validate config
        config.validate()?;

        // Create LLM client
        let llm_client = create_llm_client(&config.llm)?;

        // Create MCP clients
        let mcp_clients: HashMap<String, Arc<dyn MCPClient>> = config
            .graph
            .mcps
            .iter()
            .map(|(name, mcp_config)| {
                (
                    name.clone(),
                    Arc::new(DefaultMCPClient::new(mcp_config.clone())) as Arc<dyn MCPClient>,
                )
            })
            .collect();

        // Create graph
        let mut graph = Graph::new();

        // Collect names of nodes that are managed by a supervisor.
        // These should NOT be added to the graph — the supervisor calls them internally,
        // so they don't participate in the graph's own edge routing or termination checks.
        let supervisor_children: std::collections::HashSet<String> = config
            .graph
            .nodes
            .iter()
            .filter(|n| {
                matches!(
                    n.node_type.as_deref(),
                    Some("supervisor") | Some("orchestrator")
                )
            })
            .flat_map(|n| {
                n.config
                    .get("children")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
            })
            .collect();

        // Add nodes (these are placeholders; actual nodes are added via register_node)
        // Skip supervisor-managed children — they are internal to the supervisor, not graph routing nodes.
        for node_config in &config.graph.nodes {
            if supervisor_children.contains(&node_config.name) {
                continue;
            }
            let placeholder_fn: NodeFunction<DynState> =
                Box::new(|state| Box::pin(async move { Ok(state) }));
            let is_planner = node_config.handler == "builtin::planner"
                || node_config.node_type.as_deref() == Some("planner");
            let node = if is_planner {
                Node::new_planner(
                    node_config.name.clone(),
                    placeholder_fn,
                    node_config.mcps.clone(),
                    node_config.config.clone(),
                )
            } else {
                Node::new_placeholder(
                    node_config.name.clone(),
                    placeholder_fn,
                    node_config.mcps.clone(),
                    node_config.config.clone(),
                )
            };
            graph.add_node(node);
        }

        // Add edges
        let mut node_names: std::collections::HashSet<_> =
            config.graph.nodes.iter().map(|n| n.name.clone()).collect();
        node_names.insert("START".to_string());
        node_names.insert("END".to_string());

        let mut start_nodes = Vec::new();
        let mut end_nodes = Vec::new();

        for edge_config in &config.graph.edges {
            for to in &edge_config.to {
                let mut edge = Edge::new(
                    edge_config.from.clone(),
                    to.clone(),
                    None, // Condition is set during registration
                );

                if let Some(cond_name) = &edge_config.condition {
                    edge = edge.with_condition_name(cond_name.clone());
                }

                graph.add_edge(edge);

                if edge_config.from == "START" {
                    start_nodes.push(to.clone());
                }
                if *to == "END" {
                    end_nodes.push(edge_config.from.clone());
                }
            }
        }

        graph.set_start_nodes(start_nodes);
        graph.set_end_nodes(end_nodes);
        graph.set_allow_cycles(config.graph.allow_cycles);
        graph.set_max_loop_iterations(config.graph.recursion_limit);

        // Validate graph
        graph.validate()?;

        Ok(AgentRuntime {
            config,
            graph,
            llm_client,
            mcp_clients,
            middleware_pipeline: MiddlewarePipeline::new(),
            checkpointer: None,
        })
    }

    // =========================================================================
    // Handler Registration
    // =========================================================================

    /// Register a handler function for a node
    ///
    /// Replaces the placeholder function with the actual handler.
    /// Called during agent initialization.
    pub fn register_node(&mut self, name: &str, function: NodeFunction<DynState>) -> Result<()> {
        if let Some(node) = self.graph.nodes.get_mut(name) {
            // Replace the placeholder function
            let mcps = node.mcps.clone();
            let config = node.config.clone();
            let is_planner = node.is_planner;
            let new_node = if is_planner {
                Node::new_planner(name.to_string(), function, mcps, config)
            } else {
                Node::new(name.to_string(), function, mcps, config)
            };
            self.graph.nodes.insert(name.to_string(), new_node);
            Ok(())
        } else {
            Err(FlowgentraError::NodeNotFound(name.to_string()))
        }
    }

    /// Register a condition function for an edge
    ///
    /// Associates a condition function with conditional edges.
    pub fn register_edge_condition(
        &mut self,
        from: &str,
        condition_name: &str,
        condition_fn: EdgeCondition<DynState>,
    ) -> Result<()> {
        // Find and update the edge(s)
        for edge in self.graph.edges.iter_mut() {
            if edge.from == from && edge.condition_name.as_deref() == Some(condition_name) {
                edge.condition = Some(Arc::clone(&condition_fn));
            }
        }
        Ok(())
    }

    /// Set the checkpointer for thread-scoped state persistence.
    ///
    /// When set, use `execute_with_thread(thread_id, state)` to load/save state per thread.
    pub fn set_checkpointer(&mut self, checkpointer: Arc<MemoryCheckpointer>) -> &mut Self {
        self.checkpointer = Some(checkpointer);
        self
    }

    /// Builder-style: set the checkpointer.
    pub fn with_checkpointer(mut self, checkpointer: Arc<MemoryCheckpointer>) -> Self {
        self.checkpointer = Some(checkpointer);
        self
    }

    /// Get the checkpointer if set.
    pub fn checkpointer(&self) -> Option<Arc<MemoryCheckpointer>> {
        self.checkpointer.clone()
    }

    /// Add middleware to the middleware pipeline
    ///
    /// Middleware will be executed in the order they are added during node execution.
    /// Each middleware receives an ExecutionContext and can:
    /// - Log/monitor execution
    /// - Validate state
    /// - Modify state before/after execution
    /// - Skip nodes or abort execution
    ///
    /// # Arguments
    /// - `middleware` - The middleware to add (must implement Middleware trait)
    ///
    /// # Example
    /// ```ignore
    /// # use flowgentra_ai::core::runtime::AgentRuntime;
    /// # use flowgentra_ai::core::config::AgentConfig;
    /// # use flowgentra_ai::core::middleware::LoggingMiddleware;
    /// # use std::sync::Arc;
    /// # let config = AgentConfig::from_file("config.yaml")?;
    /// # let mut runtime = AgentRuntime::from_config(config)?;
    /// runtime.add_middleware(Arc::new(LoggingMiddleware::new()));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add_middleware(
        &mut self,
        middleware: Arc<dyn crate::core::middleware::Middleware<DynState>>,
    ) -> &mut Self {
        self.middleware_pipeline.use_middleware(middleware);
        self
    }

    /// Get a reference to the middleware pipeline
    pub fn middleware_pipeline(&self) -> &MiddlewarePipeline<DynState> {
        &self.middleware_pipeline
    }

    /// Get mutable reference to the middleware pipeline
    pub fn middleware_pipeline_mut(&mut self) -> &mut MiddlewarePipeline<DynState> {
        &mut self.middleware_pipeline
    }

    /// Add observability middleware (execution tracing, node timing, token usage, failure snapshots).
    pub fn with_observability(&mut self) -> &mut Self {
        let mw = ObservabilityMiddleware::new().with_agent_name(self.config.name.clone());
        self.add_middleware(Arc::new(mw));
        self
    }

    // =========================================================================
    // Execution
    // =========================================================================

    /// Execute the agent graph from start to end.
    ///
    /// For thread-scoped persistence (resume, multi-turn), use `execute_with_thread` and
    /// set a checkpointer via `set_checkpointer` or config.
    pub async fn execute(&self, initial_state: DynState) -> Result<DynState> {
        self.execute_impl(initial_state, None).await
    }

    /// Execute with a thread id for checkpointing. When a checkpointer is set, state is
    /// loaded from the last checkpoint for this thread (if any) and saved after each node.
    pub async fn execute_with_thread(&self, thread_id: &str, initial_state: DynState) -> Result<DynState> {
        self.execute_impl(initial_state, Some(thread_id)).await
    }

    async fn execute_impl(&self, initial_state: DynState, thread_id: Option<&str>) -> Result<DynState> {
        // == STATE CLONING NOTES ==
        // This execution loop clones state at several points for correctness:
        // 1. Parallel frontier execution: Arc-wrapped single clone for shared access
        // 2. Middleware context: Clone for before/after hooks (they need mutable reference)
        // 3. Node execution: Clone as handler takes state by value
        //
        // For typical workflows, these clones are acceptable and necessary.
        // For large state objects (>1MB), consider:
        // - DynState: Zero-copy shared state via Arc<Mutex>
        // - OptimizedState: Lazy cloning on mutation only
        // See STATE_OPTIMIZATION_GUIDE.md for detailed strategies

        let span = tracing::info_span!("agent_execution");
        let _guard = span.enter();

        tracing::info!("Starting agent execution");
        let _timer = TimerGuard::start("total_execution");

        // If checkpointer + thread_id: try load checkpoint; otherwise use initial_state
        let mut state = if let (Some(cp), Some(tid)) = (&self.checkpointer, thread_id) {
            match cp.load(tid)? {
                Some(checkpoint) => {
                    let loaded = checkpoint.state()?;
                    tracing::info!(thread_id = %tid, "Resumed from checkpoint");
                    loaded
                }
                None => initial_state,
            }
        } else {
            initial_state
        };

        // Validate initial state if schema is configured
        self.config.validate_state(&state)?;

        // Track execution path for checkpoint metadata
        let mut execution_path: Vec<String> = Vec::new();

        // Planner: max iterations (re-plan after each step)
        let max_plan_steps = self.config.graph.planner.max_plan_steps;
        let mut plan_steps: usize = 0;

        // Recursion / step limit — prevents infinite loops in cyclic graphs.
        let recursion_limit = self.config.graph.recursion_limit;
        let mut total_steps: usize = 0;

        // Start from START node
        let mut current_nodes = self.graph.start_nodes().to_vec();
        let mut iteration = 0;

        while !current_nodes.is_empty() {
            iteration += 1;
            total_steps += current_nodes.len();

            if total_steps > recursion_limit {
                return Err(
                    crate::core::error::FlowgentraError::RecursionLimitExceeded {
                        limit: recursion_limit,
                    },
                );
            }

            let node_count = current_nodes.len();

            tracing::debug!(iteration, node_count, total_steps, nodes = ?current_nodes, "Processing iteration");

            let mut next_nodes = Vec::new();

            // DAG-based parallel: when multiple nodes in frontier (and none are planner), run concurrently with tokio
            let run_parallel = node_count > 1
                && !current_nodes.iter().any(|n| {
                    self.graph
                        .get_node(n)
                        .map(|node| node.is_planner)
                        .unwrap_or(false)
                });

            if run_parallel {
                // Run all frontier nodes concurrently (futures::join_all)
                // ╔════════════════════════════════════════════════════════════════════════╗
                // ║ CRITICAL PERFORMANCE: State Cloning in Parallel Execution              ║
                // ├════════════════════════════════════════════════════════════════════════╣
                // ║ Issue: clone() called per node × iterations (3 nodes × 10 steps = 30×) ║
                // ║ Impact: 50-200ms overhead for state > 500KB                            ║
                // ║ Solution: Use DynState, NOT DynState!                             ║
                // ║                                                                        ║
                // ║ DynState.clone() = Cheap (Arc pointer, ~1μs)                       ║
                // ║ DynState.clone() = Expensive (full data, 5-20ms)                    ║
                // ║                                                                        ║
                // ║ Recommendation: For production, ALWAYS use DynState                 ║
                // ║ Example at: flowgentra-ai/examples/state_graph_react_agent.rs            ║
                // ╚════════════════════════════════════════════════════════════════════════╝
                let state_shared = Arc::new(state); // No clone - wrap as-is
                let futures_with_names: Vec<_> = current_nodes
                    .iter()
                    .filter(|n| *n != "END")
                    .map(|node_name| {
                        let node = self
                            .graph
                            .get_node(node_name)
                            .ok_or_else(|| FlowgentraError::NodeNotFound(node_name.clone()))?;
                        // ⚠️ DynState.clone() is cheap (Arc pointer)
                        // ⚠️ DynState.clone() is expensive (full JSON data copy)
                        let state_in = (*state_shared).clone();
                        let name = node_name.clone();
                        let fut = node.execute(state_in);
                        Ok::<_, FlowgentraError>((name, fut))
                    })
                    .collect::<Result<Vec<_>>>()?;

                let task_futures: Vec<_> = futures_with_names
                    .into_iter()
                    .map(|(name, f)| {
                        Box::pin(async move {
                            let out = f.await?;
                            Ok::<_, FlowgentraError>((name, out))
                        })
                            as std::pin::Pin<
                                Box<dyn std::future::Future<Output = Result<(String, DynState)>> + Send>,
                            >
                    })
                    .collect();

                // Use `try_join_all` to short-circuit immediately on the first error
                let results: Vec<(String, DynState)> = match try_join_all(task_futures).await {
                    Ok(res) => res,
                    Err(e) => {
                        tracing::error!(error = ?e, "Parallel node execution failed early");
                        return Err(FlowgentraError::ParallelExecutionError(e.to_string()));
                    }
                };

                // Recover the original state from the Arc, then merge branch results into it
                state = Arc::try_unwrap(state_shared).unwrap_or_else(|arc| arc.as_ref().clone());
                for (node_name, branch_state) in &results {
                    execution_path.push(node_name.clone());
                    for (k, v) in branch_state.as_map() {
                        state.set(k, v);
                    }
                }

                if let (Some(cp), Some(tid)) = (&self.checkpointer, thread_id) {
                    let last = results.first().map(|(n, _)| n.clone());
                    let meta = CheckpointMetadata {
                        last_node: last,
                        execution_path: execution_path.clone(),
                        extra: HashMap::new(),
                    };
                    if let Err(e) = cp.save(tid, &state, &meta) {
                        tracing::error!(error = %e, thread_id = %tid, "Failed to save checkpoint after parallel execution");
                    }
                }
                tracing::debug!(nodes = ?current_nodes, "Parallel step completed");

                // Collect next nodes from all branches (evaluate edges on merged state)
                let mut seen = HashSet::new();
                for node_name in &current_nodes {
                    if node_name == "END" {
                        continue;
                    }
                    for edge in self.graph.get_next_nodes(node_name) {
                        if edge.to == "END" {
                            tracing::info!(from = node_name, "Execution complete - reached END");
                            if let (Some(cp), Some(tid)) = (&self.checkpointer, thread_id) {
                                let meta = CheckpointMetadata {
                                    last_node: Some(node_name.clone()),
                                    execution_path: execution_path.clone(),
                                    extra: HashMap::new(),
                                };
                                if let Err(e) = cp.save(tid, &state, &meta) {
                                    tracing::error!(error = %e, thread_id = %tid, "Failed to save final checkpoint");
                                }
                            }
                            let _ = self.middleware_pipeline.execute_on_complete(&state).await;
                            return Ok(state);
                        }
                        if let Ok(true) = edge.should_take(&state).await {
                            let next = edge
                                .get_next_node(&state)
                                .await?
                                .unwrap_or_else(|| edge.to.clone());
                            if seen.insert(next.clone()) {
                                next_nodes.push(next);
                            }
                        }
                    }
                }

                if next_nodes.is_empty() {
                    tracing::warn!("No next nodes after parallel step - stopping");
                    break;
                }
                current_nodes = next_nodes;
                continue;
            }

            for node_name in current_nodes {
                if node_name == "END" {
                    tracing::debug!("Reached END node");
                    continue;
                }

                let node_span = tracing::debug_span!("node_execution", node = %node_name);
                let _node_guard = node_span.enter();

                // Execute the node
                let node = self.graph.get_node(&node_name).ok_or_else(|| {
                    tracing::error!(node = %node_name, "Node not found in graph");
                    FlowgentraError::NodeNotFound(node_name.clone())
                })?;

                let _node_timer = TimerGuard::start(format!("node_{}", node_name));

                // === MIDDLEWARE: before_node ===
                // Note: state.clone() is necessary here because:
                // 1. Middleware takes mutable reference and can modify state
                // 2. Node handler takes state by value
                // For large state objects (>1MB), consider using DynState for zero-copy sharing
                let start_time = Instant::now();
                let mut middleware_ctx = MiddlewareContext::new(node_name.clone(), state.clone());

                // Execute middleware before node
                if let Err(e) = self
                    .middleware_pipeline
                    .execute_before_node(&mut middleware_ctx)
                    .await
                {
                    tracing::error!(node = %node_name, error = ?e, "Middleware before_node failed");
                    return Err(e);
                }

                state = middleware_ctx.state;

                // Planner nodes: inject _current_node and _reachable_nodes before execution
                if node.is_planner {
                    state.set(
                        "_current_node",
                        serde_json::Value::String(node_name.clone()),
                    );
                    let reachable = self.graph.get_reachable_node_ids(&node_name);
                    state.set("_reachable_nodes", serde_json::json!(reachable));
                }

                // Inject per-node MCP names so handlers can use get_node_mcp_client()
                if !node.mcps.is_empty() {
                    state.set("_node_mcps", serde_json::json!(node.mcps));
                } else {
                    state.remove("_node_mcps");
                }

                // Execute node - clone is necessary as handler takes State by value
                match node.execute(state.clone()).await {
                    Ok(new_state) => {
                        state = new_state;

                        // === MIDDLEWARE: after_node ===
                        // Clone here is needed because middleware requires mutable reference
                        let elapsed = start_time.elapsed();
                        let mut after_ctx =
                            MiddlewareContext::new(node_name.clone(), state.clone());
                        after_ctx
                            .metadata
                            .insert("elapsed_ms".to_string(), elapsed.as_millis().to_string());

                        if let Err(e) = self
                            .middleware_pipeline
                            .execute_after_node(&mut after_ctx)
                            .await
                        {
                            tracing::error!(node = %node_name, error = ?e, "Middleware after_node failed");
                            return Err(e);
                        }

                        state = after_ctx.state;

                        // === INTERNAL RETRY LOOP ===
                        // If evaluation middleware flagged this node for retry, re-execute
                        // up to max_retries times (from config.evaluation).
                        let retry_key = format!("__node_retry_needed__{}", node_name);
                        let max_retries = self
                            .config
                            .evaluation
                            .as_ref()
                            .filter(|e| e.enabled)
                            .map(|e| e.max_retries)
                            .unwrap_or(0);

                        let mut retry_count = 0u32;
                        while retry_count < max_retries {
                            if let Some(flag) = state.get(&retry_key) {
                                if flag.as_bool().unwrap_or(false) {
                                    retry_count += 1;
                                    tracing::info!(
                                        node = %node_name,
                                        attempt = retry_count,
                                        max_retries = max_retries,
                                        "Retrying node due to low confidence"
                                    );

                                    // Clear the retry flag before re-execution
                                    state.set(&retry_key, serde_json::json!(false));

                                    // Re-execute the node
                                    match node.execute(state.clone()).await {
                                        Ok(new_state) => {
                                            state = new_state;

                                            // Run after_node middleware again to re-evaluate
                                            let mut retry_ctx = MiddlewareContext::new(
                                                node_name.clone(),
                                                state.clone(),
                                            );
                                            if let Err(e) = self
                                                .middleware_pipeline
                                                .execute_after_node(&mut retry_ctx)
                                                .await
                                            {
                                                tracing::error!(
                                                    node = %node_name,
                                                    error = ?e,
                                                    "Middleware after_node failed during retry"
                                                );
                                                return Err(e);
                                            }
                                            state = retry_ctx.state;

                                            // Check if retry is still needed
                                            continue;
                                        }
                                        Err(e) => {
                                            tracing::error!(
                                                node = %node_name,
                                                attempt = retry_count,
                                                error = ?e,
                                                "Node retry execution failed"
                                            );
                                            return Err(e.context(&format!(
                                                "while retrying node '{}' (attempt {})",
                                                node_name, retry_count
                                            )));
                                        }
                                    }
                                }
                            }
                            // No retry flag set — break
                            break;
                        }

                        if retry_count >= max_retries && max_retries > 0 {
                            if let Some(flag) = state.get(&retry_key) {
                                if flag.as_bool().unwrap_or(false) {
                                    tracing::warn!(
                                        node = %node_name,
                                        max_retries = max_retries,
                                        "Maximum retries reached, proceeding with best output"
                                    );
                                    // Clear the flag so downstream code doesn't see a stale retry request
                                    state.set(&retry_key, serde_json::json!(false));
                                }
                            }
                        }

                        execution_path.push(node_name.clone());
                        // Checkpoint after each node when thread_id and checkpointer are set
                        if let (Some(cp), Some(tid)) = (&self.checkpointer, thread_id) {
                            let meta = CheckpointMetadata {
                                last_node: Some(node_name.clone()),
                                execution_path: execution_path.clone(),
                                extra: HashMap::new(),
                            };
                            if let Err(e) = cp.save(tid, &state, &meta) {
                                tracing::warn!(error = ?e, "Checkpoint save failed");
                            }
                        }
                        tracing::debug!(node = %node_name, elapsed_ms = elapsed.as_millis(), "Node execution completed");

                        // Planner nodes: use _next_node from state instead of following edges
                        if node.is_planner {
                            plan_steps += 1;
                            if plan_steps >= max_plan_steps {
                                tracing::info!(plan_steps, "Max plan steps reached - stopping");
                                break;
                            }
                            let next_node = state
                                .get_string("_next_node")
                                .unwrap_or_else(|| "END".to_string());
                            if next_node == "END" {
                                tracing::info!(from = %node_name, "Planner chose END - execution complete");
                                if let (Some(cp), Some(tid)) = (&self.checkpointer, thread_id) {
                                    let meta = CheckpointMetadata {
                                        last_node: Some(node_name.clone()),
                                        execution_path: execution_path.clone(),
                                        extra: HashMap::new(),
                                    };
                                    let _ = cp.save(tid, &state, &meta);
                                }
                                let _ = self.middleware_pipeline.execute_on_complete(&state).await;
                                return Ok(state);
                            }
                            tracing::debug!(next_node = %next_node, "Planner chose next node");
                            next_nodes.push(next_node);
                            continue;
                        }
                    }
                    Err(e) => {
                        tracing::error!(node = %node_name, error = ?e, "Node execution failed");

                        // === MIDDLEWARE: on_error ===
                        let _elapsed = start_time.elapsed();
                        let error_ctx = MiddlewareContext::new(node_name.clone(), state.clone());

                        self.middleware_pipeline
                            .execute_on_error(&node_name, &e, &error_ctx)
                            .await;

                        return Err(e.context(&format!("while executing node '{}'", node_name)));
                    }
                }

                // Find next nodes
                let outgoing = self.graph.get_next_nodes(&node_name);
                tracing::trace!(node = %node_name, edge_count = outgoing.len(), "Evaluating outgoing edges");

                for edge in outgoing {
                    if edge.to == "END" {
                        tracing::info!(from = %node_name, "Execution complete - reached END");
                        execution_path.push(node_name.clone());
                        if let (Some(cp), Some(tid)) = (&self.checkpointer, thread_id) {
                            let meta = CheckpointMetadata {
                                last_node: Some(node_name.clone()),
                                execution_path: execution_path.clone(),
                                extra: HashMap::new(),
                            };
                            if let Err(e) = cp.save(tid, &state, &meta) {
                                tracing::warn!(error = ?e, "Checkpoint save failed");
                            }
                        }
                        // === MIDDLEWARE: on_complete ===
                        let _complete_result =
                            self.middleware_pipeline.execute_on_complete(&state).await;

                        return Ok(state);
                    }

                    // Check if edge condition is satisfied
                    let edge_span = tracing::debug_span!(
                        "edge_evaluation",
                        from = %edge.from,
                        to = %edge.to
                    );
                    let _edge_guard = edge_span.enter();

                    match edge.should_take(&state).await {
                        Ok(should_take) => {
                            if should_take {
                                tracing::debug!(
                                    from = %edge.from,
                                    to = %edge.to,
                                    "Edge condition satisfied"
                                );

                                match edge.get_next_node(&state).await {
                                    Ok(Some(next_node)) => {
                                        next_nodes.push(next_node);
                                    }
                                    Ok(None) => {
                                        next_nodes.push(edge.to.clone());
                                    }
                                    Err(e) => {
                                        tracing::error!(error = ?e, "Failed to get next node");
                                        return Err(e);
                                    }
                                }
                            } else {
                                tracing::trace!(
                                    from = %edge.from,
                                    to = %edge.to,
                                    "Edge condition not satisfied"
                                );
                            }
                        }
                        Err(e) => {
                            tracing::error!(error = ?e, "Error evaluating edge condition");
                            return Err(e);
                        }
                    }
                }
            }

            if next_nodes.is_empty() {
                tracing::warn!("No next nodes found - execution stopping");
                break;
            }

            current_nodes = next_nodes;
        }

        tracing::info!("Agent execution completed successfully");
        Ok(state)
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get a reference to the graph for inspection
    pub fn graph(&self) -> &Graph<DynState> {
        &self.graph
    }

    /// Get mutable access to the graph
    pub fn graph_mut(&mut self) -> &mut Graph<DynState> {
        &mut self.graph
    }

    /// Get the configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Get the LLM client
    pub fn llm_client(&self) -> Arc<dyn LLMClient> {
        Arc::clone(&self.llm_client)
    }

    /// Get an MCP client by name
    pub fn mcp_client(&self, name: &str) -> Option<Arc<dyn MCPClient>> {
        self.mcp_clients.get(name).map(Arc::clone)
    }

    // =========================================================================
    // Visualization
    // =========================================================================

    /// Visualize the agent graph as a text file
    ///
    /// Generates a text-based representation of the graph structure.
    /// This is useful for debugging and documentation.
    ///
    /// # Arguments
    /// - `output_path`: Path where to save the visualization
    ///
    /// # Example
    /// ```ignore
    /// # use flowgentra_ai::core::runtime::AgentRuntime;
    /// # use flowgentra_ai::core::config::AgentConfig;
    /// # let config = AgentConfig::from_file("config.yaml")?;
    /// # let runtime: AgentRuntime<DynState> = AgentRuntime::from_config(config)?;
    /// #[cfg(feature = "visualization")]
    /// runtime.visualize_graph("agent_graph.txt")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn visualize_graph(&self, output_path: &str) -> Result<()> {
        // Use pure-Rust visualization module with improved defaults
        let config = crate::core::utils::visualization::VisualizationConfig::new(output_path);

        crate::core::utils::visualization::visualize_graph(&self.graph, config)?;

        tracing::info!(output_path = %output_path, "Graph visualization completed");
        println!("✓ Graph visualization saved to: {}", output_path);
        Ok(())
    }
}

// Sub-modules for runtime organization
pub mod context;
pub mod optimization;
pub mod parallel;

pub use optimization::{strategies, CloneStats, OptimizedState};
pub use parallel::*;
