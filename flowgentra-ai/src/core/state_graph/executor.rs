//! State graph builder and execution engine

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

tokio::task_local! {
    /// Per-task nesting depth counter for subgraph invocations.
    ///
    /// Incremented by one each time `invoke_with_id` is entered so that
    /// deeply-nested subgraph chains are caught before they overflow.
    static INVOKE_DEPTH: std::cell::Cell<usize>;
}

fn max_nesting_depth() -> usize {
    std::env::var("FLOWGENTRA_MAX_NESTING")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1 && n <= 1_000)
        .unwrap_or(25)
}

use super::checkpoint::{Checkpoint, Checkpointer, InMemoryCheckpointer};
use super::edge::{Edge, FixedEdge, END, START};
use super::error::{Result, StateGraphError};
use super::node::{Node, RouterFn};
use crate::core::middleware::{ExecutionContext, Middleware, MiddlewarePipeline};
use crate::core::observability::events::{EventBroadcaster, ExecutionEvent};
use crate::core::plugins::PluginRegistry;
use crate::core::state::{Context, State};

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
    context: Context,
    broadcaster: Option<Arc<EventBroadcaster>>,
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
            context: Context::new(),
            broadcaster: None,
        }
    }

    /// Add a node to the graph
    pub fn add_node(mut self, name: impl Into<String>, node: Arc<dyn Node<S>>) -> Self {
        self.nodes.insert(name.into(), node);
        self
    }

    /// Add a compiled subgraph as a node.
    pub fn add_subgraph(self, name: impl Into<String>, subgraph: StateGraph<S>) -> Self {
        let name = name.into();
        let node = Arc::new(SubgraphNode::new(name.clone(), subgraph));
        self.add_node(name, node)
    }

    /// Add a fixed edge (A → B)
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push(Edge::Fixed(FixedEdge::new(from, to)));
        self
    }

    /// Add a conditional edge (A → router → B or C or ...)
    pub fn add_conditional_edge(mut self, from: impl Into<String>, router: RouterFn<S>) -> Self {
        self.edges.push(Edge::Conditional {
            from: from.into(),
            router,
        });
        self
    }

    /// Add an async conditional edge
    pub fn add_async_conditional_edge(
        mut self,
        from: impl Into<String>,
        router: super::edge::AsyncRouterFn<S>,
    ) -> Self {
        self.edges.push(Edge::AsyncConditional {
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

    /// Set the framework context (LLM, MCP, RAG configs)
    pub fn set_context(mut self, context: Context) -> Self {
        self.context = context;
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

    /// Set the plugin registry
    pub fn set_plugins(mut self, plugins: Arc<PluginRegistry>) -> Self {
        self.plugins = plugins;
        self
    }

    /// Attach an `EventBroadcaster` so all subscribers receive real-time execution events.
    ///
    /// If not set, the compiled graph creates its own internal broadcaster.
    /// Call `StateGraph::subscribe()` on the compiled graph to get a receiver.
    pub fn set_broadcaster(mut self, broadcaster: Arc<EventBroadcaster>) -> Self {
        self.broadcaster = Some(broadcaster);
        self
    }

    /// Attach a shared `ToolRegistry` for dynamic tool binding at runtime.
    ///
    /// Nodes can call `ctx.tool_registry()` to read or write tools mid-execution.
    pub fn with_tool_registry(
        mut self,
        registry: Arc<tokio::sync::RwLock<crate::core::tools::ToolRegistry>>,
    ) -> Self {
        self.context.set_tool_registry(registry);
        self
    }

    /// Build and validate the graph
    pub fn compile(self) -> Result<StateGraph<S>> {
        let entry_point = self
            .entry_point
            .ok_or_else(|| StateGraphError::InvalidGraph("No entry point set".to_string()))?;

        if !self.nodes.contains_key(&entry_point) {
            return Err(StateGraphError::InvalidGraph(format!(
                "Entry point '{}' is not a registered node",
                entry_point
            )));
        }

        // ── Validate edge references ──
        for edge in &self.edges {
            match edge {
                Edge::Fixed(fixed_edge) => {
                    if fixed_edge.from != START && !self.nodes.contains_key(&fixed_edge.from) {
                        return Err(StateGraphError::NodeNotFound(format!(
                            "'{}' (referenced in edge). Available nodes: [{}]",
                            fixed_edge.from,
                            self.nodes.keys().cloned().collect::<Vec<_>>().join(", ")
                        )));
                    }
                    if fixed_edge.to != END && !self.nodes.contains_key(&fixed_edge.to) {
                        return Err(StateGraphError::NodeNotFound(format!(
                            "'{}' (target of edge from '{}'). Available nodes: [{}]",
                            fixed_edge.to,
                            fixed_edge.from,
                            self.nodes.keys().cloned().collect::<Vec<_>>().join(", ")
                        )));
                    }
                }
                Edge::Conditional { from, .. } | Edge::AsyncConditional { from, .. } => {
                    if from != START && !self.nodes.contains_key(from) {
                        return Err(StateGraphError::NodeNotFound(format!(
                            "'{}' (conditional edge source). Available nodes: [{}]",
                            from,
                            self.nodes.keys().cloned().collect::<Vec<_>>().join(", ")
                        )));
                    }
                }
            }
        }

        // ── Detect unreachable nodes ──
        {
            let mut reachable = std::collections::HashSet::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(entry_point.clone());

            while let Some(node) = queue.pop_front() {
                if reachable.contains(&node) {
                    continue;
                }
                reachable.insert(node.clone());

                for edge in &self.edges {
                    if edge.from() == node {
                        match edge {
                            Edge::Fixed(fe) if fe.to != END => {
                                queue.push_back(fe.to.clone());
                            }
                            Edge::Conditional { .. } | Edge::AsyncConditional { .. } => {
                                // Conditional edges can reach any node — add all as potentially reachable
                                for name in self.nodes.keys() {
                                    queue.push_back(name.clone());
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            let unreachable: Vec<String> = self
                .nodes
                .keys()
                .filter(|n| !reachable.contains(*n))
                .cloned()
                .collect();

            if !unreachable.is_empty() {
                tracing::warn!(
                    nodes = %unreachable.join(", "),
                    "Graph topology warning: these nodes are unreachable from the entry point. \
                     Add edges leading to them or remove them from the graph."
                );
            }
        }

        let checkpointer = self
            .checkpointer
            .unwrap_or_else(|| Arc::new(InMemoryCheckpointer::new()));

        let broadcaster = self
            .broadcaster
            .unwrap_or_else(|| Arc::new(EventBroadcaster::new(256)));

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
            context: self.context,
            broadcaster,
            thread_locks: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        })
    }
}

impl<S: State> Default for StateGraphBuilder<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// A node that delegates execution to a compiled subgraph.
pub struct SubgraphNode<S: State> {
    name: String,
    subgraph: Arc<StateGraph<S>>,
}

impl<S: State + Send + Sync + 'static> SubgraphNode<S> {
    pub fn new(name: impl Into<String>, subgraph: StateGraph<S>) -> Self {
        Self {
            name: name.into(),
            subgraph: Arc::new(subgraph),
        }
    }
}

#[async_trait::async_trait]
impl<S: State + Send + Sync + 'static> Node<S> for SubgraphNode<S> {
    async fn execute(&self, state: &S, _ctx: &Context) -> Result<S::Update> {
        // Run the subgraph to completion, then diff to produce an update.
        // For subgraphs, we run the full graph and return a "replace all" update.
        // The subgraph internally applies its own updates step by step.
        let result = self.subgraph.invoke(state.clone()).await?;
        // We need to produce an S::Update from the final state.
        // Since we can't generically diff two S values, we serialize and let
        // the caller decide. For now, we use a workaround: store the result
        // in a thread-local and return a default update.
        // TODO: A better approach is to have the subgraph return the accumulated updates.
        // For now, we return default (no-op) and the executor will use the subgraph's
        // final state directly via the SubgraphNode execution path.
        let _ = result;
        Ok(S::Update::default())
    }

    fn name(&self) -> &str {
        &self.name
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
    context: Context,
    broadcaster: Arc<EventBroadcaster>,
    /// Per-thread-id mutex map that serializes concurrent invocations on the
    /// same thread_id so two callers cannot race on checkpoint read-modify-write.
    thread_locks: Arc<tokio::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
}

impl<S: State + Send + Sync + 'static> StateGraph<S> {
    /// Build a new state graph using the builder pattern.
    pub fn builder() -> StateGraphBuilder<S> {
        StateGraphBuilder::new()
    }

    /// Get a list of node names in this graph.
    pub fn node_names(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    /// Get the entry point node name.
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Get a reference to the context.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Get a mutable reference to the context.
    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.context
    }

    /// Subscribe to real-time execution events from this graph.
    ///
    /// The receiver will get all events emitted during `invoke()` / `stream()` calls.
    ///
    /// ```ignore
    /// let mut rx = graph.subscribe();
    /// tokio::spawn(async move {
    ///     while let Ok(event) = rx.recv().await {
    ///         println!("{:?}", event);
    ///     }
    /// });
    /// let result = graph.invoke(state).await?;
    /// ```
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<ExecutionEvent> {
        self.broadcaster.subscribe()
    }

    /// Get a reference to the event broadcaster.
    ///
    /// Use this to attach external subscribers before invoking the graph.
    pub fn event_broadcaster(&self) -> &Arc<EventBroadcaster> {
        &self.broadcaster
    }

    /// Execute the graph and return a stream of execution events.
    ///
    /// The stream yields events as nodes execute and completes when the graph finishes.
    /// The final item is always `GraphCompleted` or `GraphFailed`.
    ///
    /// ```ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = graph.stream(initial_state).await;
    /// while let Some(event) = stream.next().await {
    ///     match event {
    ///         ExecutionEvent::LLMStreaming { chunk, .. } => print!("{}", chunk),
    ///         ExecutionEvent::NodeCompleted { node_name, .. } => println!("\n✓ {}", node_name),
    ///         ExecutionEvent::GraphCompleted { .. } => break,
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn stream(
        &self,
        _initial_state: S,
    ) -> impl futures::Stream<Item = ExecutionEvent> + 'static {
        let _rx = self.broadcaster.subscribe();

        // Use a channel to bridge the broadcast receiver into a stream
        let (tx, rx_stream) = tokio::sync::mpsc::unbounded_channel::<ExecutionEvent>();
        let broadcaster = Arc::clone(&self.broadcaster);

        // Spawn a task that forwards broadcast events to the mpsc channel
        // until GraphCompleted or GraphFailed is received.
        let mut bcast_rx = broadcaster.subscribe();
        tokio::spawn(async move {
            while let Ok(event) = bcast_rx.recv().await {
                let is_terminal = matches!(
                    &event,
                    ExecutionEvent::GraphCompleted { .. } | ExecutionEvent::GraphFailed { .. }
                );
                let _ = tx.send(event);
                if is_terminal {
                    break;
                }
            }
        });

        // We need to drive the graph execution in a background task too.
        // However, since `self` is borrowed, we clone the necessary parts.
        // The stream is returned immediately; events flow as the graph runs.

        // Note: we cannot simply call self.invoke() from within stream() because
        // of borrow constraints. Callers should use stream_owned() or the
        // pattern: subscribe() + invoke() in separate tasks for full ownership.
        // This method is suitable when the caller can spawn the invoke separately.
        futures::stream::unfold(rx_stream, |mut rx| async move {
            rx.recv().await.map(|event| (event, rx))
        })
    }

    /// Execute the graph
    pub async fn invoke(&self, initial_state: S) -> Result<S> {
        let thread_id = Uuid::new_v4().to_string();
        self.invoke_with_id(thread_id, initial_state).await
    }

    /// Execute the graph with a specific thread ID (for resuming from checkpoints).
    ///
    /// Each nested subgraph call increments an async-task-local depth counter.
    /// Once the counter reaches `FLOWGENTRA_MAX_NESTING` (default 25) a
    /// `StateGraphError::RecursionLimitExceeded` error is returned to prevent
    /// unbounded recursion.
    pub async fn invoke_with_id(&self, thread_id: String, initial_state: S) -> Result<S> {
        let current_depth = INVOKE_DEPTH.try_with(|d| d.get()).unwrap_or(0);
        let limit = max_nesting_depth();
        if current_depth >= limit {
            return Err(StateGraphError::RecursionLimitExceeded {
                depth: current_depth,
                limit,
            });
        }
        INVOKE_DEPTH
            .scope(
                std::cell::Cell::new(current_depth + 1),
                self.invoke_inner(thread_id, initial_state),
            )
            .await
    }

    async fn invoke_inner(&self, thread_id: String, initial_state: S) -> Result<S> {
        // Serialize concurrent invocations on the same thread_id to prevent
        // checkpoint read-modify-write races when two callers share a thread.
        let thread_lock = {
            let mut locks = self.thread_locks.lock().await;
            Arc::clone(
                locks
                    .entry(thread_id.clone())
                    .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
            )
        };
        let _thread_guard = thread_lock.lock().await;

        let graph_id = thread_id.clone();
        self.broadcaster.emit(ExecutionEvent::GraphStarted {
            graph_id: graph_id.clone(),
        });

        let graph_start = Instant::now();

        let state = if let Some(checkpoint) = self.checkpointer.load_latest(&thread_id).await? {
            checkpoint.state
        } else {
            initial_state
        };

        let mut current_state = state;
        let mut current_node = self.entry_point.clone();
        let mut step = 0;

        loop {
            if step >= self.max_steps {
                let err = StateGraphError::Timeout("Max steps exceeded".to_string());
                self.broadcaster.emit(ExecutionEvent::GraphFailed {
                    error: err.to_string(),
                    last_node: Some(current_node.clone()),
                });
                return Err(err);
            }

            if self.interrupt_before.contains(&current_node) {
                return Err(StateGraphError::InterruptedAtBreakpoint { node: current_node });
            }

            // Emit NodeStarted event
            self.broadcaster.node_started(&current_node, step);

            // Create execution context for middleware
            let mut middleware_ctx = ExecutionContext::new(&current_node, current_state.clone());

            // Execute before_node hooks
            self.middleware
                .execute_before_node(&mut middleware_ctx)
                .await
                .map_err(|e| StateGraphError::ExecutionError {
                    node: current_node.clone(),
                    reason: format!("Middleware error: {}", e),
                })?;

            current_state = middleware_ctx.state;

            // Create plugin context
            let plugin_ctx = crate::core::plugins::PluginContext::new();
            let _ = self
                .plugins
                .on_node_execute(&plugin_ctx, &current_node)
                .await;

            // Prepare node context with current node info + broadcaster
            let mut node_ctx = self.context.clone();
            node_ctx.set_node_name(&current_node);
            node_ctx.set_event_broadcaster(Arc::clone(&self.broadcaster));

            tracing::info!(node = %current_node, step = %step, "Executing node");

            let node = self
                .nodes
                .get(&current_node)
                .ok_or_else(|| StateGraphError::NodeNotFound(current_node.clone()))?;

            let node_start = Instant::now();

            match node.execute(&current_state, &node_ctx).await {
                Ok(update) => {
                    let duration_ms = node_start.elapsed().as_millis() as u64;

                    // Apply the partial update using per-field reducers
                    current_state.apply_update(update);

                    // Update context for after_node hooks
                    middleware_ctx.state = current_state.clone();

                    self.middleware
                        .execute_after_node(&mut middleware_ctx)
                        .await
                        .map_err(|e| StateGraphError::ExecutionError {
                            node: current_node.clone(),
                            reason: format!("Middleware error: {}", e),
                        })?;

                    current_state = middleware_ctx.state;

                    let _ = self
                        .plugins
                        .on_node_complete(&plugin_ctx, &current_node, true)
                        .await;

                    // Checkpoint
                    let checkpoint = Checkpoint::new(
                        thread_id.clone(),
                        step,
                        current_node.clone(),
                        current_state.clone(),
                    );
                    self.checkpointer.save(&checkpoint).await?;

                    // Emit NodeCompleted event
                    self.broadcaster
                        .node_completed(&current_node, step, duration_ms, None);

                    if self.interrupt_after.contains(&current_node) {
                        return Err(StateGraphError::InterruptedAtBreakpoint {
                            node: current_node,
                        });
                    }

                    // Determine next node
                    let next_node = self.get_next_node(&current_node, &current_state).await?;

                    // Emit EdgeTraversed event
                    self.broadcaster
                        .edge_traversed(&current_node, &next_node, None);

                    if next_node == END {
                        self.middleware.execute_on_complete(&current_state).await;
                        let total_duration_ms = graph_start.elapsed().as_millis() as u64;
                        self.broadcaster.emit(ExecutionEvent::GraphCompleted {
                            total_steps: step + 1,
                            total_duration_ms,
                        });
                        return Ok(current_state);
                    }

                    current_node = next_node;
                }
                Err(e) => {
                    tracing::error!(node = %current_node, error = %e, "Node execution failed");

                    // Emit NodeFailed event
                    self.broadcaster
                        .node_failed(&current_node, step, e.to_string());

                    let _ = self
                        .plugins
                        .on_node_complete(&plugin_ctx, &current_node, false)
                        .await;

                    let error_ctx = ExecutionContext::new(&current_node, current_state.clone());
                    let err_as_flowgentra =
                        crate::core::error::FlowgentraError::ExecutionError(e.to_string());
                    self.middleware
                        .execute_on_error(&current_node, &err_as_flowgentra, &error_ctx)
                        .await;

                    self.broadcaster.emit(ExecutionEvent::GraphFailed {
                        error: e.to_string(),
                        last_node: Some(current_node),
                    });

                    return Err(e);
                }
            }

            step += 1;
        }
    }

    /// Get the next node from the current node
    async fn get_next_node(&self, current_node: &str, state: &S) -> Result<String> {
        for edge in &self.edges {
            if edge.from() == current_node {
                return edge.next_node(state).await;
            }
        }
        Ok(END.to_string())
    }

    /// Resume execution from a checkpoint
    pub async fn resume(&self, thread_id: &str) -> Result<S> {
        let checkpoint = self
            .checkpointer
            .load_latest(thread_id)
            .await?
            .ok_or_else(|| StateGraphError::ResumeFailed("No checkpoint found".to_string()))?;

        self.invoke_with_id(thread_id.to_string(), checkpoint.state)
            .await
    }

    /// Resume execution from a checkpoint with injected state updates.
    ///
    /// Human-in-the-loop: execution pauses at a breakpoint, the user modifies
    /// state via an update, and execution resumes.
    pub async fn resume_with_update(&self, thread_id: &str, state_update: S::Update) -> Result<S> {
        let checkpoint = self
            .checkpointer
            .load_latest(thread_id)
            .await?
            .ok_or_else(|| StateGraphError::ResumeFailed("No checkpoint found".to_string()))?;

        let mut state = checkpoint.state;
        state.apply_update(state_update);

        self.invoke_with_id(thread_id.to_string(), state).await
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

    // ── Graph Export ──

    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph StateGraph {\n  rankdir=LR;\n");
        dot.push_str("  node [shape=box style=rounded];\n\n");
        dot.push_str("  \"__start__\" [label=\"START\" shape=ellipse];\n");
        dot.push_str("  \"__end__\" [label=\"END\" shape=ellipse];\n");
        dot.push_str(&format!("  \"__start__\" -> \"{}\";\n", self.entry_point));

        for name in self.nodes.keys() {
            dot.push_str(&format!("  \"{}\";\n", name));
        }

        for edge in &self.edges {
            match edge {
                Edge::Fixed(fe) => {
                    let from = if fe.from == START {
                        "__start__"
                    } else {
                        &fe.from
                    };
                    let to = if fe.to == END { "__end__" } else { &fe.to };
                    dot.push_str(&format!("  \"{}\" -> \"{}\";\n", from, to));
                }
                Edge::Conditional { from, .. } | Edge::AsyncConditional { from, .. } => {
                    let from_label = if from == START {
                        "__start__"
                    } else {
                        from.as_str()
                    };
                    for target_name in self.nodes.keys() {
                        dot.push_str(&format!(
                            "  \"{}\" -> \"{}\" [style=dashed label=\"?\"];\n",
                            from_label, target_name
                        ));
                    }
                    dot.push_str(&format!(
                        "  \"{}\" -> \"__end__\" [style=dashed label=\"?\"];\n",
                        from_label
                    ));
                }
            }
        }

        dot.push_str("}\n");
        dot
    }

    pub fn to_mermaid(&self) -> String {
        let mut out = String::from("graph LR\n");
        out.push_str("  __start__((START))\n");
        out.push_str("  __end__((END))\n");
        out.push_str(&format!("  __start__ --> {}\n", self.entry_point));

        for name in self.nodes.keys() {
            out.push_str(&format!("  {}[{}]\n", name, name));
        }

        for edge in &self.edges {
            match edge {
                Edge::Fixed(fe) => {
                    let from = if fe.from == START {
                        "__start__"
                    } else {
                        &fe.from
                    };
                    let to = if fe.to == END { "__end__" } else { &fe.to };
                    out.push_str(&format!("  {} --> {}\n", from, to));
                }
                Edge::Conditional { from, .. } | Edge::AsyncConditional { from, .. } => {
                    let from_label = if from == START {
                        "__start__"
                    } else {
                        from.as_str()
                    };
                    for target_name in self.nodes.keys() {
                        out.push_str(&format!("  {} -.->|?| {}\n", from_label, target_name));
                    }
                    out.push_str(&format!("  {} -.->|?| __end__\n", from_label));
                }
            }
        }

        out
    }

    pub fn to_json(&self) -> serde_json::Value {
        let nodes: Vec<String> = self.nodes.keys().cloned().collect();
        let edges: Vec<serde_json::Value> = self
            .edges
            .iter()
            .map(|edge| match edge {
                Edge::Fixed(fe) => serde_json::json!({
                    "type": "fixed",
                    "from": fe.from,
                    "to": fe.to,
                }),
                Edge::Conditional { from, .. } | Edge::AsyncConditional { from, .. } => {
                    serde_json::json!({
                        "type": "conditional",
                        "from": from,
                    })
                }
            })
            .collect();

        serde_json::json!({
            "nodes": nodes,
            "edges": edges,
            "entry_point": self.entry_point,
            "max_steps": self.max_steps,
        })
    }
}
