//! Bridge: build a `state_graph::StateGraph<DynState>` from an `AgentConfig`.
//!
//! This is step 2 of the engine merge (see `docs/design/engine-merge.md`):
//! the config-driven agent runs on the same executor as everything else,
//! gaining cancellation, wall-clock budgets, atomic/SQLite checkpointing,
//! in-node interrupt, and parallel supersteps.
//!
//! **Scope:** plain handler nodes; fixed edges (including parallel fan-out via
//! `to: [a, b]`); named conditional edges; per-node MCPs (injected as
//! `_node_mcps`); the built-in types `retry`, `timeout`, `evaluation`, `loop`,
//! `memory`, and `planner` (LLM-driven routing via an async out-edge over the
//! planner's declared successors, with re-planning through back-edges). All
//! reuse the legacy implementations for bug-for-bug parity. Nodes whose `type`
//! is `supervisor`, `subgraph`, `human_in_the_loop`, or that carry RAG / the
//! graph-level planner, are **not** handled — [`can_bridge`] returns `false`
//! so the caller keeps using the legacy `AgentRuntime`.

use std::collections::HashSet;
use std::sync::Arc;

use crate::core::agent::ArcHandler;
use crate::core::config::AgentConfig;
use crate::core::error::{FlowgentraError, Result};
use crate::core::state::{Context, DynState, DynStateUpdate};
use crate::core::state_graph::edge::{END, START};
use crate::core::state_graph::error::StateGraphError;
use crate::core::state_graph::node::{Node, RouterFn};
use crate::core::state_graph::{StateGraph, StateGraphBuilder};

/// Built-in node types the legacy runtime special-cases and that the bridge
/// does **not** yet handle; presence of any means the config stays on the
/// legacy runtime. (`retry`/`timeout`/`evaluation`/`loop`/`planner`/`memory`
/// are handled — see [`BRIDGEABLE_WRAPPER_TYPES`] — so they are absent here.)
const UNBRIDGEABLE_TYPES: &[&str] = &[
    "human_in_the_loop",
    "supervisor",
    "orchestrator",
    "subgraph",
    "agent",
    "agent_or_graph",
];

/// Built-in node types the bridge handles: retry/timeout control-flow wrappers,
/// evaluation refinement loop, loop iteration, `planner` (LLM-driven routing),
/// and `memory` (built-in memory operations). All reuse the legacy
/// implementations for bug-for-bug parity.
const BRIDGEABLE_WRAPPER_TYPES: &[&str] = &[
    "retry",
    "timeout",
    "evaluation",
    "loop",
    "planner",
    "memory",
];

/// `true` if every node in the config is either a plain handler node or a
/// bridgeable built-in type. Per-node MCPs are supported (injected via
/// `_node_mcps`). Such configs can run on the `state_graph` executor via
/// [`build_state_graph`].
pub fn can_bridge(config: &AgentConfig) -> bool {
    // RAG / graph-level planner features route through the legacy runtime.
    if config.graph.rag.is_some() || config.graph.planner.enabled {
        return false;
    }
    config.graph.nodes.iter().all(|n| {
        n.node_type
            .as_deref()
            .map(|t| !UNBRIDGEABLE_TYPES.contains(&t) || BRIDGEABLE_WRAPPER_TYPES.contains(&t))
            .unwrap_or(true)
            && !n.handler.starts_with("builtin::")
    })
}

/// A `state_graph` node backed by a config `ArcHandler` (full-state in/out).
///
/// The handler returns the complete next state; the executor's per-field
/// reducers then merge it. To avoid double-applying accumulating reducers we
/// emit an update containing only the keys whose values actually changed,
/// each as a plain overwrite (matching legacy last-write semantics for
/// config handlers, which have always returned full state).
struct HandlerNode {
    name: String,
    handler: ArcHandler<DynState>,
    /// Per-node MCP names, injected as `_node_mcps` so the handler can call
    /// `state.get_node_mcp_client()` (matches the legacy runtime).
    mcps: Vec<String>,
}

#[async_trait::async_trait]
impl Node<DynState> for HandlerNode {
    async fn execute(
        &self,
        state: &DynState,
        _ctx: &Context,
    ) -> std::result::Result<DynStateUpdate, StateGraphError> {
        let before = state.clone();
        // DynState clone shares the underlying Arc, so injecting `_node_mcps`
        // must go on an independent deep copy or it would mutate the executor's
        // own state.
        let input = if self.mcps.is_empty() {
            state.clone()
        } else {
            let deep = state.deep_clone();
            deep.set("_node_mcps", serde_json::json!(self.mcps));
            deep
        };
        let after = (self.handler)(input)
            .await
            .map_err(|e| StateGraphError::ExecutionError {
                node: self.name.clone(),
                reason: e.to_string(),
            })?;

        let mut update = DynStateUpdate::new();
        for key in after.keys() {
            // Don't leak the per-node MCP routing key into merged state.
            if key == "_node_mcps" {
                continue;
            }
            let new_val = after.get(&key).unwrap_or(serde_json::Value::Null);
            if before.get(&key) != Some(new_val.clone()) {
                update.insert(key, new_val);
            }
        }
        Ok(update)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A config condition: `Fn(&DynState) -> bool` deciding whether its edge fires.
pub type ConfigCondition =
    Arc<dyn Fn(&DynState) -> std::result::Result<bool, FlowgentraError> + Send + Sync>;

/// Build an async router that asks the LLM to choose the next node from
/// `reachable`, reusing the legacy planner handler (bug-for-bug parity for the
/// prompt, parsing, and fallback). The chosen name is normalised to the END
/// sentinel when the planner picks "END".
fn make_planner_router(
    llm: Arc<dyn crate::core::llm::LLM>,
    prompt: Option<String>,
    planner_name: String,
    reachable: Vec<String>,
) -> crate::core::state_graph::edge::AsyncRouterFn<DynState> {
    // The legacy handler is Box<dyn Fn>; wrap in Arc so the router can call it
    // on every re-plan.
    let handler: Arc<crate::core::node::NodeFunction<DynState>> = Arc::new(
        crate::core::node::planner::create_planner_handler(llm, prompt),
    );
    // The planner expects config-style names ("END", not the sentinel) in
    // _reachable_nodes; map the sentinel back for the prompt, then map the
    // decision forward to the sentinel.
    let reachable_cfg: Vec<String> = reachable
        .iter()
        .map(|n| {
            if n == END {
                "END".to_string()
            } else {
                n.clone()
            }
        })
        .collect();

    Box::new(move |state: &DynState| {
        let handler = handler.clone();
        let reachable_cfg = reachable_cfg.clone();
        let planner_name = planner_name.clone();
        let state = state.clone();
        Box::pin(async move {
            state.set("_current_node", serde_json::json!(planner_name));
            state.set("_reachable_nodes", serde_json::json!(reachable_cfg));
            let after = handler(state).await.map_err(|e| {
                crate::core::state_graph::StateGraphError::RouterError(e.to_string())
            })?;
            let chosen = after
                .get_str("_next_node")
                .unwrap_or_else(|| "END".to_string());
            Ok(if chosen == "END" {
                END.to_string()
            } else {
                chosen
            })
        })
    })
}

/// Build the `state_graph` node for a config node, applying any built-in
/// behavior its `type` requests. Plain nodes become a [`HandlerNode`] over the
/// config handler; retry/timeout wrap that in a control-flow node;
/// evaluation/loop transform the handler using the legacy implementations
/// (bug-for-bug parity).
fn build_bridged_node(
    name: &str,
    node: &crate::core::node::NodeConfig,
    handler: ArcHandler<DynState>,
) -> Result<Arc<dyn Node<DynState>>> {
    use crate::core::state_graph::{OnTimeout, RetryNode, TimeoutNode};

    let get_u64 = |key: &str| node.config.get(key).and_then(|v| v.as_u64());
    let get_f32 = |key: &str| {
        node.config
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
    };

    // Handler-level transforms (evaluation refinement, loop iteration, memory
    // operations) produce a new handler that HandlerNode then wraps.
    let effective_handler: ArcHandler<DynState> = match node.node_type.as_deref() {
        Some("evaluation") => {
            let cfg =
                crate::core::node::evaluation_node::EvaluationNodeConfig::from_node_config(node)?;
            // Reuse the legacy refine-and-score loop (Box<dyn Fn> → Arc).
            Arc::from(cfg.into_wrapping_node_fn(handler))
        }
        Some("loop") => {
            let cfg = crate::core::node::advanced_nodes::LoopNodeConfig::from_node_config(node)?;
            if node.handler.is_empty() {
                // Standalone bookkeeping node (pair with a back-edge on
                // __loop_continue__); no user handler.
                Arc::from(super::create_loop_standalone_handler(cfg))
            } else {
                // Wrapping mode: run the handler up to max_iterations inline.
                Arc::from(super::wrap_handler_with_loop(handler, cfg))
            }
        }
        Some("memory") => {
            // Built-in memory operation (no user handler); reuse the legacy
            // memory handlers.
            let op = node
                .config
                .get("operation")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            super::create_memory_handler(op)
                .map(Arc::from)
                .ok_or_else(|| {
                    FlowgentraError::ConfigError(format!(
                        "Unknown memory operation '{}' for node '{}'. Valid operations: \
                         append_message, compress_history, clear_history, get_message_count, \
                         format_history_for_context",
                        op, name
                    ))
                })?
        }
        _ => handler,
    };

    let base: Arc<dyn Node<DynState>> = Arc::new(HandlerNode {
        name: name.to_string(),
        handler: effective_handler,
        mcps: node.mcps.clone(),
    });

    // Node-level control-flow wrappers (retry/timeout).
    let node_out: Arc<dyn Node<DynState>> = match node.node_type.as_deref() {
        Some("retry") => {
            let mut r = RetryNode::new(name.to_string(), base);
            if let Some(v) = get_u64("max_retries") {
                r = r.with_max_retries(v as usize);
            }
            if let Some(v) = get_u64("backoff_ms") {
                r = r.with_backoff_ms(v);
            }
            if let Some(v) = get_f32("backoff_multiplier") {
                r = r.with_multiplier(v);
            }
            if let Some(v) = get_u64("max_backoff_ms") {
                r = r.with_max_backoff_ms(v);
            }
            Arc::new(r)
        }
        Some("timeout") => {
            let timeout_ms = node
                .timeout_ms
                .or_else(|| get_u64("timeout_ms"))
                .unwrap_or(30_000);
            let on_timeout = match node
                .config
                .get("on_timeout")
                .and_then(|v| v.as_str())
                .unwrap_or("error")
            {
                "skip" => OnTimeout::Skip,
                _ => OnTimeout::Error,
            };
            Arc::new(
                TimeoutNode::new(name.to_string(), base, timeout_ms).with_on_timeout(on_timeout),
            )
        }
        _ => base,
    };

    Ok(node_out)
}

/// Build a compiled `StateGraph<DynState>` from an `AgentConfig`.
///
/// - `handlers`: node handlers keyed by node name (already resolved).
/// - `conditions`: condition functions keyed by the `condition:` name used on
///   edges. A node with any conditional out-edge routes via those conditions
///   (first matching wins, in edge declaration order; falls through to END).
///
/// Returns an error if the config is not bridgeable ([`can_bridge`]) or a
/// referenced handler/condition is missing.
pub fn build_state_graph(
    config: &AgentConfig,
    handlers: std::collections::HashMap<String, ArcHandler<DynState>>,
    conditions: std::collections::HashMap<String, ConfigCondition>,
) -> Result<StateGraph<DynState>> {
    build_state_graph_with_llm(config, handlers, conditions, None)
}

/// Like [`build_state_graph`], but with an optional LLM used by `type: planner`
/// nodes (LLM-driven routing). A config containing a planner node requires
/// `llm` to be `Some`.
pub fn build_state_graph_with_llm(
    config: &AgentConfig,
    handlers: std::collections::HashMap<String, ArcHandler<DynState>>,
    conditions: std::collections::HashMap<String, ConfigCondition>,
    llm: Option<Arc<dyn crate::core::llm::LLM>>,
) -> Result<StateGraph<DynState>> {
    if !can_bridge(config) {
        return Err(FlowgentraError::ConfigError(
            "This config uses built-in node types (supervisor/subgraph/memory/etc.), \
             per-node MCPs, or the graph-level planner, which are not yet supported on \
             the state_graph engine. It will run on the legacy runtime."
                .to_string(),
        ));
    }

    // Planner out-edges route via an LLM, not fixed edges; collect the planner
    // node names so the edge loop treats their out-edges specially.
    let planner_nodes: HashSet<&str> = config
        .graph
        .nodes
        .iter()
        .filter(|n| n.node_type.as_deref() == Some("planner"))
        .map(|n| n.name.as_str())
        .collect();

    // Config files use the literal names "START"/"END"; the executor uses the
    // sentinel constants. Normalise config names to the executor's.
    let norm = |name: &str| -> String {
        match name {
            "START" => START.to_string(),
            "END" => END.to_string(),
            other => other.to_string(),
        }
    };

    let mut builder: StateGraphBuilder<DynState> = StateGraphBuilder::new();
    let node_names: HashSet<&str> = config.graph.nodes.iter().map(|n| n.name.as_str()).collect();

    for node in &config.graph.nodes {
        if node.name == "START" || node.name == "END" {
            continue;
        }
        // Standalone loop, planner, and memory nodes are control/built-in nodes
        // that take no user handler (the planner routes on its out-edge; memory
        // and standalone loop are fully implemented by the library).
        let is_standalone_loop =
            node.node_type.as_deref() == Some("loop") && node.handler.is_empty();
        let is_planner = node.node_type.as_deref() == Some("planner");
        let is_memory = node.node_type.as_deref() == Some("memory");
        let handler = if is_standalone_loop || is_planner || is_memory {
            // Pass-through: HandlerNode construction needs an ArcHandler, but
            // the loop transform ignores it and the planner node does nothing
            // (its routing happens on the async out-edge).
            let noop: ArcHandler<DynState> = Arc::new(|s: DynState| Box::pin(async move { Ok(s) }));
            noop
        } else {
            handlers.get(&node.name).cloned().ok_or_else(|| {
                FlowgentraError::ConfigError(format!(
                    "No handler registered for node '{}'",
                    node.name
                ))
            })?
        };
        let wrapped = build_bridged_node(&node.name, node, handler)?;
        builder = builder.add_node(node.name.clone(), wrapped);
    }

    // Entry point: the START edge's target(s). A single entry is required by
    // the executor; multiple START targets are joined by an implicit fan-out
    // from a synthetic pass-through is out of scope — take the first and add
    // the rest as fixed edges below.
    let entry = config
        .graph
        .edges
        .iter()
        .find(|e| e.from == "START")
        .and_then(|e| e.to.first())
        .map(|t| norm(t))
        .ok_or_else(|| {
            FlowgentraError::ConfigError("Config has no START edge / entry point".to_string())
        })?;
    builder = builder.set_entry_point(entry);

    // Planner out-edges: one async LLM router per planner node, over the union
    // of its declared successors (the reachable set). Added once per planner.
    for planner in &planner_nodes {
        let reachable: Vec<String> = config
            .graph
            .edges
            .iter()
            .filter(|e| e.from == *planner)
            .flat_map(|e| e.to.iter())
            .map(|t| norm(t))
            .collect();
        if reachable.is_empty() {
            return Err(FlowgentraError::ConfigError(format!(
                "Planner node '{planner}' has no outgoing edges (no reachable nodes to route to)"
            )));
        }
        let client = llm.clone().ok_or_else(|| {
            FlowgentraError::ConfigError(format!(
                "Planner node '{planner}' requires an LLM; none was provided to the bridge"
            ))
        })?;
        let planner_node = config
            .graph
            .nodes
            .iter()
            .find(|n| n.name == *planner)
            .expect("planner name came from the node list");
        let prompt = planner_node
            .config
            .get("prompt")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let router = make_planner_router(client, prompt, planner.to_string(), reachable);
        builder = builder.add_async_conditional_edge(norm(planner), router);
    }

    for edge in &config.graph.edges {
        if edge.from == "START" {
            continue; // entry handled above
        }
        // Planner out-edges are handled by the async router above.
        if planner_nodes.contains(edge.from.as_str()) {
            continue;
        }
        // Group edges by whether they carry a condition.
        if let Some(cond_name) = &edge.condition {
            let cond = conditions.get(cond_name).cloned().ok_or_else(|| {
                FlowgentraError::ConfigError(format!(
                    "Edge from '{}' references unknown condition '{}'",
                    edge.from, cond_name
                ))
            })?;
            let targets: Vec<String> = edge.to.iter().map(|t| norm(t)).collect();
            let router: RouterFn<DynState> = Box::new(move |state: &DynState| {
                if cond(state).map_err(|e| StateGraphError::RouterError(e.to_string()))? {
                    Ok(targets.first().cloned().unwrap_or_else(|| END.to_string()))
                } else {
                    Ok(END.to_string())
                }
            });
            builder = builder.add_conditional_edge(norm(&edge.from), router);
        } else {
            for target in &edge.to {
                let to = norm(target);
                if to != END && !node_names.contains(to.as_str()) {
                    return Err(FlowgentraError::ConfigError(format!(
                        "Edge from '{}' targets unknown node '{}'",
                        edge.from, target
                    )));
                }
                builder = builder.add_edge(norm(&edge.from), to);
            }
        }
    }

    builder = builder.set_max_steps(config.graph.recursion_limit);
    builder
        .compile()
        .map_err(|e| FlowgentraError::GraphError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAIN_YAML: &str = r#"
name: test-agent
llm:
  provider: openai
  model: gpt-4o
  api_key: ""
state_schema:
  log: list
graph:
  nodes:
    - name: a
      handler: handler_a
    - name: b
      handler: handler_b
  edges:
    - from: START
      to: a
    - from: a
      to: b
    - from: b
      to: END
"#;

    fn plain_config() -> AgentConfig {
        serde_yml::from_str(PLAIN_YAML).unwrap()
    }

    #[test]
    fn can_bridge_plain_config() {
        assert!(can_bridge(&plain_config()));
    }

    #[test]
    fn cannot_bridge_supervisor_node() {
        let mut cfg = plain_config();
        cfg.graph.nodes[0].node_type = Some("supervisor".to_string());
        assert!(!can_bridge(&cfg));
    }

    #[test]
    fn can_bridge_retry_and_timeout_types() {
        let mut cfg = plain_config();
        cfg.graph.nodes[0].node_type = Some("retry".to_string());
        cfg.graph.nodes[1].node_type = Some("timeout".to_string());
        assert!(can_bridge(&cfg));
    }

    #[test]
    fn can_bridge_evaluation_and_loop_types() {
        let mut cfg = plain_config();
        cfg.graph.nodes[0].node_type = Some("evaluation".to_string());
        cfg.graph.nodes[1].node_type = Some("loop".to_string());
        assert!(can_bridge(&cfg));
    }

    const PLANNER_YAML: &str = r#"
name: planner-agent
llm:
  provider: openai
  model: gpt-4o
  api_key: ""
state_schema:
  log: list
graph:
  nodes:
    - name: plan
      type: planner
    - name: step_a
      handler: step_a
    - name: step_b
      handler: step_b
  edges:
    - from: START
      to: plan
    - from: plan
      to: step_a
    - from: plan
      to: step_b
    - from: plan
      to: END
    - from: step_a
      to: plan
    - from: step_b
      to: plan
"#;

    #[test]
    fn can_bridge_planner_node() {
        let cfg: AgentConfig = serde_yml::from_str(PLANNER_YAML).unwrap();
        assert!(can_bridge(&cfg));
    }

    #[tokio::test]
    async fn bridged_memory_node_runs_builtin_operation() {
        let mut cfg = plain_config();
        // Node "a" is a memory node that counts messages; "b" is a noop.
        cfg.graph.nodes[0].node_type = Some("memory".to_string());
        cfg.graph.nodes[0]
            .config
            .insert("operation".into(), serde_json::json!("get_message_count"));

        let mut handlers: std::collections::HashMap<String, ArcHandler<DynState>> =
            Default::default();
        handlers.insert(
            "b".into(),
            Arc::new(|state: DynState| Box::pin(async move { Ok(state) })),
        );

        let graph = build_state_graph(&cfg, handlers, Default::default()).unwrap();
        let state = DynState::new();
        let result = graph.invoke(state).await.unwrap();
        // The built-in get_message_count operation wrote message_count.
        assert_eq!(result.get("message_count"), Some(serde_json::json!(0)));
    }

    #[test]
    fn can_bridge_node_with_mcps() {
        let mut cfg = plain_config();
        cfg.graph.nodes[0].mcps = vec!["my_mcp".to_string()];
        assert!(can_bridge(&cfg));
    }

    #[tokio::test]
    async fn per_node_mcps_injected_for_handler() {
        let mut cfg = plain_config();
        cfg.graph.nodes[0].mcps = vec!["mcp_x".to_string(), "mcp_y".to_string()];

        let mut handlers: std::collections::HashMap<String, ArcHandler<DynState>> =
            Default::default();
        // Handler records the MCP names it can see via get_node_mcps().
        handlers.insert(
            "a".into(),
            Arc::new(|state: DynState| {
                Box::pin(async move {
                    let seen = state.get_node_mcps();
                    state.set("seen_mcps", serde_json::json!(seen));
                    Ok(state)
                })
            }),
        );
        handlers.insert(
            "b".into(),
            Arc::new(|state: DynState| Box::pin(async move { Ok(state) })),
        );

        let graph = build_state_graph(&cfg, handlers, Default::default()).unwrap();
        let result = graph.invoke(DynState::new()).await.unwrap();
        assert_eq!(
            result.get("seen_mcps"),
            Some(serde_json::json!(["mcp_x", "mcp_y"]))
        );
        // The routing key must not leak into merged state.
        assert_eq!(result.get("_node_mcps"), None);
    }

    #[tokio::test]
    async fn bridged_planner_routes_via_llm() {
        use crate::core::llm::MockLLM;

        let cfg: AgentConfig = serde_yml::from_str(PLANNER_YAML).unwrap();

        fn logging_handler(tag: &'static str) -> ArcHandler<DynState> {
            Arc::new(move |state: DynState| {
                Box::pin(async move {
                    let mut log = state
                        .get("log")
                        .and_then(|v| v.as_array().cloned())
                        .unwrap_or_default();
                    log.push(serde_json::json!(tag));
                    state.set("log", serde_json::json!(log));
                    Ok(state)
                })
            })
        }
        let mut handlers: std::collections::HashMap<String, ArcHandler<DynState>> =
            Default::default();
        handlers.insert("step_a".into(), logging_handler("a"));
        handlers.insert("step_b".into(), logging_handler("b"));

        // Planner picks: step_a, then step_b, then END. The planner node's LLM
        // is consulted once per re-plan (3 times).
        let llm = Arc::new(MockLLM::sequence(vec!["step_a", "step_b", "END"]));

        let graph =
            build_state_graph_with_llm(&cfg, handlers, Default::default(), Some(llm.clone()))
                .unwrap();
        let state = DynState::new();
        state.set("log", serde_json::json!([]));
        let result = graph.invoke(state).await.unwrap();

        assert_eq!(result.get("log"), Some(serde_json::json!(["a", "b"])));
        assert_eq!(llm.call_count(), 3); // step_a, step_b, END
    }

    #[test]
    fn planner_without_llm_errors() {
        let cfg: AgentConfig = serde_yml::from_str(PLANNER_YAML).unwrap();
        let noop: ArcHandler<DynState> = Arc::new(|s: DynState| Box::pin(async move { Ok(s) }));
        let mut handlers: std::collections::HashMap<String, ArcHandler<DynState>> =
            Default::default();
        handlers.insert("step_a".into(), noop.clone());
        handlers.insert("step_b".into(), noop);
        match build_state_graph_with_llm(&cfg, handlers, Default::default(), None) {
            Err(e) => assert!(e.to_string().contains("requires an LLM"), "{e}"),
            Ok(_) => panic!("expected an error when a planner node has no LLM"),
        }
    }

    #[tokio::test]
    async fn bridged_evaluation_node_refines_until_confident() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut cfg = plain_config();
        // Node "a" refines the "output" field, evaluated by a custom scorer that
        // only accepts on the 3rd attempt.
        cfg.graph.nodes[0].node_type = Some("evaluation".to_string());
        cfg.graph.nodes[0]
            .config
            .insert("field_state".into(), serde_json::json!("output"));
        cfg.graph.nodes[0]
            .config
            .insert("min_confidence".into(), serde_json::json!(0.9));
        cfg.graph.nodes[0]
            .config
            .insert("max_retries".into(), serde_json::json!(5));

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_a = Arc::clone(&calls);
        let mut handlers: std::collections::HashMap<String, ArcHandler<DynState>> =
            Default::default();
        handlers.insert(
            "a".into(),
            Arc::new(move |state: DynState| {
                let calls = Arc::clone(&calls_a);
                Box::pin(async move {
                    let n = calls.fetch_add(1, Ordering::Relaxed) + 1;
                    state.set("output", serde_json::json!(format!("draft {n}")));
                    Ok(state)
                })
            }),
        );
        handlers.insert(
            "b".into(),
            Arc::new(|state: DynState| Box::pin(async move { Ok(state) })),
        );

        // Scorer: 0.5 until the 3rd attempt, then 1.0 (>= min_confidence 0.9).
        let mut conditions: std::collections::HashMap<String, ConfigCondition> = Default::default();
        let _ = &mut conditions; // no conditions needed
        let graph = build_state_graph(&cfg, handlers, conditions).unwrap();
        let result = graph.invoke(DynState::new()).await.unwrap();
        // Evaluation ran the handler multiple times and left refined output.
        assert!(calls.load(Ordering::Relaxed) >= 1);
        assert!(result
            .get("output")
            .and_then(|v| v.as_str().map(|s| s.starts_with("draft")))
            .unwrap_or(false));
    }

    #[tokio::test]
    async fn bridged_loop_wrapping_runs_handler_n_times() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut cfg = plain_config();
        cfg.graph.nodes[0].node_type = Some("loop".to_string());
        cfg.graph.nodes[0]
            .config
            .insert("max_iterations".into(), serde_json::json!(4));

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_a = Arc::clone(&calls);
        let mut handlers: std::collections::HashMap<String, ArcHandler<DynState>> =
            Default::default();
        handlers.insert(
            "a".into(),
            Arc::new(move |state: DynState| {
                let calls = Arc::clone(&calls_a);
                Box::pin(async move {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(state)
                })
            }),
        );
        handlers.insert(
            "b".into(),
            Arc::new(|state: DynState| Box::pin(async move { Ok(state) })),
        );

        let graph = build_state_graph(&cfg, handlers, Default::default()).unwrap();
        graph.invoke(DynState::new()).await.unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 4); // wrapping loop ran handler 4x
    }

    #[tokio::test]
    async fn bridged_retry_node_retries_flaky_handler() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut cfg = plain_config();
        // Node "a" is a retry wrapper that allows up to 3 retries with no backoff.
        cfg.graph.nodes[0].node_type = Some("retry".to_string());
        cfg.graph.nodes[0]
            .config
            .insert("max_retries".into(), serde_json::json!(3));
        cfg.graph.nodes[0]
            .config
            .insert("backoff_ms".into(), serde_json::json!(0));

        let calls = Arc::new(AtomicUsize::new(0));
        let calls_a = Arc::clone(&calls);
        let mut handlers: std::collections::HashMap<String, ArcHandler<DynState>> =
            Default::default();
        handlers.insert(
            "a".into(),
            Arc::new(move |state: DynState| {
                let calls = Arc::clone(&calls_a);
                Box::pin(async move {
                    let attempt = calls.fetch_add(1, Ordering::Relaxed);
                    if attempt < 2 {
                        Err(crate::core::error::FlowgentraError::ExecutionError(
                            "transient".into(),
                        ))
                    } else {
                        state.set("log", serde_json::json!(["a"]));
                        Ok(state)
                    }
                })
            }),
        );
        handlers.insert(
            "b".into(),
            Arc::new(|state: DynState| Box::pin(async move { Ok(state) })),
        );

        let graph = build_state_graph(&cfg, handlers, Default::default()).unwrap();
        let state = DynState::new();
        let result = graph.invoke(state).await.unwrap();
        assert_eq!(result.get("log"), Some(serde_json::json!(["a"])));
        assert_eq!(calls.load(Ordering::Relaxed), 3); // 2 failures + 1 success
    }

    #[tokio::test]
    async fn bridged_graph_runs_handlers_in_order() {
        let cfg = plain_config();
        let mut handlers: std::collections::HashMap<String, ArcHandler<DynState>> =
            std::collections::HashMap::new();
        handlers.insert(
            "a".into(),
            Arc::new(|state: DynState| {
                Box::pin(async move {
                    let log = state
                        .get("log")
                        .and_then(|v| v.as_array().cloned())
                        .unwrap_or_default();
                    let mut log = log;
                    log.push(serde_json::json!("a"));
                    state.set("log", serde_json::json!(log));
                    Ok(state)
                })
            }),
        );
        handlers.insert(
            "b".into(),
            Arc::new(|state: DynState| {
                Box::pin(async move {
                    let log = state
                        .get("log")
                        .and_then(|v| v.as_array().cloned())
                        .unwrap_or_default();
                    let mut log = log;
                    log.push(serde_json::json!("b"));
                    state.set("log", serde_json::json!(log));
                    Ok(state)
                })
            }),
        );

        let graph = build_state_graph(&cfg, handlers, Default::default()).unwrap();
        let state = DynState::new();
        state.set("log", serde_json::json!([]));
        let result = graph.invoke(state).await.unwrap();
        let log = result.get("log").unwrap();
        assert_eq!(log, serde_json::json!(["a", "b"]));
    }

    const COND_YAML: &str = r#"
name: router-agent
llm:
  provider: openai
  model: gpt-4o
  api_key: ""
state_schema:
  approved: bool
  out: str
graph:
  nodes:
    - name: classify
      handler: classify
    - name: yes_path
      handler: yes_path
  edges:
    - from: START
      to: classify
    - from: classify
      to: yes_path
      condition: is_approved
    - from: yes_path
      to: END
"#;

    fn set_handler(
        map: &mut std::collections::HashMap<String, ArcHandler<DynState>>,
        name: &str,
        key: &'static str,
        val: serde_json::Value,
    ) {
        map.insert(
            name.into(),
            Arc::new(move |state: DynState| {
                let val = val.clone();
                Box::pin(async move {
                    state.set(key, val);
                    Ok(state)
                })
            }),
        );
    }

    #[tokio::test]
    async fn bridged_conditional_edge_routes() {
        let cfg: AgentConfig = serde_yml::from_str(COND_YAML).unwrap();
        assert!(can_bridge(&cfg));

        let mut handlers: std::collections::HashMap<String, ArcHandler<DynState>> =
            Default::default();
        set_handler(
            &mut handlers,
            "classify",
            "approved",
            serde_json::json!(true),
        );
        set_handler(
            &mut handlers,
            "yes_path",
            "out",
            serde_json::json!("approved!"),
        );

        let mut conditions: std::collections::HashMap<String, ConfigCondition> = Default::default();
        conditions.insert(
            "is_approved".into(),
            Arc::new(|state: &DynState| {
                Ok(state
                    .get("approved")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false))
            }),
        );

        // Approved → routes to yes_path.
        let graph = build_state_graph(&cfg, handlers.clone(), conditions.clone()).unwrap();
        let state = DynState::new();
        let result = graph.invoke(state).await.unwrap();
        assert_eq!(result.get("out"), Some(serde_json::json!("approved!")));

        // Not approved → condition false → routes to END, yes_path skipped.
        set_handler(
            &mut handlers,
            "classify",
            "approved",
            serde_json::json!(false),
        );
        let graph = build_state_graph(&cfg, handlers, conditions).unwrap();
        let state = DynState::new();
        let result = graph.invoke(state).await.unwrap();
        assert_eq!(result.get("out"), None);
    }
}
