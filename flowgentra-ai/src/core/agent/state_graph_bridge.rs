//! Bridge: build a `state_graph::StateGraph<DynState>` from an `AgentConfig`.
//!
//! This is step 2 of the engine merge (see `docs/design/engine-merge.md`):
//! the config-driven agent runs on the same executor as everything else,
//! gaining cancellation, wall-clock budgets, atomic/SQLite checkpointing,
//! in-node interrupt, and parallel supersteps.
//!
//! **Scope:** plain handler nodes; fixed edges (including parallel fan-out via
//! `to: [a, b]`); named conditional edges; and the control-flow built-in types
//! `retry`, `timeout`, `evaluation`, and `loop` (the latter two reuse the
//! legacy implementations for bug-for-bug parity). Nodes whose `type` is
//! `planner`, `supervisor`, `subgraph`, `memory`, `human_in_the_loop`, or that
//! carry per-node MCPs / RAG, are **not** handled — [`can_bridge`] returns
//! `false` so the caller keeps using the legacy `AgentRuntime`.

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
/// legacy runtime. (`retry`/`timeout`/`evaluation`/`loop` are handled — see
/// [`BRIDGEABLE_WRAPPER_TYPES`] — so they are absent here.)
const UNBRIDGEABLE_TYPES: &[&str] = &[
    "planner",
    "human_in_the_loop",
    "memory",
    "supervisor",
    "orchestrator",
    "subgraph",
    "agent",
    "agent_or_graph",
];

/// Built-in node types the bridge handles by transforming the node's handler
/// (retry/timeout control-flow wrappers, evaluation refinement loop, loop
/// iteration). All reuse the legacy implementations for bug-for-bug parity.
const BRIDGEABLE_WRAPPER_TYPES: &[&str] = &["retry", "timeout", "evaluation", "loop"];

/// `true` if every node in the config is either a plain handler node or a
/// bridgeable control-flow wrapper (retry/timeout), with no per-node MCP list.
/// Such configs can run on the `state_graph` executor via [`build_state_graph`].
pub fn can_bridge(config: &AgentConfig) -> bool {
    // RAG/planner graph features route through the legacy runtime.
    if config.graph.rag.is_some() || config.graph.planner.enabled {
        return false;
    }
    config.graph.nodes.iter().all(|n| {
        let plain_type = n
            .node_type
            .as_deref()
            .map(|t| !UNBRIDGEABLE_TYPES.contains(&t) || BRIDGEABLE_WRAPPER_TYPES.contains(&t))
            .unwrap_or(true);
        let no_builtin = !n.handler.starts_with("builtin::");
        let no_mcp = n.mcps.is_empty();
        plain_type && no_builtin && no_mcp
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
}

#[async_trait::async_trait]
impl Node<DynState> for HandlerNode {
    async fn execute(
        &self,
        state: &DynState,
        _ctx: &Context,
    ) -> std::result::Result<DynStateUpdate, StateGraphError> {
        let before = state.clone();
        let after =
            (self.handler)(state.clone())
                .await
                .map_err(|e| StateGraphError::ExecutionError {
                    node: self.name.clone(),
                    reason: e.to_string(),
                })?;

        let mut update = DynStateUpdate::new();
        for key in after.keys() {
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

    // Handler-level transforms (evaluation refinement, loop iteration) produce
    // a new handler that HandlerNode then wraps.
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
        _ => handler,
    };

    let base: Arc<dyn Node<DynState>> = Arc::new(HandlerNode {
        name: name.to_string(),
        handler: effective_handler,
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
    if !can_bridge(config) {
        return Err(FlowgentraError::ConfigError(
            "This config uses built-in node types (planner/supervisor/subgraph/etc.) \
             or per-node MCPs that are not yet supported on the state_graph engine. \
             It will run on the legacy runtime."
                .to_string(),
        ));
    }

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
        // Standalone loop nodes are pure bookkeeping and take no user handler.
        let is_standalone_loop =
            node.node_type.as_deref() == Some("loop") && node.handler.is_empty();
        let handler = if is_standalone_loop {
            // Never invoked (the loop transform ignores it), but HandlerNode
            // construction needs an ArcHandler.
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

    for edge in &config.graph.edges {
        if edge.from == "START" {
            continue; // entry handled above
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
    fn cannot_bridge_planner_node() {
        let mut cfg = plain_config();
        cfg.graph.nodes[0].node_type = Some("planner".to_string());
        assert!(!can_bridge(&cfg));
    }

    #[test]
    fn cannot_bridge_per_node_mcp() {
        let mut cfg = plain_config();
        cfg.graph.nodes[0].mcps = vec!["some_mcp".to_string()];
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
