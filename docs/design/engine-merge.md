# Design note: merging the two graph engines

Status: **approved direction (migrate + deprecate), mechanical port pending.**
Owner sign-off given 2026-07-15; this note pins the mapping before the port.

## Problem

Two engines exist (AUDIT.md F-16):

| | `core::graph` + `core::runtime` (legacy) | `core::state_graph` (surviving) |
|---|---|---|
| Used by | `core::agent` (config-driven path) only | Everything else: bindings, prebuilt agents, supervisor |
| Execution | BFS waves; multiple nodes per wave run concurrently | Single current node, sequential |
| Routing | `Edge` + named `Condition` registered post-hoc | `RouterFn` closures on conditional edges |
| Checkpointing | `MemoryCheckpointer` (state::checkpointer family) | `Checkpointer<S>` trait: memory / file / SQLite |
| Budgets | `recursion_limit` per BFS wave | max_steps, max_duration, cancel flag |
| Special nodes | planner (re-plans each wave), per-node MCP lists | tool/LLM/retry/timeout/eval/cached/subgraph nodes |

~4.8k LOC duplicated concepts; three checkpointer families; every reliability
feature (cancellation, wall-clock budget, atomic checkpoints, in-node
interrupt) landed on `state_graph` only — the legacy engine falls further
behind with each release.

## Target

`core::agent` builds a `state_graph::StateGraph<DynState>` from `AgentConfig`.
`core::graph` and `core::runtime` are deprecated at 0.3, deleted at 1.0.

## Mapping

1. **Nodes**: config handlers are `ArcHandler<DynState>` =
   `Fn(DynState) -> Future<Result<DynState>>` (full-state in/out). Wrap each in
   an adapter `Node<DynState>` that diffs returned state against input to emit a
   `DynStateUpdate` (reuse `compute_subgraph_field_delta` semantics from the
   bindings), or — simpler and faithful to current behavior — return a
   full-overwrite update for every key.
2. **Edges**: `from → [to…]` fan-out with optional named condition maps to one
   conditional edge whose router evaluates the registered `Condition` against
   state and returns the first matching target (documented order). Multiple
   unconditional targets from one node = the parallel superstep case (below).
3. **Parallel waves**: `state_graph` needs a superstep mode: when a node has
   multiple outgoing unconditional edges, execute the target set concurrently
   and merge updates with per-key channel reducers (semantics already
   implemented and tested for `ParallelGraphNode` in the bindings — promote
   that merge into the executor). This is the one genuinely missing feature
   and the prerequisite for the port.
4. **Planner nodes**: the runtime's planner loop (re-plan after each wave)
   becomes a conditional edge whose router calls the planner LLM — same shape
   as the bindings' `create_planner_graph_node`.
5. **Per-node MCP lists**: carried via `Context` (`set_node_name` +
   `_node_mcps` already exist) — no engine feature needed.
6. **Checkpointers**: `Agent.run_with_thread` switches from
   `MemoryCheckpointer` to the `Checkpointer<DynState>` trait (memory default,
   file/SQLite opt-in). The `state::checkpointer` family is then deprecated
   with the engine.
7. **Config validation**: cycle detection / termination-path analysis exists in
   both; keep the legacy error messages (they are better) by porting the
   message text into `state_graph`'s compile-time validation.

## Sequencing

1. ✅ Executor superstep support + per-key reducer merge (core, tested).
2. ✅ `AgentConfig → StateGraph<DynState>` builder fn:
   `core::agent::build_state_graph` + `can_bridge`
   (`agent/state_graph_bridge.rs`). Handles plain handler nodes, fixed edges
   (incl. parallel fan-out), and named conditional edges; `can_bridge`
   returns false for planner/supervisor/subgraph/eval/retry/timeout/loop/
   memory/human-in-the-loop node types, per-node MCPs, and RAG/planner graph
   features. Tested (bridge runs handlers in order + conditional routing).
   ✅ `Agent::run*` now selects the bridge when `can_bridge` and falls back to
   the legacy runtime otherwise. Escape hatch:
   `FLOWGENTRA_FORCE_LEGACY_RUNTIME=1`.
   ✅ Control-flow types **retry**, **timeout**, **evaluation**, **loop**
   ported and handled by the bridge (retry/timeout via
   `state_graph::{RetryNode, TimeoutNode}`; evaluation/loop reuse the legacy
   `into_wrapping_node_fn` / `create_loop_standalone_handler` /
   `wrap_handler_with_loop` for bug-for-bug parity).
   ✅ `MockLLM` (`core::llm::mock`) provides offline scripted responses — the
   fixture foundation for porting LLM-driven types.
   **Remaining:** `planner` → router-that-calls-LLM (MockLLM-testable),
   `memory`, per-node `mcp` → Context injection, `subgraph`, and `supervisor`
   (biggest).
3. `#[deprecated]` on `Graph`, `AgentRuntime`, `state::MemoryCheckpointer`
   (rustdoc deprecation-planned notices already in place).
4. Delete at 1.0.

## Risks

- Planner and MCP paths have no offline tests today — port needs recorded
  fixtures first.
- BFS-wave state merge in the legacy engine is last-write-wins and emergent;
  the port makes it reducer-specified, which may change outputs for configs
  that (accidentally) relied on wave ordering. Release-note this.
