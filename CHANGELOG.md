# Changelog

All notable changes to the `flowgentra-ai` crate. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows SemVer
(0.x: minor bumps may break).

## [0.3.2] - 2026-07-19

No functional changes in the core crate — version-sync release with the
Python bindings 0.3.2 (which fix the `Conversational.run()` and
`document_loaders` wrapper regressions).

## [0.3.1] - 2026-07-17

The config-driven `Agent` now runs on the `state_graph` executor for every
valid config. The legacy `Graph`/`AgentRuntime` engine is deprecated and built
only as a fallback.

### Security
- Dependency advisories cleared (`cargo deny` green): quick-xml 0.41
  (RUSTSEC-2026-0194/0195 DoS fixes; ArXiv Atom parsing ported to the new
  entity-event API with a regression test), pdf-extract 0.12 / lopdf 0.42
  (RUSTSEC-2026-0187 stack overflow), scraper 0.27 (drops unmaintained
  fxhash), dotenv → dotenvy (maintained fork), crossbeam-epoch 0.9.20.
  Remaining: ttf-parser unmaintained (transitive via the fixed lopdf; ignored
  in deny.toml with rationale).

### Changed
- **BREAKING**: `Agent::runtime_mut()` returns `Option<&mut AgentRuntime>`.
  The runtime is now built only when the state_graph bridge cannot compile the
  config (an unresolvable handler, or a supervisor cycle) or when the legacy
  engine is forced, so it is `None` for essentially every real config. Callers
  that unconditionally used the returned reference must handle `None`.
  `set_checkpointer`, `with_checkpointer`, `register_memory_node`, and the
  auto-evaluation middleware are no-ops on the bridged path.
- `flowgentra-ai-macros` is now resolved from the workspace via `path`. The
  dependency previously pinned `0.2.5` with no path, so the crate compiled
  against the published macros crate and local edits to the workspace member
  had no effect.

### Deprecated
- `core::runtime::AgentRuntime` and `core::graph::Graph` — use
  `core::state_graph::StateGraph`. Both are removed at 1.0 along with the
  `FLOWGENTRA_FORCE_LEGACY_RUNTIME` escape hatch.

### Added
- The bridge covers every built-in node type: `retry`, `timeout`,
  `evaluation`, `loop`, `planner`, `memory`, `human_in_the_loop`, `subgraph`,
  and `supervisor`/`orchestrator`, plus per-node MCP lists and RAG configs.
  Supervisors nest recursively (cycles are rejected at build time). Wrapped
  node types reuse the legacy handler builders, so behavior matches the old
  engine bug-for-bug.
- `core::llm::mock::MockLLM` — scripted offline LLM (`always`, `sequence`,
  `when_contains`, `when`, `otherwise`, usage, streaming, `call_count`) for
  deterministic tests of LLM-driven paths such as planner routing.
- Panic-site CI gate (`scripts/check_panics.py`, run in the lint job): every
  `.unwrap()`/`.expect()`/`panic!`/`unreachable!`/`todo!` in production code
  must carry a `// PANIC-OK: <reason>` marker, so a new bare unwrap on a
  fallible value trips CI (F-9).
- `FilesTool::try_default()` — fallible sandbox constructor for callers that
  need to detect an unresolvable current directory instead of degrading.
- Cost budget (F-10): `StateGraphBuilder::set_max_cost(usd)` +
  `StateGraphError::CostBudgetExceeded`, enforced between nodes against a
  `_cost_usd` state field. `observability::record_usage_with_cost` accumulates
  per-call cost at each call's model price, so multi-model runs are summed
  correctly; unpriced models count as $0 and warn once. `llm::set_model_price`
  overrides the built-in table without a release. Config gains a `budget`
  section (`max_tokens`/`max_cost_usd`/`max_duration_secs`) and a
  `model_pricing` map — the bridge now applies token/cost/duration budgets to
  config-driven agents (previously only `max_steps` was wired).

### Removed
- Dead tombstone modules under `core::state`: `dynamic`, `scoped`, `shared`,
  `state_ext`, `typed`. Each was an empty file exporting nothing.
- Orphan `core::examples` module (never declared, never compiled).

### Fixed
- `CachedNode` hashed serialized state with nondeterministic map ordering, so
  equivalent states could miss the cache. Hashing is now canonical (sorted
  keys).
- Per-node MCP injection mutated the executor's live state, because cloning a
  `DynState` shares the inner `Arc<RwLock>`. The bridge now deep-clones the
  node input and keeps `_node_mcps` out of the emitted update.
- Reachable panics removed (F-9): `ToolRegistry::with_builtins` no longer
  `.expect()`s each registration; the default file-tool sandbox falls back to
  the temp dir (then a deny-all root) instead of panicking when the current
  directory is unavailable; `OptimizedState::as_mut` uses `Arc::make_mut`,
  which cannot panic on a live `Weak`.

## [0.3.0] - 2026-07-16

### Security
- `LLMConfig.api_key` is now a `Secret`: redacted in `Debug`/`Display` and all
  serde serialization, zeroized on drop, read via `.expose()`. Checkpoints no
  longer contain raw API keys; `DynState::get_llm()` re-resolves the key from
  the provider's environment variable.
- Checkpoint `thread_id` validated as a single path component (path-traversal
  fix). `ShellTool` restricted mode uses a quote-aware tokenizer; timed-out
  subprocesses are killed (`kill_on_drop`); unrestricted mode uses `cmd /C`
  on Windows.
- New `SECURITY.md` and `docs/threat-model.md`.

### Added
- Parallel supersteps: multiple fixed edges from a node run concurrently and
  merge by per-field reducer in sorted node order (`execute_superstep`).
- `set_max_tokens` — total-token budget (`TokenBudgetExceeded`).
- In-node `interrupt()` for human-in-the-loop.
- `build_state_graph`/`can_bridge` — compile an AgentConfig onto the
  state_graph executor (engine-merge step 2).
- RAG/embeddings config API keys redacted on serialize + re-resolved from env
  (checkpoint leak fixed, matching the LLM key).
- `StateGraphBuilder::set_max_duration` — wall-clock budget
  (`StateGraphError::WallClockExceeded`).
- `StateGraphBuilder::set_cancel_flag` — cooperative cancellation
  (`StateGraphError::Cancelled`); powers Python Ctrl+C.
- `state_graph::interrupt(payload)` — in-node human-in-the-loop pause
  (`StateGraphError::InterruptedByNode`); resume re-runs the interrupted node.
- `SqliteCheckpointer` (behind the `sqlite` feature) — durable transactional
  checkpointing via the shared `Checkpointer` trait.
- `CachedNode` — input-state-keyed node memoization with TTL and size bound.
- Criterion benchmarks (`benches/core_benches.rs`); baselines in the
  repo-root BENCHMARKS.md.
- CI: fmt/clippy(-D warnings)/tests on 3 OSes, feature-gate checks,
  cargo-deny.

### Fixed
- `resume()`/`resume_with_update()` re-ran the graph from the entry point and
  re-triggered the pausing breakpoint forever; they now continue after the
  last checkpointed node (or re-run the interrupt()ing node).
- Checkpoint files are written atomically (temp + rename) with a
  `schema_version` field; corrupt files fail with an error naming the file.
- Panic sweep: unwraps on LLM responses, retry loops, schema validation, and
  clock math replaced with non-panicking handling.

### Deprecation notices
- `core::graph::Graph` and `core::runtime::AgentRuntime` are planned for
  removal at 1.0 — see `docs/design/engine-merge.md`.
