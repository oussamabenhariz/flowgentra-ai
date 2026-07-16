# Changelog

All notable changes to the `flowgentra-ai` crate. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows SemVer
(0.x: minor bumps may break).

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
