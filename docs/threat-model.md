# FlowgentraAI Threat Model

This document states what the framework protects against, what it does not,
and where the trust boundaries sit. It applies to both the Rust crate
(`flowgentra-ai`) and the Python package (`flowgentra-ai` on PyPI).

## Actors and trust levels

| Input | Trust level | Why |
|---|---|---|
| Application code (Rust/Python) | Trusted | It links the library; it can do anything anyway. |
| Agent config YAML | **Trusted** | Configs can name Python handler modules that are imported at load time — importing executes code. Loading a config is equivalent to running a script. |
| LLM output (completions, tool calls) | **Untrusted** | Models can be manipulated via prompt injection; treat every completion as adversarial. |
| Retrieved documents (RAG) | **Untrusted** | Classic indirect prompt-injection vector. |
| Checkpoint files | Semi-trusted | Written by the framework, but an attacker with file access can tamper. Loads are bounds-checked and never execute code. |
| Tool arguments | **Untrusted** | Chosen by the LLM. Validated against the tool's JSON schema before dispatch. |

## Trust boundary 1: config loading = code execution

`Agent.from_config_path()` (Python) supports two mechanisms that import code
named by the YAML file:

1. `python_handler_module: <module>` — the module is imported and scanned for
   `@register_handler` functions. **Import executes the module's top-level code.**
2. `handler: python.<module>:<function>` on a node — the module is imported.

Consequence: **a config file is a program.** Loading a YAML file from an
untrusted source (download, shared repo, user upload) is remote code execution
by design. The Python API requires opting in to handler imports
(`allow_python_handlers`, warned since 0.2.7) so this cannot happen silently.

Mitigations:
- Only load configs you or your CI wrote.
- `${ENV_VAR}` interpolation substitutes values only; it never evaluates code.
- Rust handler registration (`#[register_handler]`) happens at compile time and
  is not reachable from config content.

## Trust boundary 2: tool execution

Tools run with the process's privileges on arguments chosen by an untrusted
model. Defenses, per layer:

- **Schema validation**: tool args are validated against the tool's
  `JsonSchema` (type, required fields, min/max, length) before dispatch.
- **Calculator**: structured operations on two numbers. No `eval`, no shell.
- **File tools**: sandboxed to a root directory; paths are canonicalized and
  must remain inside the root; `..` components are rejected.
- **ShellTool**: default deny-all allowlist. Restricted mode tokenizes
  (quote-aware) and spawns directly — no shell, so metacharacters are inert.
  `unrestricted()` runs `sh -c` / `cmd /C` and is documented as
  developer-input-only.
- **REPL tools** (`PythonReplTool`, `NodeJsReplTool`): arbitrary code execution
  by design. Do not expose them to model-chosen input outside a sandboxed
  environment (container, VM). The framework does not provide OS-level
  sandboxing.
- **Timeouts**: subprocess tools enforce a timeout and kill the child on breach.

What the framework does **not** do: OS sandboxing, network egress filtering
for tools, resource quotas (CPU/memory). If you need those walls, run the agent
in a container.

## Trust boundary 3: model providers and network

- API keys live in `Secret` wrappers: redacted from `Debug`/`Display`/serde,
  zeroized on drop, exposed only when building auth headers.
- Keys resolve from explicit config, provider env vars, or a `.env` file in the
  working directory — in that order.
- `base_url` / custom providers: the framework sends requests to whatever
  endpoint the (trusted) config names. It does not currently block link-local
  or metadata addresses; if configs can be influenced by less-trusted parties,
  filter egress at the network layer. (Tracked as a hardening candidate.)

## Trust boundary 4: persistence

- Checkpoints are JSON; loading them never executes code (no pickle anywhere).
- Writes are atomic (temp + rename); a crash cannot truncate an existing
  checkpoint. Corrupt files fail with a typed error naming the file.
- `thread_id` must be a single path component — `../` traversal is rejected.
- Checkpoint files are **not** encrypted or signed. Anything in graph state is
  written in plaintext; do not put secrets in state.

## Runaway protection

Agents can loop. Budgets, all enforced between nodes:

- `set_max_steps(n)` — step budget (default 1000).
- `set_max_duration(d)` — wall-clock budget.
- `set_cancel_flag(flag)` — cooperative cancellation (used by Python Ctrl+C).
- Nesting depth limit for subgraph/supervisor recursion
  (`FLOWGENTRA_MAX_NESTING`, default 25).

Token/cost budgets are not yet enforced by the executor; track usage via
`TokenUsage`/`estimated_cost` and enforce in middleware if you need a hard cap.

## Known gaps (tracked, not yet mitigated)

- No SSRF guardrails on configurable endpoints (see boundary 3).
- No per-tool resource caps beyond timeouts.
- `State.from_json` bounds: size/depth limits enforced in the Python bindings;
  the Rust `serde_json` path relies on serde's recursion limit (128).
