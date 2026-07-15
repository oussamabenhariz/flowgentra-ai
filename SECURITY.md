# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately via GitHub Security Advisories
("Report a vulnerability" on the repository's Security tab) or by email to the
maintainer. Do **not** open a public issue for security reports.

You can expect an acknowledgement within 7 days. Please include a minimal
reproduction and the version you tested.

Supported versions: the latest published 0.x release. Older releases do not
receive security fixes.

## Security model in brief

FlowgentraAI executes workflows that act on **untrusted model output**. The
detailed trust boundaries are documented in
[docs/threat-model.md](docs/threat-model.md). Highlights:

- **Config files are trusted input.** A YAML agent config can name Python
  handler modules that get imported (imports execute code). Never load a config
  file from an untrusted source. See the threat model for the exact surface.
- **Built-in tools are safe-by-default.** The calculator is a structured
  arithmetic evaluator (no `eval`). `ShellTool` defaults to a deny-all
  allowlist and executes without a shell in restricted mode;
  `ShellTool::unrestricted` and the Python/Node REPL tools execute arbitrary
  code and must only be given developer-controlled input.
- **Secrets**: `LLMConfig.api_key` is a `Secret` — redacted in `Debug`,
  `Display`, and serde serialization, zeroized on drop. A regression test
  asserts keys cannot leak through formatted output.
- **Checkpoints** are written atomically (temp file + rename) and validated on
  load; `thread_id` is restricted to a single path component to prevent path
  traversal.

## Hardening checklist for deployers

- Expose to each agent only the tools it needs (allowlist, not the full registry).
- Set execution budgets: `set_max_steps`, `set_max_duration`.
- Run file tools with an explicit sandbox root (`FilesTool::new_with_root`).
- Treat retrieved documents as adversarial (prompt injection) — do not wire
  retrieval output into tools with side effects without human review.
- Keep API keys in environment variables, not in config files.
