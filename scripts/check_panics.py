#!/usr/bin/env python3
"""Fail CI on unjustified panic sites in production code (audit F-9).

A "panic site" is a call that aborts the process on the unhappy path:
`.unwrap()`, `.expect(`, `panic!(`, `unreachable!(`, `todo!(`,
`unimplemented!(`. Every one that survives in non-test code must carry a
`// PANIC-OK: <reason>` marker on, or within a few lines above, the site.
That forces a human to write down why the panic cannot fire (a local
invariant, a compile-time constant, an unrecoverable environment failure)
instead of reaching for `.unwrap()` on a fallible runtime value.

Scope: `flowgentra-ai/src/**/*.rs`, excluding each file's test module. Tests
legitimately use `.unwrap()` everywhere, so everything from the first
`#[cfg(test)]` / `#[cfg(all(test, ...))]` attribute to end-of-file is skipped
(the repo convention is tests-at-the-bottom).

Usage:
    python scripts/check_panics.py          # check, exit 1 on violations
    python scripts/check_panics.py --list    # list every justified site too

Exit status is non-zero if any unjustified site is found.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Order matters only for reporting; each pattern is matched independently.
PANIC_RE = re.compile(
    r"\.unwrap\(\)|\.expect\(|panic!\(|unreachable!\(|todo!\(|unimplemented!\("
)
TEST_CUTOFF_RE = re.compile(r"#\[cfg\(\s*(all\(\s*)?test\b")
MARKER = "PANIC-OK"
# How many lines below a marker it may still justify a panic site. Method
# chains (builder().a().b().expect(...)) put the token a few lines under the
# marker, so this is generous but bounded.
ARM_WINDOW = 10


def is_comment(line: str) -> bool:
    return line.lstrip().startswith("//")


def check_file(path: Path) -> list[tuple[int, str]]:
    """Return a list of (line_number, text) for unjustified panic sites."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    # Find where test code starts; scan only production code above it.
    cutoff = len(lines)
    for i, line in enumerate(lines):
        if TEST_CUTOFF_RE.search(line):
            cutoff = i
            break

    violations: list[tuple[int, str]] = []
    armed_until = -1  # last line index (inclusive) a live marker still covers
    for i in range(cutoff):
        line = lines[i]
        if MARKER in line:
            armed_until = i + ARM_WINDOW
            continue
        if is_comment(line):
            continue
        if PANIC_RE.search(line):
            if i <= armed_until:
                armed_until = -1  # consume the marker
            else:
                violations.append((i + 1, line.strip()))
    return violations


def justified_sites(path: Path) -> int:
    """Count panic sites preceded by a marker (for --list reporting)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cutoff = len(lines)
    for i, line in enumerate(lines):
        if TEST_CUTOFF_RE.search(line):
            cutoff = i
            break
    count = 0
    armed_until = -1
    for i in range(cutoff):
        line = lines[i]
        if MARKER in line:
            armed_until = i + ARM_WINDOW
            continue
        if is_comment(line):
            continue
        if PANIC_RE.search(line) and i <= armed_until:
            count += 1
            armed_until = -1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="also report justified sites per file"
    )
    parser.add_argument(
        "--src",
        default=str(Path(__file__).resolve().parent.parent / "flowgentra-ai" / "src"),
        help="source root to scan",
    )
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        print(f"error: source root not found: {src}", file=sys.stderr)
        return 2

    all_violations: list[tuple[Path, int, str]] = []
    total_justified = 0
    for path in sorted(src.rglob("*.rs")):
        for lineno, text in check_file(path):
            all_violations.append((path, lineno, text))
        if args.list:
            j = justified_sites(path)
            if j:
                total_justified += j
                print(f"  {path.relative_to(src)}: {j} justified")

    if args.list:
        print(f"justified panic sites: {total_justified}")

    if all_violations:
        print(
            f"\nERROR: {len(all_violations)} unjustified panic site(s) in production code.",
            file=sys.stderr,
        )
        print(
            "Add a `// PANIC-OK: <reason>` marker on/above the line, or return a "
            "typed error instead.\n",
            file=sys.stderr,
        )
        for path, lineno, text in all_violations:
            rel = path.relative_to(src)
            print(f"  {rel}:{lineno}: {text}", file=sys.stderr)
        return 1

    print("OK: no unjustified panic sites in production code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
