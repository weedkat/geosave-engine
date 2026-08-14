"""Fail on docstring/comment bloat CLAUDE.md's Docstring Writing Guide bans.

Mechanical gate, not a style suggestion — run this on any file with a
touched docstring/comment before calling that work done. Checks:
  - Docstring body (summary line + blank, up to Args/Returns/Raises/
    Examples) is at most MAX_BODY_LINES lines.
  - No run of MAX_COMMENT_LINES+ consecutive standalone "#" comment lines.

Usage: python scripts/check_docstrings.py <file_or_dir> [...]
Exit 0 if clean, 1 with one finding per line if not.
"""
from __future__ import annotations

import ast
import sys
import tokenize
from pathlib import Path

MAX_BODY_LINES = 3
MAX_COMMENT_LINES = 1
SECTION_HEADERS = ("Args:", "Returns:", "Raises:", "Examples:", "Yields:", "Attributes:")


def _docstring_findings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            continue
        lines = doc.splitlines()
        body_end = next((i for i, line in enumerate(lines) if line.strip() in SECTION_HEADERS), len(lines))
        body = [line for line in lines[1:body_end] if line.strip()]
        if len(body) > MAX_BODY_LINES:
            name = getattr(node, "name", "<module>")
            lineno = getattr(node, "lineno", 1)
            findings.append(f"{path}:{lineno}: {name}'s docstring body is {len(body)} lines, max {MAX_BODY_LINES}")
    return findings


def _comment_findings(path: Path) -> list[str]:
    """Flag a run of standalone "#" lines — a trailing inline comment doesn't count."""
    findings = []
    run_start: int | None = None
    run_len = 0
    with open(path, "rb") as f:
        tokens = tokenize.tokenize(f.readline)
        prev_line = -2
        for tok in tokens:
            is_standalone = tok.type == tokenize.COMMENT and tok.line.strip().startswith("#")
            if not is_standalone:
                continue
            line = tok.start[0]
            if line == prev_line + 1:
                run_len += 1
            else:
                if run_len > MAX_COMMENT_LINES:
                    findings.append(f"{path}:{run_start}: {run_len}-line comment run, max {MAX_COMMENT_LINES}")
                run_start, run_len = line, 1
            prev_line = line
        if run_len > MAX_COMMENT_LINES:
            findings.append(f"{path}:{run_start}: {run_len}-line comment run, max {MAX_COMMENT_LINES}")
    return findings


def check(path: Path) -> list[str]:
    """Run both checks on one .py file."""
    return _docstring_findings(path) + _comment_findings(path)


def main(argv: list[str]) -> int:
    targets: list[Path] = []
    for arg in argv:
        p = Path(arg)
        targets.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])

    findings = [f for t in targets for f in check(t)]
    for f in findings:
        print(f)
    if findings:
        print(f"\n{len(findings)} violation(s).")
        return 1
    print("clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
