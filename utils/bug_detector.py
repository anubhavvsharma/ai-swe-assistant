
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BugFinding:
    severity: str  # "error" | "warning" | "info"
    message: str
    line: int | None
    suggestion: str = ""


def _python_unused_and_antipatterns(tree: ast.AST, source: str) -> list[BugFinding]:
    findings: list[BugFinding] = []
    assigned: set[str] = set()
    used: set[str] = set()

    class V(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used.add(node.id)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            for a in node.args.args:
                used.add(a.arg)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            for a in node.args.args:
                used.add(a.arg)
            self.generic_visit(node)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if node.name:
                used.add(node.name)
            self.generic_visit(node)

    V().visit(tree)

    builtins = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
    for name in sorted(assigned - used):
        if name.startswith("_") or name in {"self", "cls"}:
            continue
        if name in builtins:
            continue
        findings.append(
            BugFinding(
                severity="warning",
                message=f"Potentially unused variable: `{name}`",
                line=None,
                suggestion="Remove the assignment or use the variable; confirm with your test suite.",
            )
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if (
                len(node.body) >= 1
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                if not node.body[0].value.value.strip():
                    findings.append(
                        BugFinding(
                            severity="info",
                            message=f"Empty or placeholder docstring in `{node.name}`",
                            line=node.lineno,
                            suggestion="Add a short docstring describing parameters and return value.",
                        )
                    )

    lines = source.splitlines()
    for i, line in enumerate(lines, start=1):
        if re.search(r"\bprint\s*\(", line):
            findings.append(
                BugFinding(
                    severity="info",
                    message="Use of print() — consider logging for production code.",
                    line=i,
                    suggestion="Replace with `logging.getLogger(__name__).info(...)`.",
                )
            )
            break
    return findings


def analyze_python_file(path: Path) -> list[BugFinding]:
    text = path.read_text(encoding="utf-8", errors="replace")
    findings: list[BugFinding] = []
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        findings.append(
            BugFinding(
                severity="error",
                message=f"Syntax error: {e.msg}",
                line=e.lineno,
                suggestion="Fix the syntax near the reported line and re-run analysis.",
            )
        )
        return findings

    findings.extend(_python_unused_and_antipatterns(tree, text))
    return findings


def analyze_javascriptish_file(path: Path) -> list[BugFinding]:
    text = path.read_text(encoding="utf-8", errors="replace")
    findings: list[BugFinding] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if re.search(r"[^=!]==[^=]", line) and "===" not in line:
            findings.append(
                BugFinding(
                    severity="warning",
                    message="Loose equality (`==`) detected — prefer `===` in JS/TS when possible.",
                    line=i,
                    suggestion="Use strict equality to avoid type coercion bugs.",
                )
            )
            break
    if "var " in text:
        findings.append(
            BugFinding(
                severity="info",
                message="Use of `var` — prefer `let` or `const`.",
                line=None,
                suggestion="Use `const` by default; `let` when reassignment is required.",
            )
        )
    return findings


def analyze_path(path: Path) -> list[BugFinding]:
    path = Path(path)
    if path.suffix.lower() == ".py":
        return analyze_python_file(path)
    if path.suffix.lower() in {".js", ".mjs", ".cjs", ".ts", ".tsx"}:
        return analyze_javascriptish_file(path)
    return []


def analyze_directory(root: Path, max_files: int = 80) -> list[tuple[str, list[BugFinding]]]:
    from utils.code_parser import iter_supported_files

    out: list[tuple[str, list[BugFinding]]] = []
    for i, f in enumerate(iter_supported_files(root)):
        if i >= max_files:
            break
        out.append((str(f), analyze_path(f)))
    return out
