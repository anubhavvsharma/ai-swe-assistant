"""Extract functions, classes, and imports from source files (Python + basic JS/TS)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class ParsedSymbol:
    kind: str  # "function" | "class" | "import"
    name: str
    line: int
    snippet: str = ""


@dataclass
class ParsedFile:
    path: str
    language: str
    symbols: list[ParsedSymbol] = field(default_factory=list)
    raw: str = ""


def _detect_language(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext in (".js", ".mjs", ".cjs"):
        return "javascript"
    if ext in (".ts", ".tsx"):
        return "typescript"
    return "other"


def _snippet_lines(text: str, start: int, end: int, max_lines: int = 40) -> str:
    lines = text.splitlines()
    chunk = lines[start - 1 : end]
    if len(chunk) > max_lines:
        chunk = chunk[:max_lines] + ["..."]
    return "\n".join(chunk)


def parse_python_source(path: Path, text: str) -> list[ParsedSymbol]:
    symbols: list[ParsedSymbol] = []
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        symbols.append(
            ParsedSymbol(
                kind="syntax_error",
                name=str(e.msg),
                line=e.lineno or 1,
                snippet=text.splitlines()[e.lineno - 1] if e.lineno else "",
            )
        )
        return symbols

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end = getattr(node, "end_lineno", None) or node.lineno + 5
            symbols.append(
                ParsedSymbol(
                    kind="function",
                    name=node.name,
                    line=node.lineno,
                    snippet=_snippet_lines(text, node.lineno, end),
                )
            )
        elif isinstance(node, ast.ClassDef):
            end = getattr(node, "end_lineno", None) or node.lineno + 8
            symbols.append(
                ParsedSymbol(
                    kind="class",
                    name=node.name,
                    line=node.lineno,
                    snippet=_snippet_lines(text, node.lineno, end),
                )
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                symbols.append(
                    ParsedSymbol(
                        kind="import",
                        name=alias.name,
                        line=node.lineno,
                        snippet=f"import {alias.name}",
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                symbols.append(
                    ParsedSymbol(
                        kind="import",
                        name=f"{mod}.{alias.name}" if mod else alias.name,
                        line=node.lineno,
                        snippet=f"from {mod} import {alias.name}",
                    )
                )
    return symbols


def parse_js_ts_source(path: Path, text: str) -> list[ParsedSymbol]:
    """Lightweight regex-based extraction for JS/TS (good enough for demos)."""
    symbols: list[ParsedSymbol] = []
    for i, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import "):
            symbols.append(ParsedSymbol(kind="import", name=stripped[:120], line=i, snippet=stripped))
    for m in re.finditer(
        r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(",
        text,
        re.MULTILINE,
    ):
        line = text[: m.start()].count("\n") + 1
        symbols.append(ParsedSymbol(kind="function", name=m.group(1), line=line, snippet=m.group(0).strip()))
    for m in re.finditer(
        r"^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
        text,
        re.MULTILINE,
    ):
        line = text[: m.start()].count("\n") + 1
        symbols.append(ParsedSymbol(kind="function", name=m.group(1), line=line, snippet=m.group(0).strip()[:200]))
    for m in re.finditer(r"^\s*(?:export\s+)?class\s+(\w+)", text, re.MULTILINE):
        line = text[: m.start()].count("\n") + 1
        symbols.append(ParsedSymbol(kind="class", name=m.group(1), line=line, snippet=m.group(0).strip()))
    return symbols


def parse_file(path: Path, text: str | None = None) -> ParsedFile:
    path = Path(path)
    lang = _detect_language(path)
    raw = text if text is not None else path.read_text(encoding="utf-8", errors="replace")

    if lang == "python":
        symbols = parse_python_source(path, raw)
    elif lang in ("javascript", "typescript"):
        symbols = parse_js_ts_source(path, raw)
    else:
        symbols = []

    return ParsedFile(path=str(path), language=lang, symbols=symbols, raw=raw)


def iter_supported_files(root: Path, ignore_dirs: set[str] | None = None) -> Iterator[Path]:
    ignore_dirs = ignore_dirs or {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".chroma",
    }
    exts = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx"}
    root = Path(root)
    if not root.exists():
        return
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in ignore_dirs for part in p.parts):
            continue
        if p.suffix.lower() in exts:
            yield p
