

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from langchain_core.documents import Document

from utils.code_parser import iter_supported_files, parse_file


def load_repo_documents(root: Path, max_files: int = 120, max_chars_per_file: int = 12000) -> list[Document]:
    root = Path(root)
    docs: list[Document] = []
    for i, file_path in enumerate(iter_supported_files(root)):
        if i >= max_files:
            break
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > max_chars_per_file:
            text = text[:max_chars_per_file] + "\n\n... [truncated] ..."
        parsed = parse_file(file_path, text)
        header = f"File: {file_path.relative_to(root)}\nLanguage: {parsed.language}\n"
        symbols_lines = []
        for s in parsed.symbols[:80]:
            symbols_lines.append(f"- {s.kind} {s.name} (line {s.line})")
        sym_block = "\n".join(symbols_lines) if symbols_lines else "(no symbols extracted)"
        page = f"{header}\nSymbols:\n{sym_block}\n\n--- Source ---\n{text}"
        docs.append(
            Document(
                page_content=page,
                metadata={
                    "source": str(file_path),
                    "relpath": str(file_path.relative_to(root)),
                    "language": parsed.language,
                },
            )
        )
    return docs


def extract_uploaded_zip(uploaded_bytes: bytes, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(uploaded_bytes)) as zf:
        zf.extractall(dest)
    return dest


def resolve_repo_root(path: Path) -> Path:
    """If the archive unpacks to a single top-level folder, use that as project root."""
    path = Path(path)
    entries = [e for e in path.iterdir() if e.name not in {".DS_Store", "__MACOSX"}]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return path
