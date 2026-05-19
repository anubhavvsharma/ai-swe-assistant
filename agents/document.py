
from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from agents.llm_config import get_chat_llm
from tools.source_reader import load_repo_documents


def _summarize_tree(root: Path, max_lines: int = 80) -> str:
    lines: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        if any(x in p.parts for x in {".git", "__pycache__", "node_modules", ".venv", "venv"}):
            continue
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        lines.append(str(rel))
        if len(lines) >= max_lines:
            lines.append("... (truncated)")
            break
    return "\n".join(lines)


def generate_project_docs(root: Path, model: str | None = None) -> str:
    llm = get_chat_llm(model=model, temperature=0.25)
    docs = load_repo_documents(Path(root), max_files=40, max_chars_per_file=8000)
    digest = "\n\n".join(f"## {d.metadata.get('relpath')}\n{d.page_content[:3500]}" for d in docs[:25])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You write clear project documentation in Markdown. "
                "Include: title, short README summary, setup instructions (Python venv + pip), "
                "how to run (placeholders if unknown), module overview, and a bullet list of key functions/classes "
                "with one-line descriptions inferred from the snippets.",
            ),
            (
                "human",
                "Project file tree (sample):\n{tree}\n\nRepository digest:\n{digest}",
            ),
        ]
    )
    chain = prompt | llm
    tree = _summarize_tree(Path(root))
    msg = chain.invoke({"tree": tree, "digest": digest})
    return msg.content if hasattr(msg, "content") else str(msg)
