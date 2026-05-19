
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def _persist_dir(session_id: str) -> str:
    h = hashlib.sha256(session_id.encode()).hexdigest()[:16]
    base = Path(tempfile.gettempdir()) / "ai_se_assistant_chroma" / h
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


class RepositoryMemory:
    """Keeps embeddings for one loaded repository in a temporary Chroma folder."""

    def __init__(self, session_id: str, embeddings: Embeddings) -> None:
        self.session_id = session_id
        self._embeddings = embeddings
        self._store: Chroma | None = None

    @property
    def vectorstore(self) -> Chroma | None:
        return self._store

    def build(self, documents: list[Document]) -> Chroma:
        persist = _persist_dir(self.session_id)
        self._store = Chroma.from_documents(
            documents=documents,
            embedding=self._embeddings,
            persist_directory=persist,
        )
        return self._store

    def as_retriever(self, k: int = 6) -> Any:
        if not self._store:
            raise RuntimeError("Memory not built — load documents first.")
        return self._store.as_retriever(search_kwargs={"k": k})
