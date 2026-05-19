"""Thin wrapper for RAG question answering."""

from __future__ import annotations

from typing import Any


def ask_repository(chain: Any, question: str) -> str:
    return chain.invoke({"question": question})
