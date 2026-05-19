"""Application settings and shared Google Gemini (LangChain) clients."""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Loaded from .env: GOOGLE_API_KEY (see https://aistudio.google.com/apikey)
    google_api_key: str | None = None
    # Default avoids gemini-2.0-flash when free-tier quota for that model is exhausted (429).
    # Override in Streamlit sidebar or set CHAT_MODEL in .env.
    chat_model: str = "gemini-2.5-flash-lite"
    # Gemini embedContent model (legacy "models/embedding-001" returns 404 on current API).
    # Override with env EMBEDDING_MODEL if needed, e.g. gemini-embedding-2-preview.
    embedding_model: str = "gemini-embedding-001"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _ensure_google_env() -> None:
    """Prefer key from process env (e.g. Streamlit sidebar), then .env via settings."""
    if os.getenv("GOOGLE_API_KEY"):
        return
    s = get_settings()
    if s.google_api_key:
        os.environ["GOOGLE_API_KEY"] = s.google_api_key


def get_chat_llm(*, model: str | None = None, temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    _ensure_google_env()
    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=model or s.chat_model,
        temperature=temperature,
    )


def get_embedding_model(*, model: str | None = None) -> GoogleGenerativeAIEmbeddings:
    _ensure_google_env()
    s = get_settings()
    return GoogleGenerativeAIEmbeddings(model=model or s.embedding_model)
