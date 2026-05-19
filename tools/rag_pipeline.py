

from __future__ import annotations

from operator import itemgetter
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agents.llm_config import get_chat_llm, get_embedding_model
from tools.source_reader import load_repo_documents
from utils.memory_store import RepositoryMemory


def chunk_documents(docs: list[Document], chunk_size: int = 1200, overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)


def build_rag_chain(session_id: str, docs: list[Document], model: str | None = None):
    chunks = chunk_documents(docs)
    embeddings = get_embedding_model()
    memory = RepositoryMemory(session_id=session_id, embeddings=embeddings)
    vs = memory.build(chunks)
    retriever = vs.as_retriever(search_kwargs={"k": 8})
    llm = get_chat_llm(model=model, temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a patient software engineering tutor. Answer using the context snippets. "
                "If the answer is not in the context, say you are not sure and suggest what file to open. "
                "Use short markdown with bullet points when helpful.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}",
            ),
        ]
    )

    def format_docs(docs_: list[Document]) -> str:
        parts = []
        for d in docs_:
            rel = d.metadata.get("relpath", d.metadata.get("source", ""))
            parts.append(f"### {rel}\n{d.page_content}")
        return "\n\n".join(parts)

    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever, memory


def ingest_repository(root: Path, session_id: str, model: str | None = None):
    docs = load_repo_documents(Path(root))
    if not docs:
        raise ValueError("No supported source files found (.py, .js, .ts, .tsx).")
    return build_rag_chain(session_id, docs, model=model)
