"""High-level code review suggestions."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from agents.llm_config import get_chat_llm


def review_code(code: str, model: str | None = None) -> str:
    llm = get_chat_llm(model=model, temperature=0.35)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior engineer doing a friendly code review. Output markdown with sections: "
                "Summary, Readability, Possible bugs/edge cases, Clean code / style, "
                "Suggested refactors (small, practical). Be constructive.",
            ),
            ("human", "Review this code:\n```\n{code}\n```"),
        ]
    )
    chain = prompt | llm
    msg = chain.invoke({"code": code})
    return msg.content if hasattr(msg, "content") else str(msg)
