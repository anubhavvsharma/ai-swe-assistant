
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from agents.llm_config import get_chat_llm


def explain_code(code: str, focus: str = "general", model: str | None = None) -> str:
    llm = get_chat_llm(model=model, temperature=0.3)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You explain source code to junior developers. Use plain language, short sections, "
                "and markdown. Cover purpose, main logic, inputs/outputs, and one gotcha if relevant.",
            ),
            (
                "human",
                "Focus: {focus}\n\nCode:\n```\n{code}\n```",
                
            ),
        ]
    )
    chain = prompt | llm
    msg = chain.invoke({"code": code, "focus": focus})
    return msg.content if hasattr(msg, "content") else str(msg)
