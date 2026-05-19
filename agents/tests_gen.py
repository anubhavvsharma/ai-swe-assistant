
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from agents.llm_config import get_chat_llm


def generate_pytest_tests(function_code: str, model: str | None = None) -> str:
    llm = get_chat_llm(model=model, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You write minimal pytest tests for the given Python function. "
                "Use plain asserts or pytest.approx where needed. Include imports. "
                "Only output valid Python code in a single markdown code block or raw code.",
            ),
            ("human", "Function to test:\n```python\n{code}\n```"),
        ]
    )
    chain = prompt | llm
    msg = chain.invoke({"code": function_code})
    return msg.content if hasattr(msg, "content") else str(msg)
