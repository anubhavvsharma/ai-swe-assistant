
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import streamlit as st

from agents.document import generate_project_docs
from agents.explain import explain_code
from agents.llm_config import get_settings
from agents.rag_chat import ask_repository
from agents.review import review_code
from agents.tests_gen import generate_pytest_tests
from tools.github_tool import clone_public_repo
from tools.rag_pipeline import ingest_repository
from tools.source_reader import extract_uploaded_zip, resolve_repo_root
from utils.bug_detector import analyze_directory
from utils.storage_paths import get_data_root, make_upload_staging_dir


def _init_session() -> None:
    defaults = {
        "repo_path": None,
        "repo_label": "",
        "rag_chain": None,
        "rag_session_id": None,
        "chat_messages": [],
        "documentation_md": "",
        "bug_markdown": "",
        "status_message": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _require_api_key() -> bool:
    key = (st.session_state.get("google_api_key") or "").strip() or os.getenv("GOOGLE_API_KEY", "")
    if not key:
        return False
    os.environ["GOOGLE_API_KEY"] = key
    return True


def _build_rag_for_path(repo_path: Path) -> None:
    if not _require_api_key():
        st.session_state.status_message = "Set your Google Gemini API key (GOOGLE_API_KEY) in the sidebar."
        return
    sid = st.session_state.rag_session_id or uuid.uuid4().hex
    st.session_state.rag_session_id = sid
    with st.spinner("Indexing repository for RAG (embeddings)…"):
        chain, _retriever, _memory = ingest_repository(
            repo_path,
            sid,
            model=st.session_state.get("chat_model"),
        )
    st.session_state.rag_chain = chain
    st.session_state.status_message = f"Ready — indexed `{repo_path}`."


def _load_github(url: str) -> None:
    if not url.strip():
        st.session_state.status_message = "Enter a repository URL."
        return
    if not _require_api_key():
        st.session_state.status_message = "Set your Google Gemini API key (GOOGLE_API_KEY) in the sidebar."
        return
    try:
        with st.spinner("Cloning repository…"):
            dest = clone_public_repo(url.strip())
    except Exception as e:  # noqa: BLE001
        st.session_state.status_message = f"Clone failed: {e}"
        return
    st.session_state.repo_path = str(dest)
    st.session_state.repo_label = url.strip()
    st.session_state.chat_messages = []
    st.session_state.documentation_md = ""
    st.session_state.bug_markdown = ""
    _build_rag_for_path(Path(dest))
    st.session_state.status_message = f"Loaded Git repository into `{dest}`."


def _load_upload(uploaded) -> None:
    if uploaded is None:
        return
    if not _require_api_key():
        st.session_state.status_message = "Set your Google Gemini API key (GOOGLE_API_KEY) in the sidebar."
        return
    raw = uploaded.getvalue()
    dest_parent = make_upload_staging_dir()
    extract_uploaded_zip(raw, dest_parent)
    root = resolve_repo_root(dest_parent)
    st.session_state.repo_path = str(root)
    st.session_state.repo_label = uploaded.name
    st.session_state.chat_messages = []
    st.session_state.documentation_md = ""
    st.session_state.bug_markdown = ""
    _build_rag_for_path(root)
    st.session_state.status_message = f"Loaded upload `{uploaded.name}` from `{root}`."


def _load_local_path(text: str) -> None:
    p = Path(text.strip())
    if not text.strip():
        st.session_state.status_message = "Enter a folder path."
        return
    if not p.exists() or not p.is_dir():
        st.session_state.status_message = "Path must be an existing directory."
        return
    st.session_state.repo_path = str(p.resolve())
    st.session_state.repo_label = str(p.resolve())
    st.session_state.chat_messages = []
    st.session_state.documentation_md = ""
    st.session_state.bug_markdown = ""
    _build_rag_for_path(p.resolve())
    st.session_state.status_message = f"Loaded local path `{p}`."


def _format_bugs() -> str:
    if not st.session_state.repo_path:
        return ""
    lines: list[str] = []
    for path, findings in analyze_directory(Path(st.session_state.repo_path)):
        if not findings:
            continue
        lines.append(f"### `{path}`")
        for f in findings:
            loc = f"L{f.line}" if f.line else "—"
            lines.append(f"- **{f.severity}** ({loc}): {f.message}")
            if f.suggestion:
                lines.append(f"  - *Suggestion:* {f.suggestion}")
        lines.append("")
    return "\n".join(lines) if lines else "_No issues reported for scanned files._"


def main() -> None:
    st.set_page_config(page_title="AI SE Assistant", layout="wide")
    _init_session()

    st.title("AI Software Engineering Assistant")
    st.caption("Python · LangChain · Google Gemini · Streamlit — explain, document, review, and ask questions about a codebase.")

    with st.sidebar:
        st.subheader("Configuration")
        st.session_state.google_api_key = st.text_input(
            "Google Gemini API key",
            value=os.getenv("GOOGLE_API_KEY", ""),
            type="password",
            help="Create a key at https://aistudio.google.com/apikey — stored in session; use .env for servers.",
        )
        if (st.session_state.google_api_key or "").strip():
            os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key.strip()

        settings = get_settings()
        st.session_state.chat_model = st.text_input(
            "Gemini chat model",
            value=settings.chat_model,
            help="If you see 429 quota errors, try another model (e.g. gemini-2.5-flash-lite, gemini-1.5-flash).",
        )
        st.markdown("**Run FastAPI (optional)**  \n`uvicorn apis.main:app --reload`")

        st.divider()
        st.markdown("**Loaded repository**")
        st.write(st.session_state.repo_label or "_(none)_")
        st.code(st.session_state.repo_path or "—", language="text")

    tab_repo, tab_chat, tab_explain, tab_docs, tab_bugs, tab_review, tab_tests = st.tabs(
        ["Repository", "Chat (RAG)", "Explain code", "Documentation", "Bug detection", "Code review", "Tests"]
    )

    with tab_repo:
        st.caption(
            f"Uploads and clones are stored under **{get_data_root()}** "
            "(set `ASSISTANT_DATA_DIR` in `.env` to change)."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            url = st.text_input("Public Git URL (HTTPS)", placeholder="https://github.com/langchain-ai/langchain.git")
            if st.button("Clone & index", type="primary"):
                _load_github(url)
        with col_b:
            up = st.file_uploader("Upload project (.zip)", type=["zip"])
            if st.button("Load uploaded ZIP"):
                _load_upload(up)

        local = st.text_input("Or local folder path", placeholder=r"C:\path\to\your\project")
        if st.button("Use local folder"):
            _load_local_path(local)

        if st.session_state.status_message:
            st.info(st.session_state.status_message)

    with tab_chat:
        if not st.session_state.rag_chain:
            st.warning("Load a repository in the **Repository** tab first.")
        else:
            for m in st.session_state.chat_messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            q = st.chat_input("Ask a question about this repository…")
            if q and _require_api_key():
                st.session_state.chat_messages.append({"role": "user", "content": q})
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        ans = ask_repository(st.session_state.rag_chain, q)
                    st.markdown(ans)
                st.session_state.chat_messages.append({"role": "assistant", "content": ans})

    with tab_explain:
        code = st.text_area("Paste code to explain", height=220, placeholder="# python / JS / TS …")
        focus = st.selectbox("Focus", ["general", "functions", "classes", "APIs / public surface"])
        if st.button("Generate explanation", key="btn_explain"):
            if not code.strip():
                st.error("Paste some code first.")
            elif not _require_api_key():
                st.error("Add your Google Gemini API key in the sidebar (or set GOOGLE_API_KEY in .env).")
            else:
                with st.spinner("Explaining…"):
                    out = explain_code(code, focus=focus, model=st.session_state.get("chat_model"))
                st.markdown("### Explanation")
                st.markdown(out)

    with tab_docs:
        if not st.session_state.repo_path:
            st.warning("Load a repository first.")
        else:
            if st.button("Generate Markdown documentation", key="btn_docs"):
                if not _require_api_key():
                    st.error("Add your Google Gemini API key in the sidebar (or set GOOGLE_API_KEY in .env).")
                else:
                    with st.spinner("Writing documentation…"):
                        st.session_state.documentation_md = generate_project_docs(
                            Path(st.session_state.repo_path),
                            model=st.session_state.get("chat_model"),
                        )
            if st.session_state.documentation_md:
                st.markdown("### Documentation viewer")
                st.markdown(st.session_state.documentation_md)

    with tab_bugs:
        if not st.session_state.repo_path:
            st.warning("Load a repository first.")
        else:
            if st.button("Run bug / lint scan", key="btn_bugs"):
                st.session_state.bug_markdown = _format_bugs()
            if st.session_state.bug_markdown:
                st.markdown("### Results")
                st.markdown(st.session_state.bug_markdown)

    with tab_review:
        code_r = st.text_area("Code to review", height=240, key="review_code")
        if st.button("Run code review", key="btn_review"):
            if not code_r.strip():
                st.error("Paste code to review.")
            elif not _require_api_key():
                st.error("Add your Google Gemini API key in the sidebar (or set GOOGLE_API_KEY in .env).")
            else:
                with st.spinner("Reviewing…"):
                    rev = review_code(code_r, model=st.session_state.get("chat_model"))
                st.markdown(rev)

    with tab_tests:
        code_t = st.text_area("Python function to cover", height=240, key="test_code")
        if st.button("Generate pytest tests", key="btn_tests"):
            if not code_t.strip():
                st.error("Paste a Python function or module snippet.")
            elif not _require_api_key():
                st.error("Add your Google Gemini API key in the sidebar (or set GOOGLE_API_KEY in .env).")
            else:
                with st.spinner("Generating tests…"):
                    tests = generate_pytest_tests(code_t, model=st.session_state.get("chat_model"))
                st.markdown(tests)


if __name__ == "__main__":
    main()
