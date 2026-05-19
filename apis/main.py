

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="AI Software Engineering Assistant API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo-only in-memory session store (single-process).
_sessions: dict[str, dict] = {}


class CloneBody(BaseModel):
    url: str = Field(..., description="Public git repository URL (HTTPS)")


class CloneResponse(BaseModel):
    session_id: str
    path: str


class ExplainBody(BaseModel):
    code: str
    focus: str = "general"


class ReviewBody(BaseModel):
    code: str


class TestsBody(BaseModel):
    code: str


class DocsBody(BaseModel):
    root: str = Field(..., description="Absolute path to repository root on this machine")


class RAGBuildBody(BaseModel):
    root: str
    session_id: str | None = None


class RAGAskBody(BaseModel):
    session_id: str
    question: str


class BugsBody(BaseModel):
    root: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/clone", response_model=CloneResponse)
def clone_repo(body: CloneBody) -> CloneResponse:
    from tools.github_tool import clone_public_repo

    try:
        dest = clone_public_repo(body.url)
    except Exception as e:  # noqa: BLE001 — demo API
        raise HTTPException(status_code=400, detail=str(e)) from e
    sid = uuid.uuid4().hex
    _sessions[sid] = {"path": str(dest)}
    return CloneResponse(session_id=sid, path=str(dest))


@app.post("/v1/upload")
async def upload_repo(file: UploadFile = File(...)) -> CloneResponse:
    from tools.source_reader import extract_uploaded_zip, resolve_repo_root
    from utils.storage_paths import make_upload_staging_dir

    raw = await file.read()
    dest_parent = make_upload_staging_dir()
    extract_uploaded_zip(raw, dest_parent)
    root = resolve_repo_root(dest_parent)
    sid = uuid.uuid4().hex
    _sessions[sid] = {"path": str(root)}
    return CloneResponse(session_id=sid, path=str(root))


@app.post("/v1/rag/build")
def rag_build(body: RAGBuildBody) -> dict[str, str]:
    from tools.rag_pipeline import ingest_repository

    root = Path(body.root)
    if not root.exists():
        raise HTTPException(status_code=400, detail="root path does not exist")
    sid = body.session_id or uuid.uuid4().hex
    try:
        chain, _retriever, memory = ingest_repository(root, sid)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e)) from e
    _sessions[sid] = {"path": str(root), "chain": chain, "memory": memory}
    return {"session_id": sid, "status": "ready"}


@app.post("/v1/rag/ask")
def rag_ask(body: RAGAskBody) -> dict[str, str]:
    from agents.rag_chat import ask_repository

    data = _sessions.get(body.session_id)
    if not data or "chain" not in data:
        raise HTTPException(status_code=404, detail="Unknown session or RAG not built")
    answer = ask_repository(data["chain"], body.question)
    return {"answer": answer}


@app.post("/v1/explain")
def api_explain(body: ExplainBody) -> dict[str, str]:
    from agents.explain import explain_code

    return {"markdown": explain_code(body.code, focus=body.focus)}


@app.post("/v1/review")
def api_review(body: ReviewBody) -> dict[str, str]:
    from agents.review import review_code

    return {"markdown": review_code(body.code)}


@app.post("/v1/tests")
def api_tests(body: TestsBody) -> dict[str, str]:
    from agents.tests_gen import generate_pytest_tests

    return {"markdown": generate_pytest_tests(body.code)}


@app.post("/v1/documentation")
def api_docs(body: DocsBody) -> dict[str, str]:
    from agents.document import generate_project_docs

    root = Path(body.root)
    if not root.exists():
        raise HTTPException(status_code=400, detail="root path does not exist")
    md = generate_project_docs(root)
    return {"markdown": md}


@app.post("/v1/bugs")
def api_bugs(body: BugsBody) -> dict[str, list]:
    from utils.bug_detector import analyze_directory

    root = Path(body.root)
    if not root.exists():
        raise HTTPException(status_code=400, detail="root path does not exist")
    results = []
    for path, findings in analyze_directory(root):
        results.append(
            {
                "file": path,
                "findings": [
                    {"severity": f.severity, "message": f.message, "line": f.line, "suggestion": f.suggestion}
                    for f in findings
                ],
            }
        )
    return {"results": results}


@app.get("/v1/files/preview")
def preview_files(root: str, limit: int = 30) -> dict:
    from tools.source_reader import load_repo_documents

    path = Path(root)
    if not path.exists():
        raise HTTPException(status_code=400, detail="root path does not exist")
    docs = load_repo_documents(path, max_files=limit)
    return {"files": [d.metadata.get("relpath") for d in docs]}
