"""
FastAPI application exposing the Banking RAG pipeline via /ask.
"""

import json
from pathlib import Path
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.server.dependencies import get_pipeline
from src.server.schemas import AskRequest, AskResponse, SourceDocument

app = FastAPI(
    title="Banking RAG API",
    version="0.1.0",
    description="Ask banking questions grounded in your private PDFs.",
)

# Allow browser-based tooling (Gradio) to call the API easily.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VECTOR_DB_DIR = ARTIFACTS_DIR / "vector_db"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"
_READY_STATE: Tuple[bool, str] = (False, "Not initialized")


def _check_artifacts() -> Tuple[bool, str]:
    if not VECTOR_DB_DIR.exists():
        return False, f"Missing vector DB at {VECTOR_DB_DIR}"
    if not any(VECTOR_DB_DIR.iterdir()):
        return False, f"Vector DB directory is empty at {VECTOR_DB_DIR}"
    if not MANIFEST_PATH.exists():
        return False, f"Missing manifest at {MANIFEST_PATH}"
    return True, "Ready"


def _load_manifest() -> dict | None:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


@app.on_event("startup")
def warm_pipeline() -> None:
    """
    Build the pipeline once when the server boots so the first request is fast.
    """
    global _READY_STATE
    _READY_STATE = _check_artifacts()
    if _READY_STATE[0]:
        get_pipeline()


def _build_sources(docs, max_items: int = 3) -> List[SourceDocument]:
    """
    Convert retrieved LangChain Documents into serialisable summaries.
    """
    sources: List[SourceDocument] = []

    for doc in docs[:max_items]:
        metadata = doc.metadata or {}
        preview = doc.page_content[:280].strip().replace("\n", " ")
        sources.append(
            SourceDocument(
                source=metadata.get("source"),
                doc_type=metadata.get("doc_type"),
                preview=preview,
            )
        )

    return sources


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    """
    Answer a user question via the RAG pipeline.
    """
    if not _READY_STATE[0]:
        raise HTTPException(status_code=503, detail=_READY_STATE[1])
    pipeline = get_pipeline()

    try:
        history = [{"role": msg.role, "content": msg.content} for msg in payload.history]
        answer, docs = pipeline.answer_question(
            question=payload.question,
            history=history,
            doc_type=payload.doc_type,
        )
        sources = _build_sources(docs)
        return AskResponse(answer=answer, sources=sources)
    except Exception as exc:  # pragma: no cover - FastAPI handles logging
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/ready")
def readiness() -> dict:
    """
    Report whether prebuilt artifacts are present and loadable.
    """
    ready, detail = _READY_STATE
    manifest = _load_manifest() if ready else None
    return {"ready": ready, "detail": detail, "manifest": manifest}


@app.get("/health")
def health() -> dict:
    """
    Lightweight liveness check.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
