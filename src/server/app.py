"""
FastAPI application exposing the Banking RAG pipeline via /ask.
"""

import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock
from typing import List, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from src.server.dependencies import get_pipeline
from src.retrieval.document_retriever import RetrieverConfig
from src.server.schemas import AskRequest, AskResponse, SourceDocument

app = FastAPI(
    title="Banking RAG API",
    version="0.1.0",
    description="Ask banking questions grounded in your private PDFs.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

load_dotenv(override=False)
API_LOGGER = logging.getLogger("banking_rag.api")

_DEMO_MODE = os.getenv("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
_DEMO_API_KEY = os.getenv("DEMO_API_KEY", "").strip()
_MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", "50000"))
_RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1" if _DEMO_MODE else "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "30"))
_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
_DEMO_CORS_ORIGINS = [
    "http://localhost:7860",
    "http://127.0.0.1:7860",
]
_CORS_ENV = os.getenv("DEMO_CORS_ORIGINS", "").strip()

_cors_origins = ["*"]
if _DEMO_MODE:
    _cors_origins = _DEMO_CORS_ORIGINS
    if _CORS_ENV:
        _cors_origins = [origin.strip() for origin in _CORS_ENV.split(",") if origin.strip()]

# Allow browser-based tooling (Gradio) to call the API easily.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_SHOW_GRADIO = os.getenv("SHOW_GRADIO", "").strip().lower() in {"1", "true", "yes", "on"}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts"))).expanduser()
VECTOR_DB_DIR = Path(
    os.getenv("VECTOR_DB_DIR", str(ARTIFACTS_DIR / "vector_db"))
).expanduser()
MANIFEST_PATH = Path(
    os.getenv("MANIFEST_PATH", str(ARTIFACTS_DIR / "manifest.json"))
).expanduser()
_READY_STATE: Tuple[bool, str] = (False, "Not initialized")
_MISSING_ARTIFACTS_MESSAGE = (
    "Artifacts missing. This demo expects prebuilt artifacts. "
    "Run: make build-index (or python -m src.pipeline.build_artifacts)"
)

_REQUEST_ID_HEADER = "X-Request-ID"
_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_RATE_LIMIT_LOCK = Lock()


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip() or "unknown"
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _rate_limit_key(request: Request) -> tuple[str, str]:
    api_key = request.headers.get("x-api-key", "").strip()
    if api_key:
        return f"api_key:{api_key}", "api_key"
    return f"ip:{_client_ip(request)}", "ip"


def _allow_rate_limited_request(key: str, now: float) -> tuple[bool, int]:
    window_start = now - _RATE_LIMIT_WINDOW_SECONDS
    with _RATE_LIMIT_LOCK:
        bucket = _RATE_LIMIT_BUCKETS[key]
        while bucket and bucket[0] <= window_start:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_MAX_REQUESTS:
            retry_after = max(1, int(_RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])))
            return False, retry_after
        bucket.append(now)
    return True, 0


def _check_artifacts() -> Tuple[bool, str]:
    if not VECTOR_DB_DIR.exists():
        return False, _MISSING_ARTIFACTS_MESSAGE
    if not any(VECTOR_DB_DIR.iterdir()):
        return False, _MISSING_ARTIFACTS_MESSAGE
    if not MANIFEST_PATH.exists():
        return False, _MISSING_ARTIFACTS_MESSAGE

    manifest = _load_manifest()
    if manifest is None:
        return False, "Manifest could not be read. Rebuild artifacts."

    provider = manifest.get("embedding_provider")
    model = manifest.get("embedding_model")
    retriever_config = RetrieverConfig()
    if provider:
        expected_provider = retriever_config.provider
        if provider != expected_provider:
            return (
                False,
                "Embedding provider mismatch. "
                f"Manifest={provider}, runtime={expected_provider}. "
                "Rebuild artifacts or update config.",
            )
    if model:
        expected_model = (
            retriever_config.openai_model
            if retriever_config.provider.lower() == "openai"
            else retriever_config.hf_model
        )
        if model != expected_model:
            return (
                False,
                "Embedding model mismatch. "
                f"Manifest={model}, runtime={expected_model}. "
                "Rebuild artifacts or update config.",
            )

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
    if _DEMO_MODE and not _DEMO_API_KEY:
        raise RuntimeError(
            "Demo mode requires DEMO_API_KEY. Set DEMO_API_KEY or disable DEMO_MODE."
        )
    _READY_STATE = _check_artifacts()
    if _DEMO_MODE and not _READY_STATE[0]:
        raise RuntimeError(
            "Demo mode requires prebuilt artifacts. "
            "Build them and ship artifacts/ with the app."
        )
    if _READY_STATE[0]:
        get_pipeline()


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", "").strip() or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    response = None
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        API_LOGGER.info(
            "request request_id=%s method=%s path=%s status=%s latency_ms=%s",
            request_id,
            request.method,
            request.url.path,
            status_code,
            latency_ms,
        )


@app.middleware("http")
async def request_size_guard(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > _MAX_REQUEST_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request too large. Reduce question or history size."},
                )
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header."},
            )
    body = await request.body()
    if body and len(body) > _MAX_REQUEST_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request too large. Reduce question or history size."},
        )
    request._body = body
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request. Check question length, history length, and payload size.",
            "errors": exc.errors(),
        },
    )


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
def ask_question(payload: AskRequest, request: Request) -> AskResponse:
    """
    Answer a user question via the RAG pipeline.
    """
    if not _READY_STATE[0]:
        raise HTTPException(status_code=503, detail=_READY_STATE[1])
    if _DEMO_MODE:
        api_key = request.headers.get("x-api-key")
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API key.")
        if api_key != _DEMO_API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key.")
    if _RATE_LIMIT_ENABLED:
        rate_key, key_scope = _rate_limit_key(request)
        allowed, retry_after = _allow_rate_limited_request(rate_key, now=time.time())
        if not allowed:
            request_id = getattr(request.state, "request_id", "unknown")
            API_LOGGER.warning(
                "rate_limited request_id=%s scope=%s path=/ask retry_after=%s",
                request_id,
                key_scope,
                retry_after,
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later.",
                headers={"Retry-After": str(retry_after)},
            )
    pipeline = get_pipeline()

    try:
        started = time.perf_counter()
        history = [{"role": msg.role, "content": msg.content} for msg in payload.history]
        answer, docs = pipeline.answer_question(
            question=payload.question,
            history=history,
            doc_type=payload.doc_type,
        )
        model_name = getattr(getattr(pipeline, "config", None), "model_name", "unknown")
        request_id = getattr(request.state, "request_id", "unknown")
        ask_latency_ms = int((time.perf_counter() - started) * 1000)
        API_LOGGER.info(
            "ask request_id=%s latency_ms=%s model=%s retrieval_hits=%s status=ok",
            request_id,
            ask_latency_ms,
            model_name,
            len(docs),
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
    if not ready:
        raise HTTPException(status_code=503, detail=detail)
    return {"ready": ready, "detail": detail, "manifest": manifest}


@app.get("/health")
def health() -> dict:
    """
    Lightweight liveness check.
    """
    return {"status": "ok"}


def _mount_gradio() -> None:
    if not (_DEMO_MODE or _SHOW_GRADIO):
        return
    import gradio as gr
    from src.server.gradio_ui import build_interface, DEFAULT_API_URL

    demo = build_interface(DEFAULT_API_URL)
    gr.mount_gradio_app(app, demo, path="/gradio")


@app.get("/")
def root() -> RedirectResponse:
    """
    Send browsers to the Gradio UI while keeping /health available for ALB checks.
    """
    if _DEMO_MODE or _SHOW_GRADIO:
        return RedirectResponse(url="/gradio", status_code=307)
    return RedirectResponse(url="/api/docs", status_code=307)


_mount_gradio()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
