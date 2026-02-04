"""
Gradio chat interface connected to the FastAPI /ask endpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import gradio as gr
import httpx

from src.server.schemas import AskResponse, AskRequest

DEFAULT_API_URL = os.getenv("API_BASE_URL", "").strip()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SAMPLE_QUESTIONS_PATH = ARTIFACTS_DIR / "sample_questions.json"
DEMO_MODE = os.getenv("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
DEMO_API_KEY = os.getenv("DEMO_API_KEY", "").strip()


ChatHistory = Sequence[Union[dict, Tuple[str, str]]]


def _coerce_content_to_text(content: object) -> str:
    """
    Convert Gradio Chatbot content payloads into plain text for AskRequest.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join([p for p in parts if p])
    return str(content)


def _history_to_messages(history: ChatHistory | None) -> List[dict]:
    """
    Convert Gradio history objects into the format expected by /ask.
    """
    messages: List[dict] = []
    if not history:
        return messages

    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role")
            content = _coerce_content_to_text(entry.get("content"))
            if role and content is not None:
                messages.append({"role": role, "content": content})
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            user_msg, bot_msg = entry
            if user_msg:
                messages.append({"role": "user", "content": _coerce_content_to_text(user_msg)})
            if bot_msg:
                messages.append({"role": "assistant", "content": _coerce_content_to_text(bot_msg)})
    return messages


def _history_to_message_list(history: ChatHistory | None) -> List[dict]:
    """
    Normalize Gradio history into message dictionaries.
    """
    return _history_to_messages(history)

def _resolve_api_url(api_url: str, request: gr.Request | None) -> str:
    """
    Resolve the API base URL for calls from the Gradio backend.
    """
    if api_url:
        return api_url.rstrip("/")

    if request is not None:
        proto = request.headers.get("x-forwarded-proto") or "http"
        host = request.headers.get("x-forwarded-host") or request.headers.get("host")
        if host:
            return f"{proto}://{host}".rstrip("/")

    return "http://127.0.0.1:8000"


def _format_answer(response: AskResponse) -> str:
    """
    Append source metadata to the answer for convenient referencing in the UI.
    """
    answer_lines = [response.answer.strip()]
    if response.sources:
        answer_lines.append("")
        answer_lines.append("Sources:")
        for idx, src in enumerate(response.sources, start=1):
            doc_label = src.doc_type or "document"
            source_name = src.source or "N/A"
            answer_lines.append(f"{idx}. {doc_label} :: {source_name}")
    return "\n".join(answer_lines)


def _chat_response(
    message: str,
    history: ChatHistory | None,
    doc_type: str | None,
    api_url: str,
    request: gr.Request | None,
) -> List[dict]:
    """
    Call the FastAPI endpoint and return the updated chat history.
    """
    history_messages = _history_to_message_list(history)
    payload = AskRequest(
        question=message,
        doc_type=None if not doc_type or doc_type == "All" else doc_type,
        history=history_messages,
    ).model_dump()

    resolved_api_url = _resolve_api_url(api_url, request)
    try:
        headers = {"x-api-key": DEMO_API_KEY} if DEMO_API_KEY else None
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{resolved_api_url}/ask",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            ask_response = AskResponse(**resp.json())
            answer = _format_answer(ask_response)
    except Exception as exc:  # pragma: no cover - UI feedback only
        answer = f"Error talking to API: {exc}"

    updated_history = list(history_messages)
    updated_history.append({"role": "user", "content": message})
    updated_history.append({"role": "assistant", "content": answer})
    return updated_history


def _get_ready_payload(api_url: str, request: gr.Request | None) -> tuple[dict | None, str | None]:
    try:
        resolved_api_url = _resolve_api_url(api_url, request)
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{resolved_api_url}/ready")
            if resp.status_code == 503:
                detail = resp.json().get("detail", "Artifacts not ready")
                return None, f"Corpus not ready: {detail}"
            resp.raise_for_status()
            return resp.json(), None
    except Exception as exc:  # pragma: no cover - UI feedback only
        return None, f"Corpus info unavailable: {exc}"


def _fetch_corpus_info(api_url: str, payload: dict | None = None) -> str:
    """
    Fetch manifest-backed corpus info from the API /ready endpoint.
    """
    if payload is None:
        payload, error = _get_ready_payload(api_url, None)
        if error:
            return error

        if not payload or not payload.get("ready"):
            return f"Corpus not ready: {payload.get('detail', 'Unknown error') if payload else 'Unknown error'}"

    manifest = payload.get("manifest") or {}
    corpus_name = manifest.get("corpus_name", "Corpus")
    built_at = manifest.get("built_at", "unknown")
    doc_count = manifest.get("document_count", "unknown")
    doc_types = manifest.get("document_types", {})
    embedding_model = manifest.get("embedding_model", "unknown")
    chunk_size = manifest.get("chunk_size", "unknown")
    chunk_overlap = manifest.get("chunk_overlap", "unknown")

    lines = [
        f"### {corpus_name}",
        f"- Built at: {built_at}",
        f"- Documents: {doc_count}",
        f"- Embedding model: {embedding_model}",
        f"- Chunk size / overlap: {chunk_size} / {chunk_overlap}",
    ]
    if doc_types:
        joined = ", ".join(f"{k} ({v})" for k, v in doc_types.items())
        lines.append(f"- Doc types: {joined}")
    return "\n".join(lines)


def _fetch_corpus_info_and_doc_types(api_url: str, request: gr.Request | None = None):
    """
    Fetch manifest and return markdown plus dropdown update.
    """
    payload, error = _get_ready_payload(api_url, request)
    if error or not payload:
        info = error or "Corpus info unavailable: Unknown error"
        return info, gr.update(choices=["All"], value="All")

    info = _fetch_corpus_info(api_url, payload=payload)
    manifest = payload.get("manifest") or {}
    doc_types = list((manifest.get("document_types") or {}).keys())
    choices = ["All"] + sorted(doc_types)
    return info, gr.update(choices=choices, value="All")

def _load_suggested_questions() -> list[str]:
    if not SAMPLE_QUESTIONS_PATH.exists():
        return []
    try:
        payload = json.loads(SAMPLE_QUESTIONS_PATH.read_text(encoding="utf-8"))
        questions = []
        for row in payload:
            question = row.get("question")
            if question:
                questions.append(str(question))
        return questions
    except Exception:
        return []


def build_interface(default_api_url: str = DEFAULT_API_URL) -> gr.Blocks:
    """
    Construct the Gradio Blocks app connected to the FastAPI backend.
    """
    with gr.Blocks(title="Banking RAG Chat") as demo:
        if DEMO_MODE:
            gr.Markdown(
                """
                ## Banking RAG Demo
                Prebuilt corpus is bundled with the app. Just start the API and chat.
                """.strip()
            )
        else:
            gr.Markdown(
                """
                ## Banking RAG Chat
                1. Start the FastAPI server (`uvicorn src.server.app:app --reload`)
                2. Point this UI at the running API and start asking questions
                """.strip()
            )

        api_url_box = None
        refresh_btn = None
        if not DEMO_MODE:
            api_url_box = gr.Textbox(
                label="API base URL",
                value=default_api_url,
                placeholder="https://your-api-host",
            )
            refresh_btn = gr.Button("Load corpus info")

        corpus_info = gr.Markdown("### Corpus\n- Not loaded yet")

        doc_type_box = gr.Dropdown(
            label="Doc type filter",
            choices=["All"],
            value="All",
        )

        chatbot = gr.Chatbot(height=420, allow_tags=False)

        question = gr.Textbox(
            label="Ask a banking question",
            placeholder="Type your question and press enter",
        )
        clear_btn = gr.Button("Clear conversation")
        suggestions = _load_suggested_questions() if DEMO_MODE else []
        if suggestions:
            gr.Markdown("### Suggested questions")
            with gr.Row():
                for suggestion in suggestions[:6]:
                    gr.Button(suggestion, size="sm").click(
                        fn=lambda s=suggestion: s,
                        inputs=[],
                        outputs=question,
                    )

        def _respond(
            user_message,
            chat_history,
            doc_type_value,
            api_url_value=None,
            request: gr.Request | None = None,
        ):
            return _chat_response(
                user_message,
                chat_history or [],
                doc_type_value,
                api_url_value or default_api_url,
                request,
            )

        if refresh_btn and api_url_box:
            refresh_btn.click(
                fn=_fetch_corpus_info_and_doc_types,
                inputs=[api_url_box],
                outputs=[corpus_info, doc_type_box],
            )
        else:
            def _load(request: gr.Request | None = None):
                return _fetch_corpus_info_and_doc_types(default_api_url, request=request)

            demo.load(
                fn=_load,
                inputs=[],
                outputs=[corpus_info, doc_type_box],
            )

        question.submit(
            fn=_respond,
            inputs=[question, chatbot, doc_type_box, api_url_box] if api_url_box else [question, chatbot, doc_type_box],
            outputs=chatbot,
        )
        question.submit(lambda: "", None, question)
        clear_btn.click(lambda: [], None, chatbot)

    return demo


def launch(
    api_url: str = DEFAULT_API_URL,
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
) -> None:
    """
    Convenience launcher for local testing.
    """
    interface = build_interface(api_url)
    interface.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    launch()
