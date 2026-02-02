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

DEFAULT_API_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SAMPLE_QUESTIONS_PATH = ARTIFACTS_DIR / "sample_questions.json"
DEMO_MODE = os.getenv("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
DEMO_API_KEY = os.getenv("DEMO_API_KEY", "").strip()


ChatHistory = Sequence[Union[dict, Tuple[str, str]]]


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
            content = entry.get("content")
            if role and content is not None:
                messages.append({"role": role, "content": content})
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            user_msg, bot_msg = entry
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
    return messages


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
) -> List[dict]:
    """
    Call the FastAPI endpoint and return the updated chat history.
    """
    payload = AskRequest(
        question=message,
        doc_type=None if not doc_type or doc_type == "All" else doc_type,
        history=_history_to_messages(history),
    ).model_dump()

    try:
        headers = {"x-api-key": DEMO_API_KEY} if DEMO_API_KEY else None
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{api_url.rstrip('/')}/ask",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            ask_response = AskResponse(**resp.json())
            answer = _format_answer(ask_response)
    except Exception as exc:  # pragma: no cover - UI feedback only
        answer = f"Error talking to API: {exc}"

    updated_history: List[dict] = list(_history_to_messages(history))
    updated_history.append({"role": "user", "content": message})
    updated_history.append({"role": "assistant", "content": answer})
    return updated_history


def _get_ready_payload(api_url: str) -> tuple[dict | None, str | None]:
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{api_url.rstrip('/')}/ready")
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
        payload, error = _get_ready_payload(api_url)
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


def _fetch_corpus_info_and_doc_types(api_url: str):
    """
    Fetch manifest and return markdown plus dropdown update.
    """
    payload, error = _get_ready_payload(api_url)
    if error or not payload:
        info = error or "Corpus info unavailable: Unknown error"
        return info, gr.Dropdown.update(choices=["All"], value="All")

    info = _fetch_corpus_info(api_url, payload=payload)
    manifest = payload.get("manifest") or {}
    doc_types = list((manifest.get("document_types") or {}).keys())
    choices = ["All"] + sorted(doc_types)
    return info, gr.Dropdown.update(choices=choices, value="All")

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
                placeholder="http://localhost:8000",
            )
            refresh_btn = gr.Button("Load corpus info")

        corpus_info = gr.Markdown("### Corpus\n- Not loaded yet")

        doc_type_box = gr.Dropdown(
            label="Doc type filter",
            choices=["All"],
            value="All",
        )

        chatbot = gr.Chatbot(
            height=420,
            type="messages",
            allow_tags=False,
        )

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

        def _respond(user_message, chat_history, doc_type_value, api_url_value=None):
            return _chat_response(
                user_message,
                chat_history or [],
                doc_type_value,
                api_url_value or default_api_url,
            )

        if refresh_btn and api_url_box:
            refresh_btn.click(
                fn=_fetch_corpus_info_and_doc_types,
                inputs=[api_url_box],
                outputs=[corpus_info, doc_type_box],
            )
        else:
            demo.load(
                fn=lambda: _fetch_corpus_info_and_doc_types(default_api_url),
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
