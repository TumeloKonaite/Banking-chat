"""
Build-time entrypoint for generating precomputed RAG artifacts.

Usage:
    python -m src.build.build_index
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict

from src.chunking.document_chunking import DocumentChunking
from src.embedding.document_embedding import DocumentEmbedding
from src.ingestion.load_documents import DATA_DIR, fetch_documents
from src.vectorstore.vector_store import VectorStoreBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MANIFEST_PATH = ARTIFACTS_DIR / "manifest.json"


def _summarise_documents(docs) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for doc in docs:
        doc_type = doc.metadata.get("doc_type", "unknown")
        counter[doc_type] += 1
    return dict(counter)


def build_index() -> None:
    """
    Build artifacts for the demo (chunks, embeddings, vector DB, manifest).
    """
    docs = fetch_documents()

    chunker = DocumentChunking()
    _, chunks = chunker.create_chunks(docs)

    embedder = DocumentEmbedding()
    embedder.create_embeddings(chunks)

    builder = VectorStoreBuilder()
    builder.build_from_chunks(chunks)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "built_at": datetime.now(UTC).isoformat(),
        "data_dir": str(DATA_DIR),
        "document_count": len(docs),
        "document_types": _summarise_documents(docs),
        "embedding_provider": embedder.config.provider,
        "embedding_model": (
            embedder.config.openai_model
            if embedder.config.provider.lower() == "openai"
            else embedder.config.hf_model
        ),
        "chunk_size": chunker.config.chunk_size,
        "chunk_overlap": chunker.config.chunk_overlap,
        "vector_db_dir": builder.config.db_dir,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
