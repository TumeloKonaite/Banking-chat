"""
Utility script to rebuild artifacts (chunks, embeddings, vector DB) and
log the run metadata to MLflow for experiment tracking.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict
import shutil
import subprocess
import tempfile

import mlflow

from src.chunking.document_chunking import DocumentChunking
from src.embedding.document_embedding import DocumentEmbedding
from src.ingestion.load_documents import DATA_DIR, fetch_documents
from src.vectorstore.vector_store import VectorStoreBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _summarise_documents(docs) -> Dict[str, int]:
    """
    Return a histogram of doc_type occurrences to log alongside the run.
    """
    counter: Counter[str] = Counter()
    for doc in docs:
        doc_type = doc.metadata.get("doc_type", "unknown")
        counter[doc_type] += 1
    return dict(counter)


def _log_artifact_if_exists(path: str | Path, artifact_path: str) -> None:
    path = Path(path)
    if path.is_file():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    elif path.is_dir():
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)


def _get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def build_with_mlflow() -> None:
    """
    Run the full ingestion -> chunking -> embedding -> vector-store pipeline
    while logging parameters, metrics, and artifacts to MLflow.
    """
    experiment_name = "artifact-builds"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"build-{datetime.now(UTC).isoformat()}"):
        git_commit = _get_git_commit()
        if git_commit:
            mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("data_dir", str(DATA_DIR))
        docs = fetch_documents()
        mlflow.log_param("document_count", len(docs))
        mlflow.log_param("document_type_hist", json.dumps(_summarise_documents(docs)))

        chunker = DocumentChunking()
        mlflow.log_param("chunk_size", chunker.config.chunk_size)
        mlflow.log_param("chunk_overlap", chunker.config.chunk_overlap)
        mlflow.log_param("chunks_dir", chunker.config.chunks_dir)

        chunks_file, chunks = chunker.create_chunks(docs)
        mlflow.log_metric("chunk_count", len(chunks))
        _log_artifact_if_exists(chunks_file, artifact_path="chunks")

        embedder = DocumentEmbedding()
        mlflow.log_param("embedding_provider", embedder.config.provider)
        mlflow.log_param("embedding_model_openai", embedder.config.openai_model)
        mlflow.log_param("embedding_model_hf", embedder.config.hf_model)
        mlflow.log_param("embeddings_dir", embedder.config.embeddings_dir)

        embeddings_file, vectors, texts, _ = embedder.create_embeddings(chunks)
        mlflow.log_metric("embedding_count", len(vectors))
        mlflow.log_metric("text_count", len(texts))
        _log_artifact_if_exists(embeddings_file, artifact_path="embeddings")

        builder = VectorStoreBuilder()
        mlflow.log_param("vector_provider", builder.config.provider)
        mlflow.log_param("vector_db_dir", builder.config.db_dir)

        builder.build_from_chunks(chunks)
        _log_artifact_if_exists(builder.config.db_dir, artifact_path="vector_db")

        artifacts_root = PROJECT_ROOT / "artifacts"
        if artifacts_root.exists():
            with tempfile.TemporaryDirectory() as tmpdir:
                archive_path = shutil.make_archive(
                    base_name=str(Path(tmpdir) / "artifacts_snapshot"),
                    format="zip",
                    root_dir=str(artifacts_root),
                )
                mlflow.log_artifact(archive_path, artifact_path="artifact_snapshots")


def main() -> None:
    build_with_mlflow()


if __name__ == "__main__":
    main()
