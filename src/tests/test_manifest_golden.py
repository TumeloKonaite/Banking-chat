import json
from datetime import datetime
from pathlib import Path


REQUIRED_FIELDS = {
    "corpus_name": str,
    "built_at": str,
    "data_dir": str,
    "document_count": int,
    "document_types": dict,
    "embedding_provider": str,
    "embedding_model": str,
    "chunk_size": int,
    "chunk_overlap": int,
    "vector_db_dir": str,
}


def test_manifest_exists():
    manifest_path = Path("artifacts/manifest.json")
    assert manifest_path.exists(), "artifacts/manifest.json is required for demo readiness"


def test_manifest_has_required_fields_and_types():
    manifest = json.loads(Path("artifacts/manifest.json").read_text(encoding="utf-8"))

    for key, expected_type in REQUIRED_FIELDS.items():
        assert key in manifest, f"Missing required manifest field: {key}"
        assert isinstance(manifest[key], expected_type), (
            f"Manifest field '{key}' must be of type {expected_type.__name__}"
        )


def test_manifest_semantic_constraints():
    manifest = json.loads(Path("artifacts/manifest.json").read_text(encoding="utf-8"))

    # Ensure built_at is parseable ISO-8601 timestamp.
    datetime.fromisoformat(manifest["built_at"])

    assert manifest["corpus_name"].strip()
    assert manifest["document_count"] > 0
    assert manifest["chunk_size"] > 0
    assert 0 <= manifest["chunk_overlap"] < manifest["chunk_size"]
    assert manifest["embedding_provider"] in {"openai", "huggingface"}
    assert manifest["vector_db_dir"].strip()

    document_types = manifest["document_types"]
    assert document_types, "document_types must not be empty"
    assert all(isinstance(k, str) and k for k in document_types.keys())
    assert all(isinstance(v, int) and v >= 0 for v in document_types.values())
