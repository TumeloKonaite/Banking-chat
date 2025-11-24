import json
from pathlib import Path
from typing import Iterable, List

from pydantic import BaseModel, Field

TEST_FILE = Path(__file__).parent / "tests.jsonl"


class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")
    doc_type: str | None = Field(
        default=None,
        description="Optional document type filter to apply during retrieval",
    )


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_tests(path: str | Path | None = None) -> List[TestQuestion]:
    """Load test questions from JSONL file."""
    file_path = Path(path) if path else TEST_FILE
    return [TestQuestion(**row) for row in _iter_jsonl(file_path)]
