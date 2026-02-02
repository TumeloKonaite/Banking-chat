# Banking RAG [![CI](https://github.com/TumeloKonaite/Banking-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/TumeloKonaite/Banking-RAG/actions/workflows/ci.yml)

Modular Retrieval-Augmented Generation pipeline tuned for banking documents. Drop your PDFs into the `data/` folder, run the build script, and start asking questions grounded entirely in your own content.

## Drag-and-drop workflow

1. **Drop documents** - Organise raw PDFs under `data/<doc_type>/` (for example `data/product_terms/`, `data/pricing_guides/`). Each folder becomes a metadata tag so you can later filter answers per collection.
2. **Chunk and vectorise** - Run the ingestion script (see below) to load PDFs, create overlapping chunks, embed them, and persist a Chroma vector store to `artifacts/vector_db/`.
3. **Ask questions** - Instantiate `src.pipeline.rag_pipeline.RAGPipeline` and call `answer_question(...)`. The assistant retrieves the most relevant chunks and answers only with the supplied context while applying strict guardrails.

## Architecture at a glance

```
data/<doc_type>/*.pdf
        |
        v  fetch_documents()
src/chunking/document_chunking.py
        |
        v  create_chunks()
artifacts/chunks/chunks.csv
        |
        v
src/embedding/document_embedding.py
        |
        v  create_embeddings()
artifacts/embeddings/embeddings.pkl
        |
        v
src/vectorstore/vector_store.py
        |
        v  build_from_chunks()
artifacts/vector_db/  (Chroma)
        |
        v
src/retrieval/document_retriever.py
        |
        v
src/pipeline/rag_pipeline.py  -> ChatOpenAI
```

## Key features

- **True drag-and-drop ingestion:** add PDFs to `data/` and rerun the builder, no manual metadata work required.
- **Swappable embeddings:** flip between OpenAI (`text-embedding-3-large`) and HuggingFace (`all-MiniLM-L6-v2`) via config.
- **Persistent vector store:** Chroma DB on disk means you rebuild only when documents change.
- **Prompt-injection-resistant assistant:** the system prompt encodes strict banking-specific guardrails.
- **Notebook-friendly:** everything is modular and importable for ad-hoc experiments inside `notebooks/`.

## Getting started

### Prerequisites

- Python 3.13 (managed via `uv` or `pyenv`)
- An OpenAI API key (or HuggingFace token if you switch providers)
- System packages needed by `pypdf` and `unstructured` (Poppler/Ghostscript on some Linux distros)

### Installation

```bash
git clone https://github.com/<your-account>/banking-rag.git
cd banking-rag
uv sync           # or: python -m venv .venv && source .venv/bin/activate && pip install -e .
```

Create a `.env` file (or edit the existing one) with the credentials you want to use:

```bash
OPENAI_API_KEY=sk-...
# Optional if you switch providers
HF_TOKEN=hf-...
```

### Project layout

- `data/` - drag-and-drop PDFs grouped by document type
- `artifacts/chunks/` - CSV export of chunk metadata and text
- `artifacts/embeddings/` - pickled embeddings, texts, and metadata
- `artifacts/vector_db/` - persisted Chroma database
- `src/ingestion/` - DirectoryLoader + PyPDFLoader wiring
- `src/chunking/` - RecursiveCharacterTextSplitter wrapper
- `src/embedding/` - OpenAI or HuggingFace embedding factory
- `src/vectorstore/` - vector store builder
- `src/retrieval/` - Chroma retriever with metadata filters
- `src/pipeline/` - guardrailed ChatOpenAI RAG pipeline
- `src/server/` - FastAPI endpoint and Gradio chat UI
- `src/evaluation/` - LLM-as-a-judge tooling and CLI
- `src/tests/` - JSONL fixtures plus helper models
- `notebooks/` - optional exploratory work

## Build the knowledge base

Run the following script (or adapt it into your own CLI) whenever you drop new files in `data/`:

```python
from src.ingestion.load_documents import fetch_documents
from src.chunking.document_chunking import DocumentChunking
from src.embedding.document_embedding import DocumentEmbedding
from src.vectorstore.vector_store import VectorStoreBuilder

docs = fetch_documents()                 # scans data/<doc_type> folders
chunker = DocumentChunking()
_, chunks = chunker.create_chunks(docs)  # writes artifacts/chunks/chunks.csv

embedder = DocumentEmbedding()
embedder.create_embeddings(chunks)       # writes artifacts/embeddings/embeddings.pkl

builder = VectorStoreBuilder()
builder.build_from_chunks(chunks)        # persists Chroma DB to artifacts/vector_db/
```

All artifacts are safe to commit to `.gitignore`; regenerate them on demand.

### Build once for demo (prebuilt corpus)

For the read-only demo flow, precompute artifacts and ship them with the app:

```bash
python -m src.build.build_index
```

This writes the Chroma DB under `artifacts/vector_db/` and a `artifacts/manifest.json`
describing the corpus (document counts, build timestamp, embedding model, etc.).

For the MVP demo, the repo ships with a small prebuilt corpus under `artifacts/`.
You can run the API + Gradio immediately without re-ingesting documents.

## Ask questions

```python
from src.pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()  # uses DocumentRetriever and ChatOpenAI under the hood

answer, docs = pipeline.answer_question(
    question="Explain the monthly ATM withdrawal fees.",
    history=[],                    # optional OpenAI-style message list
    doc_type="product_terms"       # optional metadata filter
)

print(answer)
for d in docs[:3]:
    print(d.metadata["source"], d.page_content[:200])
```

The assistant enforces the banking guardrails, cites only retrieved context, and gracefully declines when the corpus lacks the requested information.

## Serve over FastAPI + Gradio

1. **Start the API**

   ```bash
   uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload
   ```

   This loads the pipeline once and exposes `POST /ask`, which accepts a JSON body.
   The API will return `503` if prebuilt artifacts are missing.

   The readiness endpoints:

   - `GET /health` returns `200` when the server is up.
   - `GET /ready` returns `200` only when prebuilt artifacts are present. When missing,
     it returns `503` with guidance to run `python -m src.build.build_index`.

   ```json
   {
     "question": "Explain the ATM withdrawal fees.",
     "doc_type": "product_terms",
     "history": [
       {"role": "user", "content": "Previous question"},
       {"role": "assistant", "content": "Earlier answer"}
     ]
   }
   ```

2. **Launch the Gradio UI** (which calls the API under the hood)

   ```bash
   python -m src.server.gradio_ui
   ```

   Point the UI at your API base URL (defaults to `http://localhost:8000`), optionally set a `doc_type` filter, and chat. Answers include a quick list of the retrieved source documents for transparency.

## Run with Docker

Build the container and run the API without installing anything locally:

```bash
docker build -t banking-rag .

docker run \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  banking-rag
```

Mounting `data/` and `artifacts/` keeps your PDFs and vector DB on the host. Once the container is running, hit `http://localhost:8000/ask` (POST) or launch the Gradio UI separately and target the same base URL.

## Evaluate the pipeline

This repo includes a lightweight evaluation harness (LLM-as-a-judge + retrieval metrics) under `src/evaluation/` and fixtures in `src/tests/tests.jsonl`. To score a single row from the test set, run:

```bash
uv run python -m src.evaluation.eval 0
```

Replace `0` with any row index from the JSONL file. The CLI prints retrieval metrics (MRR, nDCG, keyword coverage) plus judge feedback and accuracy/completeness/relevance scores. You can also import `evaluate_all_retrieval` / `evaluate_all_answers` for batch processing in notebooks or scripts.

## Track runs with MLflow

All artifact builds and evaluations can be logged as MLflow runs for reproducibility:

```bash
# rebuild artifacts + log params/metrics/artifacts under the "artifact-builds" experiment
python -m src.pipeline.build_artifacts

# run an evaluation (logs to the "evaluation-runs" experiment)
python -m src.evaluation.eval 0
```

By default MLflow writes to `./mlruns`; set `MLFLOW_TRACKING_URI` if you want to point at a remote server. Start a local UI with `mlflow ui --port 5000` to compare runs, inspect logged artifacts (chunk CSVs, embeddings, Chroma snapshots, evaluation JSON), and monitor how parameter tweaks affect retrieval and judge scores.

## Continuous integration

A lightweight GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push or pull request against `main`. The workflow:

- checks out the repository and installs dependencies with Python 3.13 (including the optional `dev` extras for Ruff),
- runs `ruff check .` and `python -m compileall src` to catch style and syntax issues early,
- builds the Docker image with `docker build -t banking-rag-ci .`,
- runs `python -m src.evaluation.eval 0` as a smoke test when the `OPENAI_API_KEY` secret is present,
- and, on successful runs against `main`, tags the commit as `ci-<run_number>` and pushes the tag back to the repo.

If you want to skip auto-tagging for a fork, delete or edit the "Tag successful build" step in the workflow.

## Customisation ideas

1. **Switch embeddings/provider:** change `provider` and model names in `EmbeddingConfig`, `VectorStoreConfig`, and `RetrieverConfig`.
2. **Adjust chunking granularity:** tune `chunk_size` and `chunk_overlap` in `DocumentChunkingConfig` to balance recall versus latency.
3. **Deploy behind an API or Gradio app:** reuse `RAGPipeline.answer_question` as the backend for a chat UI.
4. **Schedule nightly refreshes:** wrap the build code in a CI job to ingest any PDFs dropped into a shared folder.

## Troubleshooting

- `FileNotFoundError: Data folder not found` -> ensure you created the `data/` hierarchy and have readable PDFs.
- `No vector database found` -> run the "Build the knowledge base" script before starting the RAG pipeline.
- `401 from OpenAI` -> double-check `OPENAI_API_KEY` in `.env` and that your account has access to `gpt-4.1-nano`.
- LangChain import errors -> reinstall dependencies with `uv sync` (or `pip install -e .`) to pull the pinned versions from `pyproject.toml`.

## License

Specify the license you plan to use for the repository (for example MIT or Apache 2.0). Update this section before publishing publicly.
