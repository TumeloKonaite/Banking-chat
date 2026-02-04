# Banking RAG [![CI](https://github.com/TumeloKonaite/Banking-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/TumeloKonaite/Banking-RAG/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Development Status](https://img.shields.io/badge/Status-Active-success.svg)](#development-status)
[![Git Workflow](https://img.shields.io/badge/GitHub-Flow-blue.svg)](https://docs.github.com/en/get-started/quickstart/github-flow)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue.svg)](#docker-quick-start)

Demo-ready Retrieval-Augmented Generation (RAG) app for banking documents.
It ships with prebuilt artifacts, serves a FastAPI backend, and exposes a Gradio UI.

## Development Status

Active development. The deployed demo and this repo are kept aligned, with fixes shipped continuously.

## Live Demo

- Gradio UI: http://bankin-banki-xuy9rfv4r0bq-1682177220.eu-west-1.elb.amazonaws.com/gradio/
- Health: http://bankin-banki-xuy9rfv4r0bq-1682177220.eu-west-1.elb.amazonaws.com/health
- API docs: http://bankin-banki-xuy9rfv4r0bq-1682177220.eu-west-1.elb.amazonaws.com/api/docs

Deployed on AWS ECS Fargate behind an Application Load Balancer, provisioned with CDK.

## Current app behavior

- `GET /` redirects to `/gradio` in demo mode.
- `POST /ask` is protected by `X-API-Key` when `DEMO_MODE=1`.
- `GET /health` is a liveness check.
- `GET /ready` validates prebuilt artifacts and returns corpus metadata.
- Gradio is mounted under `/gradio` and calls the API on the same host.

## Safety defaults (demo mode)

When `DEMO_MODE=1`, the server enforces:

- API key auth (`DEMO_API_KEY`) for `/ask`
- locked-down CORS defaults for local UI origins
- request size and schema validation limits
- fail-fast startup if artifacts are missing

## Docker Quick Start

```bash
# 1) Build image
docker build -t banking-rag .

# 2) Run API + mounted Gradio UI in one container
docker run -p 8000:8000 \
  -e DEMO_MODE=1 \
  -e DEMO_API_KEY=demo-key \
  -e OPENAI_API_KEY=sk-... \
  banking-rag
```

Open:

- `http://localhost:8000/gradio/`
- `http://localhost:8000/api/docs`

## Build / rebuild corpus artifacts

Use this when your `data/` PDFs change:

```bash
python -m src.build.build_index
```

This writes:

- `artifacts/vector_db/` (Chroma DB)
- `artifacts/manifest.json` (corpus metadata)

## API request format (`POST /ask`)

```json
{
  "question": "Explain ATM withdrawal fees.",
  "doc_type": "product_terms",
  "history": [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous answer"}
  ]
}
```

In demo mode, include header:

```text
X-API-Key: <DEMO_API_KEY>
```

## Project layout

- `src/server/` - FastAPI app, schemas, Gradio UI
- `src/pipeline/` - RAG orchestration and guardrails
- `src/retrieval/` - document retrieval over Chroma
- `src/embedding/` - embedding provider wiring
- `src/chunking/` - text chunking
- `src/build/` - artifact build entrypoint
- `artifacts/` - prebuilt demo corpus + manifest
- `infra-cdk/` - AWS CDK deployment stack

## AWS deployment notes

- CDK app lives in `infra-cdk/`.
- ECS task image is pulled from ECR repo `banking-rag`.
- `OPENAI_API_KEY` is read from AWS Secrets Manager (`banking-rag/openai-api-key`).
- `DEMO_API_KEY` is provided at deploy/runtime.

## CI

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on pushes/PRs to `main` and does:

- Ruff lint (`ruff check .`)
- Python byte-compile (`python -m compileall src`)
- Docker build (`docker build -t banking-rag-ci .`)
- optional evaluation smoke test when `OPENAI_API_KEY` secret is present

## Troubleshooting

- `503 from /ready` -> missing/invalid artifacts; run `python -m src.build.build_index`.
- `401/403 from /ask` -> missing or wrong `X-API-Key` in demo mode.
- `401 invalid_api_key` from OpenAI -> rotate and update your OpenAI key/secret.
- Gradio follow-up validation errors -> ensure latest image is deployed.

## License

MIT. See `LICENSE`.
