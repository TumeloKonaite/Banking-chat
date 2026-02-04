# Banking RAG [![CI](https://github.com/TumeloKonaite/Banking-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/TumeloKonaite/Banking-RAG/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Development Status](https://img.shields.io/badge/Status-Active-success.svg)](#development-status)
[![Git Workflow](https://img.shields.io/badge/GitHub-Flow-blue.svg)](https://docs.github.com/en/get-started/quickstart/github-flow)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue.svg)](#docker-first-run)

Demo-ready Retrieval-Augmented Generation (RAG) app for banking documents.
It ships with prebuilt artifacts, serves a FastAPI backend, and exposes a Gradio UI.

## Important disclaimer

This is an informational demo, not a production banking system. Do not use it for real account actions, and do not submit personal data, account numbers, credentials, or other sensitive PII.

## Development Status

Active development. The deployed demo and this repo are kept aligned, with fixes shipped continuously.

## Live Demo

- Gradio UI: http://bankin-banki-xuy9rfv4r0bq-1682177220.eu-west-1.elb.amazonaws.com/gradio/
- Health: http://bankin-banki-xuy9rfv4r0bq-1682177220.eu-west-1.elb.amazonaws.com/health
- API docs: http://bankin-banki-xuy9rfv4r0bq-1682177220.eu-west-1.elb.amazonaws.com/api/docs

Deployed on AWS ECS Fargate behind an Application Load Balancer, provisioned with CDK.

## First run (foolproof path)

1) Copy env template and set secrets:

```bash
cp .env.example .env
```

Required values:

- `OPENAI_API_KEY`
- `DEMO_MODE`
- `DEMO_API_KEY`

2) Install dependencies:

```bash
pip install ".[dev]"
```

3) Build/rebuild artifacts (required if missing or stale):

```bash
make build-index
```

4) Start local dev server:

```bash
make dev
```

5) Open:

- `http://localhost:8000/gradio/`
- `http://localhost:8000/api/docs`
- `http://localhost:8000/ready`

If you do not have `make`, run:

- `python -m src.pipeline.build_artifacts`
- `uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload`

## Make commands (happy path)

```bash
make build-index   # rebuild artifacts/*
make dev           # run local API + mounted Gradio UI
make docker        # build image and run container using .env
```

## Docker first run

```bash
make docker
```

Equivalent manual commands:

```bash
docker build -t banking-rag .
docker run --rm --env-file .env -p 8000:8000 banking-rag
```

Open:

- `http://localhost:8000/gradio/`
- `http://localhost:8000/api/docs`

## Cloud first run (AWS CDK + ECS)

From repo root:

```bash
cp .env.example .env
```

Set:

- `DEMO_API_KEY` in your environment before `cdk deploy`
- `OPENAI_SECRET_NAME` (defaults to `banking-rag/openai-api-key`)

Deploy:

```bash
cd infra-cdk
pip install -r requirements.txt
cdk deploy
```

## Artifact readiness behavior (`/ready`)

`/ready` checks:

- `artifacts/vector_db/` exists and is non-empty
- `artifacts/manifest.json` exists and is valid
- embedding provider/model in manifest matches runtime config

### Expected healthy response

```bash
curl -i http://localhost:8000/ready
```

Expected status: `200 OK`

```json
{"ready": true, "detail": "Ready", "manifest": {...}}
```

### Expected missing-artifacts response

```bash
curl -i http://localhost:8000/ready
```

Expected status: `503 Service Unavailable`

```json
{
  "detail": "Artifacts missing. This demo expects prebuilt artifacts. Run: make build-index (or python -m src.pipeline.build_artifacts)"
}
```

In demo mode (`DEMO_MODE=1`), startup is fail-fast when artifacts are missing.

## Current app behavior

- `GET /` redirects to `/gradio` in demo mode.
- `POST /ask` is protected by `X-API-Key` when `DEMO_MODE=1`.
- `GET /health` is a liveness check.
- `GET /ready` validates prebuilt artifacts and returns corpus metadata.
- Gradio is mounted under `/gradio` and calls the API on the same host.

## Safety defaults (demo mode)

When `DEMO_MODE=1`, the server enforces:

- API key auth (`DEMO_API_KEY`) for `/ask`
- rate limiting on `/ask` (API-key/IP scoped)
- locked-down CORS defaults for local UI origins
- request size and schema validation limits
- correlation IDs (`X-Request-ID`) and request telemetry (latency/model/retrieval hits)
- fail-fast startup if artifacts are missing

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
- `src/pipeline/` - RAG orchestration and artifact build entrypoint
- `src/retrieval/` - document retrieval over Chroma
- `src/embedding/` - embedding provider wiring
- `src/chunking/` - text chunking
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
- deterministic reliability suite (`pytest -q src/tests`) covering API contracts, artifact manifest validation, and retrieval behavior
- Docker build (`docker build -t banking-rag-ci .`)
- optional evaluation smoke test when `OPENAI_API_KEY` secret is present

## Troubleshooting

- `503 from /ready` -> missing/invalid artifacts; run `make build-index`.
- `401/403 from /ask` -> missing or wrong `X-API-Key` in demo mode.
- `401 invalid_api_key` from OpenAI -> rotate and update your OpenAI key/secret.
- Gradio follow-up validation errors -> ensure latest image is deployed.

## License

MIT. See `LICENSE`.
