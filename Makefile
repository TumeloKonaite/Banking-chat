.PHONY: help build-index dev docker

help:
	@echo "Available targets:"
	@echo "  make build-index  - rebuild artifacts from data/"
	@echo "  make dev          - run local FastAPI + Gradio app"
	@echo "  make docker       - build and run Docker image using .env"

build-index:
	python -m src.pipeline.build_artifacts

dev:
	uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload

docker:
	docker build -t banking-rag .
	docker run --rm --env-file .env -p 8000:8000 banking-rag
