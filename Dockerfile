# syntax=docker/dockerfile:1.6

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system deps required by pypdf/unstructured tooling
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        poppler-utils \
        libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata first for better build caching
COPY pyproject.toml README.md ./

# Copy source code
COPY src ./src

# Optional: copy notebooks/artifacts if needed (commented out)
# COPY notebooks ./notebooks

# Install project dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
