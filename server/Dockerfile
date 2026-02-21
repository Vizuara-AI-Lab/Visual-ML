# ---- Stage 1: Builder ----
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies for compiled packages (psycopg2, argon2, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy full source (setuptools needs the `app` package directory)
COPY . .

# Install Python dependencies using uv
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --no-cache .

# ---- Stage 2: Runtime ----
FROM python:3.10-slim

WORKDIR /app

# Install only runtime libs (libpq for psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY . .

# Create runtime directories
RUN mkdir -p uploads models

# Expose the API port
EXPOSE 3003

# Run with uvicorn (no reload in production)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3003", "--workers", "1", "--log-level", "info"]
