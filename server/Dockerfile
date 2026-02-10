# Optimized Multi-stage Dockerfile for Visual-ML Backend
# Target: <400MB (down from 692MB)

# Stage 1: Builder - Install dependencies
FROM python:3.11-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    postgresql-dev \
    linux-headers \
    libffi-dev \
    openssl-dev

# Install uv for faster dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies using uv
RUN uv pip install --system --no-cache-dir \
    fastapi>=0.128.0 \
    uvicorn[standard]>=0.30.0 \
    pydantic[email]>=2.0.0 \
    pydantic-settings>=2.0.0 \
    python-jose[cryptography]>=3.3.0 \
    passlib[bcrypt]>=1.7.4 \
    python-multipart>=0.0.6 \
    google-auth>=2.27.0 \
    argon2-cffi>=25.1.0 \
    email-validator>=2.3.0 \
    sqlalchemy>=2.0.0 \
    alembic>=1.13.0 \
    psycopg2-binary>=2.9.11 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    scikit-learn>=1.3.0 \
    joblib>=1.3.0 \
    redis>=6.4.0 \
    celery[redis]>=5.6.2 \
    boto3>=1.34.0 \
    openai>=2.15.0 \
    anthropic>=0.76.0 \
    google-generativeai>=0.8.6 \
    loguru>=0.7.0 \
    python-dotenv>=1.0.0 \
    httpx>=0.28.1 \
    aiofiles>=25.1.0 \
    brevo-python>=1.2.0

# Remove unnecessary files to reduce size
RUN find /usr/local/lib/python3.11/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11/site-packages -name "*.pyo" -delete && \
    find /usr/local/lib/python3.11/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Stage 2: Runtime - Minimal production image
FROM python:3.11-alpine

# Install only runtime dependencies (no build tools)
RUN apk add --no-cache \
    libpq \
    curl \
    libstdc++ \
    libgcc

# Create non-root user for security
RUN adduser -D -u 1000 appuser && \
    mkdir -p /app/uploads /app/models /app/logs && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy Python packages from builder (only site-packages and necessary binaries)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=builder /usr/local/bin/alembic /usr/local/bin/alembic
COPY --from=builder /usr/local/bin/celery /usr/local/bin/celery

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
