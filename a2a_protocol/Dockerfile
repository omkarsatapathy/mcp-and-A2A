# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Research Agent (A2A v1.0)
#
# Uses UV instead of pip for blazing-fast dependency resolution.
# Targets 14 CPU cores with 90% utilization during build.
# Region: ap-south-1 (Mumbai)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM --platform=linux/arm64 python:3.11-slim AS builder

# Install uv (Rust-based Python package manager — much faster than pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# System deps needed for compilation (numpy, faiss-cpu, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies with uv
# --compile-bytecode: pre-compile .pyc for faster cold starts
# UV_CONCURRENT_DOWNLOADS: max parallel downloads
# UV_CONCURRENT_BUILDS: use all 14 cores for building C extensions
# UV_CONCURRENT_INSTALLS: parallel wheel installs
ENV UV_CONCURRENT_DOWNLOADS=14 \
    UV_CONCURRENT_BUILDS=14 \
    UV_CONCURRENT_INSTALLS=14 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

# Use uv pip install with aggressive parallelism
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        --compile-bytecode \
        -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM --platform=linux/arm64 python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Runtime thread/core tuning — push CPU utilization to 90% across 14 cores
ENV OMP_NUM_THREADS=13 \
    MKL_NUM_THREADS=13 \
    NUMEXPR_NUM_THREADS=13 \
    OPENBLAS_NUM_THREADS=13 \
    VECLIB_MAXIMUM_THREADS=13 \
    TOKENIZERS_PARALLELISM=true \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=0 \
    AWS_DEFAULT_REGION=ap-south-1 \
    AWS_REGION=ap-south-1 \
    USE_BEDROCK_MEMORY=true \
    USE_AWS_SECRETS=true

# Copy application code into a proper package directory
# so `import a2a_protocol.xxx` resolves correctly
# ONNX models are included via models/ subdirectory
# .dockerignore excludes .env, __pycache__, deploy scripts, etc.
COPY . ./a2a_protocol/

EXPOSE 8080

# Run with uvicorn using 14 workers (1 worker per core)
# --workers 14 distributes requests across all CPU cores
CMD ["python", "-m", "uvicorn", \
     "a2a_protocol.a2a_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "14", \
     "--limit-concurrency", "100", \
     "--timeout-keep-alive", "120"]
