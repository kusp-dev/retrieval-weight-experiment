FROM python:3.11-slim

WORKDIR /app

# System dependencies for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reproducible dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies
COPY pyproject.toml ./
RUN uv sync --no-dev

# Copy source code and data
COPY src/ src/
COPY configs/ configs/
COPY data/ data/
COPY scripts/ scripts/

# Create volume mount points for output and model cache
VOLUME ["/app/output", "/root/.cache/huggingface"]

# Default: run the full experiment
# Override with: docker run ... uv run python scripts/run_full_experiment.py
CMD ["uv", "run", "python", "scripts/run_full_experiment.py"]
