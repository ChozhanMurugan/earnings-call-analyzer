# syntax=docker/dockerfile:1
# ── Stage 1: builder ────────────────────────────────────────────────────────────
# Compile any native extensions in a full build environment, then throw away
# the compiler so the runtime image stays lean.
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/

# Install all runtime dependencies to an isolated prefix
RUN pip install --no-cache-dir --prefix=/install .


# ── Stage 2: runtime ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# curl is required for the HEALTHCHECK directive below
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring in the installed packages from the builder — no compiler needed at runtime
COPY --from=builder /install /usr/local

# Streamlit UI entry point
COPY streamlit_app.py .

# Pin HuggingFace cache inside the image so the model is baked in and the
# container starts offline without downloading 400 MB of weights.
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1

# Set PREFETCH_FINBERT=false via --build-arg to skip the weight download in CI
# (the CI docker-build job passes this flag to keep the job fast).
ARG PREFETCH_FINBERT=true
RUN if [ "$PREFETCH_FINBERT" = "true" ]; then \
      python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='ProsusAI/finbert')"; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "eca.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
