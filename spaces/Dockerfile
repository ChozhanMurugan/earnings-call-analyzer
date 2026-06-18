# syntax=docker/dockerfile:1
# HuggingFace Spaces — single container running FastAPI (port 8000, internal)
# + Streamlit (port 7860, exposed). The trained model is downloaded from the
# GitHub Release at startup via MODEL_RELEASE_URL env var.
#
# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as a non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY --from=builder /install /usr/local
COPY streamlit_app.py .
COPY startup.sh .

RUN chmod +x startup.sh && mkdir -p data/models data/processed

# Bake FinBERT weights into the image so inference is instant at runtime
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    ECA_DATA_DIR=/app/data \
    ECA_MODEL_PATH=/app/data/models/classifier.joblib \
    ECA_API_BASE=http://localhost:8000 \
    PYTHONUNBUFFERED=1

RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='ProsusAI/finbert')"

RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces only exposes port 7860
EXPOSE 7860

CMD ["./startup.sh"]
