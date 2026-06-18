#!/bin/bash
set -e

MODEL_PATH="/app/data/models/classifier.joblib"
MODEL_DIR="/app/data/models"

# Download the trained model from the GitHub Release if not already present
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading classifier.joblib from GitHub Release..."
  mkdir -p "$MODEL_DIR"
  curl -L "$MODEL_RELEASE_URL" -o "$MODEL_PATH"
  echo "Model downloaded."
else
  echo "Model already present, skipping download."
fi

# Start FastAPI on internal port 8000 (not exposed by HF Spaces)
uvicorn eca.api.main:app --host 0.0.0.0 --port 8000 &
echo "FastAPI started on port 8000"

# Wait briefly for the API to be ready before Streamlit tries to connect
sleep 5

# Start Streamlit on port 7860 — the only port HF Spaces exposes
exec streamlit run streamlit_app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false
