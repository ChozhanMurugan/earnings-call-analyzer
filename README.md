# Earnings Call Analyzer

[![CI](https://github.com/ChozhanMurugan/earnings-call-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/ChozhanMurugan/earnings-call-analyzer/actions/workflows/ci.yml)

> NLP-driven directional classifier on earnings-call transcripts, backtested against T+1 returns, served as a FastAPI service with a Streamlit dashboard.

**Portfolio walkthrough (start here):** [`notebooks/01_eda_and_model.ipynb`](notebooks/01_eda_and_model.ipynb) — runs end-to-end in ~30 s with no downloads, no GPU, no API keys.

## Why this project exists

Earnings calls move stocks more than the headline EPS number does. The way management *talks* — hedging, forward guidance, tone shifts versus the prior quarter — has been shown to carry alpha (e.g., Loughran & McDonald 2011; Hassan, Hollander, van Lent & Tahoun 2019). This repo turns that idea into a reproducible pipeline:

```
EDGAR 8-K transcripts ─┐
HuggingFace corpus    ─┼──► feature extraction ──► LightGBM directional classifier ──► vectorized backtest ──► FastAPI + Streamlit
                       ┘
```

## Features extracted per call

| Feature group | What it captures |
| --- | --- |
| **FinBERT sentiment** | Pos / neu / neg probs from `ProsusAI/finbert`, computed per sentence then aggregated (mean, std, share-positive, share-negative) |
| **Hedge-word frequency** | Normalised count of uncertainty / weak-modal language from the Loughran-McDonald uncertainty dictionary (`might`, `could`, `approximately`, …) |
| **Forward-guidance language** | Rule-based detector for forward-looking statements (`we expect`, `we anticipate`, `next quarter`, `full year guidance`, …) and the sentiment polarity around them |
| **QoQ tone shift** | Delta of each of the above against the same ticker's prior call — the *change* in tone is more predictive than the level |
| **Call structure** | Prepared-remarks length, Q&A length, analyst-question count |

## Pipeline

1. **Ingest** — `eca.ingest.edgar` pulls 8-K filings + exhibit transcripts from SEC EDGAR (free, official, no ToS issue). `eca.ingest.hf_dataset` loads `jlh-ibm/earnings_call` from HuggingFace for bulk training.
2. **Features** — `eca.features.build` runs FinBERT + hedging + guidance + tone-shift and writes a Parquet feature store.
3. **Labels** — `eca.prices.yfinance_loader` computes T+1 close-to-close returns relative to the SPY benchmark; label = sign of excess return.
4. **Model** — `eca.model.train` fits a LightGBM classifier under **walk-forward** cross-validation (no look-ahead). Tracked in MLflow.
5. **Backtest** — `eca.backtest.engine` simulates a long/short rule based on predicted probability; reports hit-rate, annualised Sharpe, max drawdown, vs buy-and-hold SPY.
6. **Serve** — `eca.api.main` exposes `POST /analyze` and `GET /backtest/{ticker}`. `streamlit_app.py` is a thin UI on top of the API.

## Quickstart

```powershell
# 1. install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 2. copy env and fill in your contact email for the SEC user-agent
copy .env.example .env

# 3. build a small demo dataset (uses HuggingFace earnings_call, ~5 min)
python -m eca.cli build-dataset --source hf --limit 200

# 4. train + log to MLflow
python -m eca.cli train

# 5. backtest
python -m eca.cli backtest --ticker AAPL

# 6. serve
uvicorn eca.api.main:app --reload
# in another shell:
streamlit run streamlit_app.py
```

## Results (placeholder — fill in after first run)

| Metric | Model | SPY buy-and-hold |
| --- | --- | --- |
| Directional accuracy (T+1) | _TBD_ | 0.52 |
| Annualised Sharpe | _TBD_ | _TBD_ |
| Max drawdown | _TBD_ | _TBD_ |
| Hit-rate on high-confidence calls (p>0.7) | _TBD_ | — |

A walkthrough of the EDA, feature importances, and backtest equity curve lives in [`notebooks/01_eda_and_model.ipynb`](notebooks/01_eda_and_model.ipynb).

## Roadmap

- [ ] **Docker** — `Dockerfile` + `docker-compose.yml` (API + Streamlit, FinBERT pre-baked). Coming once I have Docker Desktop available.
- [ ] **Real-data run** — execute the full HF-corpus pipeline and replace `_TBD_` above with real numbers.
- [ ] **Sector features** — add GICS sector code as a feature and sector-conditional confidence thresholds.
- [ ] **Transformer fine-tuning** — fine-tune a `DeBERTa-v3-base-finance` head on the direction label.
- [ ] **EDGAR webhook** — poll the EDGAR EFTS real-time feed for new 8-K filings and trigger the pipeline automatically.

## Project layout

```
src/eca/
  ingest/      # EDGAR + HuggingFace transcript loaders
  features/    # FinBERT, hedging, guidance, tone-shift
  prices/      # yfinance T+1 return labelling
  model/       # LightGBM + walk-forward CV + MLflow
  backtest/    # vectorized backtest + metrics
  api/         # FastAPI service
  utils/
tests/
notebooks/
scripts/
streamlit_app.py
```

## A note on data sources

This project **intentionally** uses SEC EDGAR (public, official) and licensed HuggingFace datasets rather than scraping paywalled transcript providers. That choice is itself part of the portfolio story: a quant researcher who respects data licensing is a quant researcher who won't get their employer sued.

## License

MIT
