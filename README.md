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

1. **Ingest** — `eca.ingest.edgar` pulls 8-K filings + exhibit transcripts from SEC EDGAR (free, official, no ToS issue). `eca.ingest.hf_dataset` loads `kurry/sp500_earnings_transcripts` (33 k S&P 500 calls, 2005–2025, Parquet format) from HuggingFace for bulk training.
2. **Features** — `eca.features.build` runs FinBERT + hedging + guidance + tone-shift and writes a Parquet feature store.
3. **Labels** — `eca.prices.yfinance_loader` computes T+1 close-to-close returns relative to the SPY benchmark; label = sign of excess return.
4. **Model** — `eca.model.train` fits a LightGBM classifier under **walk-forward** cross-validation (no look-ahead). Tracked in MLflow.
5. **Backtest** — `eca.backtest.engine` simulates a long/short rule based on predicted probability; reports hit-rate, annualised Sharpe, max drawdown, vs buy-and-hold SPY.
6. **Serve** — `eca.api.main` exposes `POST /analyze` and `GET /backtest/{ticker}`. `streamlit_app.py` is a thin UI on top of the API.

## Quickstart

**Docker (easiest — no local Python setup required):**
```bash
docker compose up   # API on :8000, Streamlit UI on :8501
```

**Local development:**
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

## Results

Trained on 150 S&P 500 earnings calls (2005–2025) from `kurry/sp500_earnings_transcripts`. Walk-forward CV (5 folds, `TimeSeriesSplit`). Backtest uses **out-of-fold** predictions — the model never sees the test fold during training.

| Metric | Model | SPY buy-and-hold |
| --- | --- | --- |
| Directional accuracy (T+1) | **54.4%** (5-fold walk-forward CV mean) | ~52% |
| Annualised Sharpe | **−0.47** (OOF backtest, threshold=0.10) | ~0.60 |
| Max drawdown | **−53.1%** (OOF, threshold=0.10) | — |
| Hit-rate on high-confidence calls (p>0.7) | **47.6%** (threshold=0.20, 63 trades) | — |
| CV mean AUC | **0.552** | — |

> **Honest note:** 150 transcripts is a small training set. The negative Sharpe reflects that — LightGBM needs more data to build a robust signal, and the earliest walk-forward folds barely have enough history to fit. The 54.4% directional accuracy (above the 52% SPY coin-flip) is the real take-away. Re-run with `--limit 500+` on a GPU to get a meaningfully-sized sample.

Top features by LightGBM split count: `fls_ratio`, `hedge_count`, `sent_pos_mean_dqoq`, `hedge_ratio`, `fls_negative_dqoq` — the QoQ delta and forward-guidance features dominate, consistent with the academic literature.

A walkthrough of the EDA, feature importances, and backtest equity curve lives in [`notebooks/01_eda_and_model.ipynb`](notebooks/01_eda_and_model.ipynb).

## Roadmap

- [x] **Docker** — `Dockerfile` (multi-stage, FinBERT baked in) + `docker-compose.yml` (API + Streamlit). Run with `docker compose up`.
- [x] **Real-data run** — 150-call pipeline executed; Results table filled with real walk-forward CV and OOF backtest numbers.
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
