# Earnings Call Analyzer

[![CI](https://github.com/ChozhanMurugan/earnings-call-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/ChozhanMurugan/earnings-call-analyzer/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/ChozhanM/earnings-call-analyze)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A production-grade NLP and machine learning pipeline that listens to how companies **talk** during earnings calls — and predicts whether the stock will beat the market the next day.

**Try the live demo:** [huggingface.co/spaces/ChozhanM/earnings-call-analyze](https://huggingface.co/spaces/ChozhanM/earnings-call-analyze) — paste any transcript, get a trade signal instantly.

**Read the full walkthrough:** [`notebooks/01_eda_and_model.ipynb`](notebooks/01_eda_and_model.ipynb) — end-to-end in ~30 seconds, no GPU, no API keys required.

---

## The idea

Every quarter, thousands of companies hold earnings calls. The headline numbers — EPS beat or miss — get priced in within milliseconds by algorithmic traders. But the *language* management uses tells a deeper story.

Are executives hedging more than last quarter? Are they giving confident forward guidance or retreating into vague platitudes? Is the CFO's tone shifting even when the numbers look fine?

Academic research (Loughran & McDonald 2011; Hassan et al. 2019) has shown these linguistic signals carry real predictive power for next-day returns. This project turns that research into a working, deployed system — from raw transcript text to a live trade signal — built entirely with open-source tools and publicly available data.

```
Earnings call transcript
        │
        ▼
┌───────────────────────┐
│   FinBERT sentiment   │  ← finance-tuned BERT, sentence-by-sentence
│   Hedge-word density  │  ← Loughran-McDonald uncertainty dictionary
│   Forward guidance    │  ← rule-based detector for guidance language
│   QoQ tone shift      │  ← how has management's language changed?
│   Call structure      │  ← prepared remarks length, Q&A depth
└───────────────────────┘
        │
        ▼
  LightGBM classifier
  (walk-forward CV, no look-ahead bias)
        │
        ▼
  Long / Short signal + probability score
        │
        ▼
  Vectorized backtest vs SPY benchmark
        │
        ▼
  FastAPI + Streamlit — live on HuggingFace Spaces
```

---

## What makes this project technically interesting

### 1. Finance-specific NLP, not generic sentiment
Generic sentiment models (VADER, TextBlob) were not trained on financial language. When a CFO says "we remain cautious about near-term headwinds," a generic model might score that neutrally. **FinBERT** (`ProsusAI/finbert`), fine-tuned on financial communications, correctly reads that as bearish. The pipeline runs FinBERT on every sentence of the transcript — not the whole text as a blob — and aggregates across sentences to get mean, standard deviation, and share-positive/negative scores.

### 2. Quarter-on-quarter tone shift as a feature
The single most important insight in this project is that the **change** in tone matters more than the level. A company that always sounds cautious is priced for caution. A company that suddenly sounds more cautious than last quarter is a signal. Every feature is computed both in absolute terms and as a delta versus the same ticker's prior call. The LightGBM model confirms this — `sent_pos_mean_dqoq` (QoQ delta of positive sentiment mean) is one of the top 5 features by split count.

### 3. Walk-forward cross-validation — no look-ahead bias
Standard k-fold cross-validation would train on future data and test on past data, which is cheating in time series problems. This pipeline uses `TimeSeriesSplit` (5 folds), which strictly ensures the model is always trained on the past and evaluated on the future — the same discipline a real quant fund would apply.

### 4. Excess returns, not raw returns
The label is not "did the stock go up?" — it's "did the stock beat SPY on T+1?" This isolates the idiosyncratic move driven by the earnings call itself from broad market noise. A stock that rises 1% when the market rises 2% is a short signal, not a long one.

### 5. Honest evaluation with out-of-fold predictions
The backtest uses **out-of-fold predictions only** — predictions the model made before it ever saw that data point. This is the correct way to backtest an ML model. The in-sample predictions (which showed a suspiciously perfect 100% hit rate and Sharpe of 15.8) were identified as data leakage and discarded. The reported numbers are real.

### 6. Production-grade packaging
The entire system runs in a single `docker compose up` — multi-stage Dockerfile, FinBERT weights baked in at build time, healthcheck, non-root user, GitHub Actions CI pipeline. The same image runs locally and on HuggingFace Spaces.

---

## Features extracted per transcript

| Feature group | What it captures |
| --- | --- |
| **FinBERT sentiment** | Per-sentence positive / neutral / negative probabilities from `ProsusAI/finbert`, aggregated to mean, std, share-positive, share-negative across the full call |
| **Hedge-word density** | Normalised count of uncertainty and weak-modal language from the Loughran-McDonald financial dictionary (`might`, `could`, `approximately`, `we believe`, …) |
| **Forward-guidance language** | Rule-based detector for forward-looking statements (`we expect`, `we anticipate`, `next quarter`, `full year guidance`, …) plus the FinBERT polarity of the sentences surrounding them |
| **QoQ tone shift** | Delta of every feature above against the same ticker's prior quarter call — the change in management's language, not just the level |
| **Call structure** | Prepared-remarks length, Q&A section length, number of analyst questions — structural signals about how management chose to present information |

---

## The full pipeline

| Step | Module | What happens |
| --- | --- | --- |
| **1. Ingest** | `eca.ingest.edgar` | Pulls 8-K filings and earnings call exhibits directly from SEC EDGAR — free, official, no terms-of-service issues |
| | `eca.ingest.hf_dataset` | Loads `kurry/sp500_earnings_transcripts` (33,000 S&P 500 calls, 2005–2025) from HuggingFace for bulk training |
| **2. Feature extraction** | `eca.features.build` | Runs FinBERT, computes hedge/guidance/structure features, calculates QoQ deltas, writes a Parquet feature store |
| **3. Label generation** | `eca.prices.yfinance_loader` | Downloads T+1 closing prices via yfinance, computes excess return vs SPY, assigns binary long/short label |
| **4. Model training** | `eca.model.train` | Fits LightGBM under walk-forward CV (TimeSeriesSplit, 5 folds), logs metrics and feature importances to MLflow |
| **5. Backtest** | `eca.backtest.engine` | Simulates a long/short trading rule on out-of-fold predictions; reports hit-rate, annualised Sharpe, max drawdown vs buy-and-hold SPY |
| **6. Serve** | `eca.api.main` | FastAPI service — `POST /analyze` scores a new transcript, `GET /backtest/{ticker}` returns historical performance |
| **7. UI** | `streamlit_app.py` | Streamlit dashboard on top of the API — live on HuggingFace Spaces |

---

## Results

Trained on 150 S&P 500 earnings calls (2005–2025) via `kurry/sp500_earnings_transcripts`. Walk-forward CV (5 folds, `TimeSeriesSplit`). Backtest uses **out-of-fold predictions only** — the model never touches the test fold during training.

| Metric | Model | SPY buy-and-hold |
| --- | --- | --- |
| Directional accuracy (T+1) | **54.4%** (5-fold walk-forward CV mean) | ~52% |
| CV mean AUC | **0.552** | — |
| Annualised Sharpe | **−0.47** (OOF backtest, confidence threshold = 0.10) | ~0.60 |
| Max drawdown | **−53.1%** (OOF, threshold = 0.10) | — |
| Hit-rate, high-confidence trades | **47.6%** (threshold = 0.20, 63 trades) | — |

**On the negative Sharpe:** 150 transcripts is a deliberately small dataset for this initial run — the earliest walk-forward folds have as few as 20 training rows, which is not enough for LightGBM to learn robust patterns. The 54.4% directional accuracy (above the ~52% SPY baseline) is the meaningful signal. The in-sample backtest produced a suspiciously perfect Sharpe of 15.8 and 100% hit rate — that was identified as data leakage (the final model had seen all training data), corrected, and the honest out-of-fold numbers are reported here instead. Scaling to 500+ calls on a GPU would produce meaningfully stronger results.

**Top features by LightGBM split count:**

| Rank | Feature | Interpretation |
| --- | --- | --- |
| 1 | `fls_ratio` | Share of forward-looking sentences in the call |
| 2 | `hedge_count` | Raw count of hedging / uncertainty words |
| 3 | `sent_pos_mean_dqoq` | QoQ change in mean positive sentiment |
| 4 | `hedge_ratio` | Hedge words normalised by call length |
| 5 | `fls_negative_dqoq` | QoQ change in negative forward-guidance sentiment |

The QoQ delta and forward-guidance features dominate — exactly what the academic literature predicts.

---

## Quickstart

**Try it live (no setup required):**
```
https://huggingface.co/spaces/ChozhanM/earnings-call-analyze
```

**Run with Docker:**
```bash
docker compose up
# API available at http://localhost:8000
# Streamlit UI at http://localhost:8501
```

**Run locally:**
```powershell
# Install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Configure (add your email for the SEC EDGAR user-agent header)
copy .env.example .env

# Build a dataset (downloads ~150 calls from HuggingFace, ~30 min on CPU)
python -m eca.cli build-dataset --source hf --limit 150

# Train the model
python -m eca.cli train

# Run the backtest
python -m eca.cli backtest

# Start the API and UI
uvicorn eca.api.main:app --reload
streamlit run streamlit_app.py   # in a second terminal
```

---

## Technology stack

| Layer | Technology | Why |
| --- | --- | --- |
| NLP model | `ProsusAI/finbert` (HuggingFace Transformers) | Finance-domain BERT — outperforms generic sentiment on earnings language |
| ML model | LightGBM | Fast gradient boosting, handles tabular features well, strong baseline for structured financial data |
| Data | SEC EDGAR (official) + HuggingFace `kurry/sp500_earnings_transcripts` | Free, public, legally sound — no scraping of paywalled providers |
| Prices | yfinance | T+1 return labels and SPY benchmark |
| Experiment tracking | MLflow | Logs CV metrics and feature importances per training run |
| API | FastAPI + Pydantic | Typed, async, auto-documented at `/docs` |
| UI | Streamlit | Rapid dashboard prototyping on top of the API |
| Containerisation | Docker (multi-stage) + docker-compose | Single `docker compose up` to run the full stack |
| CI | GitHub Actions | Runs tests and builds the Docker image on every push |
| Deployment | HuggingFace Spaces (Docker SDK) | Free public hosting, accessible to anyone without setup |

---

## Project layout

```
src/eca/
├── ingest/       # SEC EDGAR and HuggingFace transcript loaders
├── features/     # FinBERT inference, hedge/guidance/structure extraction, QoQ delta
├── prices/       # yfinance T+1 return labelling vs SPY benchmark
├── model/        # LightGBM training, walk-forward CV, MLflow logging
├── backtest/     # Vectorized long/short backtest engine and metrics
├── api/          # FastAPI application (analyze + backtest endpoints)
└── utils/        # Shared utilities

tests/            # pytest suite (27 tests, all passing)
notebooks/        # EDA, feature importances, backtest equity curve
scripts/          # oof_backtest.py — honest out-of-fold evaluation helper
spaces/           # HuggingFace Spaces deployment (Dockerfile + startup.sh)
streamlit_app.py  # Streamlit dashboard
```

---

## A note on data ethics

This project intentionally uses SEC EDGAR (public government data) and a licensed HuggingFace dataset rather than scraping paywalled transcript providers. That choice reflects a deliberate principle: financial ML research should be reproducible and legally sound, not dependent on grey-area data collection that would be unusable in a professional setting.

---

## Roadmap

- [x] End-to-end pipeline: ingest → features → model → backtest → API → UI
- [x] Docker packaging (multi-stage, FinBERT baked in, healthcheck)
- [x] Real-data run with honest walk-forward CV and OOF backtest results
- [x] Live deployment on HuggingFace Spaces
- [ ] Scale to 5,000+ calls for a broader ticker universe
- [ ] Add GICS sector code as a feature with sector-conditional confidence thresholds
- [ ] Fine-tune a `DeBERTa-v3-base-finance` head directly on the direction label
- [ ] EDGAR webhook — poll the real-time EFTS feed and trigger the pipeline on new 8-K filings automatically

---

## License

MIT
