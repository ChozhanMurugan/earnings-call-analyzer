"""FastAPI service exposing the earnings-call analyzer.

Endpoints
---------
GET  /healthz                 — liveness probe
POST /analyze                 — score a raw transcript text, returns features + prediction
GET  /features/{ticker}       — latest computed features for a ticker (from processed parquet)
GET  /backtest/{ticker}       — run backtest restricted to a single ticker (uses cached predictions)
GET  /backtest                — full-universe backtest

Run with:
    uvicorn eca.api.main:app --reload
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from eca.backtest import run_backtest
from eca.config import settings
from eca.features.build import FEATURE_COLUMNS, build_features
from eca.features.sentiment import FinBertSentiment
from eca.ingest.schema import Transcript
from eca.model.predict import load_model

app = FastAPI(
    title="Earnings Call Analyzer",
    version="0.1.0",
    description="FinBERT + hedging + forward-guidance features → directional classifier → backtest.",
)

_PREDICTIONS_PATH = settings.processed_dir / "predictions.parquet"
_sentiment_singleton: FinBertSentiment | None = None


def _sentiment() -> FinBertSentiment:
    global _sentiment_singleton
    if _sentiment_singleton is None:
        _sentiment_singleton = FinBertSentiment()
    return _sentiment_singleton


# ----- request / response models -----

class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., json_schema_extra={"example": "AAPL"})
    call_date: date = Field(..., json_schema_extra={"example": "2024-08-01"})
    fiscal_quarter: str | None = Field(None, json_schema_extra={"example": "Q3 2024"})
    prepared_remarks: str = Field(..., min_length=200)
    qa_section: str = Field("", json_schema_extra={"example": ""})


class AnalyzeResponse(BaseModel):
    ticker: str
    call_date: date
    features: dict
    prediction: dict | None = None
    note: str | None = None


class BacktestResponse(BaseModel):
    n_trades: int
    hit_rate: float
    mean_trade_return: float
    total_return: float
    annualised_sharpe: float
    max_drawdown: float
    benchmark_total_return: float
    threshold: float
    equity_curve: list[dict]


# ----- routes -----

@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    transcript = Transcript(
        ticker=req.ticker.upper(),
        call_date=req.call_date,
        fiscal_quarter=req.fiscal_quarter,
        prepared_remarks=req.prepared_remarks,
        qa_section=req.qa_section,
        source="api",
    )
    feats = build_features(transcript, sentiment=_sentiment())

    pred: dict | None = None
    note: str | None = None
    try:
        predictor = load_model()
        # api inputs have no QoQ deltas; fill with 0 — caller can supply them explicitly
        for c in FEATURE_COLUMNS:
            feats.setdefault(c, 0.0)
        pred = predictor.predict_row(feats).as_dict()
    except FileNotFoundError:
        note = "model not yet trained; features only"

    cleaned = {k: v for k, v in feats.items() if k not in {"ticker", "call_date", "fiscal_quarter", "source"}}
    return AnalyzeResponse(
        ticker=req.ticker.upper(),
        call_date=req.call_date,
        features=cleaned,
        prediction=pred,
        note=note,
    )


@app.get("/features/{ticker}")
def features_for_ticker(ticker: str) -> dict:
    path = settings.processed_dir / "features_labelled.parquet"
    if not path.exists():
        raise HTTPException(404, detail=f"no processed features at {path}")
    df = pd.read_parquet(path)
    sub = df[df["ticker"].str.upper() == ticker.upper()].sort_values("call_date")
    if sub.empty:
        raise HTTPException(404, detail=f"no rows for ticker {ticker}")
    return {"ticker": ticker.upper(), "rows": sub.to_dict(orient="records")}


@app.get("/backtest/{ticker}", response_model=BacktestResponse)
def backtest_ticker(ticker: str, threshold: float = Query(0.0, ge=0.0, le=0.5)) -> BacktestResponse:
    return _run_backtest_with_filter(ticker=ticker.upper(), threshold=threshold)


@app.get("/backtest", response_model=BacktestResponse)
def backtest_all(threshold: float = Query(0.0, ge=0.0, le=0.5)) -> BacktestResponse:
    return _run_backtest_with_filter(ticker=None, threshold=threshold)


# ----- helpers -----

def _run_backtest_with_filter(*, ticker: str | None, threshold: float) -> BacktestResponse:
    df = _load_predictions()
    if ticker is not None:
        df = df[df["ticker"].str.upper() == ticker]
        if df.empty:
            raise HTTPException(404, detail=f"no predictions for {ticker}")
    res = run_backtest(df, threshold=threshold)
    return BacktestResponse(**res.summary(), equity_curve=res.equity_curve.assign(
        date=res.equity_curve["date"].astype(str)
    ).to_dict(orient="records"))


def _load_predictions() -> pd.DataFrame:
    if not _PREDICTIONS_PATH.exists():
        raise HTTPException(
            404,
            detail=(
                f"no cached predictions at {_PREDICTIONS_PATH}; "
                "run `python -m eca.cli predict` after training"
            ),
        )
    return pd.read_parquet(_PREDICTIONS_PATH)


# small helper so uvicorn `--factory` works too
def create_app() -> FastAPI:
    return app
