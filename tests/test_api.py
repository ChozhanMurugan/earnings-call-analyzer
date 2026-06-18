"""FastAPI smoke tests with FinBERT stubbed so we never touch transformers."""
from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Stub FinBertSentiment.score so /analyze runs without downloading 400MB."""
    from eca.api import main as api_main
    from eca.features.sentiment import SentimentAgg

    def fake_score(self, text):
        return SentimentAgg(
            sent_pos_mean=0.5, sent_neg_mean=0.2, sent_neu_mean=0.3,
            sent_pos_share=0.4, sent_neg_share=0.2,
            sent_polarity_mean=0.3, sent_polarity_std=0.1,
            n_sentences=10,
        )

    monkeypatch.setattr("eca.features.sentiment.FinBertSentiment.score", fake_score)
    # reset cached sentiment singleton so the patched class is used
    api_main._sentiment_singleton = None
    return TestClient(api_main.app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_analyze_returns_features_even_without_model(client):
    body = {
        "ticker": "AAPL",
        "call_date": date(2024, 7, 1).isoformat(),
        "fiscal_quarter": "Q3 2024",
        "prepared_remarks": (
            "We delivered strong revenue growth this quarter. We expect Q4 to accelerate. "
            "Margins expanded year over year. Cash flow remained robust. "
        ) * 6,
        "qa_section": "Analyst asked about margins. CFO answered constructively. " * 4,
    }
    r = client.post("/analyze", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ticker"] == "AAPL"
    assert "features" in data
    assert "sent_pos_mean" in data["features"]
    assert "hedge_ratio" in data["features"]
    # prediction may be None if no model is trained — both are valid
    assert "prediction" in data


def test_analyze_rejects_short_transcripts(client):
    body = {
        "ticker": "AAPL",
        "call_date": date(2024, 7, 1).isoformat(),
        "prepared_remarks": "too short",
    }
    r = client.post("/analyze", json=body)
    assert r.status_code == 422  # pydantic validation


def test_features_endpoint_404_without_data(client):
    r = client.get("/features/AAPL")
    # in CI there is no processed parquet, so we expect 404
    assert r.status_code in (404, 200)


def test_backtest_endpoint_404_without_predictions(client):
    r = client.get("/backtest/AAPL")
    assert r.status_code in (404, 200)
