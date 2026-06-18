"""Round-trip test for the model bundle: train a tiny model, save, reload, predict."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_labelled(tmp_path):
    """A 200-row labelled dataset where the label correlates with sent_polarity_mean."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "ticker": ["AAPL"] * n,
            "call_date": [date(2023, 1, 1) + timedelta(days=i * 3) for i in range(n)],
            "sent_pos_mean": rng.uniform(0.2, 0.6, n),
            "sent_neg_mean": rng.uniform(0.1, 0.4, n),
            "sent_neu_mean": rng.uniform(0.2, 0.5, n),
            "sent_pos_share": rng.uniform(0, 1, n),
            "sent_neg_share": rng.uniform(0, 1, n),
            "sent_polarity_mean": rng.uniform(-0.3, 0.5, n),
            "sent_polarity_std": rng.uniform(0, 0.2, n),
            "hedge_count": rng.integers(0, 30, n).astype(float),
            "hedge_ratio": rng.uniform(0, 0.05, n),
            "n_words": rng.integers(500, 5000, n).astype(float),
            "fls_count": rng.integers(0, 20, n).astype(float),
            "fls_ratio": rng.uniform(0, 0.2, n),
            "fls_positive": rng.uniform(0, 1, n),
            "fls_negative": rng.uniform(0, 1, n),
            "n_sentences": rng.integers(50, 500, n).astype(float),
            "prepared_len_chars": rng.integers(5000, 50000, n).astype(float),
            "qa_len_chars": rng.integers(2000, 20000, n).astype(float),
            "qa_ratio": rng.uniform(0.2, 0.6, n),
        }
    )
    # build QoQ delta columns the model expects
    for c in ["sent_pos_mean", "sent_neg_mean", "sent_polarity_mean", "hedge_ratio",
              "fls_positive", "fls_negative", "fls_ratio"]:
        df[f"{c}_dqoq"] = df[c].diff().fillna(0)

    # label correlates with polarity so the model has signal to learn
    df["label"] = (df["sent_polarity_mean"] + rng.normal(0, 0.1, n) > 0.1).astype(int)
    return df


def test_train_and_predict_roundtrip(synthetic_labelled, tmp_path):
    import joblib

    from eca.model.predict import Predictor
    from eca.model.train import train

    model_path = tmp_path / "classifier.joblib"
    result = train(synthetic_labelled, n_splits=3, model_path=model_path)

    assert model_path.exists()
    assert 0 <= result.mean_accuracy <= 1
    assert 0 <= result.mean_auc <= 1
    assert len(result.feature_importances) > 0

    bundle = joblib.load(model_path)
    predictor = Predictor(bundle)
    pred = predictor.predict_row({c: 0.0 for c in predictor.feature_columns})
    assert 0.0 <= pred.prob_up <= 1.0
    assert pred.direction in (0, 1)
    assert 0.0 <= pred.confidence <= 1.0

    probs = predictor.predict_frame(synthetic_labelled)
    assert probs.shape == (len(synthetic_labelled),)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_train_rejects_tiny_dataset():
    from eca.model.train import train

    df = pd.DataFrame({"label": [0, 1, 0]})
    with pytest.raises((ValueError, KeyError)):
        train(df, n_splits=5)
