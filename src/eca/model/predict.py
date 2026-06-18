"""Inference wrapper around the saved LightGBM bundle."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from eca.config import settings


@dataclass
class Prediction:
    prob_up: float
    direction: int  # 1 if up, 0 if down
    confidence: float  # |prob - 0.5| * 2

    def as_dict(self) -> dict[str, float | int]:
        return {"prob_up": self.prob_up, "direction": self.direction, "confidence": self.confidence}


class Predictor:
    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.feature_columns: list[str] = bundle["feature_columns"]

    def predict_row(self, features: dict) -> Prediction:
        row = {c: float(features.get(c, 0.0) or 0.0) for c in self.feature_columns}
        X = pd.DataFrame([row], columns=self.feature_columns)
        p = float(self.model.predict_proba(X)[:, 1][0])
        return Prediction(prob_up=p, direction=int(p >= 0.5), confidence=float(abs(p - 0.5) * 2))

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.feature_columns).astype(float).fillna(0.0)
        return self.model.predict_proba(X)[:, 1]


@lru_cache(maxsize=1)
def load_model(path: Path | str | None = None) -> Predictor:
    p = Path(path) if path else settings.model_path
    if not p.exists():
        raise FileNotFoundError(f"model not found at {p}; run `eca train` first")
    return Predictor(joblib.load(p))
