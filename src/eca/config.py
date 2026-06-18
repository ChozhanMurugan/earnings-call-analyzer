"""Centralised configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("ECA_DATA_DIR", "./data")).resolve()
    model_path: Path = Path(os.getenv("ECA_MODEL_PATH", "./data/models/classifier.joblib")).resolve()
    edgar_user_agent: str = os.getenv("EDGAR_USER_AGENT", "ECA Research contact@example.com")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    finbert_model: str = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


settings = Settings()
