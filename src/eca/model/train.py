"""LightGBM directional classifier with walk-forward (time-series) CV.

Why walk-forward? Earnings-call returns are non-IID and time-ordered.
A random K-fold leaks future information into the training set and produces
optimistic accuracy. Walk-forward respects causality: each fold trains on data
strictly older than its test window.

Outputs
-------
- ``data/models/classifier.joblib`` — the fitted model + feature schema
- MLflow run with cv metrics and feature importances
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from eca.config import settings
from eca.features.build import FEATURE_COLUMNS
from eca.utils import logger

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 400,
}


@dataclass
class CVMetrics:
    accuracy: float
    log_loss: float
    roc_auc: float
    n_train: int
    n_test: int


@dataclass
class TrainResult:
    metrics: list[CVMetrics]
    mean_accuracy: float
    mean_auc: float
    feature_importances: dict[str, float]
    model_path: Path


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Drop unusable rows, return X / y / used columns. Sorted by call_date."""
    df = df.dropna(subset=["label"]).copy()
    df["call_date"] = pd.to_datetime(df["call_date"])
    df = df.sort_values("call_date").reset_index(drop=True)

    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not cols:
        raise ValueError("no FEATURE_COLUMNS present — did you run build_features_df + attach_labels?")
    X = df[cols].astype(float).fillna(0.0)
    y = df["label"].astype(int)
    return X, y, cols


def train(
    features_with_labels: pd.DataFrame,
    *,
    n_splits: int = 5,
    params: dict | None = None,
    model_path: Path | None = None,
    mlflow_experiment: str = "eca_directional",
) -> TrainResult:
    """Walk-forward CV, then refit on full data, save model, log to MLflow."""
    import lightgbm as lgb

    params = {**DEFAULT_PARAMS, **(params or {})}
    model_path = model_path or settings.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)

    X, y, cols = _prepare(features_with_labels)
    if len(X) < n_splits + 5:
        raise ValueError(f"need at least {n_splits + 5} labelled rows, got {len(X)}")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[CVMetrics] = []

    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)
        run_ctx = mlflow.start_run()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"mlflow disabled ({exc})")
        mlflow = None  # type: ignore[assignment]
        run_ctx = _NullCtx()

    with run_ctx:
        if mlflow is not None:
            mlflow.log_params(params)
            mlflow.log_param("n_splits", n_splits)
            mlflow.log_param("n_features", len(cols))
            mlflow.log_param("n_rows", len(X))

        for fold, (tr, te) in enumerate(splitter.split(X), 1):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]
            if y_tr.nunique() < 2 or y_te.nunique() < 2:
                logger.warning(f"fold {fold}: degenerate label distribution, skipping")
                continue
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(30, verbose=False)])
            p = model.predict_proba(X_te)[:, 1]
            yhat = (p >= 0.5).astype(int)
            m = CVMetrics(
                accuracy=float(accuracy_score(y_te, yhat)),
                log_loss=float(log_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6))),
                roc_auc=float(roc_auc_score(y_te, p)),
                n_train=int(len(tr)),
                n_test=int(len(te)),
            )
            fold_metrics.append(m)
            logger.info(f"fold {fold}: acc={m.accuracy:.3f} auc={m.roc_auc:.3f} ll={m.log_loss:.3f}")
            if mlflow is not None:
                mlflow.log_metrics(
                    {f"fold{fold}_acc": m.accuracy, f"fold{fold}_auc": m.roc_auc, f"fold{fold}_ll": m.log_loss}
                )

        # final fit on everything
        final = lgb.LGBMClassifier(**params)
        final.fit(X, y)
        importances = dict(sorted(zip(cols, final.feature_importances_.tolist(), strict=True), key=lambda kv: -kv[1]))

        bundle = {"model": final, "feature_columns": cols, "params": params}
        joblib.dump(bundle, model_path)
        logger.info(f"saved model -> {model_path}")

        mean_acc = float(np.mean([m.accuracy for m in fold_metrics])) if fold_metrics else float("nan")
        mean_auc = float(np.mean([m.roc_auc for m in fold_metrics])) if fold_metrics else float("nan")
        if mlflow is not None:
            mlflow.log_metric("cv_mean_accuracy", mean_acc)
            mlflow.log_metric("cv_mean_auc", mean_auc)
            mlflow.log_dict({k: float(v) for k, v in importances.items()}, "feature_importances.json")
            mlflow.log_artifact(str(model_path))

        return TrainResult(
            metrics=fold_metrics,
            mean_accuracy=mean_acc,
            mean_auc=mean_auc,
            feature_importances=importances,
            model_path=model_path,
        )


def metrics_to_frame(result: TrainResult) -> pd.DataFrame:
    return pd.DataFrame([asdict(m) for m in result.metrics])


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
