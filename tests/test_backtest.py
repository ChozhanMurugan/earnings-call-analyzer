"""Backtest engine tests with deterministic synthetic data."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from eca.backtest import run_backtest


def _make_predictions(n: int, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [date(2024, 1, 1) + timedelta(days=i * 3) for i in range(n)]
    probs = rng.uniform(0, 1, size=n)
    # construct returns so the signal has positive edge: higher prob -> higher excess return
    base = (probs - 0.5) * 0.04 + rng.normal(0, 0.01, size=n)
    bench = rng.normal(0.0005, 0.01, size=n)
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * n,
            "call_date": dates,
            "prob_up": probs,
            "ret_excess": base,
            "ret_bench": bench,
        }
    )


class TestBacktest:
    def test_empty_input_returns_zero_trades(self):
        df = pd.DataFrame(columns=["ticker", "call_date", "prob_up", "ret_excess", "ret_bench"])
        res = run_backtest(df)
        assert res.n_trades == 0
        assert res.total_return == 0
        assert res.equity_curve.empty

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"prob_up": [0.5]})
        with pytest.raises(ValueError, match="missing columns"):
            run_backtest(df)

    def test_long_signal_when_prob_above_half(self):
        df = pd.DataFrame(
            {
                "ticker": ["A"],
                "call_date": [date(2024, 1, 1)],
                "prob_up": [0.9],
                "ret_excess": [0.05],
                "ret_bench": [0.01],
            }
        )
        res = run_backtest(df, threshold=0.0)
        assert res.n_trades == 1
        assert res.hit_rate == 1.0
        assert res.mean_trade_return == pytest.approx(0.05)

    def test_short_signal_when_prob_below_half(self):
        df = pd.DataFrame(
            {
                "ticker": ["A"],
                "call_date": [date(2024, 1, 1)],
                "prob_up": [0.1],
                "ret_excess": [-0.05],  # stock falls — short profits
                "ret_bench": [0.01],
            }
        )
        res = run_backtest(df, threshold=0.0)
        assert res.n_trades == 1
        assert res.hit_rate == 1.0
        assert res.mean_trade_return == pytest.approx(0.05)

    def test_threshold_filters_low_confidence(self):
        df = pd.DataFrame(
            {
                "ticker": ["A", "B"],
                "call_date": [date(2024, 1, 1), date(2024, 1, 2)],
                "prob_up": [0.51, 0.95],  # only the second clears threshold=0.4
                "ret_excess": [0.02, 0.03],
                "ret_bench": [0.0, 0.0],
            }
        )
        res = run_backtest(df, threshold=0.4)
        assert res.n_trades == 1

    def test_sharpe_and_drawdown_finite_on_synthetic(self):
        df = _make_predictions(120)
        res = run_backtest(df, threshold=0.05)
        assert res.n_trades > 0
        assert np.isfinite(res.annualised_sharpe)
        assert -1 <= res.max_drawdown <= 0
        assert res.equity_curve.shape[0] > 0
        assert {"date", "strategy_equity", "benchmark_equity"} <= set(res.equity_curve.columns)

    def test_strategy_beats_random_when_edge_exists(self):
        # synthetic data is built so prob_up correlates with returns
        df = _make_predictions(200, seed=42)
        res = run_backtest(df, threshold=0.1)
        # with real edge baked in, hit rate should clear 50%
        assert res.hit_rate > 0.5
