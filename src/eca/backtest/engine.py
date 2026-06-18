"""Vectorized event-driven backtest.

Strategy
--------
For each labelled call we have:
    - ``prob_up`` from the classifier (out-of-fold for honesty)
    - ``ret_excess`` realised T+1 excess return vs SPY

A position is taken at the close of day T (call day) and unwound at the close
of day T+1:
    - long  if prob_up >= 0.5 + threshold
    - short if prob_up <= 0.5 - threshold
    - flat  otherwise

We sum per-day PnL across all triggered positions (equal-weighted) and report
hit-rate, mean trade return, annualised Sharpe and max drawdown. The benchmark
is buy-and-hold SPY over the same date range.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    n_trades: int
    hit_rate: float
    mean_trade_return: float
    total_return: float
    annualised_sharpe: float
    max_drawdown: float
    benchmark_total_return: float
    threshold: float
    equity_curve: pd.DataFrame  # cols: date, strategy_equity, benchmark_equity

    def summary(self) -> dict[str, float | int]:
        return {
            "n_trades": self.n_trades,
            "hit_rate": self.hit_rate,
            "mean_trade_return": self.mean_trade_return,
            "total_return": self.total_return,
            "annualised_sharpe": self.annualised_sharpe,
            "max_drawdown": self.max_drawdown,
            "benchmark_total_return": self.benchmark_total_return,
            "threshold": self.threshold,
        }


def run_backtest(
    df: pd.DataFrame,
    *,
    threshold: float = 0.0,
    prob_col: str = "prob_up",
    ret_col: str = "ret_excess",
    bench_col: str = "ret_bench",
    date_col: str = "call_date",
) -> BacktestResult:
    """Run the long/short rule on a prediction dataframe.

    ``df`` must contain ``prob_col``, ``ret_col``, ``bench_col`` and a date column.
    """
    required = {prob_col, ret_col, date_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"backtest input missing columns: {sorted(missing)}")

    work = df.dropna(subset=[prob_col, ret_col]).copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.sort_values(date_col).reset_index(drop=True)

    signal = np.where(
        work[prob_col] >= 0.5 + threshold, 1,
        np.where(work[prob_col] <= 0.5 - threshold, -1, 0),
    )
    work["signal"] = signal
    work["trade_ret"] = work["signal"] * work[ret_col]

    trades = work[work["signal"] != 0]
    n_trades = int(len(trades))
    if n_trades == 0:
        return BacktestResult(
            n_trades=0, hit_rate=float("nan"), mean_trade_return=float("nan"),
            total_return=0.0, annualised_sharpe=float("nan"), max_drawdown=0.0,
            benchmark_total_return=_safe_compound(work.get(bench_col, pd.Series(dtype=float))),
            threshold=threshold, equity_curve=pd.DataFrame(columns=["date", "strategy_equity", "benchmark_equity"]),
        )

    hit_rate = float((trades["trade_ret"] > 0).mean())
    mean_trade_return = float(trades["trade_ret"].mean())

    # equity: aggregate trades by date (equal-weight intraday-of-day average)
    daily = trades.groupby(work[date_col].dt.normalize())["trade_ret"].mean()
    eq = (1 + daily).cumprod()
    total_return = float(eq.iloc[-1] - 1) if len(eq) else 0.0

    # annualised Sharpe on daily strategy returns (assume rf=0)
    if daily.std(ddof=0) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=0) * np.sqrt(TRADING_DAYS))
    else:
        sharpe = float("nan")

    drawdown = float(_max_drawdown(eq)) if len(eq) else 0.0

    # benchmark over the same dates
    if bench_col in work.columns:
        bench_daily = work.groupby(work[date_col].dt.normalize())[bench_col].mean().reindex(eq.index).fillna(0.0)
        bench_eq = (1 + bench_daily).cumprod()
        bench_total = float(bench_eq.iloc[-1] - 1) if len(bench_eq) else 0.0
    else:
        bench_eq = pd.Series(1.0, index=eq.index)
        bench_total = 0.0

    curve = pd.DataFrame(
        {"date": eq.index, "strategy_equity": eq.values, "benchmark_equity": bench_eq.values}
    ).reset_index(drop=True)

    return BacktestResult(
        n_trades=n_trades,
        hit_rate=hit_rate,
        mean_trade_return=mean_trade_return,
        total_return=total_return,
        annualised_sharpe=sharpe,
        max_drawdown=drawdown,
        benchmark_total_return=bench_total,
        threshold=threshold,
        equity_curve=curve,
    )


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def _safe_compound(s: pd.Series) -> float:
    if s.empty:
        return 0.0
    s = s.dropna()
    if s.empty:
        return 0.0
    return float((1 + s).prod() - 1)
