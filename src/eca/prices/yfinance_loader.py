"""T+1 return labels using yfinance close prices.

Label convention
----------------
For each (ticker, call_date) row we compute the close-to-close return from the
first trading day on/after the call to the *next* trading day, then subtract
SPY's same-window return to get an *excess* return. Label = sign(excess).

Why excess vs raw? An earnings beat on a day the whole market drops 3% still
implies the call carried information; we want to isolate that.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from eca.utils import logger


def _download(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(
        tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    # Normalise to: index=date, columns=ticker, values=close
    if isinstance(data.columns, pd.MultiIndex):
        close = pd.DataFrame({t: data[t]["Close"] for t in tickers if t in data.columns.levels[0]})
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close.sort_index()


def get_excess_returns(
    df: pd.DataFrame,
    *,
    ticker_col: str = "ticker",
    date_col: str = "call_date",
    benchmark: str = "SPY",
    horizon_days: int = 1,
) -> pd.DataFrame:
    """Return ``df`` with new columns: ``ret_raw``, ``ret_bench``, ``ret_excess``, ``label``.

    ``label`` is 1 if ret_excess > 0 else 0. Rows that can't be priced are dropped.
    """
    if df.empty:
        return df.assign(ret_raw=np.nan, ret_bench=np.nan, ret_excess=np.nan, label=np.nan)

    tickers = sorted(df[ticker_col].dropna().unique().tolist())
    start = pd.to_datetime(df[date_col]).min().date() - timedelta(days=7)
    end = pd.to_datetime(df[date_col]).max().date() + timedelta(days=horizon_days + 10)

    logger.info(f"downloading prices for {len(tickers)} tickers + {benchmark} from {start} to {end}")
    prices = _download(tickers + [benchmark], start, end)

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    rets_raw, rets_bench = [], []

    for _, row in out.iterrows():
        t = row[ticker_col]
        d = row[date_col]
        rets_raw.append(_horizon_return(prices, t, d, horizon_days))
        rets_bench.append(_horizon_return(prices, benchmark, d, horizon_days))

    out["ret_raw"] = rets_raw
    out["ret_bench"] = rets_bench
    out["ret_excess"] = out["ret_raw"] - out["ret_bench"]
    out["label"] = (out["ret_excess"] > 0).astype("Int64")
    out.loc[out["ret_excess"].isna(), "label"] = pd.NA
    return out


def _horizon_return(prices: pd.DataFrame, ticker: str, call_date: pd.Timestamp, horizon: int) -> float:
    if ticker not in prices.columns:
        return np.nan
    s = prices[ticker].dropna()
    # first trading day on/after the call
    fwd = s.loc[s.index >= call_date]
    if len(fwd) < horizon + 1:
        return np.nan
    p0 = fwd.iloc[0]
    p1 = fwd.iloc[horizon]
    if p0 == 0 or np.isnan(p0):
        return np.nan
    return float(p1 / p0 - 1.0)


def attach_labels(features: pd.DataFrame) -> pd.DataFrame:
    """Convenience: drop rows we can't label."""
    labelled = get_excess_returns(features)
    before = len(labelled)
    labelled = labelled.dropna(subset=["label"])
    logger.info(f"label coverage: {len(labelled)}/{before} rows")
    return labelled
