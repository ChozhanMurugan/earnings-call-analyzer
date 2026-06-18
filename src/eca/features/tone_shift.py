"""Quarter-over-quarter tone-shift features.

For each (ticker, call_date) row we compute the delta of the listed numeric
columns versus the same ticker's previous call. The intuition: the *change* in
tone vs the prior quarter carries information that the level does not.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_SHIFT_COLS: tuple[str, ...] = (
    "sent_pos_mean",
    "sent_neg_mean",
    "sent_polarity_mean",
    "hedge_ratio",
    "fls_positive",
    "fls_negative",
    "fls_ratio",
)


def add_qoq_tone_shifts(
    df: pd.DataFrame,
    *,
    shift_cols: tuple[str, ...] = DEFAULT_SHIFT_COLS,
    ticker_col: str = "ticker",
    date_col: str = "call_date",
    suffix: str = "_dqoq",
) -> pd.DataFrame:
    """Return ``df`` with new columns ``<col>{suffix}`` per ticker, sorted by date."""
    if df.empty:
        return df.copy()
    out = df.sort_values([ticker_col, date_col]).copy()
    for col in shift_cols:
        if col not in out.columns:
            continue
        out[f"{col}{suffix}"] = out.groupby(ticker_col)[col].diff()
    return out
