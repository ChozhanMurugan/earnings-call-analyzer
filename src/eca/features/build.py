"""Orchestrates per-transcript feature extraction into a DataFrame row."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from eca.features.guidance import guidance_features
from eca.features.hedging import hedge_features
from eca.features.sentiment import FinBertSentiment
from eca.features.tone_shift import add_qoq_tone_shifts
from eca.ingest.schema import Transcript
from eca.utils import logger

# Columns the model will see (after QoQ join). Kept in one place so the API
# can validate inputs.
FEATURE_COLUMNS: list[str] = [
    "sent_pos_mean", "sent_neg_mean", "sent_neu_mean",
    "sent_pos_share", "sent_neg_share",
    "sent_polarity_mean", "sent_polarity_std",
    "hedge_count", "hedge_ratio", "n_words",
    "fls_count", "fls_ratio", "fls_positive", "fls_negative",
    "n_sentences",
    "prepared_len_chars", "qa_len_chars", "qa_ratio",
    # QoQ deltas added by add_qoq_tone_shifts
    "sent_pos_mean_dqoq", "sent_neg_mean_dqoq", "sent_polarity_mean_dqoq",
    "hedge_ratio_dqoq", "fls_positive_dqoq", "fls_negative_dqoq", "fls_ratio_dqoq",
]


def build_features(transcript: Transcript, sentiment: FinBertSentiment | None = None) -> dict:
    """Extract a flat feature dict for one transcript. No QoQ delta — that needs history."""
    sentiment = sentiment or FinBertSentiment()
    full = transcript.full_text

    sent = sentiment.score(full).as_dict()
    hedge = hedge_features(full)
    fls = guidance_features(full)

    prepared_len = len(transcript.prepared_remarks)
    qa_len = len(transcript.qa_section)
    qa_ratio = qa_len / (prepared_len + qa_len) if (prepared_len + qa_len) else 0.0

    row = {
        "ticker": transcript.ticker,
        "call_date": transcript.call_date,
        "fiscal_quarter": transcript.fiscal_quarter,
        "source": transcript.source,
        **sent,
        **hedge,
        **fls,
        "prepared_len_chars": prepared_len,
        "qa_len_chars": qa_len,
        "qa_ratio": qa_ratio,
    }
    return row


def build_features_df(transcripts: Iterable[Transcript]) -> pd.DataFrame:
    """Run features over a stream of transcripts and add QoQ tone-shifts."""
    sentiment = FinBertSentiment()
    rows: list[dict] = []
    for i, t in enumerate(transcripts, 1):
        try:
            rows.append(build_features(t, sentiment=sentiment))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"skip {t.ticker} {t.call_date}: {exc}")
        if i % 25 == 0:
            logger.info(f"featurised {i} transcripts")
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["ticker", "call_date"])
    df = pd.DataFrame(rows)
    df = add_qoq_tone_shifts(df)
    return df
