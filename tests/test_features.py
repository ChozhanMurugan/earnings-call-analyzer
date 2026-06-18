"""Unit tests for the deterministic feature extractors.

No FinBERT, no network — these run in milliseconds.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from eca.features.guidance import guidance_features
from eca.features.hedging import HEDGE_WORDS, hedge_features
from eca.features.tone_shift import add_qoq_tone_shifts

# ---------- hedging ----------

class TestHedging:
    def test_empty_text(self):
        out = hedge_features("")
        assert out == {"hedge_count": 0.0, "hedge_ratio": 0.0, "n_words": 0.0}

    def test_counts_hedge_words(self):
        text = "We might see growth. Results could vary. We expect demand to be strong."
        out = hedge_features(text)
        # "might", "could", "expect" are all in the dict
        assert out["hedge_count"] >= 3
        assert 0 < out["hedge_ratio"] < 1
        assert out["n_words"] == 13

    def test_no_hedge_words(self):
        text = "Revenue grew twenty percent year over year."
        out = hedge_features(text)
        assert out["hedge_count"] == 0
        assert out["hedge_ratio"] == 0
        assert out["n_words"] == 7

    def test_case_insensitive(self):
        out_lower = hedge_features("we might see growth")
        out_upper = hedge_features("WE MIGHT SEE GROWTH")
        assert out_lower == out_upper

    def test_dictionary_is_lowercase(self):
        assert all(w == w.lower() for w in HEDGE_WORDS)


# ---------- forward-looking statements ----------

class TestGuidance:
    def test_empty_text(self):
        out = guidance_features("")
        assert out == {"fls_count": 0.0, "fls_ratio": 0.0, "fls_positive": 0.0, "fls_negative": 0.0}

    def test_detects_forward_looking_sentence(self):
        # contains a time cue ("next quarter") AND an expectation verb ("expect")
        text = (
            "We delivered strong results this quarter. "
            "We expect revenue growth to accelerate next quarter. "
            "Margins were stable."
        )
        out = guidance_features(text)
        assert out["fls_count"] == 1
        assert out["fls_positive"] == 1.0
        assert out["fls_negative"] == 0.0

    def test_detects_negative_fls(self):
        text = (
            "We expect headwinds to continue into next quarter. "
            "Demand has been stable so far."
        )
        out = guidance_features(text)
        assert out["fls_count"] == 1
        assert out["fls_negative"] == 1.0

    def test_no_fls_when_only_time_cue(self):
        # has time cue, no expectation verb
        # "will" counts as expectation verb in the first sentence, so test a clean negative
        text2 = "Last quarter sales grew. The current quarter ended in June."
        assert guidance_features(text2)["fls_count"] == 0


# ---------- QoQ tone shift ----------

class TestToneShift:
    def test_empty_frame_returns_empty(self):
        df = pd.DataFrame(columns=["ticker", "call_date", "sent_pos_mean"])
        out = add_qoq_tone_shifts(df)
        assert out.empty

    def test_diff_within_ticker(self):
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"],
                "call_date": [
                    date(2024, 1, 1), date(2024, 4, 1), date(2024, 7, 1),
                    date(2024, 1, 15), date(2024, 4, 15),
                ],
                "sent_pos_mean": [0.5, 0.6, 0.4, 0.7, 0.8],
                "hedge_ratio": [0.01, 0.02, 0.03, 0.02, 0.01],
                "sent_neg_mean": [0.1, 0.1, 0.2, 0.05, 0.10],
                "sent_polarity_mean": [0.4, 0.5, 0.2, 0.65, 0.70],
                "fls_positive": [0.1, 0.1, 0.1, 0.1, 0.1],
                "fls_negative": [0.05, 0.10, 0.15, 0.05, 0.05],
                "fls_ratio": [0.2, 0.2, 0.2, 0.2, 0.2],
            }
        )
        out = add_qoq_tone_shifts(df)
        aapl = out[out["ticker"] == "AAPL"].sort_values("call_date").reset_index(drop=True)
        # first row is NaN (no prior call), then diffs
        assert pd.isna(aapl.loc[0, "sent_pos_mean_dqoq"])
        assert aapl.loc[1, "sent_pos_mean_dqoq"] == pytest.approx(0.1)
        assert aapl.loc[2, "sent_pos_mean_dqoq"] == pytest.approx(-0.2)
        # tickers don't leak into each other
        msft = out[out["ticker"] == "MSFT"].sort_values("call_date").reset_index(drop=True)
        assert pd.isna(msft.loc[0, "sent_pos_mean_dqoq"])
        assert msft.loc[1, "sent_pos_mean_dqoq"] == pytest.approx(0.1)

    def test_missing_shift_column_is_skipped(self):
        df = pd.DataFrame(
            {"ticker": ["A", "A"], "call_date": [date(2024, 1, 1), date(2024, 4, 1)],
             "sent_pos_mean": [0.1, 0.2]}
        )
        out = add_qoq_tone_shifts(df, shift_cols=("sent_pos_mean", "nope_not_a_col"))
        assert "sent_pos_mean_dqoq" in out.columns
        assert "nope_not_a_col_dqoq" not in out.columns


# ---------- build (with mocked sentiment) ----------

class _StubSentiment:
    """Stand-in for FinBertSentiment that doesn't touch transformers."""

    def score(self, text):
        from eca.features.sentiment import SentimentAgg

        n_sentences = max(text.count(".") + text.count("?") + text.count("!"), 1)
        return SentimentAgg(
            sent_pos_mean=0.5, sent_neg_mean=0.2, sent_neu_mean=0.3,
            sent_pos_share=0.4, sent_neg_share=0.2,
            sent_polarity_mean=0.3, sent_polarity_std=0.1,
            n_sentences=n_sentences,
        )


class TestBuild:
    def test_build_features_dict_shape(self):
        from eca.features.build import build_features
        from eca.ingest.schema import Transcript

        t = Transcript(
            ticker="AAPL",
            call_date=date(2024, 7, 1),
            fiscal_quarter="Q3 2024",
            prepared_remarks="We delivered strong revenue growth. We expect Q4 to accelerate." * 5,
            qa_section="Analyst asked about margins. CFO responded positively." * 5,
            source="test",
        )
        row = build_features(t, sentiment=_StubSentiment())
        for k in [
            "ticker", "call_date", "sent_pos_mean", "hedge_ratio",
            "fls_count", "prepared_len_chars", "qa_len_chars", "qa_ratio",
        ]:
            assert k in row
        assert row["ticker"] == "AAPL"
        assert 0 <= row["qa_ratio"] <= 1
