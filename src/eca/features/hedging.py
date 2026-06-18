"""Hedge / uncertainty word frequency, adapted from Loughran-McDonald (2011).

The list is a curated subset of the LM 'Uncertainty' + 'Weak Modal' dictionaries.
Counts are normalised by total word count, so the feature scales across calls of
different length.
"""
from __future__ import annotations

import re
from collections import Counter

HEDGE_WORDS: frozenset[str] = frozenset(
    {
        # uncertainty
        "approximately", "apparently", "appears", "around", "assume", "assumed",
        "assumes", "assuming", "assumption", "assumptions", "believe", "believed",
        "believes", "could", "depends", "depending", "estimate", "estimated",
        "estimates", "expect", "expected", "expects", "fluctuate", "fluctuation",
        "may", "maybe", "might", "perhaps", "possible", "possibly", "predict",
        "predicted", "predicts", "probable", "probably", "projection", "projections",
        "roughly", "should", "somewhat", "speculate", "speculated", "speculation",
        "suggest", "suggests", "tentative", "tentatively", "uncertain", "uncertainty",
        "unforeseen", "unknown", "unpredictable",
        # weak modals
        "can", "conceivable", "conceivably", "indefinite", "indefinitely",
        "occasionally", "possibility", "potentially", "preliminary", "presumably",
        "rarely", "seldom", "sometimes", "tend", "tends", "vague",
        "vaguely",
    }
)

_WORD_RE = re.compile(r"[A-Za-z']+")


def hedge_features(text: str) -> dict[str, float]:
    words = _WORD_RE.findall(text.lower())
    if not words:
        return {"hedge_count": 0.0, "hedge_ratio": 0.0, "n_words": 0.0}
    counts = Counter(w for w in words if w in HEDGE_WORDS)
    total_hedge = sum(counts.values())
    return {
        "hedge_count": float(total_hedge),
        "hedge_ratio": total_hedge / len(words),
        "n_words": float(len(words)),
    }
