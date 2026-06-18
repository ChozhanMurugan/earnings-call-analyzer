"""FinBERT sentence-level sentiment, aggregated per transcript."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from eca.config import settings
from eca.utils import logger

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_MAX_SENTENCES = 400  # hard cap; transcripts can have thousands of lines


@dataclass
class SentimentAgg:
    sent_pos_mean: float
    sent_neg_mean: float
    sent_neu_mean: float
    sent_pos_share: float
    sent_neg_share: float
    sent_polarity_mean: float  # pos - neg
    sent_polarity_std: float
    n_sentences: int

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


class FinBertSentiment:
    """Wrapper around HF FinBERT. Loads lazily; safe to construct in __init__."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self._model_name = model_name or settings.finbert_model
        self._device = device

    @lru_cache(maxsize=1)  # noqa: B019
    def _pipeline(self):  # noqa: ANN202
        from transformers import pipeline  # heavy import, defer

        logger.info(f"loading FinBERT: {self._model_name}")
        return pipeline(
            "sentiment-analysis",
            model=self._model_name,
            tokenizer=self._model_name,
            device=self._device if self._device is not None else -1,
            top_k=None,
            truncation=True,
        )

    def score(self, text: str) -> SentimentAgg:
        sentences = _split_sentences(text)
        if not sentences:
            return _empty_agg()
        sentences = sentences[:_MAX_SENTENCES]

        outputs = self._pipeline()(sentences, batch_size=32)
        pos, neg, neu = [], [], []
        for out in outputs:
            d = {o["label"].lower(): o["score"] for o in out}
            pos.append(d.get("positive", 0.0))
            neg.append(d.get("negative", 0.0))
            neu.append(d.get("neutral", 0.0))

        pos_a, neg_a, neu_a = np.array(pos), np.array(neg), np.array(neu)
        polarity = pos_a - neg_a
        labels = np.argmax(np.stack([pos_a, neg_a, neu_a], axis=1), axis=1)
        return SentimentAgg(
            sent_pos_mean=float(pos_a.mean()),
            sent_neg_mean=float(neg_a.mean()),
            sent_neu_mean=float(neu_a.mean()),
            sent_pos_share=float((labels == 0).mean()),
            sent_neg_share=float((labels == 1).mean()),
            sent_polarity_mean=float(polarity.mean()),
            sent_polarity_std=float(polarity.std()),
            n_sentences=int(len(sentences)),
        )


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p for p in parts if 20 <= len(p) <= 1000]


def _empty_agg() -> SentimentAgg:
    return SentimentAgg(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
