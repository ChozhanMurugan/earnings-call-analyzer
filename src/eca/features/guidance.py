"""Forward-looking statement detector.

Inspired by Hassan, Hollander, van Lent & Tahoun (2019) and the FLS literature.
We use a simple high-precision pattern set: an FLS is a sentence containing both
a forward time cue (`next quarter`, `full year`, `going forward`, ...) AND an
expectation verb (`expect`, `anticipate`, `guide`, ...).

Outputs:
    fls_count        - number of forward-looking sentences
    fls_ratio        - share of sentences flagged as FLS
    fls_positive     - share of FLS containing positive cues (`growth`, `record`, `strong`, ...)
    fls_negative     - share of FLS containing negative cues (`decline`, `headwind`, `weak`, ...)
"""
from __future__ import annotations

import re

_TIME_CUES = re.compile(
    r"\b(next (?:quarter|year|fiscal|half)|full[- ]year|fy\s?2?0?\d{2}|"
    r"going forward|in the (?:coming|next|upcoming) (?:quarter|year|months)|"
    r"second half|first half|h[12]|q[1-4]|outlook|guidance|guide)\b",
    re.IGNORECASE,
)
_EXPECT_VERBS = re.compile(
    r"\b(expect|expects|expected|anticipate|anticipates|anticipated|"
    r"project|projects|projected|forecast|forecasts|forecasted|"
    r"guide|guides|guidance|believe|believes|see|seeing|target|targets|"
    r"plan|plans|planning|intend|intends|will|should|likely)\b",
    re.IGNORECASE,
)
_POS_CUES = re.compile(
    r"\b(growth|growing|record|strong|robust|accelerate|accelerating|expand|"
    r"expansion|improve|improving|opportunity|opportunities|momentum|tailwind|"
    r"beat|exceed|outperform)\b",
    re.IGNORECASE,
)
_NEG_CUES = re.compile(
    r"\b(decline|declining|weak|weakness|headwind|headwinds|soft|softening|"
    r"pressure|pressures|challenging|challenge|slow|slowing|slowdown|"
    r"contract|contraction|miss|missed|below|disappoint|disappointing|"
    r"uncertain|uncertainty)\b",
    re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def guidance_features(text: str) -> dict[str, float]:
    sentences = [s for s in _SENT_SPLIT.split(text) if len(s) >= 20]
    if not sentences:
        return {"fls_count": 0.0, "fls_ratio": 0.0, "fls_positive": 0.0, "fls_negative": 0.0}

    fls = [s for s in sentences if _TIME_CUES.search(s) and _EXPECT_VERBS.search(s)]
    if not fls:
        return {
            "fls_count": 0.0,
            "fls_ratio": 0.0,
            "fls_positive": 0.0,
            "fls_negative": 0.0,
        }

    pos = sum(1 for s in fls if _POS_CUES.search(s))
    neg = sum(1 for s in fls if _NEG_CUES.search(s))
    return {
        "fls_count": float(len(fls)),
        "fls_ratio": len(fls) / len(sentences),
        "fls_positive": pos / len(fls),
        "fls_negative": neg / len(fls),
    }
