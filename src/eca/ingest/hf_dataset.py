"""Load a pre-scraped, license-clear earnings-call corpus from HuggingFace.

The default dataset is ``jlh-ibm/earnings_call`` which is widely used in academic
work and ships with ticker + date columns. Swap ``dataset_name`` for any
compatible corpus.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from eca.ingest.schema import Transcript
from eca.utils import logger


def load_hf_earnings_calls(
    dataset_name: str = "jlh-ibm/earnings_call",
    *,
    split: str = "train",
    limit: int | None = None,
) -> Iterator[Transcript]:
    """Stream the HF dataset and yield Transcript objects.

    The dataset is streamed (no full download) so ``limit`` controls cost.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install datasets") from exc

    logger.info(f"loading HF dataset {dataset_name} split={split} limit={limit}")
    ds = load_dataset(dataset_name, split=split, streaming=True)

    n = 0
    for row in ds:
        if limit is not None and n >= limit:
            break
        t = _row_to_transcript(row)
        if t is None:
            continue
        yield t
        n += 1


def _row_to_transcript(row: dict) -> Transcript | None:
    # The dataset's column names are slightly inconsistent across mirrors.
    # We probe a few common keys and coerce.
    ticker = row.get("ticker") or row.get("company_ticker") or row.get("symbol")
    text = row.get("transcript") or row.get("text") or row.get("content")
    date_str = row.get("date") or row.get("call_date") or row.get("event_date")
    if not (ticker and text and date_str):
        return None
    try:
        call_date = datetime.fromisoformat(str(date_str)[:10]).date()
    except ValueError:
        return None

    prepared, qa = _naive_split(text)
    return Transcript(
        ticker=str(ticker).upper(),
        call_date=call_date,
        fiscal_quarter=row.get("quarter") or None,
        prepared_remarks=prepared,
        qa_section=qa,
        source="hf",
        source_url=None,
        metadata={k: v for k, v in row.items() if k not in {"transcript", "text", "content"}},
    )


def _naive_split(text: str) -> tuple[str, str]:
    lower = text.lower()
    for marker in ("question-and-answer", "question and answer", "q&a", "q & a"):
        idx = lower.find(marker)
        if idx != -1:
            return text[:idx].strip(), text[idx:].strip()
    return text.strip(), ""
