"""Load a pre-scraped, license-clear earnings-call corpus from HuggingFace.

Default: ``kurry/sp500_earnings_transcripts`` — S&P 500 calls, 2005-2025,
modern Parquet format, no auth required.

The loader also handles the older ``jlh-ibm/earnings_call`` format so existing
callers that pass ``dataset_name`` explicitly keep working.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from eca.ingest.schema import Transcript
from eca.utils import logger


def load_hf_earnings_calls(
    dataset_name: str = "kurry/sp500_earnings_transcripts",
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
    ds = _load_dataset(load_dataset, dataset_name, split)

    n = 0
    for row in ds:
        if limit is not None and n >= limit:
            break
        t = _row_to_transcript(row)
        if t is None:
            continue
        yield t
        n += 1


def _load_dataset(load_dataset, dataset_name: str, split: str):  # noqa: ANN001
    """Try the normal loader; fall back to direct Parquet on HF Hub if the
    dataset still ships a loading script (incompatible with datasets>=3.0)."""
    try:
        return load_dataset(dataset_name, split=split, streaming=True)
    except RuntimeError as exc:
        if "loading script" not in str(exc) and "no longer supported" not in str(exc):
            raise
        logger.warning(
            f"{dataset_name} uses a deprecated loading script. "
            "Falling back to direct Parquet access via HF Hub filesystem."
        )

    try:
        from huggingface_hub import HfFileSystem
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install huggingface-hub") from exc

    hf_fs = HfFileSystem()
    for parquet_glob in (
        f"datasets/{dataset_name}/data/{split}*.parquet",
        f"datasets/{dataset_name}/**/*.parquet",
    ):
        files = hf_fs.glob(parquet_glob)
        if files:
            break

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found for {dataset_name}. "
            "The dataset may need a HF token or a different split name. "
            "Try passing dataset_name='kurry/sp500_earnings_transcripts' instead."
        )

    logger.info(f"found {len(files)} Parquet file(s) for {dataset_name}")
    hf_urls = [f"hf:///{p}" for p in files]
    return load_dataset("parquet", data_files={split: hf_urls}, split=split, streaming=True)


def _row_to_transcript(row: dict) -> Transcript | None:
    """Convert one dataset row to a Transcript.

    Handles two known column layouts:

    * ``jlh-ibm/earnings_call`` style: flat ``ticker``, ``transcript``, ``date``
    * ``kurry/sp500_earnings_transcripts`` style: ``symbol``, ``structured_content``
      (list of ``{speaker, text}`` dicts), ``date``, ``year``, ``quarter``
    """
    ticker = row.get("ticker") or row.get("company_ticker") or row.get("symbol")

    # Flat text field (jlh-ibm style)
    text: str | None = row.get("transcript") or row.get("text") or row.get("content")

    # Structured list of {speaker, text} segments (kurry/sp500 style)
    if not text:
        structured = row.get("structured_content")
        if structured and isinstance(structured, list):
            parts = []
            for seg in structured:
                if isinstance(seg, dict):
                    parts.append(seg.get("text") or "")
                elif isinstance(seg, str):
                    parts.append(seg)
            text = "\n".join(p for p in parts if p.strip())

    date_str = row.get("date") or row.get("call_date") or row.get("event_date")
    if not (ticker and text and date_str):
        return None
    try:
        call_date = datetime.fromisoformat(str(date_str)[:10]).date()
    except ValueError:
        return None

    # Derive fiscal_quarter from year + quarter number if not provided directly
    fiscal_quarter: str | None = row.get("fiscal_quarter") or row.get("quarter_label")
    if not fiscal_quarter:
        year = row.get("year")
        qtr = row.get("quarter")
        if year and qtr:
            fiscal_quarter = f"Q{qtr} {year}"

    prepared, qa = _naive_split(text)
    return Transcript(
        ticker=str(ticker).upper(),
        call_date=call_date,
        fiscal_quarter=fiscal_quarter or None,
        prepared_remarks=prepared,
        qa_section=qa,
        source="hf",
        source_url=None,
        metadata={k: v for k, v in row.items() if k not in {"transcript", "text", "content", "structured_content"}},
    )


def _naive_split(text: str) -> tuple[str, str]:
    lower = text.lower()
    for marker in ("question-and-answer", "question and answer", "q&a", "q & a"):
        idx = lower.find(marker)
        if idx != -1:
            return text[:idx].strip(), text[idx:].strip()
    return text.strip(), ""
