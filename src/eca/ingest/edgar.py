"""SEC EDGAR client for 8-K filings that carry earnings-call transcript exhibits.

Notes
-----
EDGAR requires every request to carry a descriptive `User-Agent` containing a
contact email. Set ``EDGAR_USER_AGENT`` in your ``.env`` file.

Earnings transcripts on EDGAR live inside 8-K filings as Item 2.02 ("Results of
Operations") exhibits (typically Exhibit 99.1 or 99.2). Not every issuer files
their transcript here — many only file the press release — so coverage is partial.
That's fine: this loader is for the *demo/reproducibility* path. Bulk training
should use the HuggingFace loader.
"""
from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime

import httpx
from bs4 import BeautifulSoup

from eca.config import settings
from eca.ingest.schema import Transcript
from eca.utils import logger

EDGAR_BASE = "https://www.sec.gov"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
# Polite limit: SEC says <= 10 req/s.
_MIN_INTERVAL = 0.12


@dataclass
class _Filing:
    accession: str
    filing_date: date
    primary_doc: str
    cik: int


class EdgarTranscriptClient:
    """Minimal EDGAR client. Use for tens of filings, not thousands."""

    def __init__(self, user_agent: str | None = None, timeout: float = 30.0):
        self._headers = {"User-Agent": user_agent or settings.edgar_user_agent}
        self._client = httpx.Client(headers=self._headers, timeout=timeout, follow_redirects=True)
        self._last_call = 0.0

    # ----- public -----

    def fetch(self, ticker: str, *, limit: int = 4) -> list[Transcript]:
        """Return up to ``limit`` most recent earnings-related 8-K transcripts."""
        cik = self._lookup_cik(ticker)
        filings = self._list_8k_filings(cik, limit=limit * 3)  # over-fetch then filter
        out: list[Transcript] = []
        for f in filings:
            try:
                t = self._parse_filing(ticker, f)
            except Exception as exc:  # noqa: BLE001 — best-effort scrape
                logger.warning(f"skip {f.accession}: {exc}")
                continue
            if t is not None:
                out.append(t)
            if len(out) >= limit:
                break
        return out

    # ----- internals -----

    def _get(self, url: str) -> httpx.Response:
        elapsed = time.monotonic() - self._last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        r = self._client.get(url)
        self._last_call = time.monotonic()
        r.raise_for_status()
        return r

    def _lookup_cik(self, ticker: str) -> int:
        # SEC ticker -> CIK map (small file, OK to fetch on demand)
        r = self._get(f"{EDGAR_BASE}/files/company_tickers.json")
        for row in r.json().values():
            if row["ticker"].upper() == ticker.upper():
                return int(row["cik_str"])
        raise LookupError(f"ticker {ticker!r} not found in SEC ticker map")

    def _list_8k_filings(self, cik: int, *, limit: int) -> list[_Filing]:
        r = self._get(SUBMISSIONS_URL.format(cik=cik))
        recent = r.json()["filings"]["recent"]
        out: list[_Filing] = []
        for form, acc, fdate, primary in zip(
            recent["form"], recent["accessionNumber"], recent["filingDate"], recent["primaryDocument"],
            strict=False,
        ):
            if form != "8-K":
                continue
            out.append(
                _Filing(
                    accession=acc,
                    filing_date=datetime.strptime(fdate, "%Y-%m-%d").date(),
                    primary_doc=primary,
                    cik=cik,
                )
            )
            if len(out) >= limit:
                break
        return out

    def _parse_filing(self, ticker: str, f: _Filing) -> Transcript | None:
        acc_nodash = f.accession.replace("-", "")
        index_url = f"{EDGAR_BASE}/Archives/edgar/data/{f.cik}/{acc_nodash}/"
        idx = self._get(index_url + "index.json").json()
        # find the largest .htm/.txt exhibit that looks like a transcript
        items = sorted(
            (i for i in idx["directory"]["item"] if i["name"].lower().endswith((".htm", ".html", ".txt"))),
            key=lambda x: -int(x.get("size", 0)),
        )
        for item in items:
            text = self._extract_text(self._get(index_url + item["name"]).text)
            if not _looks_like_transcript(text):
                continue
            prepared, qa = _split_prepared_qa(text)
            return Transcript(
                ticker=ticker.upper(),
                call_date=f.filing_date,
                fiscal_quarter=_infer_fq(text),
                prepared_remarks=prepared,
                qa_section=qa,
                source="edgar",
                source_url=index_url + item["name"],
                metadata={"accession": f.accession},
            )
        return None

    @staticmethod
    def _extract_text(html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()

    def __enter__(self) -> EdgarTranscriptClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()


# ----- text heuristics -----

_TRANSCRIPT_HINTS = re.compile(
    r"\b(conference call|prepared remarks|operator|question[- ]and[- ]answer|q\s*&\s*a)\b",
    re.IGNORECASE,
)
_FQ_RE = re.compile(r"\b(Q[1-4])\s*('?\d{2,4}|20\d{2})\b", re.IGNORECASE)


def _looks_like_transcript(text: str) -> bool:
    return bool(_TRANSCRIPT_HINTS.search(text)) and len(text) > 4000


def _infer_fq(text: str) -> str | None:
    m = _FQ_RE.search(text[:4000])
    if not m:
        return None
    return f"{m.group(1).upper()} {m.group(2)}"


def _split_prepared_qa(text: str) -> tuple[str, str]:
    pivot = re.search(r"(question[- ]and[- ]answer|q\s*&\s*a session)", text, re.IGNORECASE)
    if not pivot:
        return text, ""
    return text[: pivot.start()].strip(), text[pivot.start() :].strip()


def fetch_many(tickers: list[str], *, per_ticker: int = 4) -> Iterator[Transcript]:
    """Convenience generator across multiple tickers."""
    with EdgarTranscriptClient() as client:
        for t in tickers:
            try:
                yield from client.fetch(t, limit=per_ticker)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{t}: {exc}")
