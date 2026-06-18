"""Opt-in Motley Fool transcript fetcher.

LEGAL NOTE
----------
The Motley Fool Terms of Service prohibit automated scraping of their content.
This module is provided only for personal, low-volume research use. It honours
``robots.txt`` and rate-limits to one request every two seconds. **Do not use
this module to build a redistributable dataset.** For bulk training use
``eca.ingest.hf_dataset`` (licensed corpus) or ``eca.ingest.edgar`` (public
8-K exhibits).

Seeking Alpha transcripts are hard-paywalled with aggressive bot detection;
no scraper is provided for them.
"""
from __future__ import annotations

import re
import time
import urllib.robotparser
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from eca.ingest.schema import Transcript
from eca.utils import logger

BASE = "https://www.fool.com"
_INDEX_PATH = "/earnings-call-transcripts/"
_MIN_INTERVAL = 2.0  # seconds between requests — be polite
_UA = "ECA-Research/0.1 (personal research; respects robots.txt)"


@dataclass
class _Listing:
    url: str
    title: str


class MotleyFoolClient:
    """Polite, opt-in transcript fetcher. Construct only when you understand the ToS."""

    def __init__(self, user_agent: str = _UA, timeout: float = 30.0):
        self._headers = {"User-Agent": user_agent}
        self._client = httpx.Client(headers=self._headers, timeout=timeout, follow_redirects=True)
        self._last_call = 0.0
        self._robots = self._load_robots()

    def _load_robots(self) -> urllib.robotparser.RobotFileParser:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(urljoin(BASE, "/robots.txt"))
        try:
            rp.read()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"could not load robots.txt: {exc}; refusing to fetch")
            rp.disallow_all = True
        return rp

    def _allowed(self, url: str) -> bool:
        try:
            return self._robots.can_fetch(self._headers["User-Agent"], url)
        except Exception:  # noqa: BLE001
            return False

    def _get(self, url: str) -> httpx.Response:
        if not self._allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        elapsed = time.monotonic() - self._last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        r = self._client.get(url)
        self._last_call = time.monotonic()
        r.raise_for_status()
        return r

    def fetch(self, ticker: str, *, limit: int = 4) -> list[Transcript]:
        """Find recent transcripts for a ticker. Returns at most ``limit`` results."""
        listings = self._search_listings(ticker, max_pages=limit)
        out: list[Transcript] = []
        for L in listings[: limit * 2]:
            try:
                t = self._parse_transcript(ticker, L)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"motley_fool skip {L.url}: {exc}")
                continue
            if t is not None:
                out.append(t)
            if len(out) >= limit:
                break
        return out

    def _search_listings(self, ticker: str, *, max_pages: int) -> list[_Listing]:
        url = f"{BASE}{_INDEX_PATH}"
        r = self._get(url)
        soup = BeautifulSoup(r.text, "lxml")
        out: list[_Listing] = []
        for a in soup.select("a"):
            href = a.get("href") or ""
            title = (a.get_text() or "").strip()
            if "earnings-call-transcript" not in href:
                continue
            if ticker.lower() not in (title.lower() + " " + href.lower()):
                continue
            out.append(_Listing(url=urljoin(BASE, href), title=title))
            if len(out) >= max_pages:
                break
        return out

    def _parse_transcript(self, ticker: str, listing: _Listing) -> Transcript | None:
        r = self._get(listing.url)
        soup = BeautifulSoup(r.text, "lxml")
        body = soup.select_one("article") or soup
        for tag in body(["script", "style", "aside", "nav", "footer"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", body.get_text("\n")).strip()
        if len(text) < 2000:
            return None
        prepared, qa = _split_prepared_qa(text)
        return Transcript(
            ticker=ticker.upper(),
            call_date=_infer_date(text) or date.today(),
            fiscal_quarter=_infer_fq(text),
            prepared_remarks=prepared,
            qa_section=qa,
            source="motley_fool",
            source_url=listing.url,
            metadata={"title": listing.title},
        )

    def __enter__(self) -> MotleyFoolClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()


_FQ_RE = re.compile(r"\b(Q[1-4])\s*('?\d{2,4}|20\d{2})\b", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
    r")\s+(\d{1,2}),?\s+(20\d{2})\b",
    re.IGNORECASE,
)


def _infer_fq(text: str) -> str | None:
    m = _FQ_RE.search(text[:4000])
    return f"{m.group(1).upper()} {m.group(2)}" if m else None


def _infer_date(text: str) -> date | None:
    m = _DATE_RE.search(text[:4000])
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)[:3]} {m.group(2)} {m.group(3)}", "%b %d %Y").date()
    except ValueError:
        return None


def _split_prepared_qa(text: str) -> tuple[str, str]:
    pivot = re.search(r"(question[- ]and[- ]answer|q\s*&\s*a session|questions and answers)", text, re.IGNORECASE)
    if not pivot:
        return text, ""
    return text[: pivot.start()].strip(), text[pivot.start() :].strip()


def fetch_many(tickers: list[str], *, per_ticker: int = 4) -> Iterator[Transcript]:
    with MotleyFoolClient() as client:
        for t in tickers:
            try:
                yield from client.fetch(t, limit=per_ticker)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{t}: {exc}")
