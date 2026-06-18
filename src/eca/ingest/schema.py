"""Canonical Transcript object used everywhere downstream."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Transcript:
    """A single earnings call transcript in a source-agnostic shape."""

    ticker: str
    call_date: date  # date the call took place (UTC date)
    fiscal_quarter: str | None  # e.g. "Q2 2024"
    prepared_remarks: str
    qa_section: str
    source: str  # "edgar" | "hf" | ...
    source_url: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return f"{self.prepared_remarks}\n\n{self.qa_section}".strip()
