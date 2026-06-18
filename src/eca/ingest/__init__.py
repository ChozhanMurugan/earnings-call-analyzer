"""Ingest layer: pulls raw earnings-call transcripts from public sources."""

from eca.ingest.edgar import EdgarTranscriptClient
from eca.ingest.hf_dataset import load_hf_earnings_calls
from eca.ingest.motley_fool import MotleyFoolClient
from eca.ingest.schema import Transcript

__all__ = [
    "EdgarTranscriptClient",
    "MotleyFoolClient",
    "load_hf_earnings_calls",
    "Transcript",
]
