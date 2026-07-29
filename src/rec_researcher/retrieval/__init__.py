"""Retrieval preprocessing primitives."""

from rec_researcher.retrieval.chunker import PassageChunker
from rec_researcher.retrieval.dedup import (
    deduplicate_passages,
    deduplicate_sources,
    normalize_url,
    text_fingerprint,
)
from rec_researcher.retrieval.fetcher import AsyncWebFetcher, FetchResult

__all__ = [
    "AsyncWebFetcher",
    "FetchResult",
    "PassageChunker",
    "deduplicate_passages",
    "deduplicate_sources",
    "normalize_url",
    "text_fingerprint",
]
