"""Retrieval, fusion, and diversity primitives."""

from rec_researcher.retrieval.bm25 import BM25Result, BM25Retriever, mixed_tokenize
from rec_researcher.retrieval.chunker import PassageChunker
from rec_researcher.retrieval.dedup import (
    deduplicate_passages,
    deduplicate_sources,
    normalize_url,
    text_fingerprint,
)
from rec_researcher.retrieval.diversity import (
    DiversityCandidate,
    MMRResult,
    maximal_marginal_relevance,
)
from rec_researcher.retrieval.fetcher import AsyncWebFetcher, FetchResult
from rec_researcher.retrieval.fusion import (
    FusedResult,
    weighted_reciprocal_rank_fusion,
)
from rec_researcher.retrieval.vector_store import MilvusLiteIndex, VectorSearchHit

__all__ = [
    "AsyncWebFetcher",
    "BM25Result",
    "BM25Retriever",
    "DiversityCandidate",
    "FetchResult",
    "FusedResult",
    "MMRResult",
    "MilvusLiteIndex",
    "PassageChunker",
    "VectorSearchHit",
    "deduplicate_passages",
    "deduplicate_sources",
    "normalize_url",
    "maximal_marginal_relevance",
    "mixed_tokenize",
    "text_fingerprint",
    "weighted_reciprocal_rank_fusion",
]
