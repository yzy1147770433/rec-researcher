"""Protocols that isolate all external API capabilities."""

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from rec_researcher.core.models import PassageRecord, SourceRecord


class VectorSearchResult(Protocol):
    """Minimum result shape returned by a vector index."""

    passage_id: str
    source_id: str
    text: str
    score: float
    rank: int


class LanguageModel(Protocol):
    """Asynchronous text-generation capability."""

    async def generate(self, prompt: str) -> str:
        """Generate text for a prompt."""
        ...


class SearchProvider(Protocol):
    """Asynchronous web-search capability."""

    async def search(self, query: str, *, limit: int) -> list[SourceRecord]:
        """Return up to ``limit`` normalized source records."""
        ...


class TextEmbedder(Protocol):
    """Asynchronous text-embedding capability."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in input order; empty input returns an empty list."""
        ...


class RerankResult(BaseModel):
    """A reranked document tied to its original input position."""

    model_config = ConfigDict(extra="forbid")

    index: int
    document: str
    relevance_score: float


class PassageReranker(Protocol):
    """Asynchronous document reranking with stable input positions."""

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int,
    ) -> list[RerankResult]:
        """Rank documents by relevance; empty input returns an empty list."""
        ...


class VectorIndex(Protocol):
    """Asynchronous vector storage and nearest-neighbor capability."""

    async def upsert_passages(
        self, passages: Sequence[PassageRecord], vectors: Sequence[Sequence[float]]
    ) -> None:
        """Insert or replace passage/vector pairs."""
        ...

    async def search(
        self, vector: Sequence[float], *, limit: int
    ) -> list[VectorSearchResult]:
        """Return scored nearest passages."""
        ...

    def scoped(self, namespace: str) -> "VectorIndex":
        """Return an index isolated to one run/task namespace."""
        ...
