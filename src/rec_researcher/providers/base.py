"""Protocols that isolate all external API capabilities."""

from collections.abc import Sequence
from typing import Protocol

from rec_researcher.core.models import PassageRecord, SourceRecord


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


class PassageReranker(Protocol):
    """Asynchronous passage-reranking capability."""

    async def rerank(
        self,
        query: str,
        passages: Sequence[PassageRecord],
        *,
        limit: int,
    ) -> list[PassageRecord]:
        """Rank passages by relevance; empty input returns an empty list."""
        ...


class VectorIndex(Protocol):
    """Asynchronous vector storage and nearest-neighbor capability."""

    async def add(
        self, passages: Sequence[PassageRecord], vectors: Sequence[Sequence[float]]
    ) -> None:
        """Store passage/vector pairs."""
        ...

    async def search(
        self, vector: Sequence[float], *, limit: int
    ) -> list[PassageRecord]:
        """Return the nearest retained passages."""
        ...
