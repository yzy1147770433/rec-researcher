"""Deterministic, completely offline provider implementations."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from rec_researcher.core.models import PassageRecord, SourceRecord


class MockLanguageModel:
    """Return a stable synthetic response derived from the input prompt."""

    async def generate(self, prompt: str) -> str:
        """Generate deterministic text without making a network request."""

        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        return f"Mock response ({digest}): {prompt.strip()}"


class MockSearchProvider:
    """Search a fixed corpus of fictional recommender-system sources."""

    _SOURCES = (
        (
            "S1",
            "A Practical Guide to Collaborative Filtering",
            "https://example.com/recsys/collaborative-filtering",
            "A fictional overview of neighborhood and latent-factor recommenders.",
        ),
        (
            "S2",
            "Sequential Recommendation with Small Transformers",
            "https://example.com/recsys/sequential-transformers",
            "A fictional study of attention-based next-item prediction.",
        ),
        (
            "S3",
            "Offline Metrics for Top-K Recommendation",
            "https://example.com/recsys/offline-metrics",
            "A fictional comparison of Recall, NDCG, MAP, and coverage.",
        ),
        (
            "S4",
            "Open Recommendation Benchmarks and Datasets",
            "https://example.com/recsys/open-benchmarks",
            "A fictional catalog of reproducible datasets, splits, and baselines.",
        ),
        (
            "S5",
            "Reproducible Evaluation of Recommender Systems",
            "https://example.com/recsys/reproducible-evaluation",
            "A fictional checklist covering seeds, leakage, tuning, and reporting.",
        ),
    )

    async def search(self, query: str, *, limit: int) -> list[SourceRecord]:
        """Return up to ``limit`` stable fictional sources."""

        if limit <= 0:
            return []
        return [
            SourceRecord(
                id=source_id,
                title=title,
                url=url,
                snippet=f"{snippet} Query context: {query.strip()}",
                provider="mock",
            )
            for source_id, title, url, snippet in self._SOURCES[:limit]
        ]


class MockTextEmbedder:
    """Create fixed-size deterministic embeddings from SHA-256 bytes."""

    def __init__(self, dimensions: int = 8) -> None:
        """Configure the positive number of output dimensions."""

        if dimensions < 1:
            raise ValueError("dimensions must be at least 1")
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed input texts in order; empty input is safe."""

        vectors: list[list[float]] = []
        for value in texts:
            digest = hashlib.sha256(value.encode("utf-8")).digest()
            vectors.append(
                [
                    digest[index % len(digest)] / 255.0
                    for index in range(self.dimensions)
                ]
            )
        return vectors


class MockPassageReranker:
    """Rank passages by deterministic token overlap and original position."""

    async def rerank(
        self,
        query: str,
        passages: Sequence[PassageRecord],
        *,
        limit: int,
    ) -> list[PassageRecord]:
        """Return the most overlapping passages; empty input is safe."""

        if limit <= 0 or not passages:
            return []
        query_tokens = set(re.findall(r"\w+", query.casefold()))
        scored = enumerate(passages)
        ranked = sorted(
            scored,
            key=lambda pair: (
                -len(query_tokens & set(re.findall(r"\w+", pair[1].text.casefold()))),
                pair[0],
            ),
        )
        return [passage for _, passage in ranked[:limit]]
