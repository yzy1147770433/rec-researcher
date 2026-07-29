from collections.abc import Sequence

import pytest

from rec_researcher.core.models import PassageRecord
from rec_researcher.providers.mock import MockTextEmbedder
from rec_researcher.providers.siliconflow import SiliconFlowRerankResult
from rec_researcher.retrieval.pipeline import RetrievalPipeline
from rec_researcher.retrieval.vector_store import VectorSearchHit

pytestmark = pytest.mark.asyncio


def _passages() -> list[PassageRecord]:
    return [
        PassageRecord(id="p1", source_id="s1", text="vector retrieval search"),
        PassageRecord(id="p2", source_id="s2", text="collaborative filtering"),
        PassageRecord(id="p3", source_id="s3", text="ranking metrics"),
    ]


class FakeVectorIndex:
    async def upsert_passages(
        self,
        passages: Sequence[PassageRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        assert len(passages) == len(vectors)

    async def search(
        self, vector: Sequence[float], *, limit: int
    ) -> list[VectorSearchHit]:
        return [
            VectorSearchHit(
                passage_id="p2",
                source_id="s2",
                text="collaborative filtering",
                score=0.9,
                rank=1,
            )
        ]


class FakeReranker:
    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[SiliconFlowRerankResult]:
        return [
            SiliconFlowRerankResult(
                index=len(documents) - 1,
                document=documents[-1],
                relevance_score=0.95,
            ),
            SiliconFlowRerankResult(
                index=0, document=documents[0], relevance_score=0.8
            ),
        ][:top_n]


class FailingEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("embedding unavailable")


class FailingReranker:
    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[SiliconFlowRerankResult]:
        raise RuntimeError("reranker unavailable")


async def test_pipeline_runs_all_stages() -> None:
    pipeline = RetrievalPipeline(
        embedder=MockTextEmbedder(),
        vector_index=FakeVectorIndex(),
        reranker=FakeReranker(),
        mmr_top_k=2,
    )

    result = await pipeline.retrieve("vector search", _passages())

    assert len(result.passages) == 2
    assert result.warnings == []


async def test_embedding_failure_keeps_bm25_and_adds_warning() -> None:
    pipeline = RetrievalPipeline(
        embedder=FailingEmbedder(),
        vector_index=FakeVectorIndex(),
        reranker=FakeReranker(),
        mmr_top_k=2,
    )

    result = await pipeline.retrieve("vector search", _passages())

    assert result.passages
    assert "using BM25 only" in result.warnings[0]


async def test_reranker_failure_keeps_rrf_and_adds_warning() -> None:
    pipeline = RetrievalPipeline(
        embedder=MockTextEmbedder(),
        vector_index=FakeVectorIndex(),
        reranker=FailingReranker(),
        mmr_top_k=2,
    )

    result = await pipeline.retrieve("vector search", _passages())

    assert result.passages
    assert "using RRF ranking" in result.warnings[0]


async def test_pipeline_empty_corpus_is_safe() -> None:
    pipeline = RetrievalPipeline(
        embedder=FailingEmbedder(),
        vector_index=FakeVectorIndex(),
        reranker=FailingReranker(),
    )

    result = await pipeline.retrieve("query", [])

    assert result.passages == []
    assert result.warnings == []
