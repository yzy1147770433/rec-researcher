"""End-to-end hybrid passage retrieval with explicit provider degradation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from rec_researcher.core.models import PassageRecord
from rec_researcher.providers.base import TextEmbedder, VectorIndex
from rec_researcher.providers.siliconflow import SiliconFlowRerankResult
from rec_researcher.retrieval.bm25 import BM25Retriever
from rec_researcher.retrieval.diversity import (
    DiversityCandidate,
    MMRResult,
    maximal_marginal_relevance,
)
from rec_researcher.retrieval.fusion import weighted_reciprocal_rank_fusion


class DocumentReranker(Protocol):
    """Rerank plain documents and retain their input positions."""

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[SiliconFlowRerankResult]:
        """Return provider-ranked documents."""
        ...


class RetrievalPipelineResult(BaseModel):
    """Selected passages and any visible degradation notices."""

    model_config = ConfigDict(extra="forbid")

    passages: list[MMRResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RetrievalPipeline:
    """Combine BM25, vector retrieval, RRF, reranking, and MMR."""

    def __init__(
        self,
        *,
        embedder: TextEmbedder,
        vector_index: VectorIndex,
        reranker: DocumentReranker,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 10,
        mmr_top_k: int = 8,
        rrf_k: int = 60,
        mmr_lambda: float = 0.75,
    ) -> None:
        """Configure retrieval stages and their candidate limits."""

        self._embedder = embedder
        self._vector_index = vector_index
        self._reranker = reranker
        self._retrieval_top_k = retrieval_top_k
        self._rerank_top_k = rerank_top_k
        self._mmr_top_k = mmr_top_k
        self._rrf_k = rrf_k
        self._mmr_lambda = mmr_lambda

    async def retrieve(
        self, query: str, passages: Sequence[PassageRecord]
    ) -> RetrievalPipelineResult:
        """Retrieve passages, degrading individual provider stages explicitly."""

        corpus = list(passages)
        if not corpus:
            return RetrievalPipelineResult()
        by_id = {passage.id: passage for passage in corpus}
        lexical = BM25Retriever(corpus).search(query, limit=self._retrieval_top_k)
        warnings: list[str] = []
        vector = []
        try:
            passage_vectors = await self._embedder.embed(
                [passage.text for passage in corpus]
            )
            await self._vector_index.upsert_passages(corpus, passage_vectors)
            query_vectors = await self._embedder.embed([query])
            vector = await self._vector_index.search(
                query_vectors[0], limit=self._retrieval_top_k
            )
        except Exception as exc:  # noqa: BLE001 - this is a degradation boundary
            warnings.append(
                f"embedding/vector retrieval failed; using BM25 only: {exc}"
            )

        fused = weighted_reciprocal_rank_fusion(
            lexical,
            vector,
            lexical_weight=1.0,
            vector_weight=1.0,
            rrf_k=self._rrf_k,
        )
        candidates = [
            by_id[item.passage_id] for item in fused if item.passage_id in by_id
        ]
        scores = {item.passage_id: item.score for item in fused}
        try:
            reranked = await self._reranker.rerank(
                query,
                [passage.text for passage in candidates],
                top_n=min(self._rerank_top_k, len(candidates)),
            )
            ordered = [candidates[item.index] for item in reranked]
            scores = {
                candidates[item.index].id: item.relevance_score for item in reranked
            }
        except Exception as exc:  # noqa: BLE001 - this is a degradation boundary
            warnings.append(f"reranker failed; using RRF ranking: {exc}")
            ordered = candidates

        diversity = [
            DiversityCandidate(
                passage_id=passage.id,
                source_id=passage.source_id,
                text=passage.text,
                relevance_score=scores[passage.id],
            )
            for passage in ordered
        ]
        selected = maximal_marginal_relevance(
            diversity, top_k=self._mmr_top_k, lambda_=self._mmr_lambda
        )
        return RetrievalPipelineResult(passages=selected, warnings=warnings)
