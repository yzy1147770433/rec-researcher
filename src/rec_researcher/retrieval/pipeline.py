"""End-to-end hybrid passage retrieval with explicit provider degradation."""

from __future__ import annotations

import inspect
import time
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from rec_researcher.core.models import PassageRecord
from rec_researcher.providers.base import PassageReranker, TextEmbedder, VectorIndex
from rec_researcher.retrieval.bm25 import BM25Retriever
from rec_researcher.retrieval.diversity import (
    DiversityCandidate,
    MMRResult,
    maximal_marginal_relevance,
)
from rec_researcher.retrieval.fusion import weighted_reciprocal_rank_fusion


class RetrievalStatistics(BaseModel):
    """Per-query retrieval counters and provider timings."""

    model_config = ConfigDict(extra="forbid")

    bm25_candidate_count: int = 0
    dense_candidate_count: int = 0
    fused_candidate_count: int = 0
    reranked_candidate_count: int = 0
    final_passage_count: int = 0
    embedding_calls: int = 0
    embedding_text_count: int = 0
    reranker_calls: int = 0
    degradation_events: int = 0
    retrieval_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0


class RetrievalTrace(BaseModel):
    """Ranking provenance retained for one selected passage."""

    model_config = ConfigDict(extra="forbid")

    lexical_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    selection_stage: str


class RetrievalPipelineResult(BaseModel):
    """Selected passages, ranking provenance, stats, and degradation notices."""

    model_config = ConfigDict(extra="forbid")

    passages: list[MMRResult] = Field(default_factory=list)
    traces: dict[str, RetrievalTrace] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    statistics: RetrievalStatistics = Field(default_factory=RetrievalStatistics)


class RetrievalPipeline:
    """Combine existing BM25, vector retrieval, RRF, reranking, and MMR."""

    def __init__(
        self,
        *,
        embedder: TextEmbedder,
        vector_index: VectorIndex,
        reranker: PassageReranker,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 10,
        mmr_top_k: int = 8,
        rrf_k: int = 60,
        mmr_lambda: float = 0.75,
    ) -> None:
        self._embedder = embedder
        self._vector_index = vector_index
        self._reranker = reranker
        self._retrieval_top_k = retrieval_top_k
        self._rerank_top_k = rerank_top_k
        self._mmr_top_k = mmr_top_k
        self._rrf_k = rrf_k
        self._mmr_lambda = mmr_lambda

    async def retrieve(
        self,
        query: str,
        passages: Sequence[PassageRecord],
        *,
        namespace: str = "isolated",
    ) -> RetrievalPipelineResult:
        """Retrieve passages with isolated vector state and visible degradation."""

        started = time.perf_counter()
        corpus = list(passages)
        stats = RetrievalStatistics()
        if not corpus:
            stats.retrieval_latency_ms = (time.perf_counter() - started) * 1000
            return RetrievalPipelineResult(statistics=stats)

        by_id = {passage.id: passage for passage in corpus}
        if len(by_id) != len(corpus):
            raise ValueError("passage identifiers must be unique within a task")
        lexical = BM25Retriever(corpus).search(query, limit=self._retrieval_top_k)
        stats.bm25_candidate_count = len(lexical)
        warnings: list[str] = []
        dense = []

        embedding_started = time.perf_counter()
        try:
            stats.embedding_calls += 1
            stats.embedding_text_count += len(corpus)
            passage_vectors = await self._embedder.embed(
                [passage.text for passage in corpus]
            )
            self._validate_vectors(passage_vectors, expected=len(corpus))
            stats.embedding_calls += 1
            stats.embedding_text_count += 1
            query_vectors = await self._embedder.embed([query])
            query_dimension = self._validate_vectors(query_vectors, expected=1)
            if query_dimension != len(passage_vectors[0]):
                raise ValueError("query and document embedding dimensions differ")
        except Exception as exc:  # noqa: BLE001 - provider degradation boundary
            warning = f"embedding failed; using BM25 only: {type(exc).__name__}: {exc}"
            warnings.append(warning)
            stats.degradation_events += 1
            passage_vectors = []
            query_vectors = []
        stats.embedding_latency_ms += (time.perf_counter() - embedding_started) * 1000

        if passage_vectors and query_vectors:
            scoped = None
            try:
                scope = getattr(self._vector_index, "scoped", None)
                scoped = scope(namespace) if scope is not None else self._vector_index
                await scoped.upsert_passages(corpus, passage_vectors)
                dense = await scoped.search(
                    query_vectors[0], limit=self._retrieval_top_k
                )
            except Exception as exc:  # noqa: BLE001 - storage degradation boundary
                warning = (
                    f"vector store failed; using BM25 only: {type(exc).__name__}: {exc}"
                )
                warnings.append(warning)
                stats.degradation_events += 1
                dense = []
            finally:
                close = getattr(scoped, "close", None)
                if scoped is not self._vector_index and close is not None:
                    try:
                        close()
                    except Exception as exc:  # noqa: BLE001 - cleanup is non-fatal
                        warnings.append(
                            f"vector store cleanup failed: {type(exc).__name__}: {exc}"
                        )
                        stats.degradation_events += 1
        stats.dense_candidate_count = len(dense)

        fused = weighted_reciprocal_rank_fusion(
            lexical,
            dense,
            lexical_weight=1.0,
            vector_weight=1.0,
            rrf_k=self._rrf_k,
        )
        stats.fused_candidate_count = len(fused)
        fused_by_id = {item.passage_id: item for item in fused}
        candidates = [
            by_id[item.passage_id] for item in fused if item.passage_id in by_id
        ]
        relevance = {item.passage_id: item.score for item in fused}
        rerank_scores: dict[str, float] = {}
        ordered = candidates
        selection_stage = "rrf"

        if candidates:
            rerank_started = time.perf_counter()
            try:
                stats.reranker_calls += 1
                reranked = await self._reranker.rerank(
                    query,
                    [passage.text for passage in candidates],
                    top_n=min(self._rerank_top_k, len(candidates)),
                )
                if any(not 0 <= item.index < len(candidates) for item in reranked):
                    raise ValueError("reranker returned an invalid document index")
                ordered = [candidates[item.index] for item in reranked]
                rerank_scores = {
                    candidates[item.index].id: item.relevance_score for item in reranked
                }
                relevance.update(rerank_scores)
                stats.reranked_candidate_count = len(reranked)
                selection_stage = "reranker"
            except Exception as exc:  # noqa: BLE001 - provider degradation boundary
                warning = (
                    f"reranker failed; using RRF ranking: {type(exc).__name__}: {exc}"
                )
                warnings.append(warning)
                stats.degradation_events += 1
            stats.rerank_latency_ms += (time.perf_counter() - rerank_started) * 1000

        diversity = [
            DiversityCandidate(
                passage_id=passage.id,
                source_id=passage.source_id,
                text=passage.text,
                relevance_score=relevance[passage.id],
            )
            for passage in ordered
        ]
        try:
            selected = maximal_marginal_relevance(
                diversity, top_k=self._mmr_top_k, lambda_=self._mmr_lambda
            )
            final_stage = "mmr"
        except Exception as exc:  # noqa: BLE001 - algorithm degradation boundary
            warning = (
                f"MMR failed; preserving {selection_stage} ranking: "
                f"{type(exc).__name__}: {exc}"
            )
            warnings.append(warning)
            stats.degradation_events += 1
            selected = [
                MMRResult(
                    **item.model_dump(), rank=index, mmr_score=item.relevance_score
                )
                for index, item in enumerate(diversity[: self._mmr_top_k], start=1)
            ]
            final_stage = selection_stage

        traces: dict[str, RetrievalTrace] = {}
        for item in selected:
            fused_item = fused_by_id[item.passage_id]
            traces[item.passage_id] = RetrievalTrace(
                lexical_rank=fused_item.lexical_rank,
                dense_rank=fused_item.vector_rank,
                rrf_score=fused_item.score,
                rerank_score=rerank_scores.get(item.passage_id),
                selection_stage=final_stage,
            )
        stats.final_passage_count = len(selected)
        stats.retrieval_latency_ms = (time.perf_counter() - started) * 1000
        return RetrievalPipelineResult(
            passages=selected, traces=traces, warnings=warnings, statistics=stats
        )

    async def aclose(self) -> None:
        """Release provider clients and the root vector-index connection."""

        for resource in (self._embedder, self._reranker, self._vector_index):
            close = getattr(resource, "aclose", None) or getattr(
                resource, "close", None
            )
            if close is None:
                continue
            result = close()
            if inspect.isawaitable(result):
                await result

    @staticmethod
    def _validate_vectors(vectors: Sequence[Sequence[float]], *, expected: int) -> int:
        if len(vectors) != expected:
            raise ValueError(
                f"embedding count mismatch: expected {expected}, got {len(vectors)}"
            )
        if not vectors or not vectors[0]:
            raise ValueError("embedding vectors must not be empty")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("embedding dimensions are inconsistent")
        return dimension
