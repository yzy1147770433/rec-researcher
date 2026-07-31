import pytest

from rec_researcher.core.models import PassageRecord
from rec_researcher.providers.mock import MockPassageReranker, MockTextEmbedder
from rec_researcher.retrieval.pipeline import RetrievalPipeline
from rec_researcher.retrieval.vector_store import InMemoryVectorIndex


def _pipeline(mode: str) -> RetrievalPipeline:
    return RetrievalPipeline(
        embedder=MockTextEmbedder(),
        vector_index=InMemoryVectorIndex(),
        reranker=MockPassageReranker(),
        mode=mode,  # type: ignore[arg-type]
        retrieval_top_k=3,
        mmr_top_k=2,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "bm25", "dense", "reranker", "stage"),
    [
        ("bm25_only", True, False, False, "rrf"),
        ("dense_only", False, True, False, "rrf"),
        ("hybrid_rrf", True, True, False, "rrf"),
        ("hybrid_rerank", True, True, True, "reranker"),
        ("hybrid_rerank_mmr", True, True, True, "mmr"),
    ],
)
async def test_modes_execute_distinct_stages(
    mode: str, bm25: bool, dense: bool, reranker: bool, stage: str
) -> None:
    passages = [
        PassageRecord(id=f"p{i}", source_id=f"s{i}", text=text)
        for i, text in enumerate(("dense retrieval", "BM25 lexical", "unrelated"))
    ]
    result = await _pipeline(mode).retrieve("retrieval", passages)
    stats = result.statistics
    assert bool(stats.bm25_candidate_count) is bm25
    assert bool(stats.dense_candidate_count) is dense
    assert bool(stats.reranker_calls) is reranker
    assert {trace.selection_stage for trace in result.traces.values()} == {stage}
