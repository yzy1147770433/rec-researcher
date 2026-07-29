import pytest

from rec_researcher.retrieval.bm25 import BM25Result
from rec_researcher.retrieval.fusion import weighted_reciprocal_rank_fusion
from rec_researcher.retrieval.vector_store import VectorSearchHit


def test_weighted_rrf_matches_hand_calculation_and_retains_ranks() -> None:
    lexical = [
        BM25Result(passage_id="a", rank=1, score=9),
        BM25Result(passage_id="b", rank=2, score=8),
    ]
    vector = [
        VectorSearchHit(
            passage_id="b", source_id="s", text="b", rank=1, score=0.9
        ),
        VectorSearchHit(
            passage_id="c", source_id="s", text="c", rank=2, score=0.8
        ),
    ]

    results = weighted_reciprocal_rank_fusion(
        lexical,
        vector,
        lexical_weight=2.0,
        vector_weight=1.0,
        rrf_k=10,
    )

    assert [result.passage_id for result in results] == ["b", "a", "c"]
    assert results[0].score == pytest.approx(2 / 12 + 1 / 11)
    assert results[0].lexical_rank == 2
    assert results[0].vector_rank == 1
    assert results[1].vector_rank is None


def test_rrf_empty_inputs_are_safe() -> None:
    assert (
        weighted_reciprocal_rank_fusion(
            [], [], lexical_weight=1.0, vector_weight=1.0
        )
        == []
    )
