import math

import pytest

from rec_researcher.core.models import SourceRecord
from rec_researcher.evaluation.metrics import (
    duplicate_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    source_diversity,
)


def source(identifier: str, url: str) -> SourceRecord:
    return SourceRecord(
        id=identifier, title=identifier, url=url, snippet="", provider="fixture"
    )


def test_relevance_metrics_are_null_without_gold_annotations() -> None:
    retrieved = ["https://returned.example/paper"]

    assert recall_at_k(retrieved, None, k=3) is None
    assert recall_at_k(retrieved, {}, k=5) is None
    assert mean_reciprocal_rank(retrieved, {}) is None
    assert ndcg_at_k(retrieved, {}, k=5) is None


def test_graded_ndcg_matches_hand_calculation() -> None:
    gold = {"https://a.example": 3, "https://b.example": 2, "https://c.example": 1}
    retrieved = ["https://b.example", "https://missing.example", "https://a.example"]
    actual_dcg = 3.0 + 7.0 / math.log2(4)
    ideal_dcg = 7.0 + 3.0 / math.log2(3) + 1.0 / math.log2(4)

    assert ndcg_at_k(retrieved, gold, k=5) == pytest.approx(actual_dcg / ideal_dcg)


def test_mrr_uses_first_relevant_rank() -> None:
    retrieved = ["https://x.example", "https://b.example", "https://a.example"]

    assert (
        mean_reciprocal_rank(
            retrieved, {"https://a.example": 3, "https://b.example": 1}
        )
        == 0.5
    )
    assert (
        recall_at_k(retrieved, {"https://a.example": 3, "https://b.example": 1}, k=2)
        == 0.5
    )


def test_diversity_and_duplicates_are_safe_for_empty_and_repeated_urls() -> None:
    sources = [
        source("one", "https://same.example/a"),
        source("two", "https://same.example/a"),
    ]

    assert source_diversity([]) == 0.0
    assert duplicate_rate([]) == 0.0
    assert source_diversity(sources) == 0.5
    assert duplicate_rate(sources) == 0.5
