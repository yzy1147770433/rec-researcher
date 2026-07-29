"""Text-aware maximal marginal relevance selection."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from rec_researcher.retrieval.bm25 import mixed_tokenize


class DiversityCandidate(BaseModel):
    """Candidate information needed for deterministic MMR."""

    model_config = ConfigDict(extra="forbid")

    passage_id: str
    source_id: str
    text: str
    relevance_score: float


class MMRResult(DiversityCandidate):
    """Selected candidate with its MMR trace."""

    rank: int
    mmr_score: float


def _similarity(left: str, right: str) -> float:
    left_tokens = set(mixed_tokenize(left))
    right_tokens = set(mixed_tokenize(right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def maximal_marginal_relevance(
    candidates: Sequence[DiversityCandidate],
    *,
    top_k: int,
    lambda_: float = 0.7,
    same_source_penalty: float = 0.1,
) -> list[MMRResult]:
    """Select relevant, non-redundant passages with a source repeat penalty."""

    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be between 0 and 1")
    if same_source_penalty < 0:
        raise ValueError("same_source_penalty must be non-negative")
    if not candidates or top_k <= 0:
        return []
    remaining = list(enumerate(candidates))
    selected: list[DiversityCandidate] = []
    results: list[MMRResult] = []
    while remaining and len(results) < top_k:
        scored: list[tuple[float, float, int, DiversityCandidate]] = []
        for original_index, candidate in remaining:
            redundancy = max(
                (_similarity(candidate.text, item.text) for item in selected),
                default=0.0,
            )
            source_penalty = (
                same_source_penalty
                if any(item.source_id == candidate.source_id for item in selected)
                else 0.0
            )
            score = lambda_ * candidate.relevance_score - (1 - lambda_) * redundancy
            score -= source_penalty
            scored.append(
                (score, candidate.relevance_score, -original_index, candidate)
            )
        score, _, _, winner = max(scored, key=lambda item: item[:3])
        selected.append(winner)
        results.append(
            MMRResult(**winner.model_dump(), rank=len(results) + 1, mmr_score=score)
        )
        remaining = [(index, item) for index, item in remaining if item is not winner]
    return results
