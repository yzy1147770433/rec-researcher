"""Weighted reciprocal-rank fusion for independent retrieval channels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class RankedHit(Protocol):
    """Input shape required by reciprocal-rank fusion."""

    passage_id: str
    rank: int


class FusedResult(BaseModel):
    """A fused hit retaining each contributing channel rank."""

    model_config = ConfigDict(extra="forbid")

    passage_id: str
    rank: int
    score: float
    lexical_rank: int | None = None
    vector_rank: int | None = None


def weighted_reciprocal_rank_fusion(
    lexical: Sequence[RankedHit],
    vector: Sequence[RankedHit],
    *,
    lexical_weight: float,
    vector_weight: float,
    rrf_k: int = 60,
) -> list[FusedResult]:
    """Fuse rankings with ``weight / (rrf_k + rank)`` contributions."""

    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if lexical_weight < 0 or vector_weight < 0:
        raise ValueError("fusion weights must be non-negative")
    values: dict[str, dict[str, float | int | None]] = {}
    for channel, hits, weight in (
        ("lexical_rank", lexical, lexical_weight),
        ("vector_rank", vector, vector_weight),
    ):
        seen: set[str] = set()
        for hit in hits:
            if hit.passage_id in seen:
                continue
            if hit.rank <= 0 or rrf_k + hit.rank == 0:
                raise ValueError("channel ranks must be positive")
            seen.add(hit.passage_id)
            value = values.setdefault(
                hit.passage_id,
                {"score": 0.0, "lexical_rank": None, "vector_rank": None},
            )
            value[channel] = hit.rank
            value["score"] = float(value["score"]) + weight / (rrf_k + hit.rank)
    ordered = sorted(
        values.items(), key=lambda item: (-float(item[1]["score"]), item[0])
    )
    return [
        FusedResult(
            passage_id=passage_id,
            rank=rank,
            score=float(value["score"]),
            lexical_rank=value["lexical_rank"],
            vector_rank=value["vector_rank"],
        )
        for rank, (passage_id, value) in enumerate(ordered, start=1)
    ]
