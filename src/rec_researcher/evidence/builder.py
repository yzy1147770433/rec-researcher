"""Build source-traceable evidence from selected passages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rec_researcher.core.models import EvidenceRecord, PassageRecord, SourceRecord


class EvidenceBuilder:
    """Convert final retrieval selections into bounded evidence records."""

    def __init__(self, *, max_excerpt_length: int = 600) -> None:
        """Configure the maximum number of characters retained per excerpt."""

        if max_excerpt_length < 1:
            raise ValueError("max_excerpt_length must be at least 1")
        self.max_excerpt_length = max_excerpt_length

    def build(
        self,
        passages: Sequence[PassageRecord],
        sources: Sequence[SourceRecord],
        *,
        relevance_scores: Mapping[str, float] | None = None,
        claim_hints: Mapping[str, str] | None = None,
        retrieval_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[EvidenceRecord]:
        """Build evidence in passage order and reject orphaned passages."""

        source_ids = {source.id for source in sources}
        scores = relevance_scores or {}
        hints = claim_hints or {}
        metadata = retrieval_metadata or {}
        evidence: list[EvidenceRecord] = []
        for index, passage in enumerate(passages, start=1):
            if passage.source_id not in source_ids:
                raise ValueError(
                    f"passage {passage.id!r} references unknown source "
                    f"{passage.source_id!r}"
                )
            excerpt = passage.text.strip()[: self.max_excerpt_length]
            trace = metadata.get(passage.id, {})
            evidence.append(
                EvidenceRecord(
                    evidence_id=f"E{index}",
                    source_id=passage.source_id,
                    passage_id=passage.id,
                    claim_hint=hints.get(passage.id, excerpt),
                    excerpt=excerpt,
                    relevance_score=scores.get(passage.id, 0.0),
                    lexical_rank=trace.get("lexical_rank"),
                    dense_rank=trace.get("dense_rank"),
                    rrf_score=trace.get("rrf_score"),
                    rerank_score=trace.get("rerank_score"),
                    selection_stage=str(trace.get("selection_stage", "snippet")),
                )
            )
        return evidence
