import pytest

from rec_researcher.core.models import PassageRecord, SourceRecord
from rec_researcher.evidence.builder import EvidenceBuilder


def source() -> SourceRecord:
    return SourceRecord(
        id="source-a",
        title="Title",
        url="https://example.com/a",
        snippet="snippet",
        provider="test",
    )


def test_builds_bounded_traceable_evidence() -> None:
    passage = PassageRecord(id="p1", source_id="source-a", text="abcdefgh")

    result = EvidenceBuilder(max_excerpt_length=5).build(
        [passage],
        [source()],
        relevance_scores={"p1": 0.75},
        claim_hints={"p1": "claim"},
    )

    assert result[0].model_dump() == {
        "evidence_id": "E1",
        "source_id": "source-a",
        "passage_id": "p1",
        "claim_hint": "claim",
        "excerpt": "abcde",
        "relevance_score": 0.75,
    }


def test_rejects_orphaned_passage_and_handles_empty_input() -> None:
    builder = EvidenceBuilder()
    assert builder.build([], []) == []
    passage = PassageRecord(id="p1", source_id="missing", text="text")
    with pytest.raises(ValueError, match="unknown source"):
        builder.build([passage], [source()])
