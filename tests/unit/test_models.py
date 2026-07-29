import pytest
from pydantic import ValidationError

from rec_researcher.core.models import (
    EvidenceRecord,
    InquiryTask,
    PassageRecord,
    ResearchOutput,
    SourceRecord,
)


def test_list_defaults_are_not_shared() -> None:
    first = InquiryTask(id="task-1", question="What should be measured?")
    second = InquiryTask(id="task-2", question="Which baseline is appropriate?")

    first.search_queries.append("offline recommender metrics")

    assert second.search_queries == []
    assert ResearchOutput(question="q").sources == []


def test_source_passage_and_evidence_preserve_links() -> None:
    source = SourceRecord(
        id="source-1",
        title="A study",
        url="https://example.com/study",
        snippet="Summary",
        provider="fixture",
    )
    passage = PassageRecord(id="passage-1", source_id=source.id, text="Evidence text")
    evidence = EvidenceRecord(
        evidence_id="evidence-1",
        source_id=source.id,
        passage_id=passage.id,
        claim_hint="A supported claim",
        excerpt=passage.text,
    )

    assert str(source.url) == "https://example.com/study"
    assert passage.source_id == source.id
    assert evidence.source_id == source.id
    assert evidence.passage_id == passage.id


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        InquiryTask(id="task-1", question="q", unknown=True)
