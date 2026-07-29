from rec_researcher.core.models import SourceRecord
from rec_researcher.reporting.citation import CitationRegistry


def make_source(source_id: str, url: str) -> SourceRecord:
    return SourceRecord(
        id=source_id, title=source_id, url=url, snippet="", provider="test"
    )


def test_registry_is_stable_and_deduplicates_urls() -> None:
    registry = CitationRegistry(
        [
            make_source("alpha", "https://example.com/a"),
            make_source("alias", "https://example.com/a"),
            make_source("beta", "https://example.com/b"),
        ]
    )

    assert registry.labels == ("S1", "S2")
    assert registry.label_for_source("alpha") == "S1"
    assert registry.label_for_source("alias") == "S1"
    assert registry.label_for_source("beta") == "S2"
    assert registry.references_markdown().count("[S1]") == 1
    assert registry.references_markdown().splitlines()[2].startswith("[S1]")
