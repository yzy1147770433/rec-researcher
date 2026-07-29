from rec_researcher.core.models import PassageRecord, SourceRecord
from rec_researcher.retrieval.dedup import (
    deduplicate_passages,
    deduplicate_sources,
    normalize_url,
)


def _source(source_id: str, url: str) -> SourceRecord:
    return SourceRecord(
        id=source_id, title="title", url=url, snippet="", provider="fixture"
    )


def test_normalize_url_removes_fragment_trailing_slash_and_default_port() -> None:
    assert normalize_url("HTTPS://Example.COM:443/path/#section") == (
        "https://example.com/path"
    )
    assert normalize_url("https://example.com/#fragment") == "https://example.com/"


def test_duplicate_urls_keep_first_source() -> None:
    sources = [
        _source("first", "https://example.com/article/"),
        _source("second", "https://EXAMPLE.com/article#details"),
        _source("third", "https://example.com/other"),
    ]

    assert [source.id for source in deduplicate_sources(sources)] == ["first", "third"]


def test_normalized_text_fingerprint_removes_duplicate_chunks() -> None:
    passages = [
        PassageRecord(id="one", source_id="a", text="Same   TEXT"),
        PassageRecord(id="two", source_id="b", text=" same text \n"),
        PassageRecord(id="three", source_id="a", text="different"),
    ]

    assert [item.id for item in deduplicate_passages(passages)] == ["one", "three"]
