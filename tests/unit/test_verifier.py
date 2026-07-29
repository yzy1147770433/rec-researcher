from rec_researcher.core.models import SourceRecord
from rec_researcher.evidence.verifier import CitationVerifier


def sources() -> list[SourceRecord]:
    return [
        SourceRecord(
            id=f"source-{index}",
            title=f"Title {index}",
            url=f"https://example.com/{index}",
            snippet="text",
            provider="test",
        )
        for index in range(1, 3)
    ]


def test_valid_report_metrics_and_long_paragraph_warning() -> None:
    uncited = "This is a long factual paragraph without a citation. " * 4
    report = (
        "# Report\n\n## Findings\n\nSupported factual paragraph with enough detail "
        "to count for coverage [S1].\n\n"
        f"{uncited}\n\n## References\n\n"
        "- [S1] Title 1 — https://example.com/1\n"
        "- [S2] Title 2 — https://example.com/2\n"
    )

    result = CitationVerifier(long_paragraph_length=100).verify(report, sources())

    assert result.valid
    assert result.citation_coverage == 0.5
    assert result.source_diversity == 0.5
    assert result.uncited_long_paragraphs


def test_rejects_unknown_zero_gap_and_missing_major_citation() -> None:
    report = (
        "# Report\n\n## Findings\n\nNo cited findings here.\n\n"
        "## References\n\n- [S0] Bad — ftp://example.com/0\n"
        "- [S2] Title 2 — https://example.com/2\n"
    )

    result = CitationVerifier().verify(report, sources())

    assert not result.valid
    assert any("[S0]" in error for error in result.errors)
    assert any("gap" in error or "consecutive" in error for error in result.errors)
    assert any("major" in error for error in result.errors)


def test_matches_s1_through_s8_with_plain_and_bulleted_references() -> None:
    eight_sources = [
        SourceRecord(
            id=f"source-{index}",
            title=f"Title {index}",
            url=f"https://example.com/{index}",
            snippet="text",
            provider="test",
        )
        for index in range(1, 9)
    ]
    body_citations = " ".join(f"[S{index}]" for index in range(1, 9))
    references = "\n".join(
        (
            f"[S{index}] Title {index} — https://example.com/{index}"
            if index % 2
            else f"- [S{index}] Title {index} — https://example.com/{index}"
        )
        for index in range(1, 9)
    )
    report = (
        "# Report\n\n## Findings\n\n"
        f"Every source is cited in this factual summary: {body_citations}\n\n"
        f"## References\n\n{references}\n"
    )

    result = CitationVerifier().verify(report, eight_sources)

    assert result.valid, result.errors
    assert result.cited_source_count == 8
    assert result.source_diversity == 1.0
