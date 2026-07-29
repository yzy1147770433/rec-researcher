import pytest

from rec_researcher.core.exceptions import ReportValidationError
from rec_researcher.core.models import ResearchOutput, SourceRecord
from rec_researcher.reporting.writer import RealReportWriter


class StubLanguageModel:
    async def generate(self, prompt: str) -> str:
        assert "Never invent a citation" in prompt
        assert '"id": "source-1"' in prompt
        return "Supported claim [source-1]."


@pytest.mark.asyncio
async def test_real_writer_uses_structured_sources() -> None:
    result = ResearchOutput(
        question="question",
        sources=[
            SourceRecord(
                id="source-1",
                title="Title",
                url="https://example.com/source",
                snippet="Evidence",
                provider="test",
            )
        ],
    )

    report = await RealReportWriter(StubLanguageModel()).write(result)

    assert report == "Supported claim [source-1]."


def test_real_writer_rejects_unknown_citation() -> None:
    with pytest.raises(ReportValidationError, match="unknown"):
        RealReportWriter.validate_citations("Claim [unknown].", [])
