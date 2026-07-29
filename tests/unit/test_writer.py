import pytest

from rec_researcher.core.exceptions import ReportValidationError
from rec_researcher.core.models import ResearchOutput, SourceRecord
from rec_researcher.reporting.writer import RealReportWriter


class StubLanguageModel:
    async def generate(self, prompt: str) -> str:
        assert "Never invent" in prompt
        assert '"id": "source-1"' in prompt
        return (
            "# Report\n\n## Findings\n\nSupported factual claim with sufficient "
            "detail for verification [S1].\n\n## References\n\n"
            "- [S1] Title — https://example.com/source"
        )


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

    assert "[S1]" in report


class RepairingLanguageModel:
    def __init__(self, *, successful: bool) -> None:
        self.calls = 0
        self.successful = successful

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return "# Report\n\n## Findings\n\nBad claim [S9]."
        if not self.successful:
            return "Still invalid [S0]."
        return (
            "# Report\n\n## Findings\n\nCorrected factual claim with enough "
            "detail to verify [S1].\n\n## References\n\n"
            "- [S1] Title — https://example.com/source"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("successful", [True, False])
async def test_real_writer_repairs_once_and_preserves_failed_original(
    successful: bool,
) -> None:
    model = RepairingLanguageModel(successful=successful)
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
    writer = RealReportWriter(model)

    report = await writer.write(result)

    assert model.calls == 2
    assert writer.last_validation.valid is successful
    if successful:
        assert "Corrected" in report
    else:
        assert "Bad claim [S9]" in report
        assert writer.last_validation.errors


def test_real_writer_rejects_unknown_citation() -> None:
    with pytest.raises(ReportValidationError, match="unknown"):
        RealReportWriter.validate_citations("Claim [unknown].", [])
