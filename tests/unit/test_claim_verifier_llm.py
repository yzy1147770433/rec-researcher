import pytest

from rec_researcher.core.models import EvidenceRecord, ReportClaim, SourceRecord
from rec_researcher.evidence.claim_verifier import LLMClaimEvidenceVerifier


class FakeLLM:
    async def generate(self, prompt: str) -> str:
        assert '"claim_id": "C1"' in prompt
        return (
            '{"results":[{"claim_id":"C1","status":"supported",'
            '"confidence":0.91,"explanation":"entailed"}]}'
        )


class FailingLLM:
    async def generate(self, prompt: str) -> str:
        raise RuntimeError("offline failure")


def source() -> SourceRecord:
    return SourceRecord(
        id="src",
        title="Paper",
        url="https://arxiv.org/abs/1",
        snippet="x",
        provider="test",
    )


def evidence() -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id="E1",
        source_id="src",
        passage_id="P1",
        claim_hint="x",
        excerpt="模型使用 sampled softmax 训练",
    )


@pytest.mark.asyncio
async def test_llm_verifier_parses_entailment() -> None:
    claim = ReportClaim(claim_id="C1", text="模型使用 softmax", citation_ids=["S1"])
    result = await LLMClaimEvidenceVerifier(FakeLLM()).verify(
        [claim], [source()], [evidence()]
    )
    assert result[0].status == "supported"
    assert result[0].confidence == 0.91


@pytest.mark.asyncio
async def test_llm_verifier_falls_back_on_failure() -> None:
    claim = ReportClaim(
        claim_id="C1", text="模型使用 sampled softmax 训练", citation_ids=["S1"]
    )
    result = await LLMClaimEvidenceVerifier(FailingLLM()).verify(
        [claim], [source()], [evidence()]
    )
    assert result[0].status == "supported"
