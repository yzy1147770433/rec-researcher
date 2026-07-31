"""Claim-level deterministic evidence verification with safe failure behavior."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence

from rec_researcher.core.models import (
    ClaimVerificationResult,
    EvidenceRecord,
    ReportClaim,
    SourceRecord,
)
from rec_researcher.providers.base import LanguageModel
from rec_researcher.providers.llm_http import OpenAICompatibleLanguageModel

_CITATION = re.compile(r"\[(S\d+)\]")
_TOKEN = re.compile(r"[A-Za-z0-9_.+-]+|[\u4e00-\u9fff]")


class ClaimExtractor:
    """Extract citation-bearing or sufficiently factual prose sentences."""

    def extract(self, report: str) -> list[ReportClaim]:
        section: str | None = None
        claims: list[ReportClaim] = []
        for line in report.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                section = stripped[3:].strip()
                continue
            if not stripped or stripped.startswith(("#", "- [S")):
                continue
            line_citations = _CITATION.findall(stripped)
            clean_line = _CITATION.sub("", stripped)
            for sentence in re.split(r"(?<=[。！？.!?])\s*", clean_line):
                text = sentence.strip().lstrip("- ")
                citations = line_citations
                plain = text.strip()
                factual = bool(citations) or bool(
                    re.search(
                        r"\d|提出|使用|定义|结果|dataset|method|uses|reports",
                        plain,
                        re.I,
                    )
                )
                if len(plain) >= 6 and factual:
                    claims.append(
                        ReportClaim(
                            claim_id=f"C{len(claims) + 1}",
                            text=plain,
                            section=section,
                            citation_ids=citations,
                            requires_evidence=True,
                        )
                    )
        return claims


class ClaimEvidenceVerifier:
    """Judge support using source linkage followed by deterministic overlap."""

    def verify(
        self,
        claims: Sequence[ReportClaim],
        sources: Sequence[SourceRecord],
        evidence: Sequence[EvidenceRecord],
    ) -> list[ClaimVerificationResult]:
        labels = {f"S{i}": source.id for i, source in enumerate(sources, start=1)}
        by_source: dict[str, list[EvidenceRecord]] = {}
        for item in evidence:
            by_source.setdefault(item.source_id, []).append(item)
        results: list[ClaimVerificationResult] = []
        for claim in claims:
            if not claim.requires_evidence:
                continue
            if not claim.citation_ids:
                results.append(
                    self._result(
                        claim, "missing_citation", 1.0, [], "事实性陈述没有引用"
                    )
                )
                continue
            if any(label not in labels for label in claim.citation_ids):
                results.append(
                    self._result(
                        claim, "invalid_citation", 1.0, [], "引用不在来源注册表中"
                    )
                )
                continue
            candidates = [
                item
                for label in claim.citation_ids
                for item in by_source.get(labels[label], [])
            ]
            if not candidates:
                results.append(
                    self._result(
                        claim, "unsupported", 0.9, [], "引用来源没有可用证据段落"
                    )
                )
                continue
            claim_tokens = set(_TOKEN.findall(claim.text.casefold()))
            scored = []
            for item in candidates:
                evidence_tokens = set(_TOKEN.findall(item.excerpt.casefold()))
                score = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
                scored.append((score, item))
            score, best = max(scored, key=lambda pair: pair[0])
            if score >= 0.55:
                status, explanation = "supported", "引用段落与陈述具有充分词项支持"
            elif score >= 0.25:
                status, explanation = (
                    "partially_supported",
                    "引用段落仅支持陈述的一部分",
                )
            else:
                status, explanation = "unsupported", "引用段落与陈述的可验证重合不足"
            results.append(
                self._result(
                    claim,
                    status,
                    min(1.0, 0.5 + score),
                    [best.evidence_id],
                    explanation,
                )
            )
        return results

    @staticmethod
    def _result(
        claim: ReportClaim,
        status: str,
        confidence: float,
        evidence_ids: list[str],
        explanation: str,
    ) -> ClaimVerificationResult:
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            status=status,
            confidence=confidence,
            evidence_ids=evidence_ids,
            explanation=explanation,
        )


class LLMClaimEvidenceVerifier:
    """Optional entailment judge with deterministic prefilter and fallback."""

    def __init__(
        self,
        language_model: LanguageModel,
        *,
        fallback: ClaimEvidenceVerifier | None = None,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.language_model = language_model
        self.fallback = fallback or ClaimEvidenceVerifier()
        self.timeout_seconds = timeout_seconds

    async def verify(
        self,
        claims: Sequence[ReportClaim],
        sources: Sequence[SourceRecord],
        evidence: Sequence[EvidenceRecord],
    ) -> list[ClaimVerificationResult]:
        """Use deterministic linkage first and ask the LLM only for valid evidence."""

        baseline = self.fallback.verify(claims, sources, evidence)
        labels = {f"S{i}": source.id for i, source in enumerate(sources, start=1)}
        evidence_by_source: dict[str, list[EvidenceRecord]] = {}
        for item in evidence:
            evidence_by_source.setdefault(item.source_id, []).append(item)
        eligible: dict[str, tuple[ReportClaim, list[EvidenceRecord]]] = {}
        for claim, deterministic in zip(claims, baseline, strict=True):
            if deterministic.status in {"missing_citation", "invalid_citation"}:
                continue
            passages = [
                item
                for label in claim.citation_ids
                for item in evidence_by_source.get(labels.get(label, ""), [])
            ][:3]
            if not passages:
                continue
            eligible[claim.claim_id] = (claim, passages)
        if not eligible:
            return baseline
        request = [
            {
                "claim_id": claim_id,
                "claim": claim.text,
                "evidence": [
                    {"evidence_id": item.evidence_id, "text": item.excerpt}
                    for item in passages
                ],
            }
            for claim_id, (claim, passages) in eligible.items()
        ]
        prompt = (
            "Judge evidence support for every claim. Return JSON only as "
            '{"results":[{"claim_id":"C1","status":"supported|'
            'partially_supported|unsupported","confidence":0.0,'
            '"explanation":"..."}]}. Do not omit claims.\n'
            + json.dumps(request, ensure_ascii=False)
        )
        try:
            async with asyncio.timeout(self.timeout_seconds):
                raw = await self.language_model.generate(prompt)
            payload = OpenAICompatibleLanguageModel.parse_json(raw)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("results"), list
            ):
                raise ValueError("verifier response must contain results")
            judgments = {
                str(item.get("claim_id")): item
                for item in payload["results"]
                if isinstance(item, dict)
            }
            output = list(baseline)
            for index, claim in enumerate(claims):
                if claim.claim_id not in eligible or claim.claim_id not in judgments:
                    continue
                item = judgments[claim.claim_id]
                status = str(item.get("status"))
                if status not in {"supported", "partially_supported", "unsupported"}:
                    continue
                passages = eligible[claim.claim_id][1]
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
                output[index] = ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    status=status,
                    confidence=confidence,
                    evidence_ids=[passage.evidence_id for passage in passages],
                    explanation=str(item.get("explanation", "LLM entailment judgment")),
                )
            return output
        except Exception:  # noqa: BLE001 - optional verifier must degrade safely
            return baseline
