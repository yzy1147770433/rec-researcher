"""Mock and LLM-backed Markdown report generation."""

from __future__ import annotations

import json
import re

from rec_researcher.core.exceptions import ReportValidationError
from rec_researcher.core.models import (
    CitationValidation,
    ResearchOutput,
    SourceRecord,
    WorkState,
)
from rec_researcher.evidence.verifier import CitationVerifier
from rec_researcher.providers.base import LanguageModel
from rec_researcher.reporting.citation import CitationRegistry


class ReportWriter:
    """Render deterministic mock research output as traceable Markdown."""

    def __init__(self) -> None:
        """Initialize the latest validation result."""

        self.last_validation = CitationValidation()

    def write(self, result: ResearchOutput) -> str:
        """Build and locally verify a report using canonical source labels."""

        registry = CitationRegistry(result.sources)
        citation = self._citation(registry)
        task_lines = [f"- {task.question}" for task in result.tasks]
        error_lines = [
            f"- {item.task_id}: {'; '.join(item.errors)}"
            for item in result.task_results
            if item.state == WorkState.FAILED
        ]
        report = "\n".join(
            [
                f"# {result.question}",
                "",
                "## 问题拆解",
                "",
                *(task_lines or ["- 无可用子任务。"]),
                "",
                "## 技术路线",
                "",
                f"采用离线检索夹具梳理协同过滤、序列建模与评估流程。{citation}",
                "",
                "## 代表工作",
                "",
                f"代表性方向包括潜因子方法和基于注意力的序列推荐。{citation}",
                "",
                "## 实验与指标",
                "",
                f"建议同时报告 Recall、NDCG、覆盖率，并固定数据划分。{citation}",
                "",
                "## 复现建议",
                "",
                f"记录数据版本、随机种子、负采样、超参数与运行环境。{citation}",
                "",
                "## 局限性",
                "",
                "本报告完全由虚构的离线测试来源生成，不能替代真实文献检索。",
                *(error_lines or []),
                "",
                registry.references_markdown(),
                "",
            ]
        )
        self.last_validation = CitationVerifier().verify(
            report, result.sources, registry
        )
        return report

    @staticmethod
    def _citation(registry: CitationRegistry) -> str:
        labels = registry.labels[:2]
        return " " + " ".join(f"[{label}]" for label in labels) if labels else ""


MockReportWriter = ReportWriter


class RealReportWriter:
    """Generate, verify, and at most once repair an evidence-grounded report."""

    def __init__(
        self,
        language_model: LanguageModel,
        *,
        verifier: CitationVerifier | None = None,
    ) -> None:
        """Configure generation and deterministic local verification."""

        self.language_model = language_model
        self.verifier = verifier or CitationVerifier()
        self.last_validation = CitationValidation()

    async def write(self, result: ResearchOutput) -> str:
        """Generate a report and perform no more than one citation repair."""

        registry = CitationRegistry(result.sources)
        prompt = self._generation_prompt(result, registry)
        original = await self.language_model.generate(prompt)
        validation = self.verifier.verify(original, result.sources, registry)
        if validation.valid:
            self.last_validation = validation
            return original

        repair_prompt = (
            "Repair only the citation and References problems in the report below. "
            "Use exclusively the supplied labels; do not invent sources or claims. "
            "Return the complete Markdown report.\n\n"
            f"Allowed labels: {', '.join(f'[{item}]' for item in registry.labels)}\n"
            f"Canonical references:\n{registry.references_markdown()}\n"
            f"Validation errors: {json.dumps(validation.errors, ensure_ascii=False)}\n"
            f"Original report:\n{original}"
        )
        repaired = await self.language_model.generate(repair_prompt)
        repaired_validation = self.verifier.verify(
            repaired, result.sources, registry
        )
        if repaired_validation.valid:
            self.last_validation = repaired_validation
            return repaired
        # Preserve the original output when the one permitted repair is unsuccessful.
        validation.warnings.append(
            "citation repair failed: " + "; ".join(repaired_validation.errors)
        )
        self.last_validation = validation
        return original

    @staticmethod
    def _generation_prompt(
        result: ResearchOutput, registry: CitationRegistry
    ) -> str:
        sources = []
        for source in result.sources:
            item = source.model_dump(mode="json")
            item["citation"] = f"[{registry.label_for_source(source.id)}]"
            sources.append(item)
        evidence = []
        for item in result.evidence:
            value = item.model_dump(mode="json")
            value["citation"] = f"[{registry.label_for_source(item.source_id)}]"
            evidence.append(value)
        tasks = [task.model_dump(mode="json") for task in result.tasks]
        return (
            "Write a Markdown research report answering the question. Use only the "
            "evidence and source registry below. Every factual claim must retain its "
            "source citation. Cite only the exact allowed [Sx] labels. Never invent a "
            "citation, source, URL, or claim. Include the canonical References "
            "section. "
            "If evidence is insufficient, say so explicitly.\n\n"
            f"Question: {result.question}\n"
            f"Tasks: {json.dumps(tasks, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(evidence, ensure_ascii=False)}\n"
            f"Source registry: {json.dumps(sources, ensure_ascii=False)}\n"
            f"Canonical references:\n{registry.references_markdown()}"
        )

    @staticmethod
    def validate_citations(report: str, sources: list[SourceRecord]) -> None:
        """Retain the legacy source-ID validation helper for callers."""

        known_ids = {source.id for source in sources}
        cited_ids = set(re.findall(r"\[([A-Za-z0-9][A-Za-z0-9_.:-]*)\]", report))
        unknown = cited_ids - known_ids
        if unknown:
            raise ReportValidationError(
                "report contains citations absent from sources: "
                + ", ".join(sorted(unknown))
            )
