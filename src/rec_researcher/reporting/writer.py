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
from rec_researcher.domain.recommender import (
    RecommendationDomainAnalyzer,
    RecommendationPaperProfile,
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
        profiles = self._profiles(result)
        paper_lines = [
            "- "
            f"{profile.title}：{', '.join(profile.task_type) or '任务类型未知'}；"
            f"代码：{', '.join(profile.code_urls) or '未确认公开代码'}。{citation}"
            for profile in profiles
        ]
        dataset_lines = [
            "- "
            f"{profile.title}：数据集 {', '.join(profile.datasets) or 'unknown'}；"
            f"指标 {', '.join(profile.metrics) or 'unknown'}。{citation}"
            for profile in profiles
        ]
        difficulty_lines = [
            "- "
            f"{profile.title}：{profile.reproduction_difficulty.value} "
            f"(score={profile.reproduction_score})；"
            f"{'；'.join(profile.reproduction_score_reasons) or '未触发加减分规则'}。"
            f"{citation}"
            for profile in profiles
        ]
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
                "## 论文与代码对照",
                "",
                *(paper_lines or ["- 当前没有可分析的论文来源。"]),
                "",
                "## 数据集与指标",
                "",
                *(dataset_lines or ["- 当前证据未确认数据集与指标。"]),
                "",
                "## 复现难度分析",
                "",
                *(difficulty_lines or ["- 当前证据不足，无法进行规则评分。"]),
                "",
                "## 三天复现建议",
                "",
                "- 第一天：核对论文、代码、数据许可及评估口径。",
                "- 第二天：运行最小基线，固定随机种子与配置。",
                "- 第三天：复现主指标，记录差异、风险和未决证据。",
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
    def _profiles(result: ResearchOutput) -> list[RecommendationPaperProfile]:
        analyzer = RecommendationDomainAnalyzer()
        return [
            analyzer.analyze(
                [source],
                [item for item in result.evidence if item.source_id == source.id],
            )
            for source in result.sources
        ]

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
        generated = await self.language_model.generate(prompt)
        original = self._with_canonical_references(generated, registry)
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
        generated_repair = await self.language_model.generate(repair_prompt)
        repaired = self._with_canonical_references(generated_repair, registry)
        repaired_validation = self.verifier.verify(repaired, result.sources, registry)
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
    def _generation_prompt(result: ResearchOutput, registry: CitationRegistry) -> str:
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
            "citation, source, URL, or claim. Do not write a References section; "
            "the writer appends it deterministically from the source registry. "
            "Include these exact recommender-specific sections: "
            "## 论文与代码对照, ## 数据集与指标, ## 复现难度分析, and "
            "## 三天复现建议. Do not state a GPU-memory number unless the supplied "
            "evidence explicitly states it; otherwise use unknown. Explain "
            "reproduction difficulty using only these score rules: no public code +2, "
            "no public data "
            "+2, LLM training +2, multi-GPU/distributed +2, single-GPU -1, complete "
            "configuration and checkpoint -1. "
            "If evidence is insufficient, say so explicitly.\n\n"
            f"Question: {result.question}\n"
            f"Tasks: {json.dumps(tasks, ensure_ascii=False)}\n"
            f"Evidence: {json.dumps(evidence, ensure_ascii=False)}\n"
            f"Source registry: {json.dumps(sources, ensure_ascii=False)}\n"
            f"Canonical references:\n{registry.references_markdown()}"
        )

    @staticmethod
    def _with_canonical_references(report: str, registry: CitationRegistry) -> str:
        """Replace any model-written References with the registry rendering."""

        references_heading = re.search(r"(?im)^#{1,6}\s+References\s*$", report)
        body = report[: references_heading.start()] if references_heading else report
        return f"{body.rstrip()}\n\n{registry.references_markdown()}"

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
