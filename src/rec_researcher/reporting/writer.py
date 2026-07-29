"""Mock and LLM-backed Markdown report generation."""

from __future__ import annotations

import json
import re

from rec_researcher.core.exceptions import ReportValidationError
from rec_researcher.core.models import ResearchOutput, SourceRecord, WorkState
from rec_researcher.providers.base import LanguageModel


class ReportWriter:
    """Render structured mock research output as traceable Markdown."""

    def write(self, result: ResearchOutput) -> str:
        """Build a report whose numbered citations resolve to references."""

        sources = result.sources
        citation = self._citation(sources)
        task_lines = [f"- {task.question}" for task in result.tasks]
        error_lines = [
            f"- {item.task_id}: {'; '.join(item.errors)}"
            for item in result.task_results
            if item.state == WorkState.FAILED
        ]
        references = [
            f"- [{source.id}] {source.title} — {source.url}" for source in sources
        ]
        return "\n".join(
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
                "## References",
                "",
                *(references or ["- 无引用来源。"]),
                "",
            ]
        )

    @staticmethod
    def _citation(sources: list[SourceRecord]) -> str:
        if not sources:
            return ""
        return " " + " ".join(f"[{source.id}]" for source in sources[:2])


MockReportWriter = ReportWriter


class RealReportWriter:
    """Generate a report from structured sources and validate citations locally."""

    def __init__(self, language_model: LanguageModel) -> None:
        """Configure the language model used for report generation."""

        self.language_model = language_model

    async def write(self, result: ResearchOutput) -> str:
        """Generate Markdown and reject citations absent from supplied sources."""

        sources = [source.model_dump(mode="json") for source in result.sources]
        tasks = [task.model_dump(mode="json") for task in result.tasks]
        prompt = (
            "Write a Markdown research report answering the question. Use only the "
            "structured sources below as evidence. Cite a source using its exact ID "
            "in square brackets, for example [source-id]. Never invent a citation or "
            "cite an ID or URL that is not present in sources. Include a References "
            "section. If sources are insufficient, say so explicitly.\n\n"
            f"Question: {result.question}\n"
            f"Tasks: {json.dumps(tasks, ensure_ascii=False)}\n"
            f"Sources: {json.dumps(sources, ensure_ascii=False)}"
        )
        report = await self.language_model.generate(prompt)
        self.validate_citations(report, result.sources)
        return report

    @staticmethod
    def validate_citations(report: str, sources: list[SourceRecord]) -> None:
        """Ensure bracket citations resolve to one of the structured source IDs."""

        known_ids = {source.id for source in sources}
        cited_ids = set(re.findall(r"\[([A-Za-z0-9][A-Za-z0-9_.:-]*)\]", report))
        unknown = cited_ids - known_ids
        if unknown:
            raise ReportValidationError(
                "report contains citations absent from sources: "
                + ", ".join(sorted(unknown))
            )
