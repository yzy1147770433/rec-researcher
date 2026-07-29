"""Sequential asynchronous research workflow."""

from __future__ import annotations

import inspect
import os
import tempfile
import time
import uuid
from pathlib import Path

from rec_researcher.core.models import (
    ResearchOutput,
    ResearchRun,
    RunStatistics,
    SourceRecord,
    TaskResult,
    WorkState,
)
from rec_researcher.planning.planner import ResearchPlanner
from rec_researcher.providers.base import SearchProvider
from rec_researcher.providers.mock import MockSearchProvider
from rec_researcher.reporting.writer import RealReportWriter, ReportWriter
from rec_researcher.workflow.budget import RunBudget


class ResearchOrchestrator:
    """Compose planner, search provider, writer, budgets, and persistence."""

    def __init__(
        self,
        *,
        output_dir: Path = Path("outputs"),
        planner: ResearchPlanner | None = None,
        search_provider: SearchProvider | None = None,
        writer: ReportWriter | RealReportWriter | None = None,
        mode: str = "mock",
        max_tasks: int = 5,
        max_sources: int = 30,
        sources_per_query: int = 5,
    ) -> None:
        """Configure an offline workflow and its work limits."""

        self.output_dir = output_dir
        self.planner = planner or ResearchPlanner()
        self.search_provider = search_provider or MockSearchProvider()
        self.writer = writer or ReportWriter()
        self.mode = mode
        self.max_tasks = max_tasks
        self.max_sources = max_sources
        self.sources_per_query = sources_per_query

    async def run(self, question: str) -> ResearchRun:
        """Run tasks sequentially, isolate task errors, and persist artifacts."""

        started = time.monotonic()
        tasks = await self.planner.create_tasks(question)
        budget = RunBudget(max_tasks=self.max_tasks, max_sources=self.max_sources)
        budget.consume_task(len(tasks))
        task_results: list[TaskResult] = []
        sources_by_id: dict[str, SourceRecord] = {}

        for task in tasks:
            try:
                budget.record_api_call()
                found = await self.search_provider.search(
                    task.search_queries[0], limit=self.sources_per_query
                )
                new_sources = [item for item in found if item.id not in sources_by_id]
                budget.consume_sources(len(new_sources))
                sources_by_id.update((item.id, item) for item in new_sources)
                task_results.append(
                    TaskResult(
                        task_id=task.id,
                        state=WorkState.COMPLETED,
                        source_ids=[item.id for item in found],
                        findings=[item.snippet for item in found],
                    )
                )
            except Exception as exc:  # task boundary intentionally isolates providers
                task_results.append(
                    TaskResult(
                        task_id=task.id,
                        state=WorkState.FAILED,
                        errors=[f"{type(exc).__name__}: {exc}"],
                    )
                )

        sources = list(sources_by_id.values())
        statistics = RunStatistics(
            planned_tasks=len(tasks),
            completed_tasks=sum(r.state == WorkState.COMPLETED for r in task_results),
            failed_tasks=sum(r.state == WorkState.FAILED for r in task_results),
            sources_found=len(sources),
            elapsed_seconds=time.monotonic() - started,
        )
        output = ResearchOutput(
            question=question.strip(),
            tasks=tasks,
            task_results=task_results,
            sources=sources,
            statistics=statistics,
            reproduction_suggestions=[
                "固定随机种子并保存配置。",
                "记录数据集版本与划分策略。",
            ],
        )
        report = self.writer.write(output)
        output.markdown_report = await report if inspect.isawaitable(report) else report
        run = ResearchRun(run_id=uuid.uuid4().hex, mode=self.mode, output=output)
        self._persist(run)
        return run

    def _persist(self, run: ResearchRun) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{run.run_id}-", dir=self.output_dir)
        )
        (temporary / "report.md").write_text(
            run.output.markdown_report, encoding="utf-8"
        )
        (temporary / "sources.json").write_text(
            self._json(
                [source.model_dump(mode="json") for source in run.output.sources]
            ),
            encoding="utf-8",
        )
        (temporary / "run.json").write_text(
            run.model_dump_json(indent=2), encoding="utf-8"
        )
        os.replace(temporary, self.output_dir / run.run_id)

    @staticmethod
    def _json(value: object) -> str:
        import json

        return json.dumps(value, ensure_ascii=False, indent=2)
