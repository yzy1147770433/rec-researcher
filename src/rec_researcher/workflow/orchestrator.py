"""Controlled asynchronous research workflow."""

from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
import uuid
from pathlib import Path

from rec_researcher.core.models import (
    InquiryTask,
    PassageRecord,
    ResearchOutput,
    ResearchRun,
    RunBudgetRecord,
    RunStatistics,
    SourceRecord,
    TaskResult,
    WorkState,
)
from rec_researcher.evidence.builder import EvidenceBuilder
from rec_researcher.planning.planner import ResearchPlanner
from rec_researcher.providers.base import SearchProvider
from rec_researcher.providers.mock import MockSearchProvider
from rec_researcher.reporting.writer import RealReportWriter, ReportWriter
from rec_researcher.workflow.budget import RunBudget
from rec_researcher.workflow.scheduler import AsyncTaskScheduler, ScheduledTask


class ResearchOrchestrator:
    """Compose planning, bounded concurrent research, reporting, and persistence."""

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
        max_concurrency: int = 3,
        retrieval_concurrency: int = 3,
        timeout: float = 120.0,
        evidence_excerpt_length: int = 600,
    ) -> None:
        """Configure providers and independent workflow/retrieval limits."""

        if retrieval_concurrency < 1:
            raise ValueError("retrieval_concurrency must be at least 1")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.output_dir = output_dir
        self.planner = planner or ResearchPlanner()
        self.search_provider = search_provider or MockSearchProvider()
        self.writer = writer or ReportWriter()
        self.mode = mode
        self.max_tasks = max_tasks
        self.max_sources = max_sources
        self.sources_per_query = sources_per_query
        self.max_concurrency = max_concurrency
        self.retrieval_concurrency = retrieval_concurrency
        self.timeout = timeout
        self.evidence_builder = EvidenceBuilder(
            max_excerpt_length=evidence_excerpt_length
        )

    async def run(self, question: str) -> ResearchRun:
        """Run research with a global deadline and always persist terminal runs."""

        normalized = question.strip()
        if not normalized:
            raise ValueError("research question must not be empty")
        run_id = uuid.uuid4().hex
        budget = RunBudget(max_tasks=self.max_tasks, max_sources=self.max_sources)
        tasks: list[InquiryTask] = []
        task_results: list[TaskResult] = []
        sources: list[SourceRecord] = []
        limitations: list[str] = []
        interrupted: BaseException | None = None

        try:
            async with asyncio.timeout(self.timeout):
                tasks = await self.planner.create_tasks(normalized)
                budget.consume_task(len(tasks))
                task_results, sources = await self._research(tasks, budget)
        except TimeoutError:
            limitations.append(f"global timeout after {self.timeout:g} seconds")
            task_results = self._fill_missing_results(
                tasks, task_results, "global timeout"
            )
        except (KeyboardInterrupt, asyncio.CancelledError) as exc:
            limitations.append(f"run interrupted: {type(exc).__name__}")
            task_results = self._fill_missing_results(
                tasks, task_results, "run interrupted"
            )
            interrupted = exc

        failures = [
            result for result in task_results if result.state != WorkState.COMPLETED
        ]
        for result in failures:
            budget.record_failed_task(result.task_id)
        limitations.extend(
            f"{result.task_id}: {error}"
            for result in failures
            for error in result.errors[:1]
        )
        if failures and not limitations:
            limitations.append("部分研究任务失败，报告仅基于可用来源。")
        budget.warnings.extend(limitations)

        statistics = RunStatistics(
            planned_tasks=len(tasks),
            completed_tasks=sum(r.state == WorkState.COMPLETED for r in task_results),
            failed_tasks=sum(r.state == WorkState.FAILED for r in task_results),
            sources_found=len(sources),
            passages_created=len(sources),
            evidence_items=len(sources),
            elapsed_seconds=budget.elapsed_seconds,
        )
        passages = [
            PassageRecord(
                id=f"passage-{index}",
                source_id=source.id,
                text=source.snippet,
                position=index - 1,
                end_offset=len(source.snippet),
            )
            for index, source in enumerate(sources, start=1)
        ]
        evidence = self.evidence_builder.build(
            passages,
            sources,
            relevance_scores={
                passage.id: min(1.0, max(0.0, sources[index].score or 0.0))
                for index, passage in enumerate(passages)
            },
        )
        budget.record_passage(len(passages))
        output = ResearchOutput(
            question=normalized,
            tasks=tasks,
            task_results=task_results,
            sources=sources,
            passages=passages,
            evidence=evidence,
            statistics=statistics,
            limitations=limitations,
            reproduction_suggestions=[
                "固定随机种子并保存配置。",
                "记录数据集版本与划分策略。",
            ],
        )
        await self._write_report(output)
        all_failed = bool(task_results) and not any(
            item.state == WorkState.COMPLETED for item in task_results
        )
        status = (
            WorkState.FAILED
            if all_failed or (not tasks and limitations)
            else WorkState.COMPLETED
        )
        run = ResearchRun(
            run_id=run_id,
            mode=self.mode,
            status=status,
            output=output,
            budget=RunBudgetRecord(
                start_time=budget.start_time,
                elapsed_seconds=budget.elapsed_seconds,
                llm_calls=budget.llm_calls,
                search_calls=budget.search_calls,
                embedding_calls=budget.embedding_calls,
                reranker_calls=budget.reranker_calls,
                fetched_pages=budget.fetched_pages,
                source_count=budget.source_count,
                passage_count=budget.passage_count,
                warnings=budget.warnings,
                failed_tasks=budget.failed_tasks,
            ),
        )
        self._persist(run)
        if interrupted is not None:
            raise interrupted
        return run

    async def _research(
        self, tasks: list[InquiryTask], budget: RunBudget
    ) -> tuple[list[TaskResult], list[SourceRecord]]:
        retrieval_limit = asyncio.Semaphore(self.retrieval_concurrency)

        async def search(task: InquiryTask) -> list[SourceRecord]:
            async with retrieval_limit:
                budget.record_call("search")
                if not task.search_queries:
                    return []
                return await self.search_provider.search(
                    task.search_queries[0], limit=self.sources_per_query
                )

        scheduled = [
            ScheduledTask(
                id=task.id,
                operation=lambda task=task: search(task),
                dependencies=([task.parent_id] if task.parent_id else []),
            )
            for task in tasks
        ]
        scheduler = AsyncTaskScheduler(
            max_concurrency=self.max_concurrency, task_timeout=self.timeout
        )
        outcomes = await scheduler.run(scheduled)
        sources_by_id: dict[str, SourceRecord] = {}
        results: list[TaskResult] = []
        for outcome in outcomes:
            found = outcome.value or []
            retained: list[SourceRecord] = []
            for source in found:
                if source.id in sources_by_id:
                    retained.append(source)
                elif len(sources_by_id) < self.max_sources:
                    sources_by_id[source.id] = source
                    retained.append(source)
            if outcome.state == WorkState.COMPLETED:
                results.append(
                    TaskResult(
                        task_id=outcome.task_id,
                        state=outcome.state,
                        source_ids=[item.id for item in retained],
                        findings=[item.snippet for item in retained],
                    )
                )
            else:
                results.append(
                    TaskResult(
                        task_id=outcome.task_id,
                        state=outcome.state,
                        errors=[outcome.error or "task failed"],
                    )
                )
        budget.consume_sources(len(sources_by_id))
        return results, list(sources_by_id.values())

    async def _write_report(self, output: ResearchOutput) -> None:
        try:
            report = self.writer.write(output)
            output.markdown_report = (
                await report if inspect.isawaitable(report) else report
            )
            output.validation = self.writer.last_validation
        except Exception as exc:  # noqa: BLE001 - preserve the run on writer failure
            warning = (
                f"report writer failed; used fallback: {type(exc).__name__}: {exc}"
            )
            output.limitations.append(warning)
            fallback = ReportWriter()
            output.markdown_report = fallback.write(output)
            output.validation = fallback.last_validation
        if output.limitations and "## 局限性" not in output.markdown_report:
            details = "\n".join(f"- {item}" for item in output.limitations)
            output.markdown_report += f"\n\n## 局限性\n\n{details}\n"

    @staticmethod
    def _fill_missing_results(
        tasks: list[InquiryTask], results: list[TaskResult], error: str
    ) -> list[TaskResult]:
        by_id = {result.task_id: result for result in results}
        return [
            by_id.get(
                task.id,
                TaskResult(task_id=task.id, state=WorkState.FAILED, errors=[error]),
            )
            for task in tasks
        ]

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
        (temporary / "evidence.json").write_text(
            self._json([item.model_dump(mode="json") for item in run.output.evidence]),
            encoding="utf-8",
        )
        (temporary / "validation.json").write_text(
            run.output.validation.model_dump_json(indent=2), encoding="utf-8"
        )
        (temporary / "run.json").write_text(
            run.model_dump_json(indent=2), encoding="utf-8"
        )
        os.replace(temporary, self.output_dir / run.run_id)

    @staticmethod
    def _json(value: object) -> str:
        import json

        return json.dumps(value, ensure_ascii=False, indent=2)
