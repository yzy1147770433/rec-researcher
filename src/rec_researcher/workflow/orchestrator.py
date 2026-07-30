"""Controlled asynchronous research workflow."""

from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
import uuid
from collections.abc import Awaitable
from pathlib import Path
from typing import Protocol

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
from rec_researcher.retrieval.chunker import PassageChunker
from rec_researcher.retrieval.dedup import (
    deduplicate_passages,
    normalize_url,
)
from rec_researcher.retrieval.fetcher import FetchResult
from rec_researcher.workflow.budget import RunBudget
from rec_researcher.workflow.scheduler import AsyncTaskScheduler, ScheduledTask


class WebFetcher(Protocol):
    """Provider boundary used to fetch one source without coupling orchestration."""

    def fetch(self, source: SourceRecord) -> Awaitable[FetchResult]:
        """Return a structured, non-raising fetch outcome."""


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
        fetch_concurrency: int = 3,
        timeout: float = 120.0,
        evidence_excerpt_length: int = 600,
        retrieval_mode: str = "snippet",
        web_fetcher: WebFetcher | None = None,
        passage_chunker: PassageChunker | None = None,
        min_fetched_content_length: int = 50,
    ) -> None:
        """Configure providers and independent workflow/retrieval limits."""

        if mode not in {"mock", "real"}:
            raise ValueError("mode must be 'mock' or 'real'")
        if mode == "real":
            missing = [
                name
                for name, value in (
                    ("planner", planner),
                    ("search_provider", search_provider),
                    ("writer", writer),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "real mode requires explicit providers: " + ", ".join(missing)
                )
        if retrieval_concurrency < 1:
            raise ValueError("retrieval_concurrency must be at least 1")
        if fetch_concurrency < 1:
            raise ValueError("fetch_concurrency must be at least 1")
        if retrieval_mode not in {"snippet", "hybrid"}:
            raise ValueError("retrieval_mode must be 'snippet' or 'hybrid'")
        if retrieval_mode == "hybrid" and (
            web_fetcher is None or passage_chunker is None
        ):
            raise ValueError("hybrid mode requires a web_fetcher and passage_chunker")
        if min_fetched_content_length < 1:
            raise ValueError("min_fetched_content_length must be at least 1")
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
        self.fetch_concurrency = fetch_concurrency
        self.timeout = timeout
        self.retrieval_mode = retrieval_mode
        self.web_fetcher = web_fetcher
        self.passage_chunker = passage_chunker
        self.min_fetched_content_length = min_fetched_content_length
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
        passages: list[PassageRecord] = []
        raw_passage_count = 0
        limitations: list[str] = []
        interrupted: BaseException | None = None

        try:
            async with asyncio.timeout(self.timeout):
                tasks = await self.planner.create_tasks(normalized)
                budget.consume_task(len(tasks))
                task_results, sources = await self._research(tasks, budget)
                passages, raw_passage_count = await self._build_passages(
                    sources, budget
                )
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
            passages_created=len(passages),
            evidence_items=len(passages),
            elapsed_seconds=budget.elapsed_seconds,
            fetch_attempts=budget.fetch_attempts,
            fetch_successes=budget.fetch_successes,
            fetch_failures=budget.fetch_failures,
            fallback_passages=budget.fallback_passages,
            raw_passage_count=raw_passage_count,
            deduplicated_passage_count=len(passages),
        )
        evidence = self.evidence_builder.build(
            passages,
            sources,
            relevance_scores={
                passage.id: min(
                    1.0,
                    max(
                        0.0,
                        next(
                            source.score or 0.0
                            for source in sources
                            if source.id == passage.source_id
                        ),
                    ),
                )
                for passage in passages
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
                fetch_attempts=budget.fetch_attempts,
                fetch_successes=budget.fetch_successes,
                fetch_failures=budget.fetch_failures,
                fallback_passages=budget.fallback_passages,
                raw_passage_count=budget.raw_passage_count,
                deduplicated_passage_count=budget.deduplicated_passage_count,
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
        sources_by_url: dict[str, SourceRecord] = {}
        results: list[TaskResult] = []
        for outcome in outcomes:
            found = outcome.value or []
            retained: list[SourceRecord] = []
            for source in found:
                url_key = normalize_url(str(source.url))
                if url_key in sources_by_url:
                    retained.append(sources_by_url[url_key])
                elif source.id in sources_by_id:
                    retained.append(sources_by_id[source.id])
                elif len(sources_by_id) < self.max_sources:
                    sources_by_id[source.id] = source
                    sources_by_url[url_key] = source
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

    async def _build_passages(
        self, sources: list[SourceRecord], budget: RunBudget
    ) -> tuple[list[PassageRecord], int]:
        if self.retrieval_mode == "snippet":
            passages = [
                PassageRecord(
                    id=f"passage-{index}",
                    source_id=source.id,
                    text=source.snippet,
                    position=index - 1,
                    end_offset=len(source.snippet),
                    content_origin="search_snippet_fallback",
                )
                for index, source in enumerate(sources, start=1)
            ]
            budget.record_passage_counts(raw=len(passages), deduplicated=len(passages))
            return passages, len(passages)

        assert self.web_fetcher is not None
        assert self.passage_chunker is not None
        fetch_limit = asyncio.Semaphore(self.fetch_concurrency)

        async def fetch(source: SourceRecord) -> FetchResult:
            try:
                async with fetch_limit:
                    return await self.web_fetcher.fetch(source)
            except Exception as exc:  # noqa: BLE001 - isolate every source
                return FetchResult(
                    source_id=source.id,
                    url=str(source.url),
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                )

        outcomes = await asyncio.gather(*(fetch(source) for source in sources))
        raw_passages: list[PassageRecord] = []
        for source, outcome in zip(sources, outcomes, strict=True):
            usable = (
                outcome.success
                and len(outcome.text.strip()) >= self.min_fetched_content_length
            )
            budget.record_fetch(success=usable)
            if usable:
                raw_passages.extend(self.passage_chunker.chunk(source.id, outcome.text))
                continue
            reason = outcome.error or "empty or too-short extracted content"
            budget.add_warning(
                f"fetch fallback for {source.id} ({source.url}): {reason}"
            )
            snippet = source.snippet.strip()
            if snippet:
                raw_passages.append(
                    PassageRecord(
                        id=f"{source.id}:snippet",
                        source_id=source.id,
                        text=snippet,
                        end_offset=len(snippet),
                        content_origin="search_snippet_fallback",
                    )
                )
                budget.record_fallback_passage()
        passages = deduplicate_passages(raw_passages)
        budget.record_passage_counts(raw=len(raw_passages), deduplicated=len(passages))
        return passages, len(raw_passages)

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
