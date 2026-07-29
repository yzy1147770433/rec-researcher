"""Concurrent, failure-isolated benchmark execution."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from pydantic import Field

from rec_researcher.core.models import DomainModel, ResearchRun, WorkState
from rec_researcher.evaluation.metrics import (
    average_latency,
    citation_coverage,
    mean_reciprocal_rank,
    provider_failure_rate,
    recall_at_k,
    report_section_completeness,
    source_diversity,
    task_success_rate,
    valid_url_rate,
)
from rec_researcher.workflow.orchestrator import ResearchOrchestrator

CaseExecutor = Callable[["BenchmarkCase"], Awaitable[ResearchRun]]


class BenchmarkCase(DomainModel):
    """One independently executable research benchmark case."""

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    question: str = Field(min_length=1)
    gold_source_ids: list[str] | None = None


class CaseMetrics(DomainModel):
    """Metrics and optional relevance scores for one case."""

    task_success_rate: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    valid_url_rate: float = Field(ge=0.0, le=1.0)
    source_diversity: float = Field(ge=0.0, le=1.0)
    report_section_completeness: float = Field(ge=0.0, le=1.0)
    latency_seconds: float = Field(ge=0.0)
    provider_failure_rate: float = Field(ge=0.0, le=1.0)
    recall_at_k: float | None = Field(default=None, ge=0.0, le=1.0)
    mrr: float | None = Field(default=None, ge=0.0, le=1.0)


class BenchmarkCaseResult(DomainModel):
    """Persisted outcome for one benchmark case."""

    case_id: str
    question: str
    success: bool
    run_id: str | None = None
    metrics: CaseMetrics | None = None
    failure_reason: str | None = None


class BenchmarkSummary(DomainModel):
    """Aggregate benchmark statistics and all case outcomes."""

    total_cases: int = Field(ge=0)
    successful_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    mean_metrics: dict[str, float | None] = Field(default_factory=dict)
    cases: list[BenchmarkCaseResult] = Field(default_factory=list)


class BenchmarkRunner:
    """Run JSONL cases concurrently while isolating individual failures."""

    def __init__(
        self,
        *,
        output_dir: Path,
        mode: str = "mock",
        max_concurrency: int = 3,
        case_executor: CaseExecutor | None = None,
        recall_k: int = 10,
    ) -> None:
        """Configure an offline benchmark and optional test executor."""

        if mode != "mock":
            raise ValueError("only mock benchmark mode is currently supported")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if recall_k < 1:
            raise ValueError("recall_k must be at least 1")
        self.output_dir = output_dir
        self.mode = mode
        self.max_concurrency = max_concurrency
        self.case_executor = case_executor
        self.recall_k = recall_k

    async def run(self, benchmark_path: Path) -> BenchmarkSummary:
        """Execute every JSONL case and write case files plus summary.json."""

        cases = self.load_cases(benchmark_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def bounded(case: BenchmarkCase) -> BenchmarkCaseResult:
            async with semaphore:
                result = await self._run_case(case)
                self._write_json(
                    self.output_dir / "cases" / f"{case.id}.json",
                    result.model_dump(mode="json"),
                )
                return result

        results = await asyncio.gather(*(bounded(case) for case in cases))
        summary = summarize(results)
        self._write_json(
            self.output_dir / "summary.json", summary.model_dump(mode="json")
        )
        return summary

    @staticmethod
    def load_cases(path: Path) -> list[BenchmarkCase]:
        """Load non-empty JSONL lines in file order."""

        cases: list[BenchmarkCase] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                cases.append(BenchmarkCase.model_validate_json(line))
            except ValueError as exc:
                message = f"invalid benchmark line {line_number}: {exc}"
                raise ValueError(message) from exc
        identifiers = [case.id for case in cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("benchmark case ids must be unique")
        return cases

    async def _run_case(self, case: BenchmarkCase) -> BenchmarkCaseResult:
        started = time.perf_counter()
        try:
            if self.case_executor is None:
                orchestrator = ResearchOrchestrator(
                    output_dir=self.output_dir / "runs" / case.id,
                    mode=self.mode,
                )
                run = await orchestrator.run(case.question)
            else:
                run = await self.case_executor(case)
            latency = time.perf_counter() - started
            output = run.output
            retrieved = [source.id for source in output.sources]
            failed_tasks = sum(
                result.state != WorkState.COMPLETED for result in output.task_results
            )
            metrics = CaseMetrics(
                task_success_rate=task_success_rate(output.task_results),
                citation_coverage=citation_coverage(output.validation),
                valid_url_rate=valid_url_rate(output.sources),
                source_diversity=source_diversity(output.sources),
                report_section_completeness=report_section_completeness(
                    output.markdown_report
                ),
                latency_seconds=latency,
                provider_failure_rate=provider_failure_rate(
                    failed_tasks, len(output.task_results)
                ),
                recall_at_k=recall_at_k(
                    retrieved, case.gold_source_ids, k=self.recall_k
                ),
                mrr=mean_reciprocal_rank(retrieved, case.gold_source_ids),
            )
            run_succeeded = run.status != WorkState.FAILED
            failure_reason = None
            if not run_succeeded:
                reasons = output.limitations or [
                    error for result in output.task_results for error in result.errors
                ]
                failure_reason = "; ".join(reasons) or "research run failed"
            return BenchmarkCaseResult(
                case_id=case.id,
                question=case.question,
                success=run_succeeded,
                run_id=run.run_id,
                metrics=metrics,
                failure_reason=failure_reason,
            )
        except Exception as exc:  # noqa: BLE001 - each case must remain isolated
            return BenchmarkCaseResult(
                case_id=case.id,
                question=case.question,
                success=False,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def summarize(results: Sequence[BenchmarkCaseResult]) -> BenchmarkSummary:
    """Build arithmetic means over successful case metrics."""

    metrics = [
        result.metrics
        for result in results
        if result.success and result.metrics is not None
    ]
    names = (
        "task_success_rate",
        "citation_coverage",
        "valid_url_rate",
        "source_diversity",
        "report_section_completeness",
        "provider_failure_rate",
    )
    means: dict[str, float | None] = {
        name: average_latency([getattr(item, name) for item in metrics])
        for name in names
    }
    means["average_latency"] = average_latency(
        [item.latency_seconds for item in metrics]
    )
    for name in ("recall_at_k", "mrr"):
        annotated = [
            value for item in metrics if (value := getattr(item, name)) is not None
        ]
        means[name] = average_latency(annotated) if annotated else None
    successful = sum(result.success for result in results)
    return BenchmarkSummary(
        total_cases=len(results),
        successful_cases=successful,
        failed_cases=len(results) - successful,
        mean_metrics=means,
        cases=list(results),
    )
