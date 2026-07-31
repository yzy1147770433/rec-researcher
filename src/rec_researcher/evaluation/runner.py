"""Concurrent, failure-isolated execution of versioned retrieval benchmarks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl

from rec_researcher.core.models import DomainModel, ResearchRun, WorkState
from rec_researcher.evaluation.metrics import (
    average_latency,
    citation_coverage,
    duplicate_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    provider_failure_rate,
    recall_at_k,
    report_section_completeness,
    source_diversity,
    task_success_rate,
    valid_url_rate,
)
from rec_researcher.workflow.orchestrator import ResearchOrchestrator

CaseExecutor = Callable[["BenchmarkCase"], Awaitable[ResearchRun]]


class GoldSource(DomainModel):
    """A source whose graded relevance was assigned by a human annotator."""

    title: str = Field(min_length=1)
    url: HttpUrl
    relevance_grade: Literal[1, 2, 3]
    annotation_note: str = Field(min_length=1)


class BenchmarkCase(DomainModel):
    """One versioned, independently executable benchmark case."""

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    question: str = Field(min_length=1)
    category: str = Field(min_length=1)
    gold_sources: list[GoldSource]
    expected_entities: list[str] | None = None
    annotation_version: str = Field(min_length=1)
    annotated_by: str = Field(min_length=1)
    annotated_at: datetime


class AblationName(StrEnum):
    """Supported retrieval ablations."""

    SNIPPET = "snippet"
    BM25_ONLY = "bm25_only"
    DENSE_ONLY = "dense_only"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANK = "hybrid_rerank"
    HYBRID_RERANK_MMR = "hybrid_rerank_mmr"


class AblationConfig(DomainModel):
    """Explicit stage switches for a named retrieval ablation."""

    retrieval_mode: Literal["snippet", "hybrid"]
    use_bm25: bool
    use_dense: bool
    use_rrf: bool
    use_rerank: bool
    use_mmr: bool


ABLATION_CONFIGS: dict[AblationName, AblationConfig] = {
    AblationName.SNIPPET: AblationConfig(
        retrieval_mode="snippet",
        use_bm25=False,
        use_dense=False,
        use_rrf=False,
        use_rerank=False,
        use_mmr=False,
    ),
    AblationName.BM25_ONLY: AblationConfig(
        retrieval_mode="hybrid",
        use_bm25=True,
        use_dense=False,
        use_rrf=False,
        use_rerank=False,
        use_mmr=False,
    ),
    AblationName.DENSE_ONLY: AblationConfig(
        retrieval_mode="hybrid",
        use_bm25=False,
        use_dense=True,
        use_rrf=False,
        use_rerank=False,
        use_mmr=False,
    ),
    AblationName.HYBRID_RRF: AblationConfig(
        retrieval_mode="hybrid",
        use_bm25=True,
        use_dense=True,
        use_rrf=True,
        use_rerank=False,
        use_mmr=False,
    ),
    AblationName.HYBRID_RERANK: AblationConfig(
        retrieval_mode="hybrid",
        use_bm25=True,
        use_dense=True,
        use_rrf=True,
        use_rerank=True,
        use_mmr=False,
    ),
    AblationName.HYBRID_RERANK_MMR: AblationConfig(
        retrieval_mode="hybrid",
        use_bm25=True,
        use_dense=True,
        use_rrf=True,
        use_rerank=True,
        use_mmr=True,
    ),
}


def ablation_config(name: AblationName | str) -> AblationConfig:
    """Resolve an ablation name to a validated, immutable-by-copy stage mapping."""

    return ABLATION_CONFIGS[AblationName(name)].model_copy(deep=True)


class CaseMetrics(DomainModel):
    """Operational and optional human-labelled relevance metrics for one case."""

    task_success_rate: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    valid_url_rate: float = Field(ge=0.0, le=1.0)
    source_diversity: float = Field(ge=0.0, le=1.0)
    duplicate_rate: float = Field(ge=0.0, le=1.0)
    report_section_completeness: float = Field(ge=0.0, le=1.0)
    latency_seconds: float = Field(ge=0.0)
    provider_calls: int = Field(ge=0)
    provider_failure_rate: float = Field(ge=0.0, le=1.0)
    recall_at_3: float | None = Field(default=None, ge=0.0, le=1.0)
    recall_at_5: float | None = Field(default=None, ge=0.0, le=1.0)
    mrr: float | None = Field(default=None, ge=0.0, le=1.0)
    ndcg_at_5: float | None = Field(default=None, ge=0.0, le=1.0)


class BenchmarkCaseResult(DomainModel):
    """Persisted outcome for one benchmark case."""

    case_id: str
    question: str
    category: str
    success: bool
    run_id: str | None = None
    metrics: CaseMetrics | None = None
    failure_reason: str | None = None
    config_fingerprint: str | None = None


class BenchmarkSummary(DomainModel):
    """Aggregate benchmark statistics and all case outcomes."""

    ablation: AblationName
    total_cases: int = Field(ge=0)
    successful_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    mean_metrics: dict[str, float | None] = Field(default_factory=dict)
    cases: list[BenchmarkCaseResult] = Field(default_factory=list)
    config_fingerprint: str


class BenchmarkRunner:
    """Run JSONL cases concurrently while isolating individual failures."""

    def __init__(
        self,
        *,
        output_dir: Path,
        mode: str = "mock",
        max_concurrency: int = 3,
        case_executor: CaseExecutor | None = None,
        ablation: AblationName | str = AblationName.SNIPPET,
        resume: bool = False,
        execution_config: dict[str, object] | None = None,
    ) -> None:
        """Configure an offline benchmark and optional test executor."""

        if mode not in {"mock", "real"}:
            raise ValueError("mode must be 'mock' or 'real'")
        if mode == "real" and case_executor is None:
            raise ValueError("real benchmark mode requires a case_executor")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self.output_dir = output_dir
        self.mode = mode
        self.max_concurrency = max_concurrency
        self.case_executor = case_executor
        self.ablation = AblationName(ablation)
        self.resume = resume
        self.execution_config = execution_config or {}
        self.config_fingerprint = ""

    async def run(self, benchmark_path: Path) -> BenchmarkSummary:
        """Execute JSONL cases and write all benchmark output artifacts."""

        cases = self.load_cases(benchmark_path)
        self.config_fingerprint = self._fingerprint(benchmark_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def bounded(case: BenchmarkCase) -> BenchmarkCaseResult:
            resumed = self._load_resumable(case)
            if resumed is not None:
                return resumed
            async with semaphore:
                result = await self._run_case(case)
                result.config_fingerprint = self.config_fingerprint
                self._write_json(
                    self.output_dir / "cases" / f"{case.id}.json",
                    result.model_dump(mode="json"),
                )
                return result

        results = await asyncio.gather(*(bounded(case) for case in cases))
        summary = summarize(
            results,
            ablation=self.ablation,
            config_fingerprint=self.config_fingerprint,
        )
        self._write_json(
            self.output_dir / "summary.json", summary.model_dump(mode="json")
        )
        self._write_json(self.output_dir / "per_category.json", per_category(results))
        (self.output_dir / "comparison.md").write_text(
            comparison_markdown([summary]), encoding="utf-8"
        )
        return summary

    def _fingerprint(self, benchmark_path: Path) -> str:
        payload = {
            "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
            "mode": self.mode,
            "ablation": self.ablation.value,
            "execution_config": self.execution_config,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _load_resumable(self, case: BenchmarkCase) -> BenchmarkCaseResult | None:
        if not self.resume:
            return None
        path = self.output_dir / "cases" / f"{case.id}.json"
        if not path.exists():
            return None
        try:
            previous = BenchmarkCaseResult.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            raise ValueError(f"invalid existing case result: {path}") from exc
        if previous.config_fingerprint != self.config_fingerprint:
            raise ValueError(
                f"cannot resume {case.id}: benchmark or execution configuration changed"
            )
        return previous if previous.success else None

    @staticmethod
    def load_cases(path: Path) -> list[BenchmarkCase]:
        """Load and validate non-empty JSONL lines in file order."""

        cases: list[BenchmarkCase] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                cases.append(BenchmarkCase.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(
                    f"invalid benchmark line {line_number}: {exc}"
                ) from exc
        identifiers = [case.id for case in cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("benchmark case ids must be unique")
        return cases

    async def _run_case(self, case: BenchmarkCase) -> BenchmarkCaseResult:
        started = time.perf_counter()
        try:
            if self.case_executor is None:
                config = ablation_config(self.ablation)
                orchestrator = ResearchOrchestrator(
                    output_dir=self.output_dir / "runs" / case.id,
                    mode=self.mode,
                    retrieval_mode=config.retrieval_mode,
                )
                run = await orchestrator.run(case.question)
            else:
                run = await self.case_executor(case)
            output = run.output
            retrieved = [str(source.url) for source in output.sources]
            gold = {
                str(source.url): source.relevance_grade for source in case.gold_sources
            }
            failed_tasks = sum(
                result.state != WorkState.COMPLETED for result in output.task_results
            )
            budget = run.budget
            metrics = CaseMetrics(
                task_success_rate=task_success_rate(output.task_results),
                citation_coverage=citation_coverage(output.validation),
                valid_url_rate=valid_url_rate(output.sources),
                source_diversity=source_diversity(output.sources),
                duplicate_rate=duplicate_rate(output.sources),
                report_section_completeness=report_section_completeness(
                    output.markdown_report
                ),
                latency_seconds=time.perf_counter() - started,
                provider_calls=(
                    budget.llm_calls
                    + budget.search_calls
                    + budget.embedding_calls
                    + budget.reranker_calls
                    + budget.fetch_attempts
                ),
                provider_failure_rate=provider_failure_rate(
                    failed_tasks, len(output.task_results)
                ),
                recall_at_3=recall_at_k(retrieved, gold, k=3),
                recall_at_5=recall_at_k(retrieved, gold, k=5),
                mrr=mean_reciprocal_rank(retrieved, gold),
                ndcg_at_5=ndcg_at_k(retrieved, gold, k=5),
            )
            succeeded = run.status != WorkState.FAILED
            reasons = output.limitations or [
                error for result in output.task_results for error in result.errors
            ]
            return BenchmarkCaseResult(
                case_id=case.id,
                question=case.question,
                category=case.category,
                success=succeeded,
                run_id=run.run_id,
                metrics=metrics,
                failure_reason=None
                if succeeded
                else "; ".join(reasons) or "research run failed",
                config_fingerprint=self.config_fingerprint,
            )
        except Exception as exc:  # noqa: BLE001 - isolation is the benchmark contract
            return BenchmarkCaseResult(
                case_id=case.id,
                question=case.question,
                category=case.category,
                success=False,
                failure_reason=f"{type(exc).__name__}: {exc}",
                config_fingerprint=self.config_fingerprint,
            )

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _means(results: Sequence[BenchmarkCaseResult]) -> dict[str, float | None]:
    metrics = [
        item.metrics for item in results if item.success and item.metrics is not None
    ]
    names = (
        "task_success_rate",
        "citation_coverage",
        "valid_url_rate",
        "source_diversity",
        "duplicate_rate",
        "report_section_completeness",
        "latency_seconds",
        "provider_calls",
        "provider_failure_rate",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "ndcg_at_5",
    )
    means: dict[str, float | None] = {}
    for name in names:
        values = [
            value for item in metrics if (value := getattr(item, name)) is not None
        ]
        means[name] = average_latency(values) if values else None
    return means


def summarize(
    results: Sequence[BenchmarkCaseResult],
    *,
    ablation: AblationName = AblationName.SNIPPET,
    config_fingerprint: str = "test",
) -> BenchmarkSummary:
    """Build means over successful cases, ignoring null values independently."""

    successful = sum(result.success for result in results)
    return BenchmarkSummary(
        ablation=ablation,
        total_cases=len(results),
        successful_cases=successful,
        failed_cases=len(results) - successful,
        mean_metrics=_means(results),
        cases=list(results),
        config_fingerprint=config_fingerprint,
    )


def per_category(results: Sequence[BenchmarkCaseResult]) -> dict[str, object]:
    """Aggregate outcomes and null-aware means by benchmark category."""

    categories: dict[str, list[BenchmarkCaseResult]] = {}
    for result in results:
        categories.setdefault(result.category, []).append(result)
    return {
        category: {
            "total_cases": len(items),
            "successful_cases": sum(item.success for item in items),
            "failed_cases": sum(not item.success for item in items),
            "mean_metrics": _means(items),
        }
        for category, items in sorted(categories.items())
    }


def comparison_markdown(summaries: Sequence[BenchmarkSummary]) -> str:
    """Render ablation summaries as a compact Markdown comparison table."""

    def cell(value: float | None) -> str:
        return "null" if value is None else f"{value:.4f}"

    rows = [
        "| Ablation | Recall@5 | MRR | nDCG@5 | Latency (s) | API calls |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        metrics = summary.mean_metrics
        rows.append(
            f"| {summary.ablation.value} | {cell(metrics.get('recall_at_5'))} | "
            f"{cell(metrics.get('mrr'))} | {cell(metrics.get('ndcg_at_5'))} | "
            f"{cell(metrics.get('latency_seconds'))} | "
            f"{cell(metrics.get('provider_calls'))} |"
        )
    return "\n".join(rows) + "\n"
