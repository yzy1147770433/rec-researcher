import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rec_researcher.core.models import (
    CitationValidation,
    ResearchOutput,
    ResearchRun,
    RunBudgetRecord,
    SourceRecord,
)
from rec_researcher.evaluation.runner import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkRunner,
    CaseMetrics,
    summarize,
)


def make_run(case: BenchmarkCase) -> ResearchRun:
    source = SourceRecord(
        id="S1",
        title="Fixture",
        url="https://example.com/source",
        snippet="Fixture evidence",
        provider="mock",
    )
    return ResearchRun(
        run_id=f"run-{case.id}",
        mode="mock",
        output=ResearchOutput(
            question=case.question,
            sources=[source],
            markdown_report="\n".join(
                [
                    "## 论文与代码对照",
                    "## 数据集与指标",
                    "## 复现难度分析",
                    "## 三天复现建议",
                ]
            ),
            validation=CitationValidation(citation_coverage=0.5),
        ),
        budget=RunBudgetRecord(
            start_time=datetime.now(UTC),
            elapsed_seconds=0.1,
        ),
    )


@pytest.mark.asyncio
async def test_all_five_cases_execute(tmp_path: Path) -> None:
    seen: list[str] = []

    async def execute(case: BenchmarkCase) -> ResearchRun:
        seen.append(case.id)
        return make_run(case)

    benchmark = Path("examples/bench/smoke5.jsonl")
    summary = await BenchmarkRunner(
        output_dir=tmp_path,
        case_executor=execute,
        max_concurrency=2,
    ).run(benchmark)

    assert len(seen) == 5
    assert summary.total_cases == 5
    assert summary.successful_cases == 5
    assert len(list((tmp_path / "cases").glob("*.json"))) == 5
    assert (tmp_path / "summary.json").is_file()
    assert all(result.metrics.recall_at_k is None for result in summary.cases)


@pytest.mark.asyncio
async def test_one_case_failure_does_not_stop_other_cases(tmp_path: Path) -> None:
    async def execute(case: BenchmarkCase) -> ResearchRun:
        if case.id == "semantic-id":
            raise RuntimeError("provider unavailable")
        return make_run(case)

    summary = await BenchmarkRunner(
        output_dir=tmp_path,
        case_executor=execute,
    ).run(Path("examples/bench/smoke5.jsonl"))

    assert summary.successful_cases == 4
    assert summary.failed_cases == 1
    failed = next(result for result in summary.cases if not result.success)
    assert failed.failure_reason == "RuntimeError: provider unavailable"
    persisted = json.loads(
        (tmp_path / "cases" / "semantic-id.json").read_text(encoding="utf-8")
    )
    assert persisted["failure_reason"] == failed.failure_reason


def test_summary_means_are_arithmetic_and_ignore_failed_cases() -> None:
    def result(case_id: str, value: float) -> BenchmarkCaseResult:
        return BenchmarkCaseResult(
            case_id=case_id,
            question=case_id,
            success=True,
            metrics=CaseMetrics(
                task_success_rate=value,
                citation_coverage=value,
                valid_url_rate=value,
                source_diversity=value,
                report_section_completeness=value,
                latency_seconds=value * 10,
                provider_failure_rate=1 - value,
                recall_at_k=None,
                mrr=None,
            ),
        )

    summary = summarize(
        [
            result("one", 0.25),
            result("two", 0.75),
            BenchmarkCaseResult(
                case_id="failed",
                question="failed",
                success=False,
                failure_reason="failure",
            ),
        ]
    )

    assert summary.mean_metrics["task_success_rate"] == 0.5
    assert summary.mean_metrics["average_latency"] == 5.0
    assert summary.mean_metrics["provider_failure_rate"] == 0.5
    assert summary.mean_metrics["recall_at_k"] is None
