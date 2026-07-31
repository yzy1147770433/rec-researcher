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
    AblationName,
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkRunner,
    CaseMetrics,
    ablation_config,
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
            start_time=datetime.now(UTC), elapsed_seconds=0.1, search_calls=2
        ),
    )


@pytest.mark.asyncio
async def test_outputs_are_written_and_empty_gold_stays_null(tmp_path: Path) -> None:
    async def execute(case: BenchmarkCase) -> ResearchRun:
        return make_run(case)

    summary = await BenchmarkRunner(output_dir=tmp_path, case_executor=execute).run(
        Path("examples/bench/smoke5.jsonl")
    )

    assert summary.total_cases == 5
    assert all(
        result.metrics and result.metrics.recall_at_5 is None
        for result in summary.cases
    )
    assert len(list((tmp_path / "cases").glob("*.json"))) == 5
    for filename in ("summary.json", "comparison.md", "per_category.json"):
        assert (tmp_path / filename).is_file()
    assert "| Ablation | Recall@5 | MRR | nDCG@5 | Latency (s) | API calls |" in (
        tmp_path / "comparison.md"
    ).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_one_case_failure_does_not_stop_other_cases(tmp_path: Path) -> None:
    async def execute(case: BenchmarkCase) -> ResearchRun:
        if case.id == "semantic-id":
            raise RuntimeError("provider unavailable")
        return make_run(case)

    summary = await BenchmarkRunner(output_dir=tmp_path, case_executor=execute).run(
        Path("examples/bench/smoke5.jsonl")
    )

    assert (summary.successful_cases, summary.failed_cases) == (4, 1)
    persisted = json.loads((tmp_path / "cases" / "semantic-id.json").read_text())
    assert persisted["failure_reason"] == "RuntimeError: provider unavailable"


@pytest.mark.asyncio
async def test_resume_skips_successes_and_retries_only_failures(tmp_path: Path) -> None:
    first_calls: list[str] = []

    async def first_execute(case: BenchmarkCase) -> ResearchRun:
        first_calls.append(case.id)
        if case.id == "semantic-id":
            raise RuntimeError("temporary outage")
        return make_run(case)

    benchmark = Path("examples/bench/smoke5.jsonl")
    first = await BenchmarkRunner(
        output_dir=tmp_path,
        case_executor=first_execute,
        execution_config={"request_timeout": 30},
    ).run(benchmark)
    assert first.failed_cases == 1

    resumed_calls: list[str] = []

    async def resumed_execute(case: BenchmarkCase) -> ResearchRun:
        resumed_calls.append(case.id)
        return make_run(case)

    resumed = await BenchmarkRunner(
        output_dir=tmp_path,
        case_executor=resumed_execute,
        execution_config={"request_timeout": 30},
        resume=True,
    ).run(benchmark)

    assert len(first_calls) == 5
    assert resumed_calls == ["semantic-id"]
    assert resumed.successful_cases == 5


@pytest.mark.asyncio
async def test_resume_rejects_changed_execution_configuration(tmp_path: Path) -> None:
    benchmark = Path("examples/bench/smoke5.jsonl")

    async def execute(case: BenchmarkCase) -> ResearchRun:
        return make_run(case)

    await BenchmarkRunner(
        output_dir=tmp_path,
        case_executor=execute,
        execution_config={"request_timeout": 30},
    ).run(benchmark)

    with pytest.raises(ValueError, match="configuration changed"):
        await BenchmarkRunner(
            output_dir=tmp_path,
            case_executor=execute,
            execution_config={"request_timeout": 120},
            resume=True,
        ).run(benchmark)


def test_ablation_configuration_mapping_is_exact() -> None:
    assert ablation_config("snippet").retrieval_mode == "snippet"
    assert ablation_config("bm25_only").model_dump() == {
        "retrieval_mode": "hybrid",
        "use_bm25": True,
        "use_dense": False,
        "use_rrf": False,
        "use_rerank": False,
        "use_mmr": False,
    }
    assert ablation_config("hybrid_rerank_mmr").model_dump() == {
        "retrieval_mode": "hybrid",
        "use_bm25": True,
        "use_dense": True,
        "use_rrf": True,
        "use_rerank": True,
        "use_mmr": True,
    }
    assert set(AblationName) == {
        AblationName(value)
        for value in (
            "snippet",
            "bm25_only",
            "dense_only",
            "hybrid_rrf",
            "hybrid_rerank",
            "hybrid_rerank_mmr",
        )
    }


def test_summary_mean_ignores_null_values_and_failed_cases() -> None:
    def result(case_id: str, recall: float | None) -> BenchmarkCaseResult:
        return BenchmarkCaseResult(
            case_id=case_id,
            question=case_id,
            category="category",
            success=True,
            metrics=CaseMetrics(
                task_success_rate=1,
                citation_coverage=1,
                valid_url_rate=1,
                source_diversity=1,
                duplicate_rate=0,
                report_section_completeness=1,
                latency_seconds=2,
                provider_calls=4,
                provider_failure_rate=0,
                recall_at_3=recall,
                recall_at_5=recall,
                mrr=recall,
                ndcg_at_5=recall,
            ),
        )

    summary = summarize(
        [
            result("labelled", 0.5),
            result("unlabelled", None),
            BenchmarkCaseResult(
                case_id="failed", question="failed", category="category", success=False
            ),
        ]
    )

    assert summary.mean_metrics["recall_at_5"] == 0.5
    assert summary.mean_metrics["latency_seconds"] == 2.0
    assert summary.mean_metrics["provider_calls"] == 4.0
