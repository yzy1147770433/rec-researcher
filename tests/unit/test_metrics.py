from rec_researcher.core.models import (
    CitationValidation,
    SourceRecord,
    TaskResult,
    WorkState,
)
from rec_researcher.evaluation.metrics import (
    average_latency,
    mean_reciprocal_rank,
    recall_at_k,
    report_section_completeness,
    source_diversity,
    task_success_rate,
    valid_url_rate,
)


def test_operational_metrics_are_deterministic() -> None:
    tasks = [
        TaskResult(task_id="one", state=WorkState.COMPLETED),
        TaskResult(task_id="two", state=WorkState.FAILED),
    ]
    sources = [
        SourceRecord(
            id="one",
            title="One",
            url="https://one.example/paper",
            snippet="",
            provider="fixture",
        ),
        SourceRecord(
            id="two",
            title="Two",
            url="https://two.example/paper",
            snippet="",
            provider="fixture",
        ),
    ]
    report = "\n".join(
        [
            "## 论文与代码对照",
            "## 数据集与指标",
            "## 复现难度分析",
        ]
    )

    assert task_success_rate(tasks) == 0.5
    assert valid_url_rate(sources) == 1.0
    assert source_diversity(sources) == 1.0
    assert report_section_completeness(report) == 0.75
    assert average_latency([1.0, 3.0]) == 2.0
    assert CitationValidation(citation_coverage=0.5).citation_coverage == 0.5


def test_relevance_metrics_are_null_without_gold_annotations() -> None:
    retrieved = ["some-returned-url"]

    assert recall_at_k(retrieved, None, k=10) is None
    assert recall_at_k(retrieved, [], k=10) is None
    assert mean_reciprocal_rank(retrieved, None) is None


def test_relevance_metrics_use_gold_source_ids() -> None:
    retrieved = ["irrelevant", "gold-two", "gold-one"]

    assert recall_at_k(retrieved, ["gold-one", "gold-two"], k=2) == 0.5
    assert mean_reciprocal_rank(retrieved, ["gold-one", "gold-two"]) == 0.5
