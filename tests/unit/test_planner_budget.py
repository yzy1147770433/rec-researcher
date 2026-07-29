import pytest

from rec_researcher.core.exceptions import BudgetExceededError
from rec_researcher.planning.planner import ResearchPlanner
from rec_researcher.workflow.budget import RunBudget


@pytest.mark.asyncio
async def test_planner_creates_deterministic_bounded_tasks() -> None:
    planner = ResearchPlanner()

    first = await planner.create_tasks("如何评估推荐系统？")
    second = await planner.create_tasks("如何评估推荐系统？")

    assert first == second
    assert 3 <= len(first) <= 5
    assert [task.question.rsplit("研究", 1)[-1] for task in first] == [
        "技术原理",
        "代表工作",
        "实验与指标",
        "开源代码与数据集",
        "复现建议",
    ]


@pytest.mark.asyncio
async def test_planner_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        await ResearchPlanner().create_tasks("  ")


def test_budget_tracks_and_enforces_counts() -> None:
    budget = RunBudget(max_tasks=1, max_sources=2, max_api_calls=1)
    budget.consume_task()
    budget.consume_sources(2)
    budget.record_api_call()

    assert budget.task_count == 1
    assert budget.source_count == 2
    assert budget.api_calls == 1
    assert budget.elapsed_seconds >= 0
    with pytest.raises(BudgetExceededError):
        budget.consume_task()
    with pytest.raises(BudgetExceededError):
        budget.consume_sources(1)
