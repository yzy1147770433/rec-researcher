import asyncio

import pytest

from rec_researcher.core.models import WorkState
from rec_researcher.workflow.scheduler import AsyncTaskScheduler, ScheduledTask

pytestmark = pytest.mark.asyncio


async def test_semaphore_bounds_concurrency_and_order_is_stable() -> None:
    active = 0
    peak = 0

    async def work(value: int) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep((5 - value) / 1000)
        active -= 1
        return value

    results = await AsyncTaskScheduler(max_concurrency=2, task_timeout=1).run(
        [
            ScheduledTask(str(value), lambda value=value: work(value))
            for value in range(5)
        ]
    )

    assert peak == 2
    assert [result.value for result in results] == list(range(5))


async def test_timeout_is_marked_without_cancelling_sibling() -> None:
    async def slow() -> None:
        await asyncio.sleep(1)

    async def fast() -> str:
        return "ok"

    results = await AsyncTaskScheduler(max_concurrency=2, task_timeout=0.01).run(
        [ScheduledTask("slow", slow), ScheduledTask("fast", fast)]
    )

    assert results[0].state == WorkState.FAILED
    assert results[0].timed_out is True
    assert results[1].state == WorkState.COMPLETED


async def test_failure_skips_dependent_but_not_independent_task() -> None:
    async def fail() -> None:
        raise RuntimeError("boom")

    async def succeed() -> str:
        return "ok"

    results = await AsyncTaskScheduler(max_concurrency=3, task_timeout=1).run(
        [
            ScheduledTask("failed", fail),
            ScheduledTask("dependent", succeed, ["failed"]),
            ScheduledTask("independent", succeed),
        ]
    )

    assert [result.state for result in results] == [
        WorkState.FAILED,
        WorkState.SKIPPED,
        WorkState.COMPLETED,
    ]
