"""Dependency-aware, bounded asynchronous task scheduling."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from rec_researcher.core.models import WorkState

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ScheduledTask(Generic[T]):
    """A named asynchronous operation and its prerequisite task IDs."""

    id: str
    operation: Callable[[], Awaitable[T]]
    dependencies: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ScheduledTaskResult(Generic[T]):
    """The stable, serializable-friendly outcome of a scheduled operation."""

    task_id: str
    state: WorkState
    value: T | None = None
    error: str | None = None
    timed_out: bool = False


class AsyncTaskScheduler:
    """Run ready tasks concurrently while isolating timeouts and failures."""

    def __init__(self, *, max_concurrency: int, task_timeout: float) -> None:
        """Configure positive concurrency and per-task timeout bounds."""

        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if task_timeout <= 0:
            raise ValueError("task_timeout must be positive")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._task_timeout = task_timeout

    async def run(
        self, tasks: Sequence[ScheduledTask[T]]
    ) -> list[ScheduledTaskResult[T]]:
        """Run tasks when dependencies are ready and preserve input order."""

        ordered = list(tasks)
        by_id = {task.id: task for task in ordered}
        if len(by_id) != len(ordered):
            raise ValueError("scheduled task IDs must be unique")
        unknown = {
            dependency
            for task in ordered
            for dependency in task.dependencies
            if dependency not in by_id
        }
        if unknown:
            raise ValueError(f"unknown task dependencies: {', '.join(sorted(unknown))}")
        self._validate_acyclic(ordered)

        events = {task.id: asyncio.Event() for task in ordered}
        outcomes: dict[str, ScheduledTaskResult[T]] = {}

        async def execute(task: ScheduledTask[T]) -> ScheduledTaskResult[T]:
            for dependency in task.dependencies:
                await events[dependency].wait()
            failed_dependencies = [
                dependency
                for dependency in task.dependencies
                if outcomes[dependency].state != WorkState.COMPLETED
            ]
            if failed_dependencies:
                result = ScheduledTaskResult[T](
                    task_id=task.id,
                    state=WorkState.SKIPPED,
                    error="dependency failed: " + ", ".join(failed_dependencies),
                )
            else:
                try:
                    async with self._semaphore:
                        async with asyncio.timeout(self._task_timeout):
                            value = await task.operation()
                    result = ScheduledTaskResult(
                        task_id=task.id, state=WorkState.COMPLETED, value=value
                    )
                except TimeoutError:
                    result = ScheduledTaskResult(
                        task_id=task.id,
                        state=WorkState.FAILED,
                        error=f"task timed out after {self._task_timeout:g} seconds",
                        timed_out=True,
                    )
                except Exception as exc:  # noqa: BLE001 - task isolation boundary
                    result = ScheduledTaskResult(
                        task_id=task.id,
                        state=WorkState.FAILED,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            outcomes[task.id] = result
            events[task.id].set()
            return result

        gathered = await asyncio.gather(
            *(execute(task) for task in ordered), return_exceptions=True
        )
        results: list[ScheduledTaskResult[T]] = []
        for task, item in zip(ordered, gathered, strict=True):
            if isinstance(item, BaseException):
                item = ScheduledTaskResult(
                    task_id=task.id,
                    state=WorkState.FAILED,
                    error=f"{type(item).__name__}: {item}",
                )
            results.append(item)
        return results

    @staticmethod
    def _validate_acyclic(tasks: Sequence[ScheduledTask[T]]) -> None:
        dependencies = {task.id: tuple(task.dependencies) for task in tasks}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("scheduled task dependencies contain a cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in dependencies[task_id]:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task in tasks:
            visit(task.id)
