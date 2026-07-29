"""Workflow budget accounting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rec_researcher.core.exceptions import BudgetExceededError


@dataclass(slots=True)
class RunBudget:
    """Track bounded work and elapsed monotonic time for one run."""

    max_tasks: int
    max_sources: int
    max_api_calls: int | None = None
    api_calls: int = 0
    source_count: int = 0
    task_count: int = 0
    _started_at: float = field(default_factory=time.monotonic, repr=False)

    @property
    def elapsed_seconds(self) -> float:
        """Return elapsed monotonic seconds."""

        return time.monotonic() - self._started_at

    def consume_task(self, count: int = 1) -> None:
        """Record tasks or raise before exceeding the task limit."""

        self._validate_count(count)
        if self.task_count + count > self.max_tasks:
            raise BudgetExceededError("maximum task budget exceeded")
        self.task_count += count

    def consume_sources(self, count: int) -> None:
        """Record retained sources or raise before exceeding the source limit."""

        self._validate_count(count)
        if self.source_count + count > self.max_sources:
            raise BudgetExceededError("maximum source budget exceeded")
        self.source_count += count

    def record_api_call(self, count: int = 1) -> None:
        """Record provider calls and enforce an optional call limit."""

        self._validate_count(count)
        over_limit = (
            self.max_api_calls is not None
            and self.api_calls + count > self.max_api_calls
        )
        if over_limit:
            raise BudgetExceededError("maximum API call budget exceeded")
        self.api_calls += count

    @staticmethod
    def _validate_count(count: int) -> None:
        if count < 0:
            raise ValueError("budget count must not be negative")
