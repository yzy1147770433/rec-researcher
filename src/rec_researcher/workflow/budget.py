"""Workflow budget accounting."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rec_researcher.core.exceptions import BudgetExceededError


@dataclass(slots=True)
class RunBudget:
    """Track bounded work and elapsed monotonic time for one run."""

    max_tasks: int
    max_sources: int
    max_api_calls: int | None = None
    api_calls: int = 0
    llm_calls: int = 0
    search_calls: int = 0
    embedding_calls: int = 0
    reranker_calls: int = 0
    fetched_pages: int = 0
    source_count: int = 0
    passage_count: int = 0
    fetch_attempts: int = 0
    fetch_successes: int = 0
    fetch_failures: int = 0
    fallback_passages: int = 0
    raw_passage_count: int = 0
    deduplicated_passage_count: int = 0
    embedding_text_count: int = 0
    degradation_events: int = 0
    warnings: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    task_count: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
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

    def record_call(self, kind: str, count: int = 1) -> None:
        """Record one typed external call and the aggregate API count."""

        if kind not in {"llm", "search", "embedding", "reranker"}:
            raise ValueError(f"unknown call kind: {kind}")
        self.record_api_call(count)
        setattr(self, f"{kind}_calls", getattr(self, f"{kind}_calls") + count)

    def record_failed_task(self, task_id: str) -> None:
        """Record a failed task once."""

        if task_id not in self.failed_tasks:
            self.failed_tasks.append(task_id)

    def record_fetched_page(self, count: int = 1) -> None:
        """Record successfully fetched pages."""

        self._validate_count(count)
        self.fetched_pages += count

    def record_fetch(self, *, success: bool) -> None:
        """Record one attempted page fetch and its terminal outcome."""

        self.fetch_attempts += 1
        if success:
            self.fetch_successes += 1
            self.record_fetched_page()
        else:
            self.fetch_failures += 1

    def record_fallback_passage(self, count: int = 1) -> None:
        """Record passages created from search snippets after fetch degradation."""

        self._validate_count(count)
        self.fallback_passages += count

    def record_passage_counts(self, *, raw: int, deduplicated: int) -> None:
        """Record corpus sizes before and after text deduplication."""

        self._validate_count(raw)
        self._validate_count(deduplicated)
        self.raw_passage_count += raw
        self.deduplicated_passage_count += deduplicated

    def record_passage(self, count: int = 1) -> None:
        """Record passages made available to retrieval."""

        self._validate_count(count)
        self.passage_count += count

    def add_warning(self, warning: str) -> None:
        """Append a non-empty degradation warning."""

        if warning and warning not in self.warnings:
            self.warnings.append(warning)

    def record_degradation(self, warning: str) -> None:
        """Record one explicitly surfaced retrieval degradation."""

        self.degradation_events += 1
        self.add_warning(warning)

    @staticmethod
    def _validate_count(count: int) -> None:
        if count < 0:
            raise ValueError("budget count must not be negative")
