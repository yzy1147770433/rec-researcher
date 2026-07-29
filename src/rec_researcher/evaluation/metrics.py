"""Deterministic metrics for lightweight research benchmarks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from urllib.parse import urlparse

from rec_researcher.core.models import (
    CitationValidation,
    SourceRecord,
    TaskResult,
    WorkState,
)

REQUIRED_REPORT_SECTIONS = (
    "论文与代码对照",
    "数据集与指标",
    "复现难度分析",
    "三天复现建议",
)


def task_success_rate(results: Sequence[TaskResult]) -> float:
    """Return the fraction of planned tasks that completed successfully."""

    if not results:
        return 0.0
    completed = sum(item.state == WorkState.COMPLETED for item in results)
    return completed / len(results)


def citation_coverage(validation: CitationValidation) -> float:
    """Return citation coverage produced by deterministic report validation."""

    return validation.citation_coverage


def valid_url_rate(sources: Sequence[SourceRecord]) -> float:
    """Return the fraction of source URLs with an HTTP(S) host."""

    if not sources:
        return 0.0
    valid = 0
    for source in sources:
        parsed = urlparse(str(source.url))
        valid += parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    return valid / len(sources)


def source_diversity(sources: Sequence[SourceRecord]) -> float:
    """Return unique source domains divided by the number of sources."""

    if not sources:
        return 0.0
    domains = {urlparse(str(source.url)).netloc.casefold() for source in sources}
    domains.discard("")
    return len(domains) / len(sources)


def report_section_completeness(
    report: str,
    required_sections: Sequence[str] = REQUIRED_REPORT_SECTIONS,
) -> float:
    """Return the fraction of required level-two Markdown sections present."""

    if not required_sections:
        return 1.0
    headings = {
        match.group(1).strip() for match in re.finditer(r"(?m)^##\s+(.+?)\s*$", report)
    }
    return sum(section in headings for section in required_sections) / len(
        required_sections
    )


def average_latency(latencies_seconds: Sequence[float]) -> float:
    """Return arithmetic mean latency, or zero for no completed attempts."""

    if not latencies_seconds:
        return 0.0
    return sum(latencies_seconds) / len(latencies_seconds)


def provider_failure_rate(failed_attempts: int, total_attempts: int) -> float:
    """Return failed provider attempts divided by all provider attempts."""

    if failed_attempts < 0 or total_attempts < 0:
        raise ValueError("attempt counts must be non-negative")
    if failed_attempts > total_attempts:
        raise ValueError("failed_attempts cannot exceed total_attempts")
    return failed_attempts / total_attempts if total_attempts else 0.0


def recall_at_k(
    retrieved_source_ids: Sequence[str],
    gold_source_ids: Sequence[str] | None,
    *,
    k: int,
) -> float | None:
    """Compute Recall@K only when human relevance identifiers are supplied."""

    if k < 1:
        raise ValueError("k must be at least 1")
    if not gold_source_ids:
        return None
    gold = set(gold_source_ids)
    return len(set(retrieved_source_ids[:k]) & gold) / len(gold)


def mean_reciprocal_rank(
    retrieved_source_ids: Sequence[str],
    gold_source_ids: Sequence[str] | None,
) -> float | None:
    """Compute reciprocal rank only when human relevance identifiers are supplied."""

    if not gold_source_ids:
        return None
    gold = set(gold_source_ids)
    for rank, source_id in enumerate(retrieved_source_ids, start=1):
        if source_id in gold:
            return 1.0 / rank
    return 0.0
