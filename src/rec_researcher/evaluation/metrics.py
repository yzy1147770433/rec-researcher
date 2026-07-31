"""Deterministic metrics for versioned retrieval benchmarks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

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


def _canonical_url(value: object) -> str:
    """Normalize a URL for exact, deterministic benchmark matching."""

    parsed = urlsplit(str(value))
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (parsed.scheme.casefold(), parsed.netloc.casefold(), path, parsed.query, "")
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
    valid = sum(
        urlsplit(str(source.url)).scheme in {"http", "https"}
        and bool(urlsplit(str(source.url)).netloc)
        for source in sources
    )
    return valid / len(sources)


def source_diversity(sources: Sequence[SourceRecord]) -> float:
    """Return unique source domains divided by the number of returned sources."""

    if not sources:
        return 0.0
    domains = {urlsplit(str(source.url)).netloc.casefold() for source in sources}
    domains.discard("")
    return len(domains) / len(sources)


def duplicate_rate(sources: Sequence[SourceRecord]) -> float:
    """Return the fraction of results duplicating an earlier canonical URL."""

    if not sources:
        return 0.0
    unique = {_canonical_url(source.url) for source in sources}
    return (len(sources) - len(unique)) / len(sources)


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


def average_latency(values: Sequence[float]) -> float:
    """Return an arithmetic mean, or zero for an empty sequence."""

    return sum(values) / len(values) if values else 0.0


def provider_failure_rate(failed_attempts: int, total_attempts: int) -> float:
    """Return failed provider attempts divided by all provider attempts."""

    if failed_attempts < 0 or total_attempts < 0:
        raise ValueError("attempt counts must be non-negative")
    if failed_attempts > total_attempts:
        raise ValueError("failed_attempts cannot exceed total_attempts")
    return failed_attempts / total_attempts if total_attempts else 0.0


def recall_at_k(
    retrieved_urls: Sequence[str],
    gold_relevance: Mapping[str, int] | None,
    *,
    k: int,
) -> float | None:
    """Compute Recall@K, returning null when no human gold labels exist."""

    if k < 1:
        raise ValueError("k must be at least 1")
    if not gold_relevance:
        return None
    gold = {_canonical_url(url) for url in gold_relevance}
    retrieved = {_canonical_url(url) for url in retrieved_urls[:k]}
    return len(retrieved & gold) / len(gold)


def mean_reciprocal_rank(
    retrieved_urls: Sequence[str], gold_relevance: Mapping[str, int] | None
) -> float | None:
    """Compute reciprocal rank, returning null when labels are absent."""

    if not gold_relevance:
        return None
    gold = {_canonical_url(url) for url in gold_relevance}
    for rank, url in enumerate(retrieved_urls, start=1):
        if _canonical_url(url) in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_urls: Sequence[str],
    gold_relevance: Mapping[str, int] | None,
    *,
    k: int,
) -> float | None:
    """Compute graded nDCG@K with gains ``2**grade - 1``."""

    if k < 1:
        raise ValueError("k must be at least 1")
    if not gold_relevance:
        return None
    grades = {_canonical_url(url): grade for url, grade in gold_relevance.items()}
    actual = [grades.get(_canonical_url(url), 0) for url in retrieved_urls[:k]]
    ideal = sorted(grades.values(), reverse=True)[:k]

    def dcg(values: Sequence[int]) -> float:
        return sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(values, start=1)
        )

    ideal_dcg = dcg(ideal)
    return dcg(actual) / ideal_dcg if ideal_dcg else 0.0
