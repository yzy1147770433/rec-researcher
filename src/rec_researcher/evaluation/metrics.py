"""Deterministic metrics for versioned retrieval benchmarks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from rec_researcher.core.models import (
    CitationValidation,
    ClaimVerificationResult,
    GoldDocument,
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


def precision_at_k(
    retrieved_urls: Sequence[str],
    gold_relevance: Mapping[str, int] | None,
    *,
    k: int,
) -> float | None:
    """Compute Precision@K over the returned prefix (not padded results)."""

    if k < 1:
        raise ValueError("k must be at least 1")
    if not gold_relevance:
        return None
    prefix = retrieved_urls[:k]
    if not prefix:
        return 0.0
    gold = {_canonical_url(url) for url in gold_relevance}
    relevant = sum(_canonical_url(url) in gold for url in prefix)
    return relevant / len(prefix)


def evidence_support_metrics(
    results: Sequence[ClaimVerificationResult],
) -> dict[str, float]:
    """Return mutually interpretable claim-level status rates."""

    if not results:
        return {
            "evidence_support_rate": 0.0,
            "partial_support_rate": 0.0,
            "unsupported_claim_rate": 0.0,
            "missing_citation_rate": 0.0,
            "invalid_citation_rate": 0.0,
        }
    total = len(results)
    names = {
        "supported": "evidence_support_rate",
        "partially_supported": "partial_support_rate",
        "unsupported": "unsupported_claim_rate",
        "missing_citation": "missing_citation_rate",
        "invalid_citation": "invalid_citation_rate",
    }
    return {
        metric: sum(item.status == status for item in results) / total
        for status, metric in names.items()
    }


def expected_fact_coverage(report: str, expected_facts: Sequence[str]) -> float | None:
    """Measure exact normalized expected-fact presence when annotations exist."""

    if not expected_facts:
        return None
    normalized_report = " ".join(report.casefold().split())
    matched = sum(
        " ".join(fact.casefold().split()) in normalized_report
        for fact in expected_facts
    )
    return matched / len(expected_facts)


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


_ARXIV_RE = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arxiv[:.])(\d{4}\.\d{4,5})", re.I)
_DOI_RE = re.compile(r"(?:doi\.org/|doi:\s*)(10\.\d{4,9}/[^\s?#]+)", re.I)
_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _arxiv_id(value: str) -> str | None:
    match = _ARXIV_RE.search(value)
    return match.group(1).casefold() if match else None


def _doi(value: str) -> str | None:
    match = _DOI_RE.search(value)
    return match.group(1).rstrip("./").casefold() if match else None


def _title_tokens(value: str) -> set[str]:
    return {
        token for token in _TITLE_TOKEN_RE.findall(value.casefold()) if len(token) > 1
    }


def _same_document(source: SourceRecord, gold: GoldDocument) -> bool:
    """Match stable identifiers first and conservative long-title overlap last."""

    url = str(source.url)
    accepted = {_canonical_url(item) for item in gold.accepted_urls}
    if _canonical_url(url) in accepted:
        return True
    source_arxiv = _arxiv_id(url + " " + source.title)
    if gold.arxiv_id and source_arxiv == gold.arxiv_id.casefold():
        return True
    if source_arxiv and source_arxiv == _arxiv_id(gold.document_id):
        return True
    source_doi = _doi(url + " " + source.title)
    if gold.doi and source_doi == gold.doi.casefold():
        return True
    if source_doi and source_doi == _doi(gold.document_id):
        return True
    gold_tokens = _title_tokens(gold.title)
    source_tokens = _title_tokens(source.title)
    if len(gold_tokens) < 5:
        return False
    containment = len(gold_tokens & source_tokens) / len(gold_tokens)
    return containment >= 0.8


def document_identity_metrics(
    retrieved: Sequence[SourceRecord],
    gold_documents: Sequence[GoldDocument],
    *,
    cutoffs: Sequence[int] = (3, 5, 10),
) -> dict[str, float | None]:
    """Compute document-level metrics without counting URL aliases as new gold."""

    names = {k: f"document_recall_at_{k}" for k in cutoffs}
    if not gold_documents:
        return {
            **{name: None for name in names.values()},
            "document_mrr": None,
            "document_ndcg_at_5": None,
        }
    matched_by_rank: list[GoldDocument | None] = []
    seen: set[str] = set()
    for source in retrieved:
        match = next(
            (
                gold
                for gold in gold_documents
                if gold.document_id not in seen and _same_document(source, gold)
            ),
            None,
        )
        matched_by_rank.append(match)
        if match is not None:
            seen.add(match.document_id)
    result: dict[str, float | None] = {}
    for cutoff, name in names.items():
        found = {
            item.document_id for item in matched_by_rank[:cutoff] if item is not None
        }
        result[name] = len(found) / len(gold_documents)
    first = next(
        (
            rank
            for rank, item in enumerate(matched_by_rank, start=1)
            if item is not None
        ),
        None,
    )
    result["document_mrr"] = 1.0 / first if first is not None else 0.0
    actual = [
        item.relevance_grade if item is not None else 0 for item in matched_by_rank[:5]
    ]
    ideal = sorted((item.relevance_grade for item in gold_documents), reverse=True)[:5]

    def dcg(grades: Sequence[int]) -> float:
        return sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(grades, start=1)
        )

    result["document_ndcg_at_5"] = dcg(actual) / dcg(ideal) if ideal else 0.0
    return result
