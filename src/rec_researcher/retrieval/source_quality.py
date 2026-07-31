"""Transparent source-quality rules independent of query relevance."""

from __future__ import annotations

from urllib.parse import urlsplit

from rec_researcher.core.models import SourceQualityScore, SourceRecord


def score_source(source: SourceRecord) -> SourceQualityScore:
    """Score authority/originality with conservative URL and title rules."""

    host = urlsplit(str(source.url)).netloc.casefold().removeprefix("www.")
    text = f"{source.title} {source.snippet}".casefold()
    authority, originality, freshness = 0.45, 0.45, 0.5
    reasons: list[str] = []
    if host == "arxiv.org":
        authority, originality = 0.92, 0.95
        reasons.append("论文原文/arXiv")
    elif host == "doi.org":
        authority, originality = 0.93, 0.92
        reasons.append("DOI 学术原文标识")
    elif host.endswith(
        (
            "acm.org",
            "ieee.org",
            "springer.com",
            "openreview.net",
            "ijcai.org",
            "neurips.cc",
        )
    ):
        authority, originality = 0.95, 0.9
        reasons.append("会议或期刊官网")
    elif (
        host.endswith(("github.com", "readthedocs.io"))
        or "docs." in host
        or host.startswith("research.")
        or ".research." in host
    ):
        authority, originality = 0.85, 0.9
        reasons.append("代码或官方文档候选")
    elif host.endswith(".edu") or ".edu." in host:
        authority, originality = 0.8, 0.7
        reasons.append("高校来源")
    elif any(item in host for item in ("medium.com", "blogspot.", "csdn.net")):
        authority, originality = 0.3, 0.25
        reasons.append("普通博客或聚合内容")
    domain_terms = (
        "recommend",
        "retrieval",
        "search",
        "ranking",
        "llm",
        "transformer",
        "推荐",
        "检索",
        "排序",
        "大模型",
    )
    relevance = min(1.0, 0.35 + 0.13 * sum(term in text for term in domain_terms))
    if relevance > 0.35:
        reasons.append("主题与目标领域相关")
    final = 0.4 * authority + 0.3 * originality + 0.2 * relevance + 0.1 * freshness
    return SourceQualityScore(
        authority=authority,
        originality=originality,
        domain_relevance=relevance,
        freshness=freshness,
        final_score=round(final, 6),
        reasons=reasons,
    )
