"""Deterministic, bounded search-query rewriting."""

from __future__ import annotations

import re

from rec_researcher.core.models import RewrittenQuery


class QueryRewriter:
    """Produce useful query intents without requiring an external model."""

    def __init__(self, *, max_queries: int = 5) -> None:
        if not 1 <= max_queries <= 10:
            raise ValueError("max_queries must be between 1 and 10")
        self.max_queries = max_queries

    def rewrite(self, query: str) -> list[RewrittenQuery]:
        """Return stable, case-insensitively deduplicated variants."""

        normalized = " ".join(query.split())
        if not normalized:
            return []
        keywords = " ".join(
            re.findall(r"[A-Za-z][\w.+-]*|[\u4e00-\u9fff]{2,}", normalized)
        )
        entities = self._technical_entities(normalized) or keywords or normalized
        candidates = (
            (normalized, "original", 1.0, "原始问题"),
            (
                f"{entities} original paper PDF arXiv DOI",
                "primary_paper",
                0.98,
                "论文原文优先",
            ),
            (
                f"{entities} site:arxiv.org",
                "arxiv",
                0.96,
                "arXiv 定向检索",
            ),
            (keywords or normalized, "keywords", 0.95, "核心技术词"),
            (
                f"{normalized} official documentation research publication",
                "official_docs",
                0.9,
                "官方文档或机构论文页",
            ),
            (f"{normalized} benchmark Recall NDCG MRR", "metrics", 0.75, "评测指标"),
            (f"{normalized} GitHub implementation", "github", 0.7, "开源实现"),
        )
        seen: set[str] = set()
        results: list[RewrittenQuery] = []
        for text, kind, priority, reason in candidates:
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            results.append(
                RewrittenQuery(
                    query=text, query_type=kind, priority=priority, reason=reason
                )
            )
            if len(results) == self.max_queries:
                break
        return results

    @staticmethod
    def _technical_entities(query: str) -> str:
        """Keep model names/acronyms so Chinese prose does not dilute paper search."""

        stopwords = {
            "how",
            "what",
            "why",
            "does",
            "the",
            "and",
            "with",
            "model",
            "paper",
            "method",
        }
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9.+_-]*", query)
        selected = [
            token
            for token in tokens
            if token.casefold() not in stopwords
            and (
                any(character.isupper() for character in token)
                or any(character.isdigit() for character in token)
                or len(token) <= 8
            )
        ]
        return " ".join(dict.fromkeys(selected))
