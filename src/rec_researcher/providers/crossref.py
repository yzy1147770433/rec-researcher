"""Crossref-backed academic metadata retrieval and provider composition."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import httpx

from rec_researcher.core.models import SourceRecord
from rec_researcher.providers.base import SearchProvider
from rec_researcher.retrieval.dedup import normalize_url


class CrossrefSearchProvider:
    """Search DOI metadata without credentials, with bounded network behavior."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 20.0,
    ) -> None:
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._timeout = timeout

    async def search(self, query: str, *, limit: int) -> list[SourceRecord]:
        """Return normalized scholarly works ordered by Crossref relevance."""

        if limit <= 0 or not query.strip():
            return []
        response = await self._client.get(
            "https://api.crossref.org/works",
            params={
                "query.title": query,
                "rows": min(limit, 20),
                "select": "DOI,title,URL,publisher,published",
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
        results: list[SourceRecord] = []
        for rank, item in enumerate(items, start=1):
            titles = item.get("title") or []
            title = titles[0].strip() if titles and isinstance(titles[0], str) else ""
            doi = item.get("DOI")
            if not title or not isinstance(doi, str) or not doi:
                continue
            url = f"https://doi.org/{doi}"
            publisher = item.get("publisher") or "Crossref"
            identifier = hashlib.sha256(url.encode()).hexdigest()[:16]
            results.append(
                SourceRecord(
                    id=f"crossref-{identifier}",
                    title=title,
                    url=url,
                    snippet=f"Scholarly metadata from {publisher}. DOI: {doi}",
                    provider="crossref",
                    score=1.0 / rank,
                )
            )
        return results

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class CompositeSearchProvider:
    """Merge failure-isolated providers while preferring academic metadata."""

    def __init__(self, providers: Sequence[SearchProvider]) -> None:
        self.providers = list(providers)

    async def search(self, query: str, *, limit: int) -> list[SourceRecord]:
        batches: list[list[SourceRecord]] = []
        for provider in self.providers:
            if isinstance(provider, CrossrefSearchProvider) and not any(
                marker in query.casefold()
                for marker in ("original paper", "doi", "publication")
            ):
                continue
            try:
                found = await provider.search(query, limit=limit)
            except Exception:  # noqa: BLE001 - one search backend must not abort another
                continue
            batches.append(found)
        merged: dict[str, SourceRecord] = {}
        for rank in range(limit):
            for batch in batches:
                if rank >= len(batch):
                    continue
                source = batch[rank]
                merged.setdefault(normalize_url(str(source.url)), source)
                if len(merged) == limit:
                    return list(merged.values())
        return list(merged.values())

    async def aclose(self) -> None:
        for provider in self.providers:
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()
