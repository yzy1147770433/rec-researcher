"""Tavily web-search provider."""

from __future__ import annotations

import hashlib
from urllib.parse import urlparse

import httpx
from pydantic import SecretStr, ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

from rec_researcher.core.exceptions import ProviderError
from rec_researcher.core.models import SourceRecord
from rec_researcher.core.settings import Settings

_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})
_MAX_ERROR_BODY_LENGTH = 500


class _RetryableTavilyError(ProviderError):
    """Mark a transient Tavily failure as safe to retry."""


def _is_example_dot_com(url: str) -> bool:
    """Reject reserved example.com hosts from real search results."""

    hostname = (urlparse(url).hostname or "").casefold().rstrip(".")
    return hostname == "example.com" or hostname.endswith(".example.com")


class TavilySearchProvider:
    """Map Tavily search responses to provider-independent source records."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure Tavily from settings."""

        secret = settings.tavily_api_key
        if not isinstance(secret, SecretStr) or not secret.get_secret_value():
            raise ValueError("tavily_api_key must be configured")
        self.base_url = settings.tavily_base_url.rstrip("/")
        self._api_key = secret.get_secret_value()
        self._timeout = settings.request_timeout_seconds
        self._max_attempts = settings.max_retries + 1
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def search(self, query: str, *, limit: int) -> list[SourceRecord]:
        """Search Tavily, safely handling empty and malformed individual results."""

        if limit <= 0:
            return []
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            retry=retry_if_exception_type(_RetryableTavilyError),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                try:
                    response = await self._client.post(
                        f"{self.base_url}/search",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json={
                            "query": query,
                            "search_depth": "basic",
                            "include_answer": False,
                            "include_raw_content": False,
                            "max_results": limit,
                        },
                        timeout=self._timeout,
                    )
                except httpx.RequestError as exc:
                    raise _RetryableTavilyError(
                        "Tavily network request failed: " + type(exc).__name__
                    ) from exc
                self._raise_for_status(response)
                break
        else:
            raise AssertionError("retry loop completed without a response")
        payload = response.json()
        raw_results = payload.get("results", []) if isinstance(payload, dict) else []
        if not isinstance(raw_results, list):
            raise ProviderError("Tavily response field 'results' was not a list")

        sources: list[SourceRecord] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url or url in seen_urls or _is_example_dot_com(url):
                continue
            try:
                source = SourceRecord(
                    id=f"tavily-{hashlib.sha256(url.encode()).hexdigest()[:16]}",
                    title=str(item.get("title") or ""),
                    url=url,
                    snippet=str(item.get("content") or item.get("snippet") or ""),
                    score=item.get("score"),
                    provider="tavily",
                )
            except ValidationError:
                continue
            seen_urls.add(url)
            sources.append(source)
            if len(sources) >= limit:
                break
        return sources

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        body = response.text[:_MAX_ERROR_BODY_LENGTH].replace(
            self._api_key, "[REDACTED]"
        )
        message = f"Tavily request failed with status {response.status_code}: {body}"
        if response.status_code in _RETRYABLE_STATUS_CODES:
            raise _RetryableTavilyError(message)
        raise ProviderError(message)

    async def aclose(self) -> None:
        """Close the internally-created HTTP client."""

        if self._owns_client:
            await self._client.aclose()
