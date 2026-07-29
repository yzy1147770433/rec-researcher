"""Bounded, failure-isolated web document fetching and text extraction."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from urllib.parse import urlsplit

import httpx
import trafilatura
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

from rec_researcher.core.models import SourceRecord
from rec_researcher.core.settings import Settings

_RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_BINARY_TYPES = (
    "application/octet-stream",
    "application/pdf",
    "audio/",
    "image/",
    "video/",
)


class FetchResult(BaseModel):
    """Structured result for one URL, including non-fatal failures."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    url: str
    text: str = ""
    success: bool
    status_code: int | None = None
    content_type: str | None = None
    error: str | None = None


class _RetryableFetchError(Exception):
    """A transient fetch error that may be retried."""


class AsyncWebFetcher:
    """Fetch web pages with strict bounds and extract their main text."""

    def __init__(
        self, settings: Settings, *, client: httpx.AsyncClient | None = None
    ) -> None:
        """Configure limits and optionally use an injected offline client."""

        self._client = client or httpx.AsyncClient(follow_redirects=True)
        self._owns_client = client is None
        self._timeout = settings.request_timeout_seconds
        self._max_attempts = settings.max_retries + 1
        self._max_response_bytes = settings.max_response_bytes

    async def fetch(self, source: SourceRecord) -> FetchResult:
        """Fetch one source, converting every expected failure into a result."""

        url = str(source.url)
        if urlsplit(url).scheme.lower() not in {"http", "https"}:
            return self._failure(source.id, url, "unsupported URL scheme")
        try:
            retrying = AsyncRetrying(
                stop=stop_after_attempt(self._max_attempts),
                retry=retry_if_exception_type(
                    (httpx.TimeoutException, httpx.TransportError, _RetryableFetchError)
                ),
                reraise=True,
            )
            async for attempt in retrying:
                with attempt:
                    return await self._fetch_once(source.id, url)
        except (httpx.HTTPError, _RetryableFetchError, ValueError) as exc:
            return self._failure(source.id, url, str(exc))
        return self._failure(source.id, url, "fetch did not produce a result")

    async def fetch_many(self, sources: Sequence[SourceRecord]) -> list[FetchResult]:
        """Fetch sources concurrently without one failure cancelling its siblings."""

        return list(await asyncio.gather(*(self.fetch(source) for source in sources)))

    async def aclose(self) -> None:
        """Close the internally-created HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    async def _fetch_once(self, source_id: str, url: str) -> FetchResult:
        async with self._client.stream("GET", url, timeout=self._timeout) as response:
            if response.status_code in _RETRYABLE_STATUSES:
                raise _RetryableFetchError(
                    f"temporary HTTP status {response.status_code}"
                )
            if not response.is_success:
                return self._failure(
                    source_id, url, f"HTTP status {response.status_code}", response
                )
            content_type = response.headers.get("content-type", "").lower()
            if any(marker in content_type for marker in _BINARY_TYPES):
                return self._failure(
                    source_id, url, f"binary content type: {content_type}", response
                )
            declared_size = response.headers.get("content-length")
            if declared_size and int(declared_size) > self._max_response_bytes:
                return self._failure(
                    source_id, url, "response body too large", response
                )
            body = bytearray()
            async for part in response.aiter_bytes():
                body.extend(part)
                if len(body) > self._max_response_bytes:
                    return self._failure(
                        source_id, url, "response body too large", response
                    )
            html = bytes(body).decode(response.encoding or "utf-8", errors="replace")
            text = self._extract_text(html)
            return FetchResult(
                source_id=source_id,
                url=url,
                text=text,
                success=True,
                status_code=response.status_code,
                content_type=response.headers.get("content-type"),
            )

    @staticmethod
    def _extract_text(html: str) -> str:
        if not html.strip():
            return ""
        try:
            extracted = trafilatura.extract(
                html, include_comments=False, include_tables=False, output_format="txt"
            )
        except Exception:  # trafilatura exposes multiple parser-specific failures
            extracted = None
        if extracted:
            return extracted.strip()
        soup = BeautifulSoup(html, "html.parser")
        for unwanted in soup(["script", "style", "noscript"]):
            unwanted.decompose()
        return soup.get_text("\n", strip=True)

    @staticmethod
    def _failure(
        source_id: str,
        url: str,
        error: str,
        response: httpx.Response | None = None,
    ) -> FetchResult:
        return FetchResult(
            source_id=source_id,
            url=url,
            success=False,
            status_code=response.status_code if response else None,
            content_type=response.headers.get("content-type") if response else None,
            error=error,
        )
