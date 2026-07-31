import httpx
import pytest

from rec_researcher.core.models import SourceRecord
from rec_researcher.core.settings import Settings
from rec_researcher.retrieval.fetcher import AsyncWebFetcher

pytestmark = pytest.mark.asyncio


def _source(source_id: str, url: str) -> SourceRecord:
    return SourceRecord(
        id=source_id, title="title", url=url, snippet="", provider="fixture"
    )


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"max_retries": 0, **overrides}
    return Settings(_env_file=None, **values)


async def test_failed_url_does_not_stop_other_fetches() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/failed":
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, html="<main><p>Useful article text.</p></main>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = AsyncWebFetcher(_settings(), client=client)
        results = await fetcher.fetch_many(
            [
                _source("bad", "https://example.test/failed"),
                _source("good", "https://example.test/article"),
            ]
        )

    assert results[0].success is False
    assert results[0].error is not None
    assert results[1].success is True
    assert "Useful article text" in results[1].text


async def test_empty_html_is_safe() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=b"", headers={"content-type": "text/html"}
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        result = await AsyncWebFetcher(_settings(), client=client).fetch(
            _source("empty", "https://example.test/empty")
        )

    assert result.success is True
    assert result.text == ""


async def test_extracts_chinese_page() -> None:
    html = """<html><body><article><h1>推荐系统研究</h1>
    <p>这是一个用于验证中文正文提取的段落，包含足够的信息。</p>
    <p>第二个段落讨论离线评估指标和实验设计。</p></article></body></html>"""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, text=html, headers={"content-type": "text/html; charset=utf-8"}
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        result = await AsyncWebFetcher(_settings(), client=client).fetch(
            _source("zh", "https://example.test/zh")
        )

    assert result.success is True
    assert "推荐系统研究" in result.text
    assert "离线评估指标" in result.text


async def test_rejects_binary_and_oversized_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/binary":
            return httpx.Response(
                200, content=b"binary", headers={"content-type": "image/png"}
            )
        return httpx.Response(200, content=b"x" * 11)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = AsyncWebFetcher(_settings(max_response_bytes=10), client=client)
        binary = await fetcher.fetch(_source("binary", "https://example.test/binary"))
        large = await fetcher.fetch(_source("large", "https://example.test/large"))

    assert binary.success is False
    assert "binary" in (binary.error or "")
    assert large.success is False
    assert "too large" in (large.error or "")


async def test_retries_timeout_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(
            200, text="<p>recovered</p>", headers={"content-type": "text/html"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AsyncWebFetcher(_settings(max_retries=1), client=client).fetch(
            _source("retry", "https://example.test/retry")
        )

    assert result.success is True
    assert calls == 2


async def test_rejects_non_html_content() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, text="plain response", headers={"content-type": "text/plain"}
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        result = await AsyncWebFetcher(_settings(), client=client).fetch(
            _source("plain", "https://example.test/plain")
        )

    assert result.success is False
    assert "non-HTML" in (result.error or "")
