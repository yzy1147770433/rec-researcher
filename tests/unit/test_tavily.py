import httpx
import pytest

from rec_researcher.core.settings import Settings
from rec_researcher.providers.tavily import TavilySearchProvider

pytestmark = pytest.mark.asyncio


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        tavily_base_url="https://tavily.invalid",
        tavily_api_key="test-secret",
    )


async def test_duplicate_urls_are_removed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read()
        assert b'"search_depth":"basic"' in payload
        assert b'"include_answer":false' in payload
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "First",
                        "url": "https://example.com/a",
                        "content": "content",
                        "score": 0.9,
                    },
                    {
                        "title": "Duplicate",
                        "url": "https://example.com/a",
                        "content": "other",
                        "score": 0.8,
                    },
                    {"title": "No URL", "content": "ignored"},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await TavilySearchProvider(_settings(), client=client).search(
            "query", limit=5
        )

    assert len(results) == 1
    assert results[0].title == "First"
    assert results[0].snippet == "content"
    assert results[0].score == 0.9


async def test_empty_results_return_empty_list() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"results": []})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        results = await TavilySearchProvider(_settings(), client=client).search(
            "query", limit=5
        )

    assert results == []
