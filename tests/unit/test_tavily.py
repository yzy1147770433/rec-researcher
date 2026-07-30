import httpx
import pytest

from rec_researcher.core.exceptions import ProviderError
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
        assert request.method == "POST"
        assert request.url.path == "/search"
        assert request.headers["Authorization"] == "Bearer test-secret"
        assert b'"search_depth":"basic"' in payload
        assert b'"include_answer":false' in payload
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "First",
                        "url": "https://arxiv.org/abs/one",
                        "content": "content",
                        "score": 0.9,
                    },
                    {
                        "title": "Duplicate",
                        "url": "https://arxiv.org/abs/one",
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


async def test_trailing_slash_does_not_duplicate_search_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        return httpx.Response(200, json={"results": []})

    settings = _settings().model_copy(
        update={"tavily_base_url": "https://tavily.invalid/"}
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert (
            await TavilySearchProvider(settings, client=client).search("query", limit=1)
            == []
        )


async def test_error_does_not_expose_api_key() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(403, text="echoed test-secret")
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ProviderError) as caught:
            await TavilySearchProvider(_settings(), client=client).search(
                "query", limit=1
            )

    assert "test-secret" not in str(caught.value)


@pytest.mark.parametrize("status", [429, 502, 503, 504])
async def test_retries_transient_statuses(status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status, text="temporary")
        return httpx.Response(200, json={"results": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert (
            await TavilySearchProvider(_settings(), client=client).search(
                "query", limit=1
            )
            == []
        )

    assert calls == 2


@pytest.mark.parametrize("status", [401, 403])
async def test_does_not_retry_auth_errors(status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, text="denied")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderError, match=f"status {status}"):
            await TavilySearchProvider(_settings(), client=client).search(
                "query", limit=1
            )

    assert calls == 1


async def test_reserved_example_dot_com_sources_are_excluded() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Reserved", "url": "https://example.com/paper"},
                    {
                        "title": "Reserved subdomain",
                        "url": "https://papers.example.com/paper",
                    },
                    {"title": "Real", "url": "https://arxiv.org/abs/one"},
                ]
            },
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        results = await TavilySearchProvider(_settings(), client=client).search(
            "query", limit=5
        )

    assert [str(item.url) for item in results] == ["https://arxiv.org/abs/one"]


async def test_network_error_is_wrapped_without_details() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret network details", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(
            ProviderError, match="Tavily network request failed: ConnectError"
        ) as caught:
            await TavilySearchProvider(_settings(), client=client).search(
                "query", limit=1
            )

    assert "secret network details" not in str(caught.value)
