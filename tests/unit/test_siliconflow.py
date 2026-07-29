import json

import httpx
import pytest

from rec_researcher.core.exceptions import ProviderError
from rec_researcher.core.settings import Settings
from rec_researcher.providers.siliconflow import (
    SiliconFlowEmbedder,
    SiliconFlowReranker,
)

pytestmark = pytest.mark.asyncio


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        siliconflow_base_url="https://siliconflow.invalid/v1",
        siliconflow_api_key="test-secret",
        embedding_model="embedding-model",
        reranker_model="reranker-model",
        max_retries=2,
    )


async def test_embed_batch_restores_index_order_and_authorizes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer test-secret"
        assert json.loads(request.content) == {
            "model": "embedding-model",
            "input": ["first", "second"],
        }
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, 1]},
                    {"index": 0, "embedding": [1, 0]},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        embedder = SiliconFlowEmbedder(_settings(), client=client)
        result = await embedder.embed(["first", "second"])

    assert result == [[1.0, 0.0], [0.0, 1.0]]


async def test_embed_empty_input_and_blank_value_never_call_api() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        embedder = SiliconFlowEmbedder(_settings(), client=client)
        assert await embedder.embed([]) == []
        with pytest.raises(ValueError, match="empty strings"):
            await embedder.embed(["valid", "  "])

    assert calls == 0


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([{"index": 0, "embedding": [1]}], "count"),
        (
            [
                {"index": 0, "embedding": [1]},
                {"index": 1, "embedding": [1, 2]},
            ],
            "dimensions",
        ),
        (
            [
                {"index": 0, "embedding": [1]},
                {"index": 2, "embedding": [2]},
            ],
            "index is invalid",
        ),
    ],
)
async def test_embed_validates_response(data: object, message: str) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": data})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        embedder = SiliconFlowEmbedder(_settings(), client=client)
        with pytest.raises(ProviderError, match=message):
            await embedder.embed(["one", "two"])


@pytest.mark.parametrize("status", [429, 502, 503, 504])
async def test_embed_retries_transient_statuses(status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status, text="temporary")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await SiliconFlowEmbedder(_settings(), client=client).embed(["one"])

    assert calls == 2


@pytest.mark.parametrize("status", [401, 403])
async def test_embed_does_not_retry_auth_errors_or_expose_key(status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, text="denied")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ProviderError) as caught:
            await SiliconFlowEmbedder(_settings(), client=client).embed(["one"])

    assert calls == 1
    assert "test-secret" not in str(caught.value)


async def test_rerank_maps_documents_and_clamps_top_n() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/v1/rerank"
        assert payload["top_n"] == 2
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.4},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reranker = SiliconFlowReranker(_settings(), client=client)
        result = await reranker.rerank("query", ["first", "second"], top_n=99)

    assert [(item.index, item.document, item.relevance_score) for item in result] == [
        (1, "second", 0.9),
        (0, "first", 0.4),
    ]


async def test_rerank_empty_documents_never_call_api() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reranker = SiliconFlowReranker(_settings(), client=client)
        assert await reranker.rerank("", [], top_n=5) == []
        with pytest.raises(ValueError, match="query"):
            await reranker.rerank("  ", ["document"], top_n=1)

    assert calls == 0


async def test_rerank_rejects_invalid_response_index() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"results": [{"index": 2, "relevance_score": 0.9}]}
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        reranker = SiliconFlowReranker(_settings(), client=client)
        with pytest.raises(ProviderError, match="index is invalid"):
            await reranker.rerank("query", ["document"], top_n=1)
