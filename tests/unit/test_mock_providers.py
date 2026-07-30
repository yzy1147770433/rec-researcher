import pytest

from rec_researcher.providers.mock import (
    MockLanguageModel,
    MockPassageReranker,
    MockSearchProvider,
    MockTextEmbedder,
)

pytestmark = pytest.mark.asyncio


async def test_mock_providers_are_deterministic_and_offline() -> None:
    search = MockSearchProvider()
    first = await search.search("推荐系统", limit=5)
    second = await search.search("推荐系统", limit=5)

    assert first == second
    assert len(first) >= 5
    assert all(source.url.host == "example.com" for source in first)
    first_generation = await MockLanguageModel().generate("prompt")
    second_generation = await MockLanguageModel().generate("prompt")
    assert first_generation == second_generation
    assert await MockTextEmbedder().embed(["a"]) == await MockTextEmbedder().embed(
        ["a"]
    )


async def test_mock_reranker_handles_empty_and_stable_order() -> None:
    reranker = MockPassageReranker()
    documents = ["unrelated", "ranking metrics"]

    assert await reranker.rerank("metrics", [], top_n=2) == []
    results = await reranker.rerank("metrics", documents, top_n=2)

    assert [result.index for result in results] == [1, 0]
    assert [result.document for result in results] == [documents[1], documents[0]]
