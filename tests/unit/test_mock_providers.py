import pytest

from rec_researcher.core.models import PassageRecord
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
    passages = [
        PassageRecord(id="p1", source_id="S1", text="unrelated"),
        PassageRecord(id="p2", source_id="S2", text="ranking metrics"),
    ]

    assert await reranker.rerank("metrics", [], limit=2) == []
    assert await reranker.rerank("metrics", passages, limit=2) == [
        passages[1],
        passages[0],
    ]
