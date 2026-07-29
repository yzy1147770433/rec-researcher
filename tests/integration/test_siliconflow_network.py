import os

import pytest

from rec_researcher.core.settings import Settings
from rec_researcher.providers.siliconflow import (
    SiliconFlowEmbedder,
    SiliconFlowReranker,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.network]


def _network_settings() -> Settings:
    if not os.getenv("REC_SILICONFLOW_API_KEY"):
        pytest.skip("REC_SILICONFLOW_API_KEY is not configured")
    return Settings()


async def test_siliconflow_embedding_and_reranking() -> None:
    settings = _network_settings()
    embedder = SiliconFlowEmbedder(settings)
    reranker = SiliconFlowReranker(settings)
    try:
        vectors = await embedder.embed(["vector search", "collaborative filtering"])
        results = await reranker.rerank(
            "retrieval", ["vector search", "collaborative filtering"], top_n=1
        )
    finally:
        await embedder.aclose()
        await reranker.aclose()

    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert len(results) == 1
