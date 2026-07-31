import httpx
import pytest

from rec_researcher.core.models import SourceRecord
from rec_researcher.providers.crossref import (
    CompositeSearchProvider,
    CrossrefSearchProvider,
)


@pytest.mark.asyncio
async def test_crossref_normalizes_doi_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query.title"] == "ColBERT"
        return httpx.Response(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.1145/3397271.3401075",
                            "title": ["ColBERT"],
                            "URL": "https://doi.org/10.1145/3397271.3401075",
                            "publisher": "ACM",
                        }
                    ]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await CrossrefSearchProvider(client=client).search("ColBERT", limit=5)
    assert len(results) == 1
    assert str(results[0].url) == "https://doi.org/10.1145/3397271.3401075"
    assert results[0].provider == "crossref"


@pytest.mark.asyncio
async def test_composite_isolates_failure_and_interleaves() -> None:
    class Failing:
        async def search(self, query: str, *, limit: int) -> list[SourceRecord]:
            raise RuntimeError("unavailable")

    class Working:
        async def search(self, query: str, *, limit: int) -> list[SourceRecord]:
            return [
                SourceRecord(
                    id="ok",
                    title="Paper",
                    url="https://arxiv.org/abs/2004.12832",
                    snippet=query,
                    provider="test",
                )
            ]

    results = await CompositeSearchProvider([Failing(), Working()]).search(
        "query", limit=5
    )
    assert [item.id for item in results] == ["ok"]


@pytest.mark.asyncio
async def test_composite_routes_only_academic_queries_to_crossref() -> None:
    calls: list[str] = []

    class RecordingCrossref(CrossrefSearchProvider):
        async def search(self, query: str, *, limit: int) -> list[SourceRecord]:
            calls.append(query)
            return []

    class Web:
        async def search(self, query: str, *, limit: int) -> list[SourceRecord]:
            return []

    provider = CompositeSearchProvider([RecordingCrossref(), Web()])
    await provider.search("ColBERT explanation", limit=5)
    await provider.search("ColBERT original paper DOI", limit=5)
    await provider.aclose()
    assert calls == ["ColBERT original paper DOI"]
