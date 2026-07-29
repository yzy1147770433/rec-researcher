from pathlib import Path

import pytest

from rec_researcher.core.models import PassageRecord
from rec_researcher.retrieval.vector_store import MilvusLiteIndex


def _passage(identifier: str, source_id: str, text: str) -> PassageRecord:
    return PassageRecord(id=identifier, source_id=source_id, text=text)


@pytest.mark.asyncio
async def test_milvus_lite_upsert_and_search(tmp_path: Path) -> None:
    database = tmp_path / "vectors.db"
    index = MilvusLiteIndex(database)
    try:
        await index.upsert_passages(
            [_passage("p1", "s1", "first"), _passage("p2", "s2", "second")],
            [[1.0, 0.0], [0.0, 1.0]],
        )

        results = await index.search([0.9, 0.1], limit=2)

        assert [result.passage_id for result in results] == ["p1", "p2"]
        assert results[0].source_id == "s1"
        assert results[0].text == "first"
        assert results[0].rank == 1
    finally:
        index.close()

    reopened = MilvusLiteIndex(database)
    try:
        with pytest.raises(ValueError, match="expects 2, got 3"):
            await reopened.search([1.0, 0.0, 0.0], limit=1)
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_milvus_lite_empty_and_dimension_errors(tmp_path: Path) -> None:
    index = MilvusLiteIndex(tmp_path / "vectors.db")
    try:
        await index.upsert_passages([], [])
        assert await index.search([1.0], limit=3) == []
        with pytest.raises(ValueError, match="same number"):
            await index.upsert_passages([_passage("p", "s", "text")], [])
        with pytest.raises(ValueError, match="at least one dimension"):
            await index.upsert_passages([_passage("p", "s", "text")], [[]])
        await index.upsert_passages([_passage("p", "s", "text")], [[1.0, 0.0]])
        with pytest.raises(ValueError, match="expects 2, got 3"):
            await index.upsert_passages(
                [_passage("q", "s", "other")], [[1.0, 0.0, 0.0]]
            )
        with pytest.raises(ValueError, match="must not be empty"):
            await index.search([], limit=1)
    finally:
        index.close()
