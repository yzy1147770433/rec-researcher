"""Milvus Lite backed passage vector index."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
from pymilvus import MilvusClient

from rec_researcher.core.models import PassageRecord


class VectorSearchHit(BaseModel):
    """A scored and source-linked nearest-neighbor result."""

    model_config = ConfigDict(extra="forbid")

    passage_id: str
    source_id: str
    text: str
    rank: int
    score: float


class MilvusLiteIndex:
    """Persist passage vectors in a local Milvus Lite database."""

    def __init__(
        self,
        uri: str | Path = "./data/rec_researcher.db",
        *,
        collection_name: str = "passages",
    ) -> None:
        """Connect to a database; defer collection creation until first upsert."""

        self.uri = str(uri)
        self.collection_name = collection_name
        if self.uri != ":memory:":
            Path(self.uri).parent.mkdir(parents=True, exist_ok=True)
        self._client = MilvusClient(uri=self.uri)
        self._dimension = self._read_dimension()

    def _read_dimension(self) -> int | None:
        if not self._client.has_collection(self.collection_name):
            return None
        description = self._client.describe_collection(self.collection_name)
        for field in description.get("fields", []):
            if field.get("name") == "vector":
                params = field.get("params", {})
                dimension = params.get("dim")
                return int(dimension) if dimension is not None else None
        raise RuntimeError(
            f"collection {self.collection_name!r} has no 'vector' field"
        )

    @staticmethod
    def _validate_vectors(vectors: Sequence[Sequence[float]]) -> int:
        if not vectors:
            raise ValueError("vectors must not be empty when passages are provided")
        dimension = len(vectors[0])
        if dimension == 0:
            raise ValueError("vectors must have at least one dimension")
        for index, vector in enumerate(vectors):
            if len(vector) == 0:
                raise ValueError(f"vector at index {index} is empty")
            if len(vector) != dimension:
                raise ValueError(
                    f"vector dimension mismatch at index {index}: "
                    f"expected {dimension}, got {len(vector)}"
                )
        return dimension

    def _ensure_collection(self, dimension: int) -> None:
        if self._dimension is None:
            self._client.create_collection(
                collection_name=self.collection_name,
                dimension=dimension,
                primary_field_name="passage_id",
                id_type="string",
                vector_field_name="vector",
                metric_type="COSINE",
                max_length=2048,
            )
            self._dimension = dimension
        elif dimension != self._dimension:
            raise ValueError(
                f"vector dimension mismatch: collection expects "
                f"{self._dimension}, got {dimension}"
            )

    async def upsert_passages(
        self,
        passages: Sequence[PassageRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Insert or replace passage/vector pairs in input order."""

        if not passages and not vectors:
            return
        if len(passages) != len(vectors):
            raise ValueError(
                "passages and vectors must contain the same number of items"
            )
        dimension = self._validate_vectors(vectors)
        await asyncio.to_thread(self._upsert_sync, passages, vectors, dimension)

    def _upsert_sync(
        self,
        passages: Sequence[PassageRecord],
        vectors: Sequence[Sequence[float]],
        dimension: int,
    ) -> None:
        self._ensure_collection(dimension)
        data = [
            {
                "passage_id": passage.id,
                "source_id": passage.source_id,
                "text": passage.text,
                "vector": list(vector),
            }
            for passage, vector in zip(passages, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self.collection_name, data=data)

    async def add(
        self,
        passages: Sequence[PassageRecord],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Compatibility alias for :meth:`upsert_passages`."""

        await self.upsert_passages(passages, vectors)

    async def search(
        self, vector: Sequence[float], *, limit: int
    ) -> list[VectorSearchHit]:
        """Return cosine-similar passages, best match first."""

        if len(vector) == 0:
            raise ValueError("search vector must not be empty")
        if limit <= 0:
            return []
        if self._dimension is None:
            return []
        if len(vector) != self._dimension:
            raise ValueError(
                f"search vector dimension mismatch: collection expects "
                f"{self._dimension}, got {len(vector)}"
            )
        rows = await asyncio.to_thread(self._search_sync, vector, limit)
        return [self._to_hit(row, rank) for rank, row in enumerate(rows, start=1)]

    def _search_sync(self, vector: Sequence[float], limit: int) -> list[dict[str, Any]]:
        result = self._client.search(
            collection_name=self.collection_name,
            data=[list(vector)],
            limit=limit,
            output_fields=["source_id", "text"],
        )
        return result[0] if result else []

    @staticmethod
    def _to_hit(row: dict[str, Any], rank: int) -> VectorSearchHit:
        entity = row.get("entity", {})
        passage_id = row.get("id", entity.get("passage_id"))
        return VectorSearchHit(
            passage_id=str(passage_id),
            source_id=str(entity["source_id"]),
            text=str(entity["text"]),
            rank=rank,
            score=float(row.get("distance", 0.0)),
        )

    def close(self) -> None:
        """Release the underlying local database connection."""

        self._client.close()
