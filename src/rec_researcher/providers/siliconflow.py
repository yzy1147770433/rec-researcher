"""SiliconFlow embedding and reranking providers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

from rec_researcher.core.exceptions import ProviderError
from rec_researcher.core.settings import Settings

_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})
_MAX_ERROR_BODY_LENGTH = 500


def _normalize_v1_base_url(value: str) -> str:
    """Return an API base ending in exactly one ``/v1`` segment."""

    normalized = value.rstrip("/")
    while normalized.endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return f"{normalized}/v1"


class _RetryableSiliconFlowError(ProviderError):
    """Mark a transient HTTP response as retryable."""


class SiliconFlowRerankResult(BaseModel):
    """A reranked document tied back to its original input position."""

    model_config = ConfigDict(extra="forbid")

    index: int
    document: str
    relevance_score: float


class _SiliconFlowClient:
    """Shared HTTP configuration and error handling."""

    def __init__(
        self, settings: Settings, *, client: httpx.AsyncClient | None = None
    ) -> None:
        secret = settings.siliconflow_api_key
        if not isinstance(secret, SecretStr) or not secret.get_secret_value():
            raise ValueError("siliconflow_api_key must be configured")
        self.base_url = _normalize_v1_base_url(settings.siliconflow_base_url)
        self._api_key = secret.get_secret_value()
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._timeout = settings.request_timeout_seconds
        self._max_attempts = settings.max_retries + 1

    async def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            retry=retry_if_exception_type(_RetryableSiliconFlowError),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                response = await self._client.post(
                    f"{self.base_url}/{path}",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                    timeout=self._timeout,
                )
                self._raise_for_status(response)
                return response
        raise AssertionError("retry loop completed without a response")

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        body = response.text[:_MAX_ERROR_BODY_LENGTH].replace(
            self._api_key, "[REDACTED]"
        )
        message = (
            f"SiliconFlow request failed with status {response.status_code}: {body}"
        )
        if response.status_code in _RETRYABLE_STATUS_CODES:
            raise _RetryableSiliconFlowError(message)
        raise ProviderError(message)

    async def aclose(self) -> None:
        """Close an internally-created HTTP client."""

        if self._owns_client:
            await self._client.aclose()


class SiliconFlowEmbedder(_SiliconFlowClient):
    """Call SiliconFlow's OpenAI-compatible embeddings endpoint."""

    def __init__(
        self, settings: Settings, *, client: httpx.AsyncClient | None = None
    ) -> None:
        """Configure the embedding model and shared HTTP client."""

        super().__init__(settings, client=client)
        if not settings.embedding_model:
            raise ValueError("embedding_model must be configured")
        self.model = settings.embedding_model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a non-blank text batch and restore response index order."""

        values = list(texts)
        if not values:
            return []
        if any(not value.strip() for value in values):
            raise ValueError("embedding texts must not contain empty strings")
        response = await self._post(
            "embeddings", {"model": self.model, "input": values}
        )
        try:
            payload = response.json()
            data = payload["data"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                "SiliconFlow embedding response has no data list"
            ) from exc
        if not isinstance(data, list):
            raise ProviderError("SiliconFlow embedding response data was not a list")
        if len(data) != len(values):
            raise ProviderError(
                "SiliconFlow embedding count does not match input count: "
                f"expected {len(values)}, got {len(data)}"
            )

        ordered: list[list[float] | None] = [None] * len(values)
        for item in data:
            if not isinstance(item, dict):
                raise ProviderError("SiliconFlow embedding item was not an object")
            index = item.get("index")
            embedding = item.get("embedding")
            if not isinstance(index, int) or not 0 <= index < len(values):
                raise ProviderError(
                    f"SiliconFlow embedding index is invalid: {index!r}"
                )
            if ordered[index] is not None:
                raise ProviderError(
                    f"SiliconFlow embedding index is duplicated: {index}"
                )
            if not isinstance(embedding, list) or not embedding:
                raise ProviderError(f"SiliconFlow embedding at index {index} is empty")
            if any(not isinstance(value, (int, float)) for value in embedding):
                raise ProviderError(
                    f"SiliconFlow embedding at index {index} contains non-numbers"
                )
            ordered[index] = [float(value) for value in embedding]

        vectors = [vector for vector in ordered if vector is not None]
        if len(vectors) != len(values):
            raise ProviderError("SiliconFlow embedding response has missing indices")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ProviderError("SiliconFlow embedding dimensions are inconsistent")
        return vectors


class SiliconFlowReranker(_SiliconFlowClient):
    """Call SiliconFlow's reranking endpoint."""

    def __init__(
        self, settings: Settings, *, client: httpx.AsyncClient | None = None
    ) -> None:
        """Configure the reranker model and shared HTTP client."""

        super().__init__(settings, client=client)
        if not settings.reranker_model:
            raise ValueError("reranker_model must be configured")
        self.model = settings.reranker_model

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int
    ) -> list[SiliconFlowRerankResult]:
        """Rerank documents while preserving their original indices."""

        values = list(documents)
        if not values or top_n <= 0:
            return []
        if not query.strip():
            raise ValueError("rerank query must not be empty")
        effective_top_n = min(top_n, len(values))
        response = await self._post(
            "rerank",
            {
                "model": self.model,
                "query": query,
                "documents": values,
                "top_n": effective_top_n,
            },
        )
        try:
            payload = response.json()
            results = payload["results"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(
                "SiliconFlow rerank response has no results list"
            ) from exc
        if not isinstance(results, list):
            raise ProviderError("SiliconFlow rerank results was not a list")

        output: list[SiliconFlowRerankResult] = []
        seen: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                raise ProviderError("SiliconFlow rerank result was not an object")
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int) or not 0 <= index < len(values):
                raise ProviderError(f"SiliconFlow rerank index is invalid: {index!r}")
            if index in seen:
                raise ProviderError(f"SiliconFlow rerank index is duplicated: {index}")
            if not isinstance(score, (int, float)):
                raise ProviderError(
                    f"SiliconFlow rerank score at index {index} is invalid"
                )
            seen.add(index)
            output.append(
                SiliconFlowRerankResult(
                    index=index,
                    document=values[index],
                    relevance_score=float(score),
                )
            )
        return output
