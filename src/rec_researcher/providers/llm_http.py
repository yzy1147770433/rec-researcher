"""OpenAI-compatible asynchronous language-model provider."""

from __future__ import annotations

import json
import re
from typing import TypeAlias, cast

import httpx
from pydantic import SecretStr
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt

from rec_researcher.core.exceptions import LanguageModelResponseError, ProviderError
from rec_researcher.core.settings import Settings

_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_MAX_ERROR_BODY_LENGTH = 500
JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


class _RetryableHTTPError(ProviderError):
    """Internal marker for HTTP errors that are safe to retry."""


def _configured(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"{name} must be configured")
    return value


def _normalize_v1_base_url(value: str) -> str:
    """Return an API base ending in exactly one ``/v1`` segment."""

    normalized = value.rstrip("/")
    while normalized.endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return f"{normalized}/v1"


class OpenAICompatibleLanguageModel:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the endpoint from settings without exposing its secret."""

        self.base_url = _normalize_v1_base_url(
            _configured(settings.llm_base_url, "llm_base_url")
        )
        self.model = _configured(settings.llm_model, "llm_model")
        secret = settings.llm_api_key
        if not isinstance(secret, SecretStr) or not secret.get_secret_value():
            raise ValueError("llm_api_key must be configured")
        self._api_key = secret.get_secret_value()
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._timeout = settings.request_timeout_seconds
        self._max_attempts = settings.max_retries + 1

    async def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Generate text, retrying only explicitly transient status codes."""

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            retry=retry_if_exception(lambda exc: isinstance(exc, _RetryableHTTPError)),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                try:
                    response = await self._client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json={"model": self.model, "messages": messages},
                        timeout=self._timeout,
                    )
                except httpx.RequestError as exc:
                    raise _RetryableHTTPError(
                        "LLM network request failed: " + type(exc).__name__
                    ) from exc
                self._raise_for_status(response)
                try:
                    message = response.json()["choices"][0]["message"]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise LanguageModelResponseError(
                        "LLM response did not contain choices[0].message"
                    ) from exc
                if not isinstance(message, dict):
                    raise LanguageModelResponseError(
                        "LLM response choices[0].message was not an object"
                    )
                content = message.get("content")
                if content is None or content == "":
                    content = message.get("reasoning_content")
                if not isinstance(content, str):
                    raise LanguageModelResponseError(
                        "LLM response content and reasoning_content were not text"
                    )
                return content
        raise AssertionError("retry loop completed without a result")

    async def aclose(self) -> None:
        """Close the internally-created HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def parse_json(text: str) -> JsonValue:
        """Parse JSON from plain text or a Markdown JSON code fence."""

        match = _JSON_FENCE.search(text)
        candidate = match.group(1) if match else text.strip()
        try:
            return cast(JsonValue, json.loads(candidate))
        except json.JSONDecodeError as exc:
            raise LanguageModelResponseError(
                f"LLM response is not valid JSON: {exc.msg} at line "
                f"{exc.lineno} column {exc.colno}"
            ) from exc

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        body = response.text[:_MAX_ERROR_BODY_LENGTH].replace(
            self._api_key, "[REDACTED]"
        )
        message = f"LLM request failed with status {response.status_code}: {body}"
        if response.status_code in _RETRYABLE_STATUS_CODES:
            raise _RetryableHTTPError(message)
        raise ProviderError(message)
