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


class OpenAICompatibleLanguageModel:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the endpoint from settings without exposing its secret."""

        self.base_url = _configured(settings.llm_base_url, "llm_base_url").rstrip("/")
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
                response = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self.model, "messages": messages},
                    timeout=self._timeout,
                )
                self._raise_for_status(response)
                try:
                    content = response.json()["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    raise LanguageModelResponseError(
                        "LLM response did not contain choices[0].message.content"
                    ) from exc
                if not isinstance(content, str):
                    raise LanguageModelResponseError(
                        "LLM response content was not text"
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

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        body = response.text[:_MAX_ERROR_BODY_LENGTH]
        message = f"LLM request failed with status {response.status_code}: {body}"
        if response.status_code in _RETRYABLE_STATUS_CODES:
            raise _RetryableHTTPError(message)
        raise ProviderError(message)
