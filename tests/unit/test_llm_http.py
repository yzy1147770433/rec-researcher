import httpx
import pytest

from rec_researcher.core.exceptions import ProviderError
from rec_researcher.core.settings import Settings
from rec_researcher.providers.llm_http import OpenAICompatibleLanguageModel

pytestmark = pytest.mark.asyncio


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_base_url="https://llm.invalid/v1",
        llm_api_key="test-secret",
        llm_model="test-model",
        max_retries=2,
    )


async def test_retries_429_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["Authorization"] == "Bearer test-secret"
        assert request.url.path == "/v1/chat/completions"
        if calls == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleLanguageModel(_settings(), client=client)
        result = await model.generate("question", system="follow instructions")

    assert result == "done"
    assert calls == 2


async def test_does_not_retry_401_and_does_not_expose_key() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="unauthorized")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleLanguageModel(_settings(), client=client)
        with pytest.raises(ProviderError) as caught:
            await model.generate("question")

    assert calls == 1
    assert "status 401" in str(caught.value)
    assert "test-secret" not in str(caught.value)


async def test_markdown_json_code_block_is_parsed() -> None:
    parsed = OpenAICompatibleLanguageModel.parse_json(
        'Here is the result:\n```json\n{"tasks": []}\n```'
    )

    assert parsed == {"tasks": []}
