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


@pytest.mark.parametrize("status", [429, 502, 503, 504])
async def test_retries_transient_status_then_succeeds(status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["Authorization"] == "Bearer test-secret"
        assert request.url.path == "/v1/chat/completions"
        if calls == 1:
            return httpx.Response(status, text="temporary")
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleLanguageModel(_settings(), client=client)
        result = await model.generate("question", system="follow instructions")

    assert result == "done"
    assert calls == 2


@pytest.mark.parametrize("status", [401, 403])
async def test_does_not_retry_auth_errors_and_does_not_expose_key(
    status: int,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, text="echoed test-secret")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleLanguageModel(_settings(), client=client)
        with pytest.raises(ProviderError) as caught:
            await model.generate("question")

    assert calls == 1
    assert f"status {status}" in str(caught.value)
    assert "test-secret" not in str(caught.value)


async def test_retries_network_errors_and_wraps_final_failure() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("secret details", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleLanguageModel(_settings(), client=client)
        with pytest.raises(
            ProviderError, match="LLM network request failed: ReadTimeout"
        ):
            await model.generate("question")

    assert calls == 3


@pytest.mark.parametrize(
    "base_url",
    [
        "https://llm.invalid",
        "https://llm.invalid/",
        "https://llm.invalid/v1/",
        "https://llm.invalid/v1/v1",
    ],
)
async def test_normalizes_base_url_to_exactly_one_v1(base_url: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    settings = _settings().model_copy(update={"llm_base_url": base_url})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await OpenAICompatibleLanguageModel(settings, client=client).generate(
            "question"
        )

    assert result == "done"


@pytest.mark.parametrize("content", [None, ""])
async def test_uses_reasoning_content_when_content_is_empty(
    content: str | None,
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": content, "reasoning_content": "answer"}}
                ]
            },
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        result = await OpenAICompatibleLanguageModel(
            _settings(), client=client
        ).generate("question")

    assert result == "answer"


async def test_markdown_json_code_block_is_parsed() -> None:
    parsed = OpenAICompatibleLanguageModel.parse_json(
        'Here is the result:\n```json\n{"tasks": []}\n```'
    )

    assert parsed == {"tasks": []}
