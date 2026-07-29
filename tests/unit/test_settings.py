from pathlib import Path

import pytest

from rec_researcher.core.settings import Settings


def test_settings_load_all_real_provider_fields_from_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_names = (
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "TAVILY_BASE_URL",
        "TAVILY_API_KEY",
        "SILICONFLOW_BASE_URL",
        "SILICONFLOW_API_KEY",
        "EMBEDDING_MODEL",
        "RERANKER_MODEL",
        "MILVUS_URI",
    )
    for name in field_names:
        monkeypatch.delenv(f"REC_{name}", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "REC_LLM_BASE_URL=https://llm.invalid/v1",
                "REC_LLM_API_KEY=llm-secret",
                "REC_LLM_MODEL=llm-model",
                "REC_TAVILY_BASE_URL=https://tavily.invalid",
                "REC_TAVILY_API_KEY=tavily-secret",
                "REC_SILICONFLOW_BASE_URL=https://siliconflow.invalid/v1",
                "REC_SILICONFLOW_API_KEY=siliconflow-secret",
                "REC_EMBEDDING_MODEL=embedding-model",
                "REC_RERANKER_MODEL=reranker-model",
                "REC_MILVUS_URI=./test.db",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.llm_base_url == "https://llm.invalid/v1"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "llm-secret"
    assert settings.llm_model == "llm-model"
    assert settings.tavily_base_url == "https://tavily.invalid"
    assert settings.tavily_api_key is not None
    assert settings.tavily_api_key.get_secret_value() == "tavily-secret"
    assert settings.siliconflow_base_url == "https://siliconflow.invalid/v1"
    assert settings.siliconflow_api_key is not None
    assert settings.siliconflow_api_key.get_secret_value() == "siliconflow-secret"
    assert settings.embedding_model == "embedding-model"
    assert settings.reranker_model == "reranker-model"
    assert settings.milvus_uri == "./test.db"


def test_settings_use_rec_environment_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REC_MAX_CONCURRENCY", "7")
    monkeypatch.setenv("MAX_CONCURRENCY", "99")

    settings = Settings(_env_file=None)

    assert settings.max_concurrency == 7


def test_safe_summary_never_exposes_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "super-secret-value"
    monkeypatch.setenv("REC_LLM_API_KEY", secret)

    settings = Settings(_env_file=None)
    summary = settings.safe_summary()

    assert summary["llm_api_key"] is True
    assert secret not in str(summary)
    assert all(isinstance(value, bool) for value in summary.values())


def test_real_configuration_reports_missing_fields() -> None:
    settings = Settings(_env_file=None, llm_api_key="")

    assert "llm_api_key" in settings.missing_real_configuration()
    assert settings.safe_summary()["llm_api_key"] is False
