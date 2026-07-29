import pytest

from rec_researcher.core.settings import Settings


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
