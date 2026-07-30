import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from rec_researcher import __version__
from rec_researcher.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_doctor_is_offline_by_default() -> None:
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Local dependencies: ok" in result.stdout
    assert "Network checks: skipped" in result.stdout


def test_doctor_real_fails_when_configuration_is_missing() -> None:
    result = runner.invoke(app, ["doctor", "--real"], env={"REC_LLM_API_KEY": ""})

    assert result.exit_code == 1
    assert "Missing real-mode configuration" in result.output


def test_doctor_real_only_checks_configuration_and_redacts_keys() -> None:
    secret = "doctor-complete-secret"
    result = runner.invoke(
        app,
        ["doctor", "--real"],
        env={
            "REC_LLM_BASE_URL": "https://llm.invalid/v1",
            "REC_LLM_API_KEY": secret,
            "REC_LLM_MODEL": "model",
            "REC_TAVILY_API_KEY": "tavily-secret",
        },
    )

    assert result.exit_code == 0
    assert "Real-mode configuration: ok" in result.stdout
    assert "Network checks: skipped" in result.stdout
    assert secret not in result.output


def test_run_mock_creates_output(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "如何评估推荐系统？",
            "--mode",
            "mock",
            "--output-dir",
            str(tmp_path),
            "--max-concurrency",
            "2",
            "--retrieval-concurrency",
            "1",
            "--timeout",
            "2",
            "--max-sources",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert "Run ID:" in result.stdout


def test_run_rejects_empty_question() -> None:
    result = runner.invoke(app, ["run", " ", "--mode", "mock"])

    assert result.exit_code == 2
    assert "must not be empty" in result.output


def test_run_real_fails_early_when_configuration_is_missing(
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "question",
            "--mode",
            "real",
            "--search-provider",
            "tavily",
            "--output-dir",
            str(tmp_path),
        ],
        env={
            "REC_LLM_BASE_URL": "",
            "REC_LLM_API_KEY": "",
            "REC_LLM_MODEL": "",
            "REC_TAVILY_API_KEY": "",
        },
    )

    assert result.exit_code == 2
    assert "Missing real-mode configuration" in result.output
    assert list(tmp_path.iterdir()) == []


def test_run_real_hybrid_does_not_require_siliconflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_run(_orchestrator: object, _question: str) -> object:
        return SimpleNamespace(run_id="hybrid-run")

    monkeypatch.setattr("rec_researcher.cli.ResearchOrchestrator.run", fake_run)
    result = runner.invoke(
        app,
        [
            "run",
            "question",
            "--mode",
            "real",
            "--retrieval-mode",
            "hybrid",
            "--output-dir",
            str(tmp_path),
        ],
        env={
            "REC_LLM_BASE_URL": "https://llm.invalid/v1",
            "REC_LLM_API_KEY": "llm-secret",
            "REC_LLM_MODEL": "model",
            "REC_TAVILY_API_KEY": "tavily-secret",
            "REC_SILICONFLOW_API_KEY": "",
            "REC_EMBEDDING_MODEL": "",
            "REC_RERANKER_MODEL": "",
        },
    )

    assert result.exit_code == 0
    assert "Run ID: hybrid-run" in result.output


def test_run_real_applies_cli_timeout_to_provider_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[float] = []

    def capture_settings(factory: object) -> None:
        captured.append(factory.settings.request_timeout_seconds)  # type: ignore[attr-defined]
        raise ValueError("stop after settings capture")

    monkeypatch.setattr(
        "rec_researcher.cli.ProviderFactory.create_language_model", capture_settings
    )
    result = runner.invoke(
        app,
        ["run", "question", "--mode", "real", "--timeout", "600"],
        env={
            "REC_LLM_BASE_URL": "https://llm.invalid/v1",
            "REC_LLM_API_KEY": "llm-secret",
            "REC_LLM_MODEL": "model",
            "REC_TAVILY_API_KEY": "tavily-secret",
        },
    )

    assert result.exit_code == 2
    assert captured == [600.0]


def test_run_help_lists_retrieval_provider_options() -> None:
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    for option in (
        "--retrieval-mode",
        "--fetch-concurrency",
        "--embedding-provider",
        "--reranker-provider",
        "--vector-store",
    ):
        assert option in result.stdout


def test_run_real_does_not_create_unused_milvus_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "nested" / "configured.db"

    async def fake_run(_orchestrator: object, _question: str) -> object:
        return SimpleNamespace(run_id="test-run")

    monkeypatch.setattr(
        "rec_researcher.cli.ResearchOrchestrator.run",
        fake_run,
    )
    result = runner.invoke(
        app,
        ["run", "question", "--mode", "real", "--output-dir", str(tmp_path)],
        env={
            "REC_LLM_BASE_URL": "https://llm.invalid/v1",
            "REC_LLM_API_KEY": "llm-secret",
            "REC_LLM_MODEL": "model",
            "REC_TAVILY_API_KEY": "tavily-secret",
            "REC_MILVUS_URI": str(database),
        },
    )

    assert result.exit_code == 0
    assert not database.exists()


def test_benchmark_mock_writes_summary(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            "examples/bench/smoke5.jsonl",
            "--mode",
            "mock",
            "--output-dir",
            str(tmp_path),
            "--max-concurrency",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert "Cases: 5/5 successful" in result.stdout
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_cases"] == 5
    assert summary["mean_metrics"]["recall_at_k"] is None
