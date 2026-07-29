from pathlib import Path

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


def test_run_mock_creates_output(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["run", "如何评估推荐系统？", "--mode", "mock", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Run ID:" in result.stdout


def test_run_rejects_empty_question() -> None:
    result = runner.invoke(app, ["run", " ", "--mode", "mock"])

    assert result.exit_code == 2
    assert "must not be empty" in result.output
