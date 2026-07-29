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
