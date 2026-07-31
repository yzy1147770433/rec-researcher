import json
from pathlib import Path

from typer.testing import CliRunner

from rec_researcher.cli import app

runner = CliRunner()


def test_run_loads_json_config_and_mock_alias(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"max_tasks": 3, "query_rewrite_count": 2}), encoding="utf-8"
    )
    output = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "run",
            "ranking metrics",
            "--mock",
            "--config",
            str(config),
            "--concurrency",
            "1",
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0
    run_dir = next(output.iterdir())
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert len(payload["output"]["tasks"]) == 3
    assert all(len(item["search_queries"]) == 2 for item in payload["output"]["tasks"])


def test_ablate_runs_all_modes(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "ablate",
            "examples/bench/smoke5.jsonl",
            "--mock",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert (tmp_path / "ablation-summary.md").is_file()
    assert (tmp_path / "ablation-results.json").is_file()
    assert len(json.loads((tmp_path / "ablation-results.json").read_text())) == 6


def test_benchmark_compatibility_shortcut(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["--benchmark", "examples/bench/smoke5.jsonl"],
        env={"REC_OUTPUT_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0
