import json
import re
from pathlib import Path

import pytest

from rec_researcher.core.models import SourceRecord, WorkState
from rec_researcher.providers.mock import MockSearchProvider
from rec_researcher.workflow.orchestrator import ResearchOrchestrator

pytestmark = pytest.mark.asyncio


class FailingSearchProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, *, limit: int) -> list[SourceRecord]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("fixture task failure")
        return await MockSearchProvider().search(query, limit=limit)


class AlwaysFailingSearchProvider:
    async def search(self, query: str, *, limit: int) -> list[SourceRecord]:
        raise RuntimeError("all unavailable")


async def test_normal_run_writes_traceable_outputs(tmp_path: Path) -> None:
    orchestrator = ResearchOrchestrator(output_dir=tmp_path)
    run = await orchestrator.run("序列推荐如何评估？")
    run_dir = tmp_path / run.run_id

    assert (run_dir / "report.md").is_file()
    assert (run_dir / "sources.json").is_file()
    assert (run_dir / "run.json").is_file()
    assert json.loads((run_dir / "run.json").read_text())["run_id"] == run.run_id

    report = (run_dir / "report.md").read_text()
    body, references = report.split("## References", maxsplit=1)
    cited = set(re.findall(r"\[(S\d+)\]", body))
    defined = set(re.findall(r"\[(S\d+)\]", references))
    assert cited
    assert cited <= defined


async def test_task_failure_is_recorded_and_tasks_continue(
    tmp_path: Path,
) -> None:
    run = await ResearchOrchestrator(
        output_dir=tmp_path, search_provider=FailingSearchProvider()
    ).run("推荐系统复现")

    states = [result.state for result in run.output.task_results]
    assert states.count(WorkState.FAILED) == 1
    assert states.count(WorkState.COMPLETED) == 4
    assert "fixture task failure" in run.output.task_results[1].errors[0]


async def test_two_runs_do_not_overwrite_each_other(tmp_path: Path) -> None:
    orchestrator = ResearchOrchestrator(output_dir=tmp_path)
    first = await orchestrator.run("推荐系统")
    second = await orchestrator.run("推荐系统")

    assert first.run_id != second.run_id
    assert (tmp_path / first.run_id / "report.md").exists()
    assert (tmp_path / second.run_id / "report.md").exists()


async def test_all_tasks_fail_but_run_json_is_saved(tmp_path: Path) -> None:
    run = await ResearchOrchestrator(
        output_dir=tmp_path, search_provider=AlwaysFailingSearchProvider()
    ).run("推荐系统")

    assert run.status == WorkState.FAILED
    assert run.output.limitations
    assert run.budget.failed_tasks == [f"task-{index}" for index in range(1, 6)]
    assert run.budget.search_calls == 5
    assert (tmp_path / run.run_id / "run.json").exists()
