import asyncio
import json
import re
from pathlib import Path

import pytest

from rec_researcher.core.models import InquiryTask, SourceRecord, WorkState
from rec_researcher.core.settings import Settings
from rec_researcher.providers.mock import MockSearchProvider, MockWebFetcher
from rec_researcher.retrieval.chunker import PassageChunker
from rec_researcher.retrieval.fetcher import FetchResult
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


class OneTaskPlanner:
    async def create_tasks(self, question: str) -> list[InquiryTask]:
        return [InquiryTask(id="task-1", question=question, search_queries=[question])]


class TwoTaskPlanner:
    async def create_tasks(self, question: str) -> list[InquiryTask]:
        return [
            InquiryTask(
                id="task-a", question=f"{question} A", search_queries=[question]
            ),
            InquiryTask(
                id="task-b", question=f"{question} B", search_queries=[question]
            ),
        ]


class SlowPlanner:
    async def create_tasks(self, question: str) -> list[InquiryTask]:
        await asyncio.sleep(0.05)
        return [InquiryTask(id="too-late", question=question)]


class RecordingWriter:
    def __init__(self) -> None:
        self.calls = 0

    def write(self, result: object) -> str:
        self.calls += 1
        return "external report"


class StaticSearchProvider:
    def __init__(self, sources: list[SourceRecord]) -> None:
        self.sources = sources

    async def search(self, query: str, *, limit: int) -> list[SourceRecord]:
        return self.sources[:limit]


class RecordingFetcher:
    def __init__(self, outcomes: dict[str, tuple[bool, str]]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []
        self.active = 0
        self.max_active = 0

    async def fetch(self, source: SourceRecord) -> FetchResult:
        self.calls.append(str(source.url))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        success, text = self.outcomes[source.id]
        return FetchResult(
            source_id=source.id,
            url=str(source.url),
            success=success,
            text=text if success else "",
            error=None if success else text,
        )


def _source(
    identifier: str, url: str, snippet: str = "fallback snippet"
) -> SourceRecord:
    return SourceRecord(
        id=identifier,
        title=identifier,
        url=url,
        snippet=snippet,
        provider="fixture",
    )


def _hybrid(
    tmp_path: Path,
    sources: list[SourceRecord],
    fetcher: RecordingFetcher,
    *,
    chunk_size: int = 40,
) -> ResearchOrchestrator:
    settings = Settings(_env_file=None, chunk_size=chunk_size, chunk_overlap=5)
    return ResearchOrchestrator(
        output_dir=tmp_path,
        planner=OneTaskPlanner(),  # type: ignore[arg-type]
        search_provider=StaticSearchProvider(sources),
        retrieval_mode="hybrid",
        web_fetcher=fetcher,
        passage_chunker=PassageChunker(settings),
        fetch_concurrency=2,
        min_fetched_content_length=10,
    )


async def test_real_mode_requires_explicit_non_mock_composition(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="real mode requires explicit providers"):
        ResearchOrchestrator(output_dir=tmp_path, mode="real")


async def test_normal_run_writes_traceable_outputs(tmp_path: Path) -> None:
    orchestrator = ResearchOrchestrator(output_dir=tmp_path)
    run = await orchestrator.run("序列推荐如何评估？")
    run_dir = tmp_path / run.run_id

    assert (run_dir / "report.md").is_file()
    assert (run_dir / "sources.json").is_file()
    assert (run_dir / "evidence.json").is_file()
    assert (run_dir / "validation.json").is_file()
    assert (run_dir / "run.json").is_file()
    assert json.loads((run_dir / "run.json").read_text())["run_id"] == run.run_id

    report = (run_dir / "report.md").read_text()
    body, references = report.split("## References", maxsplit=1)
    cited = set(re.findall(r"\[(S\d+)\]", body))
    defined = set(re.findall(r"\[(S\d+)\]", references))
    assert cited
    assert cited <= defined
    assert run.output.evidence
    assert json.loads((run_dir / "validation.json").read_text())["valid"]


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


async def test_case_timeout_is_failed_and_skips_external_report(
    tmp_path: Path,
) -> None:
    writer = RecordingWriter()
    run = await ResearchOrchestrator(
        output_dir=tmp_path,
        planner=SlowPlanner(),  # type: ignore[arg-type]
        writer=writer,  # type: ignore[arg-type]
        case_timeout=0.01,
        task_timeout=1,
        report_timeout=1,
    ).run("timeout")

    assert run.status == WorkState.FAILED
    assert writer.calls == 0
    assert "global timeout after 0.01 seconds" in run.output.limitations
    assert run.output.markdown_report != "external report"


async def test_hybrid_fetches_concurrently_and_isolates_failure(tmp_path: Path) -> None:
    sources = [
        _source("good", "https://example.test/good"),
        _source("bad", "https://example.test/bad", "usable failure fallback"),
    ]
    fetcher = RecordingFetcher(
        {
            "good": (True, "A sufficiently long fetched article body."),
            "bad": (False, "boom"),
        }
    )

    run = await _hybrid(tmp_path, sources, fetcher).run("question")

    assert fetcher.max_active == 2
    assert run.output.statistics.fetch_attempts == 2
    assert run.output.statistics.fetch_successes == 1
    assert run.output.statistics.fetch_failures == 1
    assert run.output.statistics.fallback_passages == 1
    assert run.output.statistics.degradation_events == 1
    assert {passage.content_origin for passage in run.output.passages} == {
        "fetched",
        "search_snippet_fallback",
    }
    assert any("fetch fallback for bad" in warning for warning in run.budget.warnings)


async def test_hybrid_fetches_normalized_url_once_and_preserves_source_link(
    tmp_path: Path,
) -> None:
    sources = [
        _source("first", "https://EXAMPLE.test/article/"),
        _source("second", "https://example.test/article#section"),
    ]
    fetcher = RecordingFetcher(
        {"first": (True, "Fetched content long enough to make a linked passage.")}
    )

    run = await _hybrid(tmp_path, sources, fetcher).run("question")

    assert len(fetcher.calls) == 1
    assert [source.id for source in run.output.sources] == ["first"]
    assert all(passage.source_id == "first" for passage in run.output.passages)
    assert run.output.task_results[0].source_ids == ["first", "first"]


async def test_empty_body_falls_back_to_snippet(tmp_path: Path) -> None:
    source = _source("empty", "https://example.test/empty", "snippet survives")
    run = await _hybrid(
        tmp_path, [source], RecordingFetcher({"empty": (True, "")})
    ).run("question")

    assert [passage.text for passage in run.output.passages] == ["snippet survives"]
    assert run.output.passages[0].content_origin == "search_snippet_fallback"
    assert run.output.statistics.fetch_failures == 1


async def test_hybrid_outputs_multiple_source_linked_chunks(tmp_path: Path) -> None:
    source = _source("many", "https://example.test/many")
    body = (
        "First paragraph has useful details.\n\n"
        "Second paragraph also has useful details."
    )
    run = await _hybrid(
        tmp_path, [source], RecordingFetcher({"many": (True, body)}), chunk_size=35
    ).run("question")

    assert len(run.output.passages) > 1
    assert all(passage.source_id == source.id for passage in run.output.passages)
    assert sorted(passage.chunk_index for passage in run.output.passages) == list(
        range(len(run.output.passages))
    )


async def test_snippet_mode_keeps_one_passage_per_source(tmp_path: Path) -> None:
    run = await ResearchOrchestrator(
        output_dir=tmp_path,
        planner=OneTaskPlanner(),  # type: ignore[arg-type]
        search_provider=StaticSearchProvider(
            [_source("one", "https://example.test/one", "original snippet")]
        ),
    ).run("question")

    assert [passage.text for passage in run.output.passages] == ["original snippet"]
    assert run.output.statistics.fetch_attempts == 0


async def test_hybrid_can_cap_final_passages_globally(tmp_path: Path) -> None:
    sources = [
        _source(str(index), f"https://example.test/{index}") for index in range(3)
    ]
    fetcher = RecordingFetcher(
        {
            source.id: (True, f"Fetched article {source.id} with enough useful text.")
            for source in sources
        }
    )
    orchestrator = _hybrid(tmp_path, sources, fetcher)
    orchestrator.final_passage_limit = 2

    run = await orchestrator.run("question")

    assert len(run.output.passages) == 2
    assert run.output.statistics.final_passage_count == 2


async def test_hybrid_isolates_task_and_run_passage_ids(tmp_path: Path) -> None:
    source = _source("shared", "https://example.test/shared", "shared useful text")
    orchestrator = ResearchOrchestrator(
        output_dir=tmp_path,
        planner=TwoTaskPlanner(),  # type: ignore[arg-type]
        search_provider=StaticSearchProvider([source]),
        retrieval_mode="hybrid",
        web_fetcher=RecordingFetcher(
            {"shared": (True, "A sufficiently long shared fetched document body.")}
        ),
        passage_chunker=PassageChunker(
            Settings(_env_file=None, chunk_size=100, chunk_overlap=0)
        ),
        min_fetched_content_length=10,
    )

    first = await orchestrator.run("question")
    second = await orchestrator.run("question")
    first_ids = {passage.id for passage in first.output.passages}
    second_ids = {passage.id for passage in second.output.passages}

    assert len(first_ids) == 2
    assert any(":task-a:" in identifier for identifier in first_ids)
    assert any(":task-b:" in identifier for identifier in first_ids)
    assert first_ids.isdisjoint(second_ids)


async def test_hybrid_run_json_contains_retrieval_statistics(tmp_path: Path) -> None:
    default_database = Path("data/rec_researcher.db")
    database_state = (
        (default_database.stat().st_size, default_database.stat().st_mtime_ns)
        if default_database.exists()
        else None
    )
    orchestrator = ResearchOrchestrator(
        output_dir=tmp_path,
        planner=OneTaskPlanner(),  # type: ignore[arg-type]
        search_provider=MockSearchProvider(),
        retrieval_mode="hybrid",
        web_fetcher=MockWebFetcher(),
        passage_chunker=PassageChunker(Settings(_env_file=None)),
    )

    run = await orchestrator.run("vector recommendation")
    persisted = json.loads((tmp_path / run.run_id / "run.json").read_text())
    statistics = persisted["output"]["statistics"]
    expected = {
        "bm25_candidate_count",
        "dense_candidate_count",
        "fused_candidate_count",
        "reranked_candidate_count",
        "final_passage_count",
        "embedding_calls",
        "embedding_text_count",
        "reranker_calls",
        "degradation_events",
        "retrieval_latency_ms",
        "fetch_latency_ms",
        "embedding_latency_ms",
        "rerank_latency_ms",
    }

    assert expected <= statistics.keys()
    assert statistics["embedding_calls"] == 2
    assert statistics["reranker_calls"] == 1
    assert all(item.selection_stage == "mmr" for item in run.output.evidence)
    current_database_state = (
        (default_database.stat().st_size, default_database.stat().st_mtime_ns)
        if default_database.exists()
        else None
    )
    assert current_database_state == database_state
