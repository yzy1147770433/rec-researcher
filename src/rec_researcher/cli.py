"""Command-line interface for local project diagnostics."""

import asyncio
import json
import tomllib
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated, Literal

import typer

from rec_researcher import __version__
from rec_researcher.core.exceptions import RecResearcherError
from rec_researcher.core.models import ResearchRun
from rec_researcher.core.settings import (
    EmbeddingProvider,
    RerankerProvider,
    RetrievalMode,
    Settings,
    VectorStore,
)
from rec_researcher.evaluation.runner import (
    AblationName,
    BenchmarkCase,
    BenchmarkRunner,
    BenchmarkSummary,
    comparison_markdown,
)
from rec_researcher.evidence.claim_verifier import LLMClaimEvidenceVerifier
from rec_researcher.planning.planner import ResearchPlanner
from rec_researcher.planning.query_rewrite import QueryRewriter
from rec_researcher.providers.factory import ProviderFactory
from rec_researcher.providers.mock import MockWebFetcher
from rec_researcher.reporting.writer import RealReportWriter
from rec_researcher.retrieval.chunker import PassageChunker
from rec_researcher.retrieval.fetcher import AsyncWebFetcher
from rec_researcher.workflow.orchestrator import ResearchOrchestrator

app = typer.Typer(no_args_is_help=True, help="Research recommender-system questions.")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    benchmark_path: Annotated[
        Path | None,
        typer.Option("--benchmark", help="Compatibility shortcut for benchmark."),
    ] = None,
) -> None:
    """Dispatch the optional legacy benchmark shortcut."""

    if benchmark_path is None or ctx.invoked_subcommand is not None:
        return
    benchmark(
        benchmark_path=benchmark_path,
        mode="mock",
        output_dir=None,
        max_concurrency=3,
        retrieval_mode="snippet",
        request_timeout=None,
        task_timeout=None,
        case_timeout=None,
        report_timeout=None,
        max_retries=None,
        vector_store=None,
        retrieval_concurrency=None,
        fetch_concurrency=None,
        resume=False,
        config=None,
        mock=True,
    )


def _settings_from_file(path: Path | None) -> Settings:
    """Load JSON or TOML settings while retaining environment defaults."""

    settings = Settings()
    if path is None:
        return settings
    try:
        raw = path.read_text(encoding="utf-8")
        payload = (
            tomllib.loads(raw) if path.suffix.casefold() == ".toml" else json.loads(raw)
        )
        if not isinstance(payload, dict):
            raise ValueError("configuration root must be an object")
        values = payload.get("rec_researcher", payload)
        if not isinstance(values, dict):
            raise ValueError("rec_researcher configuration must be an object")
        merged = settings.model_dump()
        merged.update(values)
        return Settings.model_validate(merged)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise typer.BadParameter(f"invalid config file {path}: {exc}") from exc


@app.command()
def benchmark(
    benchmark_path: Annotated[Path, typer.Argument(help="JSONL benchmark file.")],
    mode: Annotated[
        Literal["mock", "real"], typer.Option(help="Benchmark provider mode.")
    ] = "mock",
    output_dir: Annotated[
        Path | None, typer.Option(help="Directory for benchmark artifacts.")
    ] = None,
    max_concurrency: Annotated[int, typer.Option("--max-concurrency", min=1)] = 3,
    retrieval_mode: Annotated[
        RetrievalMode, typer.Option("--retrieval-mode", help="Retrieval strategy.")
    ] = "snippet",
    request_timeout: Annotated[
        float | None, typer.Option("--request-timeout", min=0.001)
    ] = None,
    task_timeout: Annotated[
        float | None, typer.Option("--task-timeout", min=0.001)
    ] = None,
    case_timeout: Annotated[
        float | None, typer.Option("--case-timeout", min=0.001)
    ] = None,
    report_timeout: Annotated[
        float | None, typer.Option("--report-timeout", min=0.001)
    ] = None,
    max_retries: Annotated[int | None, typer.Option("--max-retries", min=0)] = None,
    vector_store: Annotated[VectorStore | None, typer.Option("--vector-store")] = None,
    retrieval_concurrency: Annotated[
        int | None, typer.Option("--retrieval-concurrency", min=1)
    ] = None,
    fetch_concurrency: Annotated[
        int | None, typer.Option("--fetch-concurrency", min=1)
    ] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    mock: Annotated[
        bool, typer.Option("--mock", help="Alias for --mode mock.")
    ] = False,
) -> None:
    """Run a failure-isolated lightweight benchmark."""

    settings = _settings_from_file(config)
    if mock:
        mode = "mock"
    updates: dict[str, object] = {}
    if request_timeout is not None:
        updates["request_timeout_seconds"] = request_timeout
    if task_timeout is not None:
        updates["task_timeout_seconds"] = task_timeout
    if case_timeout is not None:
        updates["case_timeout_seconds"] = case_timeout
    if report_timeout is not None:
        updates["report_timeout_seconds"] = report_timeout
    if max_retries is not None:
        updates["max_retries"] = max_retries
    settings = settings.model_copy(update=updates)
    destination = output_dir or (
        settings.output_dir / "benchmarks" / benchmark_path.stem
    )
    try:
        case_executor = None
        if mode == "real":
            factory = ProviderFactory(
                settings,
                mode="real",
                retrieval_mode=retrieval_mode,
                vector_store=vector_store,
            )
            factory.validate_configuration()

            async def execute_real(case: BenchmarkCase) -> ResearchRun:
                question = case.question
                case_id = case.id
                case_factory = ProviderFactory(
                    settings,
                    mode="real",
                    retrieval_mode=retrieval_mode,
                    vector_store=vector_store,
                )
                llm = case_factory.create_language_model()
                search = case_factory.create_search_provider()
                pipeline = case_factory.create_retrieval_pipeline()
                fetcher = (
                    AsyncWebFetcher(settings) if retrieval_mode != "snippet" else None
                )
                orchestrator = ResearchOrchestrator(
                    output_dir=destination / "runs" / case_id,
                    planner=ResearchPlanner(llm, max_tasks=settings.max_tasks),
                    search_provider=search,
                    writer=RealReportWriter(llm),
                    mode="real",
                    max_tasks=settings.max_tasks,
                    max_sources=settings.max_total_sources,
                    sources_per_query=settings.max_sources_per_query,
                    max_concurrency=settings.max_concurrency,
                    retrieval_concurrency=(
                        retrieval_concurrency or settings.max_concurrency
                    ),
                    fetch_concurrency=fetch_concurrency or settings.fetch_concurrency,
                    task_timeout=settings.task_timeout_seconds,
                    case_timeout=settings.case_timeout_seconds,
                    report_timeout=settings.report_timeout_seconds,
                    evidence_excerpt_length=settings.evidence_excerpt_length,
                    retrieval_mode=retrieval_mode,
                    web_fetcher=fetcher,
                    passage_chunker=(
                        PassageChunker(settings)
                        if retrieval_mode != "snippet"
                        else None
                    ),
                    retrieval_pipeline=pipeline,
                    query_rewriter=QueryRewriter(
                        max_queries=settings.query_rewrite_count
                    ),
                    claim_verifier=(
                        LLMClaimEvidenceVerifier(llm)
                        if settings.claim_verifier == "llm"
                        else None
                    ),
                )
                try:
                    return await orchestrator.run(question)
                finally:
                    await _close_resources(fetcher, pipeline, search, llm)

            case_executor = execute_real
        summary = asyncio.run(
            BenchmarkRunner(
                output_dir=destination,
                mode=mode,
                max_concurrency=max_concurrency,
                case_executor=case_executor,
                ablation=(
                    AblationName.HYBRID_RERANK_MMR
                    if retrieval_mode == "hybrid"
                    else AblationName(retrieval_mode)
                ),
                resume=resume,
                execution_config={
                    "max_concurrency": max_concurrency,
                    "request_timeout": settings.request_timeout_seconds,
                    "task_timeout": settings.task_timeout_seconds,
                    "case_timeout": settings.case_timeout_seconds,
                    "report_timeout": settings.report_timeout_seconds,
                    "max_retries": settings.max_retries,
                    "vector_store": vector_store or settings.vector_store,
                    "retrieval_concurrency": (
                        retrieval_concurrency or settings.max_concurrency
                    ),
                    "fetch_concurrency": fetch_concurrency
                    or settings.fetch_concurrency,
                    "llm_base_url": settings.llm_base_url,
                    "llm_model": settings.llm_model,
                    "tavily_base_url": settings.tavily_base_url,
                    "siliconflow_base_url": settings.siliconflow_base_url,
                    "embedding_provider": settings.embedding_provider,
                    "embedding_model": settings.embedding_model,
                    "reranker_provider": settings.reranker_provider,
                    "reranker_model": settings.reranker_model,
                    "retrieval_top_k": settings.retrieval_top_k,
                    "rerank_top_k": settings.rerank_top_k,
                    "mmr_top_k": settings.mmr_top_k,
                    "rrf_k": settings.rrf_k,
                    "mmr_lambda": settings.mmr_lambda,
                    "llm_cost_per_call": settings.llm_cost_per_call,
                    "search_cost_per_call": settings.search_cost_per_call,
                    "embedding_cost_per_call": settings.embedding_cost_per_call,
                    "reranker_cost_per_call": settings.reranker_cost_per_call,
                },
            ).run(benchmark_path)
        )
    except (OSError, ValueError, RecResearcherError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Cases: {summary.successful_cases}/{summary.total_cases} successful")
    typer.echo(f"Summary: {destination / 'summary.json'}")
    if summary.failed_cases:
        raise typer.Exit(code=1)


async def _close_resources(*resources: object | None) -> None:
    """Best-effort close all provider resources without masking case results."""

    for resource in resources:
        if resource is None:
            continue
        close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
        if close is None:
            continue
        try:
            result = close()
            if hasattr(result, "__await__"):
                await result
        except Exception:  # noqa: BLE001 - cleanup must not mask benchmark results
            continue


@app.command()
def run(
    question: Annotated[str, typer.Argument(help="Research question.")],
    mode: Annotated[
        Literal["mock", "real"], typer.Option(help="Provider mode.")
    ] = "mock",
    output_dir: Annotated[
        Path | None, typer.Option(help="Directory for run artifacts.")
    ] = None,
    search_provider: Annotated[
        Literal["tavily"], typer.Option(help="Search provider for real mode.")
    ] = "tavily",
    max_concurrency: Annotated[
        int | None, typer.Option("--max-concurrency", min=1)
    ] = None,
    retrieval_concurrency: Annotated[
        int | None, typer.Option("--retrieval-concurrency", min=1)
    ] = None,
    fetch_concurrency: Annotated[
        int | None, typer.Option("--fetch-concurrency", min=1)
    ] = None,
    timeout: Annotated[float | None, typer.Option("--timeout", min=0.001)] = None,
    max_sources: Annotated[int | None, typer.Option("--max-sources", min=1)] = None,
    retrieval_mode: Annotated[
        RetrievalMode, typer.Option("--retrieval-mode", help="Retrieval strategy.")
    ] = "snippet",
    embedding_provider: Annotated[
        EmbeddingProvider,
        typer.Option("--embedding-provider", help="Embedding provider."),
    ] = "siliconflow",
    reranker_provider: Annotated[
        RerankerProvider,
        typer.Option("--reranker-provider", help="Reranking provider."),
    ] = "siliconflow",
    vector_store: Annotated[
        VectorStore, typer.Option("--vector-store", help="Vector index backend.")
    ] = "milvus",
    top_k: Annotated[int | None, typer.Option("--top-k", min=1)] = None,
    rerank_top_k: Annotated[int | None, typer.Option("--rerank-top-k", min=1)] = None,
    mmr_lambda: Annotated[
        float | None, typer.Option("--mmr-lambda", min=0.0, max=1.0)
    ] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    mock: Annotated[
        bool, typer.Option("--mock", help="Alias for --mode mock.")
    ] = False,
    concurrency: Annotated[int | None, typer.Option("--concurrency", min=1)] = None,
) -> None:
    """Run the research workflow."""

    settings = _settings_from_file(config).model_copy(
        update={
            key: value
            for key, value in {
                "retrieval_top_k": top_k,
                "bm25_top_k": top_k,
                "dense_top_k": top_k,
                "rerank_top_k": rerank_top_k,
                "mmr_lambda": mmr_lambda,
            }.items()
            if value is not None
        }
    )
    if mock:
        mode = "mock"
    if concurrency is not None:
        max_concurrency = concurrency
    try:
        if mode == "real":
            provider_settings = (
                settings.model_copy(update={"request_timeout_seconds": timeout})
                if timeout is not None
                else settings
            )
            factory = ProviderFactory(
                provider_settings,
                mode=mode,
                retrieval_mode=retrieval_mode,
                embedding_provider=embedding_provider,
                reranker_provider=reranker_provider,
                vector_store=vector_store,
            )
            factory.validate_configuration()
            llm = factory.create_language_model()
            search = factory.create_search_provider()
            fetcher = (
                AsyncWebFetcher(provider_settings)
                if retrieval_mode != "snippet"
                else None
            )
            orchestrator = ResearchOrchestrator(
                output_dir=output_dir or settings.output_dir,
                planner=ResearchPlanner(llm, max_tasks=settings.max_tasks),
                search_provider=search,
                writer=RealReportWriter(llm),
                mode="real",
                max_tasks=settings.max_tasks,
                max_sources=max_sources or settings.max_total_sources,
                sources_per_query=settings.max_sources_per_query,
                max_concurrency=max_concurrency or settings.max_concurrency,
                retrieval_concurrency=(
                    retrieval_concurrency or settings.max_concurrency
                ),
                fetch_concurrency=fetch_concurrency or settings.fetch_concurrency,
                timeout=timeout or settings.request_timeout_seconds,
                evidence_excerpt_length=settings.evidence_excerpt_length,
                retrieval_mode=retrieval_mode,
                web_fetcher=fetcher,
                passage_chunker=(
                    PassageChunker(provider_settings)
                    if retrieval_mode != "snippet"
                    else None
                ),
                retrieval_pipeline=factory.create_retrieval_pipeline(),
                query_rewriter=QueryRewriter(max_queries=settings.query_rewrite_count),
                claim_verifier=(
                    LLMClaimEvidenceVerifier(llm)
                    if settings.claim_verifier == "llm"
                    else None
                ),
            )

            async def run_real() -> object:
                try:
                    return await orchestrator.run(question)
                finally:
                    if fetcher is not None:
                        await fetcher.aclose()

            result = asyncio.run(run_real())
        else:
            factory = ProviderFactory(
                settings, mode="mock", retrieval_mode=retrieval_mode
            )
            fetcher = MockWebFetcher() if retrieval_mode != "snippet" else None
            orchestrator = ResearchOrchestrator(
                output_dir=output_dir or settings.output_dir,
                planner=ResearchPlanner(max_tasks=settings.max_tasks),
                max_tasks=settings.max_tasks,
                max_sources=max_sources or settings.max_total_sources,
                sources_per_query=settings.max_sources_per_query,
                max_concurrency=max_concurrency or settings.max_concurrency,
                retrieval_concurrency=retrieval_concurrency or settings.max_concurrency,
                fetch_concurrency=fetch_concurrency or settings.fetch_concurrency,
                timeout=timeout or settings.request_timeout_seconds,
                evidence_excerpt_length=settings.evidence_excerpt_length,
                retrieval_mode=retrieval_mode,
                web_fetcher=fetcher,
                passage_chunker=(
                    PassageChunker(settings) if retrieval_mode != "snippet" else None
                ),
                retrieval_pipeline=factory.create_retrieval_pipeline(),
                query_rewriter=QueryRewriter(max_queries=settings.query_rewrite_count),
            )
            result = asyncio.run(orchestrator.run(question))
    except (ValueError, RecResearcherError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Run ID: {result.run_id}")
    typer.echo(f"Report: {orchestrator.output_dir / result.run_id / 'report.md'}")


@app.command()
def ablate(
    benchmark_path: Annotated[Path, typer.Argument(help="JSONL benchmark file.")],
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path(
        "outputs/ablations"
    ),
    mode: Annotated[Literal["mock", "real"], typer.Option("--mode")] = "mock",
    max_concurrency: Annotated[int, typer.Option("--concurrency", min=1)] = 3,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    mock: Annotated[bool, typer.Option("--mock")] = False,
) -> None:
    """Run all six retrieval modes with one dataset and configuration."""

    if mock:
        mode = "mock"
    summaries: list[BenchmarkSummary] = []
    for ablation_name in AblationName:
        destination = output_dir / ablation_name.value
        benchmark(
            benchmark_path=benchmark_path,
            mode=mode,
            output_dir=destination,
            max_concurrency=max_concurrency,
            retrieval_mode=ablation_name.value,
            request_timeout=None,
            task_timeout=None,
            case_timeout=None,
            report_timeout=None,
            max_retries=None,
            vector_store=None,
            retrieval_concurrency=None,
            fetch_concurrency=None,
            resume=resume,
            config=config,
            mock=mock,
        )
        summaries.append(
            BenchmarkSummary.model_validate_json(
                (destination / "summary.json").read_text(encoding="utf-8")
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    table = comparison_markdown(summaries)
    (output_dir / "ablation-summary.md").write_text(table, encoding="utf-8")
    (output_dir / "ablation-results.json").write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in summaries],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"Ablation summary: {output_dir / 'ablation-summary.md'}")


@app.command()
def version() -> None:
    """Print the installed RecResearcher version."""

    typer.echo(__version__)


@app.command()
def doctor(
    real: Annotated[
        bool,
        typer.Option("--real", help="Check configuration required by real providers."),
    ] = False,
) -> None:
    """Check local dependencies and configuration without network requests."""

    required_modules = ("pydantic", "pydantic_settings", "typer")
    missing_modules = [name for name in required_modules if find_spec(name) is None]
    if missing_modules:
        typer.echo(
            f"Missing local dependencies: {', '.join(missing_modules)}", err=True
        )
        raise typer.Exit(code=1)

    settings = Settings()
    typer.echo("Local dependencies: ok")
    typer.echo("Configuration fields: ok")

    if real:
        missing_settings = settings.missing_real_configuration()
        if missing_settings:
            typer.echo(
                f"Missing real-mode configuration: {', '.join(missing_settings)}",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo("Real-mode configuration: ok")

    typer.echo("Network checks: skipped")
