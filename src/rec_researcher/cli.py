"""Command-line interface for local project diagnostics."""

import asyncio
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated, Literal

import typer

from rec_researcher import __version__
from rec_researcher.core.exceptions import RecResearcherError
from rec_researcher.core.settings import Settings
from rec_researcher.planning.planner import ResearchPlanner
from rec_researcher.providers.llm_http import OpenAICompatibleLanguageModel
from rec_researcher.providers.tavily import TavilySearchProvider
from rec_researcher.reporting.writer import RealReportWriter
from rec_researcher.workflow.orchestrator import ResearchOrchestrator

app = typer.Typer(no_args_is_help=True, help="Research recommender-system questions.")


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
    timeout: Annotated[float | None, typer.Option("--timeout", min=0.001)] = None,
    max_sources: Annotated[int | None, typer.Option("--max-sources", min=1)] = None,
) -> None:
    """Run the research workflow."""

    settings = Settings()
    try:
        if mode == "real":
            missing = settings.missing_real_configuration(
                search_provider=search_provider
            )
            if missing:
                raise ValueError(
                    "Missing real-mode configuration: " + ", ".join(missing)
                )
            llm = OpenAICompatibleLanguageModel(settings)
            search = TavilySearchProvider(settings)
            orchestrator = ResearchOrchestrator(
                output_dir=output_dir or settings.output_dir,
                planner=ResearchPlanner(llm),
                search_provider=search,
                writer=RealReportWriter(llm),
                mode="real",
                max_tasks=settings.max_tasks,
                max_sources=max_sources or settings.max_total_sources,
                sources_per_query=settings.max_sources_per_query,
                max_concurrency=max_concurrency or settings.max_concurrency,
                retrieval_concurrency=retrieval_concurrency
                or settings.max_concurrency,
                timeout=timeout or settings.request_timeout_seconds,
            )
        else:
            orchestrator = ResearchOrchestrator(
                output_dir=output_dir or settings.output_dir,
                max_tasks=settings.max_tasks,
                max_sources=max_sources or settings.max_total_sources,
                sources_per_query=settings.max_sources_per_query,
                max_concurrency=max_concurrency or settings.max_concurrency,
                retrieval_concurrency=retrieval_concurrency
                or settings.max_concurrency,
                timeout=timeout or settings.request_timeout_seconds,
            )
        result = asyncio.run(orchestrator.run(question))
    except (ValueError, RecResearcherError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Run ID: {result.run_id}")
    typer.echo(f"Report: {orchestrator.output_dir / result.run_id / 'report.md'}")


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
