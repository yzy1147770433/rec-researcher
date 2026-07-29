"""Command-line interface for local project diagnostics."""

import asyncio
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated, Literal

import typer

from rec_researcher import __version__
from rec_researcher.core.settings import Settings
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
) -> None:
    """Run the research workflow."""

    if mode != "mock":
        typer.echo("Real mode is not implemented.", err=True)
        raise typer.Exit(code=2)
    settings = Settings()
    orchestrator = ResearchOrchestrator(
        output_dir=output_dir or settings.output_dir,
        max_tasks=settings.max_tasks,
        max_sources=settings.max_total_sources,
        sources_per_query=settings.max_sources_per_query,
    )
    try:
        result = asyncio.run(orchestrator.run(question))
    except ValueError as exc:
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
