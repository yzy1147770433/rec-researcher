"""Command-line interface for local project diagnostics."""

from importlib.util import find_spec
from typing import Annotated

import typer

from rec_researcher import __version__
from rec_researcher.core.settings import Settings

app = typer.Typer(no_args_is_help=True, help="Research recommender-system questions.")


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
