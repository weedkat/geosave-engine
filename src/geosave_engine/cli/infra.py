from __future__ import annotations

import shutil
import subprocess
import importlib.resources as pkg_resources
from pathlib import Path

import typer

infra_app = typer.Typer(help="Manage Geosave dev infrastructure (Docker Compose).")

_COMPOSE_NAME = "docker-compose.yml"
_ENV_EXAMPLE = ".env.example"
_LOCAL_INFRA_DIR = "docker"


def _bundled(filename: str) -> Path:
    return Path(str(pkg_resources.files("geosave_engine") / "infra" / filename))


def _local_compose() -> Path | None:
    p = Path.cwd() / _LOCAL_INFRA_DIR / _COMPOSE_NAME
    return p if p.exists() else None


@infra_app.command()
def init() -> None:
    """Copy docker-compose.yml and .env to ./docker/.

    Run once to get editable local copies. Edit .env to override defaults.
    """
    dest_dir = Path.cwd() / _LOCAL_INFRA_DIR
    dest_dir.mkdir(exist_ok=True, parents=True)
    compose_dest = dest_dir / _COMPOSE_NAME
    env_dest = dest_dir / ".env"

    if compose_dest.exists():
        typer.echo(f"{_COMPOSE_NAME} already exists — skipped.")
    else:
        shutil.copy2(_bundled(_COMPOSE_NAME), compose_dest)
        typer.echo(f"Created docker/{_COMPOSE_NAME}")

    if env_dest.exists():
        typer.echo(".env already exists — skipped.")
    else:
        shutil.copy2(_bundled(_ENV_EXAMPLE), env_dest)
        typer.echo("Created docker/.env from defaults")

    typer.echo("\nNext: geosave infra up")


@infra_app.command()
def up(
    profile: list[str] = typer.Option([], "--profile", "-p", help="Enable a service profile."),
    detach: bool = typer.Option(True, "--detach/--no-detach", "-d/-D", help="Run in background."),
) -> None:
    """Start infrastructure services.

    Uses local docker/docker-compose.yml if present, otherwise the bundled default.

    Examples:
        geosave infra up
        geosave infra up -p mlflow
        geosave infra up -p mlflow -p viz -p db-admin
    """
    _run_compose(["up", *(["-d"] if detach else [])], profile)


@infra_app.command()
def down(
    profile: list[str] = typer.Option([], "--profile", "-p"),
) -> None:
    """Stop infrastructure services."""
    _run_compose(["down"], profile)


@infra_app.command()
def status() -> None:
    """Show running infrastructure containers."""
    _run_compose(["ps"], [])


def _run_compose(args: list[str], profiles: list[str]) -> None:
    profile_flags = [flag for p in profiles for flag in ("--profile", p)]
    compose_path = _local_compose() or _bundled(_COMPOSE_NAME)
    cmd = ["docker", "compose", "-f", str(compose_path), *profile_flags, *args]
    subprocess.run(cmd, check=True)
