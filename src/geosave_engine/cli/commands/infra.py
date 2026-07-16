from __future__ import annotations

import importlib.resources as pkg_resources
import shutil
import subprocess
from pathlib import Path

import typer

infra_app = typer.Typer(help="Manage GeoSave dev infrastructure (Docker Compose).")

_COMPOSE_NAME = "docker-compose.yml"
_ENV_EXAMPLE = ".env.example"
_LOCAL_INFRA_DIR = "docker"


def _bundled(filename: str) -> Path:
    return Path(str(pkg_resources.files("geosave_engine") / "infra" / filename))


def _local_compose() -> Path | None:
    path = Path.cwd() / _LOCAL_INFRA_DIR / _COMPOSE_NAME
    return path if path.exists() else None


@infra_app.command()
def init() -> None:
    """Copy docker-compose.yml and .env to ./docker/."""
    destination = Path.cwd() / _LOCAL_INFRA_DIR
    destination.mkdir(exist_ok=True, parents=True)
    compose_destination = destination / _COMPOSE_NAME
    env_destination = destination / ".env"

    if compose_destination.exists():
        typer.echo(f"{_COMPOSE_NAME} already exists — skipped.")
    else:
        shutil.copy2(_bundled(_COMPOSE_NAME), compose_destination)
        typer.echo(f"Created docker/{_COMPOSE_NAME}")

    if env_destination.exists():
        typer.echo(".env already exists — skipped.")
    else:
        shutil.copy2(_bundled(_ENV_EXAMPLE), env_destination)
        typer.echo("Created docker/.env from defaults")

    typer.echo("\nNext: geosave infra up")


@infra_app.command()
def up(
    profile: list[str] = typer.Option([], "--profile", "-p", help="Enable a service profile."),
    detach: bool = typer.Option(
        True,
        "--detach/--no-detach",
        "-d/-D",
        help="Run in background.",
    ),
) -> None:
    """Start infrastructure containers."""
    _run_compose(["up", *(["-d"] if detach else [])], profile)


@infra_app.command()
def down(profile: list[str] = typer.Option([], "--profile", "-p")) -> None:
    """Stop infrastructure containers."""
    _run_compose(["down"], profile)


@infra_app.command()
def status() -> None:
    """Show running infrastructure containers."""
    _run_compose(["ps"], [])


def _run_compose(args: list[str], profiles: list[str]) -> None:
    profile_flags = [flag for profile in profiles for flag in ("--profile", profile)]
    compose_path = _local_compose() or _bundled(_COMPOSE_NAME)
    command = ["docker", "compose", "-f", str(compose_path), *profile_flags, *args]
    subprocess.run(command, check=True)
