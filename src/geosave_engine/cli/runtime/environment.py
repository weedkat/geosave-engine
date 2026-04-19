from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from geosave_engine.cli.errors import WorkspaceError
from geosave_engine.cli.io import Console


@dataclass(frozen=True)
class EnvState:
    """Resolved subprocess environment plus the `.env` file we actually used."""

    env: dict[str, str]
    env_file: Path
    env_file_created_from_example: bool = False
    notes: list[str] = field(default_factory=list)


def load_environment(project_dir: Path, current_dir: Path, console: Console) -> EnvState:
    """Discover a `.env`, inject it into `os.environ.copy()`, and set PYTHONPATH.

    Search order: `project_dir/.env`, `project_dir/.env.example`, `current_dir/.env`,
    `current_dir/.env.example`, then the same pair under `current_dir.parent`. If
    only an example is found, copy it into `.env` alongside.
    """
    env_file, created_from_example = _discover_env_file(project_dir, current_dir, console)
    env = os.environ.copy()
    notes: list[str] = []

    if created_from_example:
        notes.append(f"Copied {env_file.with_name('.env.example')} → {env_file}")

    try:
        env.update(_parse_env_file(env_file))
    except Exception as error:
        console.warn(f"Failed to load .env file at {env_file}: {error}")

    env["GEOSAVE_PROJECT_DIR"] = str(project_dir.resolve())
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{project_dir.resolve()}:{existing_pythonpath}"
        if existing_pythonpath
        else str(project_dir.resolve())
    )

    return EnvState(
        env=env,
        env_file=env_file,
        env_file_created_from_example=created_from_example,
        notes=notes,
    )


def _discover_env_file(
    project_dir: Path, current_dir: Path, console: Console
) -> tuple[Path, bool]:
    search_paths = [
        project_dir / ".env",
        project_dir / ".env.example",
        current_dir / ".env",
        current_dir / ".env.example",
        current_dir.parent / ".env",
        current_dir.parent / ".env.example",
    ]

    for path in search_paths:
        if path.is_file() and path.name == ".env":
            return path, False

    for path in search_paths:
        if path.is_file() and path.name == ".env.example":
            target = path.parent / ".env"
            try:
                shutil.copy2(path, target)
            except Exception as error:
                raise WorkspaceError(f"Error copying {path} to {target}: {error}") from error
            console.warn(f"Copied {path} to {target} as .env was missing.")
            return target, True

    raise WorkspaceError(
        "No .env or .env.example file found in "
        f"{project_dir}, {current_dir}, or {current_dir.parent}."
    )


def _parse_env_file(env_file: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    with open(env_file, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            parsed[key.strip()] = value.strip().strip("'\"")
    return parsed
