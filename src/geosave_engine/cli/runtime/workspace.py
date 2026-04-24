from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml

from geosave_engine.cli.errors import WorkspaceError
from geosave_engine.cli.io import Console


@dataclass(frozen=True)
class Workspace:
    """Parsed view of a workspace's `geosave.toml`."""

    root: Path
    project_name: str
    task: str
    method: str
    description: str
    models: list[str]
    raw: dict[str, Any]


def load_workspace(project_dir: Path) -> Workspace:
    """Read `geosave.toml` from `project_dir` into a `Workspace`.

    ``project_name``, ``task``, and ``method`` are required string fields.
    ``description`` is optional (defaults to ``""``).
    ``models`` is optional (defaults to ``[]``); each element must be a string.

    Raises:
        WorkspaceError: if the file is missing, unreadable, or any required
            field is absent or has the wrong type.
    """
    config_path = project_dir / "geosave.toml"
    if not config_path.is_file():
        raise WorkspaceError(
            f"GeoSave workspace not found: expected {config_path}. "
            "Make sure you are in a directory created by 'geosave build' "
            "or pass the correct path."
        )

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = toml.load(handle)
    except Exception as error:
        raise WorkspaceError(f"Could not read {config_path}: {error}") from error

    project_name = _require_string(raw, "project_name", config_path)
    task         = _require_string(raw, "task",         config_path)
    method       = _require_string(raw, "method",       config_path)

    description_raw = raw.get("description", "")
    if not isinstance(description_raw, str):
        raise WorkspaceError(
            f"{config_path}: 'description' must be a string, got {type(description_raw).__name__!r}"
        )
    description = description_raw

    models_raw = raw.get("models", [])
    if not isinstance(models_raw, list):
        raise WorkspaceError(
            f"{config_path}: 'models' must be a list, got {type(models_raw).__name__!r}"
        )
    if not all(isinstance(item, str) for item in models_raw):
        raise WorkspaceError(f"{config_path}: every item in 'models' must be a string")
    models: list[str] = models_raw

    return Workspace(
        root=project_dir,
        project_name=project_name,
        task=task,
        method=method,
        description=description,
        models=models,
        raw=raw,
    )


def announce_workspace(workspace: Workspace, console: Console) -> None:
    """Print a short confirmation that the workspace was found."""
    console.info(f"Found GeoSave project workspace: '{workspace.project_name}'")


def _require_string(raw: dict[str, Any], key: str, config_path: Path) -> str:
    value = raw.get(key)
    if value is None:
        raise WorkspaceError(f"{config_path}: required field '{key}' is missing")
    if not isinstance(value, str):
        raise WorkspaceError(
            f"{config_path}: '{key}' must be a string, got {type(value).__name__!r}"
        )
    return value
