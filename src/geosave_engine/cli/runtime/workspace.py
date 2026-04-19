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

    Raises `WorkspaceError` if the file is missing or malformed.
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

    project_name = _coerce_string(raw.get("project_name"), "Unknown Project")
    task = _coerce_string(raw.get("task"), "")
    method = _coerce_string(raw.get("method"), "")
    description = _coerce_string(raw.get("description"), "")
    models_value = raw.get("models") or []
    models = [str(item) for item in models_value] if isinstance(models_value, list) else []

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


def _coerce_string(value: Any, fallback: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        return value[0]
    return fallback
