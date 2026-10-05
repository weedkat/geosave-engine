from __future__ import annotations

from pathlib import Path

from geosave_engine.cli.core.copy import safe_copy

from .templates import COMMON_DIR, WORKSPACES_DIR, get_workspaces

_REQUIRED_DIRS = frozenset(
    (
        "artifacts",
        "configs",
        "data",
        "logs",
        "modules",
        "notebooks",
        "predictions",
        "scripts",
    )
)
_EXCLUDE = frozenset(("__pycache__", ".ipynb_checkpoints", "description.txt"))


def create_workspace(root: Path, workspace: str | None = None) -> None:
    """Create a workspace from shared startup and an optional starter.

    Args:
        root: Destination workspace directory.
        workspace: Bundled starter name. None creates a bare workspace.

    Raises:
        ValueError: The workspace starter does not exist.
    """
    if workspace is not None and workspace not in get_workspaces():
        raise ValueError(f"Workspace {workspace!r} is not a valid workspace.")
    for directory_name in _REQUIRED_DIRS:
        (root / directory_name).mkdir(parents=True, exist_ok=True)

    safe_copy(COMMON_DIR, root, exclude=_EXCLUDE)
    if workspace is not None:
        safe_copy(WORKSPACES_DIR / workspace, root, exclude=_EXCLUDE)
