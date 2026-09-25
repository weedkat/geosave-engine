from __future__ import annotations

from pathlib import Path

from geosave_engine.utils.file_ops import safe_copy

from .templates import COMMON_DIR, TASK_DIR

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


def create_workspace(
    root: Path, task: str | None = None, method: str | None = None
) -> None:
    """Create directories and copy files for one workspace.

    Args:
        root: The root directory for the workspace.
        task: The task for the workspace, or None for a bare workspace.
        method: The method for the workspace, or None for a bare workspace.
    """
    for directory_name in _REQUIRED_DIRS:
        (root / directory_name).mkdir(parents=True, exist_ok=True)

    safe_copy(COMMON_DIR, root, exclude=_EXCLUDE)

    if task and method:
        safe_copy(TASK_DIR / task / method, root, exclude=_EXCLUDE)
