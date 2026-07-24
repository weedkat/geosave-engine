from __future__ import annotations

from typing import TYPE_CHECKING

import toml  # type: ignore[import-untyped]

from geosave_engine.utils.file_ops import safe_copy

from .templates import common_template_dir, templates_dir

if TYPE_CHECKING:
    from .model import Workspace

_REQUIRED_DIRS = ("data", "configs", "artifacts", "logs", "modules", "predictions")


def scaffold_workspace(workspace: Workspace) -> None:
    """Create directories and copy files for one workspace.

    Args:
        workspace: Workspace location and selected template specification.
    """
    for directory_name in _REQUIRED_DIRS:
        (workspace.root / directory_name).mkdir(parents=True, exist_ok=True)

    safe_copy(common_template_dir(), workspace.root)
    with (workspace.root / "geosave.toml").open("w") as file:
        toml.dump(workspace.spec.__dict__, file)

    method_dir = templates_dir() / workspace.spec.project_task / workspace.spec.project_method
    safe_copy(method_dir, workspace.root)
