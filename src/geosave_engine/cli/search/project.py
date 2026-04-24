from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectLayout:
    """Absolute paths to conventional subdirectories of a workspace."""

    root: Path

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"
    
    @property
    def noteooks_dir(self) -> Path:
        return self.root


def find_configs(layout: ProjectLayout) -> list[Path]:
    """Return all `.yaml`/`.yml` files directly under `configs/`."""
    configs_dir = layout.configs_dir
    if not configs_dir.is_dir():
        return []
    return sorted(list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml")))


def find_artifact_parents(layout: ProjectLayout) -> list[Path]:
    """Return artifact sub-directories that contain at least one YAML config."""
    artifacts_dir = layout.artifacts_dir
    if not artifacts_dir.is_dir():
        return []
    return [
        directory
        for directory in sorted(artifacts_dir.iterdir())
        if directory.is_dir()
        and (list(directory.glob("*.yaml")) or list(directory.glob("*.yml")))
    ]


def find_entrypoint(layout: ProjectLayout, script_name: str = "main.py") -> Path | None:
    """Recursively locate the workspace entry script (default `main.py`)."""
    matches = list(layout.root.rglob(script_name))
    return matches[0] if matches else None


def resolve_user_script(layout: ProjectLayout, script_name: str) -> Path | None:
    """Resolve a user-supplied script name inside `scripts/`.

    Accepts with or without a `.py` suffix; rejects non-`.py` suffixes by
    returning None (the caller converts that to a CLI error).
    """
    scripts_dir = layout.scripts_dir
    if not scripts_dir.is_dir():
        return None

    path = Path(script_name)
    if path.suffix and path.suffix != ".py":
        return None

    candidate = scripts_dir / (f"{script_name}.py" if path.suffix == "" else script_name)
    return candidate if candidate.is_file() else None


def list_user_scripts(layout: ProjectLayout) -> list[Path]:
    """Return all `.py` files under `scripts/`, excluding `__init__.py` and pycache."""
    scripts_dir = layout.scripts_dir
    if not scripts_dir.is_dir():
        return []

    return sorted(
        (
            path
            for path in scripts_dir.rglob("*.py")
            if path.is_file()
            and path.name != "__init__.py"
            and "__pycache__" not in path.parts
        ),
        key=lambda path: str(path.relative_to(scripts_dir)).lower(),
    )
