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
    """Return artifact directories that contain a ``config.yaml``.

    Supports the TensorBoard-style nesting written by the auto-injected logger:
    ``artifacts/<exp_name>/version_<n>/config.yaml``. Falls back to direct
    children of ``artifacts/`` if a config sits one level up.
    """
    artifacts_dir = layout.artifacts_dir
    if not artifacts_dir.is_dir():
        return []

    # Recursive scan up to two levels deep — enough for both the historical
    # `artifacts/<run>/config.yaml` layout and the current
    # `artifacts/<exp>/version_<n>/config.yaml` layout.
    candidates: set[Path] = set()
    for depth_glob in ("*/config.yaml", "*/*/config.yaml", "*/config.yml", "*/*/config.yml"):
        for cfg in artifacts_dir.glob(depth_glob):
            candidates.add(cfg.parent)
    return sorted(candidates)


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
