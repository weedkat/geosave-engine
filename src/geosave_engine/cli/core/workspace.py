from __future__ import annotations

from pathlib import Path

from geosave_engine.utils.file_ops import safe_copy

from .templates import common_dir, task_dir

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

def create_workspace(root, task=None, method=None) -> None:
    """Create directories and copy files for one workspace.

    Args:
        root: The root directory for the workspace.
        task: The task for the workspace.
        method: The method for the workspace.
    """
    for directory_name in _REQUIRED_DIRS:
        (root / directory_name).mkdir(parents=True, exist_ok=True)

    safe_copy(common_dir(), root, exclude=_EXCLUDE)

    if task and method:

        method_dir = task_dir() / task / method

        safe_copy(method_dir, root, exclude=_EXCLUDE)

class Workspace:
    """Load or create one GeoSave workspace."""
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        
    @property
    def root(self) -> Path:
        return self._root

    @root.setter
    def root(self, value: Path):
        path = value / "geosave.toml"
        if not path.exists():
            raise FileNotFoundError(f"geosave.toml not found in {value}")
        self._root = value

    @property
    def toml_path(self) -> Path:
        return self.root / "geosave.toml"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def config_dir(self) -> Path:
        return self.root / "configs"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    @property
    def module_dir(self) -> Path:
        return self.root / "modules"

    @property
    def notebook_dir(self) -> Path:
        return self.root / "notebooks"

    @property
    def prediction_dir(self) -> Path:
        return self.root / "predictions"

    @property
    def script_dir(self) -> Path:
        return self.root / "scripts"

    def __repr__(self) -> str:
        return f"Workspace(root={self.root})"