import os
import toml
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass

from geosave_engine.utils import safe_copy
from geosave_engine.cli.paths import common_template_dir, templates_dir, get_workspace_artifacts
from geosave_engine.cli.errors import WorkspaceError

REQUIRED_FIELDS = ["project_name"]
REQUIRED_DIRS = ["data", "configs", "artifacts", "logs", "modules", "predictions"]


@dataclass
class WorkspaceSpec:
    project_name: str
    project_task: str
    project_method: str
    catalog: str | None = None
    description: str | None = None


class Workspace:
    def __init__(self, root: Path | str, spec: WorkspaceSpec):
        rp = Path(root)
        self.root = rp if (rp / "geosave.toml").exists() else rp / spec.project_name
        self.spec = spec
        self._artifacts = get_workspace_artifacts(self.artifacts_dir)

    @property
    def config_dir(self) -> Path:
        return self.root / "configs"

    @property
    def src_dir(self) -> Path:
        return self.root / "src"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def predictions_dir(self) -> Path:
        return self.root / "predictions"

    @property
    def artifacts(self) -> dict[str, Path]:
        """Cached mapping of artifact keys (model_name/version_0) to config.yaml paths."""
        return self._artifacts

    @classmethod
    def load_workspace(cls, root: Path | str) -> "Workspace":
        try:
            data = cls._read_toml(Path(root))
            cls._validate_data(data)
            spec = WorkspaceSpec(
                project_name=data["project_name"],
                project_task=data["project_task"],
                project_method=data["project_method"],
                catalog=data.get("catalog"),
                description=data.get("description"),
            )
            return cls(root=root, spec=spec)
        except (FileNotFoundError, ValueError) as e:
            raise WorkspaceError(f"Failed to load workspace: {e}")

    def setup_workspace(self) -> None:
        for d in REQUIRED_DIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        safe_copy(common_template_dir(), self.root)
        self._write_toml()
        self._add_task()

    def run_lightning(self, command: str, args: list[str]) -> None:
        """Run the workspace's main.py with a LightningCLI command.

        Args:
            command: Lightning command name (e.g., 'fit', 'test', 'predict').
            args: Additional arguments to pass to the command.
        """
        main_py = self.root / "main.py"
        if not main_py.exists():
            raise WorkspaceError(f"main.py not found at: {main_py}")
        cmd = [sys.executable, str(main_py), command, *args]
        self._execute_command(cmd, cwd=self.root, script_path=main_py)

    def run_ingest(self, config_path: str, splits: list[str] | None = None) -> None:
        """Run the workspace's ingest.py to populate raw cache + derived layers.

        Args:
            config_path: Path to an ingest.yaml-shaped config.
            splits: Limit to these split names. None runs every split.
        """
        ingest_py = self.root / "ingest.py"
        if not ingest_py.exists():
            raise WorkspaceError(f"ingest.py not found at: {ingest_py}")
        cmd = [sys.executable, str(ingest_py), "--config", config_path]
        if splits:
            cmd += ["--splits", *splits]
        self._execute_command(cmd, cwd=self.root, script_path=ingest_py)

    def _execute_command(self, command: list[str], cwd: Path, script_path: Path) -> None:
        env = {**os.environ, "PYTHONPATH": str(cwd)}
        try:
            subprocess.run(command, cwd=cwd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            raise WorkspaceError(f"Script failed (exit {e.returncode}): {script_path}")

    @staticmethod
    def _read_toml(root: Path) -> dict:
        path = root / "geosave.toml"
        if not path.exists():
            raise FileNotFoundError(f"geosave.toml not found in {root}")
        with open(path, "r") as f:
            data = toml.load(f)
        return data

    def _write_toml(self) -> None:
        path = self.root / "geosave.toml"
        with open(path, "w") as f:
            toml.dump(self.spec.__dict__, f)

    def _add_task(self) -> None:
        """Copy catalog template into workspace.

        Each catalog is a self-contained subdir with its own ``modules/`` and ``configs/``.
        If no catalog is set, the method dir is copied directly (single-catalog methods).
        """
        method_dir = templates_dir() / self.spec.project_task / self.spec.project_method
        src = method_dir / self.spec.catalog if self.spec.catalog else method_dir
        safe_copy(src, self.root)

    @staticmethod
    def _validate_data(data: dict) -> None:
        missing_fields = [field for field in REQUIRED_FIELDS if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields in geosave.toml: {', '.join(missing_fields)}")
