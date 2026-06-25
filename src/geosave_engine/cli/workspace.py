import os
import toml
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass

from geosave_engine.utils import safe_copy
from geosave_engine.cli.paths import common_template_dir, templates_dir, get_workspace_scripts, get_workspace_artifacts
from geosave_engine.cli.errors import WorkspaceError

REQUIRED_FIELDS = ["project_name"]
REQUIRED_DIRS = ["data", "configs", "artifacts", "scripts", "logs", "modules", "predictions", "notebooks"]


@dataclass
class WorkspaceSpec:
    project_name: str
    project_task: str
    project_method: str
    description: str | None = None


class Workspace:
    def __init__(self, root: Path | str, spec: WorkspaceSpec):
        rp = Path(root)
        self.root = rp if (rp / "geosave.toml").exists() else rp / spec.project_name
        self.spec = spec
        # Cache discovered scripts and artifacts on init
        self._scripts = get_workspace_scripts(self.scripts_dir)
        self._artifacts = get_workspace_artifacts(self.artifacts_dir)

    # standard directory properties for clarity and discovery
    @property
    def config_dir(self) -> Path:
        return self.root / "configs"

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def notebooks_dir(self) -> Path:
        return self.root / "notebooks"

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
    def scripts(self) -> dict[str, Path]:
        """Cached mapping of script keys to paths."""
        return self._scripts

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
                project_task=data['project_task'],
                project_method=data['project_method'],
                description=data.get("description")
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

    def add_plugin(self, plugin_source: Path, flat: bool = False) -> None:
        """Copy plugin template into workspace folder tree."""
        target = self.root / plugin_source.parent.name
        if not flat:
            target = target / plugin_source.name
        safe_copy(plugin_source, target)

    def run_script(self, script_path: Path, args: list[str]) -> None:
        """Execute an already-resolved script path inside the workspace.

        This method intentionally does not perform discovery or key->path mapping.
        """
        if not script_path.exists() or not script_path.is_file():
            raise WorkspaceError(f"Script not found: {script_path}")

        cmd = self._build_python_command(script_path, args)
        self._execute_command(cmd, cwd=self.root, script_path=script_path)

    def run_lightning(self, command: str, args: list[str]) -> None:
        """Run the workspace's src/main.py with a lightning CLI command (fit, test, predict).

        Args:
            command: Lightning command name (e.g., 'fit', 'test', 'predict').
            args: Additional arguments to pass to the lightning command.
        """
        main_py = self.root / "main.py"
        if not main_py.exists():
            raise WorkspaceError(f"Lightning main.py not found at: {main_py}")

        cmd = self._build_python_command(main_py, [command, *args])
        self._execute_command(cmd, cwd=self.root, script_path=main_py)

    def _build_python_command(self, script_path: Path, args: list[str]) -> list[str]:
        return [sys.executable, str(script_path), *args]

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
        """Copy task template into workspace src tree."""
        templates = templates_dir() / self.spec.project_task / "methods" / self.spec.project_method
        safe_copy(templates, self.root)

    @staticmethod
    def _validate_data(data: dict) -> None:
        missing_fields = [field for field in REQUIRED_FIELDS if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields in geosave.toml: {', '.join(missing_fields)}")

if __name__ == "__main__":
    # Example usage
    pass