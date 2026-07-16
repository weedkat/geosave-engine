from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import toml  # type: ignore[import-untyped]

from geosave_engine.cli.errors import WorkspaceError

from .artifact import discover_artifacts

_REQUIRED_FIELDS = ("project_name", "project_task", "project_method")


@dataclass
class WorkspaceSpec:
    """Describe one generated workspace."""

    project_name: str
    project_task: str
    project_method: str
    description: str | None = None


class Workspace:
    """Load or create one GeoSave workspace."""

    def __init__(self, root: Path | str, spec: WorkspaceSpec) -> None:
        root_path = Path(root).resolve()
        self.root = root_path if (root_path / "geosave.toml").exists() else root_path / spec.project_name
        self.spec = spec
        self._artifacts = discover_artifacts(self.artifacts_dir)

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
    def modules_dir(self) -> Path:
        return self.root / "modules"

    @property
    def predictions_dir(self) -> Path:
        return self.root / "predictions"

    @property
    def artifacts(self) -> list[Path]:
        """Return cached model run directories."""
        return self._artifacts

    @classmethod
    def load_workspace(cls, root: Path | str) -> Workspace:
        """Load a workspace from its geosave.toml file.

        Args:
            root: Existing workspace directory.

        Raises:
            WorkspaceError: If workspace metadata is missing or invalid.
        """
        try:
            data = cls._read_toml(Path(root))
            cls._validate_data(data)
            return cls(
                root=root,
                spec=WorkspaceSpec(
                    project_name=cls._required_string(data, "project_name"),
                    project_task=cls._required_string(data, "project_task"),
                    project_method=cls._required_string(data, "project_method"),
                    description=cls._optional_string(data, "description"),
                ),
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            raise WorkspaceError(f"Failed to load workspace: {error}") from error

    def setup_workspace(self) -> None:
        """Create this workspace from its selected template."""
        from .scaffold import scaffold_workspace

        scaffold_workspace(self)

    @staticmethod
    def _read_toml(root: Path) -> dict[str, object]:
        path = root / "geosave.toml"
        if not path.exists():
            raise FileNotFoundError(f"geosave.toml not found in {root}")
        return toml.load(path)

    @staticmethod
    def _validate_data(data: dict[str, object]) -> None:
        missing_fields = [field for field in _REQUIRED_FIELDS if field not in data]
        if missing_fields:
            raise ValueError(f"Missing fields in geosave.toml: {', '.join(missing_fields)}")

    @staticmethod
    def _required_string(data: dict[str, object], field: str) -> str:
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} in geosave.toml must be a non-empty string")
        return value.strip()

    @staticmethod
    def _optional_string(data: dict[str, object], field: str) -> str | None:
        value = data.get(field)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{field} in geosave.toml must be a string")
        return value.strip() or None
