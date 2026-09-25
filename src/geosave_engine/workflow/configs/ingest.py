"""Complete primitive parameters accepted by the ingestion flow."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import field_validator

from .anchor import AnchorConfig
from .base import ConfigModel, Name
from .source import SourceConfig


class IngestConfig(ConfigModel):
    """Validate an ingestion deployment before any task is submitted."""

    sources: dict[Name, SourceConfig]
    anchor: AnchorConfig
    output: Path
    spec: Path

    @field_validator("output", mode="before")
    @classmethod
    def validate_output(cls, value: Any) -> Path:
        """Require a new local Zarr destination."""
        if "://" in str(value):
            raise ValueError("Raster stack output requires a local path")
        path = Path(value)
        if path.suffix != ".zarr":
            raise ValueError("Raster stack output must end in .zarr")
        if path.exists():
            raise ValueError(f"Output already exists: {path}")
        return path

    @field_validator("spec", mode="before")
    @classmethod
    def validate_spec(cls, value: Any) -> Path:
        """Require a local YAML file or artifact-directory path."""
        if "://" in str(value):
            raise ValueError("Model spec requires a local path")
        path = Path(value)
        if path.suffix and path.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError("Model spec files must use a .yaml or .yml suffix")
        return path

    def validate_sources(self, requirements: Mapping[str, object]) -> None:
        """Require exact source bindings without opening any source."""
        if not requirements:
            raise ValueError("At least one model source is required")
        if missing := requirements.keys() - self.sources.keys():
            raise ValueError(f"Source bindings are missing: {sorted(missing)}")
        if extra := self.sources.keys() - requirements.keys():
            raise ValueError(f"Unknown source bindings: {sorted(extra)}")
