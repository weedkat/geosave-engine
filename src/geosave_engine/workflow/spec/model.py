"""Portable call declarations and safe YAML round trips."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import ConfigDict, Field, field_validator
import yaml

from geosave_engine.geodata import attrs

from .base import Name, SpecModel
from .references import Ref, validate_value
from .requirements import RasterRequirement


class OperationSpec(SpecModel):
    """Declare a callable and its actual keyword arguments without executing it.

    Args:
        call: Import path or reference to a supplied callable/bound method.
        kwargs: YAML literals and references, resolved when the stage runs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    call: str | Ref
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("call")
    @classmethod
    def _call_path(cls, value: str | Ref) -> str | Ref:
        path = value.path if isinstance(value, Ref) else value
        Ref(path)
        if isinstance(value, str) and "." not in value:
            raise ValueError("An imported call needs a module and callable name")
        return value

    @field_validator("kwargs", mode="before")
    @classmethod
    def _arguments(cls, value: Any) -> Any:
        return validate_value(value)


class OutputSpec(SpecModel):
    """Declare the semantic legend attached to a named postprocessing output."""

    legend: attrs.Legend | None = None


class _Loader(yaml.SafeLoader):
    """Read ordinary mappings strictly and retain inert `!ref` values."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key, value in self.construct_pairs(node, deep=deep):
            # Legend class/color maps use integer pixel codes; typed fields
            # and operation argument validation constrain all other mappings.
            if type(key) not in (str, int):
                raise ValueError("YAML mapping keys must be strings or integers")
            if key in result:
                raise ValueError(f"Duplicate YAML key {key!r}")
            result[key] = value
        return result


class _Dumper(yaml.SafeDumper):
    """Write references as tags and all other values as ordinary safe YAML."""

    def represent_data(self, data):
        if isinstance(data, Ref):
            return self.represent_scalar("!ref", data.path)
        return super().represent_data(data)


_Loader.add_constructor("!ref", lambda loader, node: Ref(loader.construct_scalar(node)))


class ModelSpec(SpecModel):
    """Describe source requirements and independent named processing stages.

    Args:
        schema_version: Explicitly 2; older workflow specifications are rejected.
        sources: Requirements for externally supplied native rasters.
        preprocessing: Ordered calls preparing native values.
        inference: Ordered call declarations; no inference mechanism is implied.
        postprocessing: Ordered calls interpreting supplied predictions.
        outputs: Named results with optional legends attached after postprocessing.
    """

    filename: ClassVar[str] = "model_spec.yaml"

    schema_version: Literal[2]
    sources: dict[Name, RasterRequirement]
    preprocessing: dict[Name, OperationSpec] = Field(default_factory=dict)
    inference: dict[Name, OperationSpec] = Field(default_factory=dict)
    postprocessing: dict[Name, OperationSpec] = Field(default_factory=dict)
    outputs: dict[Name, OutputSpec] = Field(default_factory=dict)

    def validated_copy(self) -> Self:
        """Copy and revalidate mutable nested declarations without loading code."""
        return type(self).model_validate(self.model_dump())

    def save(self, path: str | Path) -> Path:
        """Save YAML to a local file or artifact directory, preserving other files.

        Args:
            path: YAML filename or directory receiving `model_spec.yaml`.

        Returns:
            Path to the saved YAML document.
        """
        target = _spec_path(path)
        payload = yaml.dump(
            self.validated_copy().model_dump(), Dumper=_Dumper, sort_keys=False
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read a local YAML file or artifact directory without importing callables."""
        payload = _spec_path(path).read_text(encoding="utf-8")
        return cls.model_validate(yaml.load(payload, Loader=_Loader))


def _spec_path(path: str | Path) -> Path:
    if "://" in str(path):
        raise ValueError("ModelSpec save/load requires a local path")
    target = Path(path)
    if target.is_dir():
        return target / ModelSpec.filename
    if target.suffix.lower() in (".yaml", ".yml"):
        return target
    if target.suffix or target.is_file():
        raise ValueError("ModelSpec files must use a .yaml or .yml YAML suffix")
    return target / ModelSpec.filename
