"""Portable model declarations and strict YAML persistence."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import Field
import yaml
from yaml.nodes import ScalarNode

from .base import Name, SpecModel
from .call import Ref
from .rasters import RasterRequirement
from .stage import StageSpec


class _Loader(yaml.SafeLoader):
    """Read ordinary mappings strictly and retain inert references."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key, value in self.construct_pairs(node, deep=deep):
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings")
            if key in result:
                raise ValueError(f"Duplicate YAML key {key!r}")
            result[key] = value
        return result


class _Dumper(yaml.SafeDumper):
    """Write references as tags and literals through the safe dumper."""

    def represent_data(self, data):
        if isinstance(data, Ref):
            return self.represent_scalar("!ref", data.path)
        return super().represent_data(data)


def _construct_ref(loader: _Loader, node: ScalarNode) -> Ref:
    """Construct one inert scalar reference."""
    return Ref(loader.construct_scalar(node))


_Loader.add_constructor("!ref", _construct_ref)


class ModelSpec(SpecModel):
    """Describe model raster requirements and inert processing declarations."""

    filename: ClassVar[str] = "model_spec.yaml"

    schema_version: Literal[2]
    rasters: dict[Name, RasterRequirement]
    preprocessing: StageSpec = Field(default_factory=StageSpec)

    def validated_copy(self) -> Self:
        """Copy and revalidate mutable nested declarations without loading code."""
        return type(self).model_validate(self.model_dump())

    @staticmethod
    def resolve_path(path: str | Path) -> Path:
        """Resolve a local YAML filename or model artifact directory."""
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

    def save(self, path: str | Path) -> Path:
        """Save this document to a local YAML file or artifact directory."""
        target = self.resolve_path(path)
        payload = yaml.dump(
            self.validated_copy().model_dump(), Dumper=_Dumper, sort_keys=False
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read a local model document without importing declared calls."""
        payload = cls.resolve_path(path).read_text(encoding="utf-8")
        return cls.model_validate(yaml.load(payload, Loader=_Loader))
