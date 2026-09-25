"""Portable YAML model usage settings saved beside architecture and weights."""

from __future__ import annotations

from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator
import yaml

from .base import RasterName, SpecModel
from .inference import InferenceSpec, SegmentationSpec
from .preprocessing import OperationSpec, PreprocessingSpec
from .requirements import RasterRequirement


class _UniqueSafeLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys rather than silently replacing settings."""


def _unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError("YAML mapping keys must be scalar values") from error
        if duplicate:
            raise ValueError(f"Duplicate YAML key {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


class ModelSpec(SpecModel):
    """Declare sources, named preprocessing recipes, and model execution settings."""

    filename: ClassVar[str] = "model_spec.yaml"

    schema_version: Literal[1] = 1
    sources: Annotated[dict[RasterName, RasterRequirement], Field(min_length=1)]
    preprocessing: dict[RasterName, PreprocessingSpec] = Field(default_factory=dict)
    inference: InferenceSpec
    postprocessing: SegmentationSpec | OperationSpec | None = None

    @model_validator(mode="after")
    def _validate_bindings(self) -> Self:
        available = self.sources.keys() | self.preprocessing.keys()
        if any(name in (".", "..") for name in available):
            raise ValueError("Raster names must be flat group names")
        self.preparation_order()
        if self.inference.tiling.raster not in available:
            raise ValueError(
                f"Tiling references unknown raster {self.inference.tiling.raster!r}"
            )
        for name, binding in self.inference.inputs.items():
            if binding.raster not in available:
                raise ValueError(
                    f"Input {name!r} references unknown raster {binding.raster!r}"
                )
            known = self.sources.get(binding.raster)
            selected = binding.variables
            if known is not None:
                selected = selected if selected is not None else known.variables
                if missing := set(selected) - set(known.variables):
                    raise ValueError(
                        f"Input {name!r} selects unavailable variables {sorted(missing)}"
                    )
            if (
                selected is not None
                and binding.normalize is not None
                and len(binding.normalize.mean) != len(selected)
            ):
                raise ValueError(
                    f"Input {name!r} normalization must match its {len(selected)} channels"
                )
        if (
            isinstance(self.postprocessing, OperationSpec)
            and self.postprocessing.inputs
        ):
            raise ValueError("Postprocessing operations cannot reference raster inputs")
        return self

    def preparation_order(self) -> tuple[str, ...]:
        """Validate recipe references and return their dependency order."""
        if collision := self.sources.keys() & self.preprocessing.keys():
            raise ValueError(
                f"Preprocessing names collide with sources: {sorted(collision)}"
            )
        available = self.sources.keys() | self.preprocessing.keys()
        graph = {}
        for name, recipe in self.preprocessing.items():
            references = {recipe.raster} | {
                raster
                for operation in recipe.operations
                for raster in operation.inputs.values()
            }
            if unknown := references - available:
                raise ValueError(
                    f"Recipe {name!r} references unknown rasters: {sorted(unknown)}"
                )
            graph[name] = references & self.preprocessing.keys()
        try:
            return tuple(TopologicalSorter(graph).static_order())
        except CycleError as error:
            raise ValueError(
                f"Preprocessing dependency cycle: {error.args[1]}"
            ) from error

    def validated_copy(self) -> Self:
        """Revalidate nested mutable settings before saving or execution."""
        return type(self).model_validate(self.model_dump())

    def save(self, path: str | Path) -> Path:
        """Write YAML to a local file or artifact directory, preserving weights."""
        target = _spec_path(path, self.filename)
        validated = self.validated_copy()
        payload = yaml.safe_dump(validated.model_dump(), sort_keys=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load safe YAML from a local file or artifact directory."""
        payload = _spec_path(path, cls.filename).read_text(encoding="utf-8")
        return cls.model_validate(yaml.load(payload, Loader=_UniqueSafeLoader))


def _spec_path(path: str | Path, filename: str) -> Path:
    if "://" in str(path):
        raise ValueError("ModelSpec save/load requires a local path")
    target = Path(path)
    if target.is_dir():
        return target / filename
    if target.suffix.lower() in (".yaml", ".yml"):
        return target
    if target.suffix or target.is_file():
        raise ValueError("ModelSpec files must use a .yaml or .yml YAML suffix")
    return target / filename
