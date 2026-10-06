"""Portable model declarations and strict YAML persistence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator
import xarray as xr
import yaml
from yaml.nodes import ScalarNode

from .base import Name, SpecModel, Text
from .call import CallSpec, Ref
from .cuts import FramesSpec, ChipsSpec
from .rasters import RasterRequirement
from .stage import StageSpec
from geosave_engine.geodata.core import GeoAnchor

if TYPE_CHECKING:
    import pandas as pd


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


class TransformSpec(SpecModel):
    """Declare one tensor transform by its Kornia class name.

    Args:
        name: Class in `kornia.augmentation`, such as `Normalize`.
        init_args: Its constructor arguments.
    """

    name: Text
    init_args: dict[str, JsonValue] = Field(default_factory=dict)


class ModelSpec(SpecModel):
    """Describe model raster requirements and inert processing declarations.

    Args:
        schema_version: Document format version.
        rasters: Rasters the model reads, by name.
        preprocessing: Named raster calls run on each frame.
        frames: How a sample's time axis is cut. None cuts nothing.
        chips: How a raster is cut in space and merged back.
        inputs: What the model chain takes, by input name. A reference is a
            raster's pixels.
        context: Optional row-based model context recipe, shared by training and inference.
        transforms: Tensor transforms per pixel input, run after augmentation.

    Raises:
        ValueError: Preprocessing reads an undeclared raster, an input reads
            neither a preprocessing result nor a raster, or transforms name
            an input that is not pixels.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filename: ClassVar[str] = "model_spec.yaml"

    schema_version: Literal[2]
    rasters: dict[Name, RasterRequirement]
    preprocessing: StageSpec = Field(default_factory=StageSpec)
    frames: FramesSpec | None = None
    chips: ChipsSpec | None = None
    inputs: dict[Name, Ref] = Field(default_factory=dict)
    context: CallSpec | None = None
    transforms: dict[Name, list[TransformSpec]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_preprocessing_inputs(self) -> Self:
        unknown = set(self.preprocessing.external_inputs) - self.rasters.keys()
        if unknown:
            raise ValueError(
                f"Preprocessing inputs must be declared rasters: {sorted(unknown)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_inputs(self) -> Self:
        known = self.rasters.keys() | set(self.preprocessing)
        if unknown := set(self.input_rasters) - known:
            raise ValueError(
                "Model inputs reference neither a preprocessing result nor a "
                f"raster: {sorted(unknown)}"
            )
        if self.context is not None and (unknown := self.context.inputs - {"row"}):
            raise ValueError(
                f"Context may reference only the sample row: {sorted(unknown)}"
            )
        if unknown := self.transforms.keys() - set(self.pixel_inputs):
            raise ValueError(
                f"Transforms name inputs that are not pixels: {sorted(unknown)}"
            )
        return self

    @property
    def pixel_inputs(self) -> tuple[str, ...]:
        """Return the model inputs that are a raster's pixels."""
        return tuple(name for name in self.inputs)

    @property
    def input_rasters(self) -> tuple[str, ...]:
        """Return the results and rasters model inputs read, in first-use order."""
        return tuple(dict.fromkeys(value.root for value in self.inputs.values()))

    def load_rasters(self, anchor: GeoAnchor, /) -> dict[str, xr.Dataset]:
        """Load every declared raster from STAC onto an anchor.

        Args:
            anchor: Grid and time the rasters are loaded on.

        Returns:
            Raster name mapped to its selected lazy raster.

        Raises:
            ValueError: A raster declares no `stac` block, or a loaded raster
                does not meet its requirement.
        """
        recipes = {
            name: requirement.stac
            for name, requirement in self.rasters.items()
            if requirement.stac is not None
        }
        if missing := [name for name in self.rasters if name not in recipes]:
            raise ValueError(f"Rasters declare no `stac` block: {missing}")

        rasters = {}
        for name, recipe in recipes.items():
            try:
                rasters[name] = self.rasters[name].select_raster(
                    recipe.load_raster(anchor)
                )
            except Exception as error:
                error.add_note(f"While loading raster {name!r}")
                raise
        return rasters

    def preprocess(self, rasters: Mapping[str, Any], /) -> dict[str, Any]:
        """Select the declared rasters and run preprocessing on them.

        Args:
            rasters: Rasters by the names this spec declares.

        Returns:
            Each selected raster and each preprocessing result, by name.

        Raises:
            TypeError: A raster is not an xarray Dataset.
            KeyError: A supplied raster lacks a declared variable or coordinate.
            ValueError: A raster does not meet its requirement, or
                preprocessing reads one that was not supplied.

        Examples:
            >>> sorted(spec.preprocess({"sentinel_2_l2a": scene}))
            ['image', 'sentinel_2_l2a', 'valid_pixels']
        """
        selected = {}
        for name, requirement in self.rasters.items():
            if name not in rasters:
                continue
            try:
                selected[name] = requirement.select_raster(rasters[name])
            except Exception as error:
                error.add_note(f"While validating raster {name!r}")
                raise
        return {**selected, **self.preprocessing.run({**rasters, **selected})}

    def model_context(self, row: pd.Series | None) -> dict[str, Any]:
        """Encode a sample row with the declared context recipe.

        Args:
            row: Native catalog row with metadata for the actual input window.

        Returns:
            Model-specific context values, or an empty dictionary.

        Raises:
            ValueError: A declared context recipe receives no row.
            TypeError: The recipe returns something other than a dictionary.
        """
        if self.context is None:
            return {}
        if row is None:
            raise ValueError("model context requires a sample row")
        values = self.context.invoke(self.context.select_inputs({"row": row}))
        if not isinstance(values, dict):
            raise TypeError("model context must return a dictionary")
        return values

    def model_inputs(
        self,
        rasters: Mapping[str, Any],
        row: pd.Series | None = None,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Combine selected raster inputs with live or explicitly cached context.

        Args:
            rasters: Prepared rasters named by the input references.
            row: Sample metadata passed to the context recipe.
            context: Cached encoding for this row and recipe. None computes it.

        Returns:
            Named raster objects and context values for the model chain.

        Raises:
            ValueError: Context attempts to overwrite a raster input.
        """
        pixels = {name: value.resolve(rasters) for name, value in self.inputs.items()}
        encoded = self.model_context(row) if context is None else context
        if overlap := pixels.keys() & encoded.keys():
            raise ValueError(
                f"context cannot overwrite raster inputs: {sorted(overlap)}"
            )
        return {**pixels, **encoded}

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
        payload = yaml.dump(self.model_dump(), Dumper=_Dumper, sort_keys=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read a local model document without importing declared calls."""
        payload = cls.resolve_path(path).read_text(encoding="utf-8")
        return cls.model_validate(yaml.load(payload, Loader=_Loader))
