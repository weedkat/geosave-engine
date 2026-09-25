"""Shared temporal sampling and named tensor inputs for training and inference."""

from collections.abc import Callable, Mapping
from typing import Any

from odc.geo.geobox import GeoBox
import torch
from torch.utils.data import Dataset
import xarray as xr

from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.transform.time import window_stack
from geosave_engine.geodata.transform.tiling import Tiles

from .spec import InferenceSpec


def sample_windows(
    prepared: xr.DataTree, *, settings: InferenceSpec
) -> tuple[xr.DataTree, ...]:
    """Select model rasters and form lazy temporal samples on one exact grid.

    Args:
        prepared: Prepared raster stack, including optional unused groups.
        settings: Named tensor bindings and native sampling settings.

    Returns:
        Temporal window stacks, or one full stack when windowing is absent.

    Raises:
        ValueError: A raster, channel, layout, or shared grid is incompatible.
    """
    settings = InferenceSpec.model_validate(settings)
    rasters = prepared.gs.rasters
    names = dict.fromkeys(
        [
            settings.tiling.raster,
            *(binding.raster for binding in settings.inputs.values()),
        ]
    )
    if missing := names.keys() - rasters.keys():
        raise ValueError(f"Missing prepared rasters: {sorted(missing)}")
    reference = rasters[settings.tiling.raster].gs.geobox
    if not isinstance(reference, GeoBox):
        raise ValueError("The tiling reference requires a regular geospatial grid")
    for name in names:
        if rasters[name].gs.geobox != reference:
            raise ValueError(
                f"Raster {name!r} does not share the reference grid; align it explicitly"
            )
    selected_variables: dict[str, dict[str, None]] = {}
    for name, binding in settings.inputs.items():
        data = rasters[binding.raster]
        variables = binding.variables or tuple(data.data_vars)
        if not variables or set(variables) - data.data_vars.keys():
            raise ValueError(
                f"Input {name!r} selects missing or empty variables: {variables}"
            )
        expected = set(reference.dimensions)
        if binding.layout == "TCHW":
            expected.add("time")
            if "time" not in data.coords or data.coords["time"].ndim != 1:
                raise ValueError(
                    f"Input {name!r} with TCHW requires a labelled time dimension"
                )
        for variable in variables:
            if set(data[variable].dims) != expected:
                raise ValueError(
                    f"Input {name!r} {binding.layout} requires dimensions {sorted(expected)}"
                )
        if binding.normalize is not None and len(binding.normalize.mean) != len(
            variables
        ):
            raise ValueError(
                f"Input {name!r} normalization must match its {len(variables)} channels"
            )
        selected_variables.setdefault(binding.raster, {}).update(
            dict.fromkeys(variables)
        )
    # Only model-bound variables participate in temporal alignment. The tiling
    # reference contributes its grid, never unrelated acquisition dates.
    selected = stack(
        {
            name: rasters[name][list(variables)]
            for name, variables in selected_variables.items()
        }
    )
    if settings.time_window is None:
        return (selected,)
    return window_stack(selected, **settings.time_window.model_dump())


class RasterSamples(Dataset):
    """Encode lazy spatial tiles as named model inputs and per-tile context.

    This PyTorch Dataset binds individual raster variables, dtypes and layouts;
    a DataLoader batches its ordinary dictionaries for training or inference.

    Args:
        tiles: Native spatial tiles for one or more prepared windows.
        settings: Named model argument encodings.
        model_context: Extract collatable context before reading tile pixels.
        context_keys: Caller-provided context names, reserved against collisions.
    """

    def __init__(
        self,
        tiles: Tiles,
        settings: InferenceSpec,
        *,
        model_context: Callable[[xr.DataTree], Mapping[str, Any]] | None = None,
        context_keys: frozenset[str] = frozenset(),
    ):
        self.tiles = tiles
        self.settings = settings
        self.model_context = model_context
        self.context_keys = context_keys
        self._derived_keys: frozenset[str] | None = None

    def __len__(self) -> int:
        """Return the number of spatial samples."""
        return len(self.tiles)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Read only this tile, extracting context before tensor conversion."""
        tile = self.tiles[index]
        context = {} if self.model_context is None else self.model_context(tile)
        if not isinstance(context, Mapping) or any(
            not isinstance(key, str) for key in context
        ):
            raise TypeError("Model context must be a mapping with string keys")
        keys = frozenset(context)
        if collision := keys & (self.settings.inputs.keys() | self.context_keys):
            raise ValueError(f"Model context arguments collide: {sorted(collision)}")
        if self._derived_keys is not None and keys != self._derived_keys:
            raise ValueError("Model context must supply the same keys for every tile")
        self._derived_keys = keys
        inputs = {}
        for name, binding in self.settings.inputs.items():
            data = tile.gs.rasters[binding.raster]
            variables = binding.variables or tuple(data.data_vars)
            encoded = data[list(variables)].gs.to_tensor(
                dtype=getattr(torch, binding.dtype)
            )
            if binding.normalize is not None:
                shape = (len(variables), 1, 1)
                mean = encoded.new_tensor(binding.normalize.mean).reshape(shape)
                std = encoded.new_tensor(binding.normalize.std).reshape(shape)
                encoded = (encoded - mean) / std
            inputs[name] = encoded
        return {"inputs": inputs, "context": dict(context), "index": index}
