"""Combine bands already describing the same pixels and observations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata import attrs

if TYPE_CHECKING:
    from geosave_engine.geodata import Dataset


def merge_bands(rasters: Sequence[xr.Dataset]) -> Dataset:
    """Combine uniquely named bands without aligning or broadcasting observations.

    Args:
        rasters: Non-empty rasters on one exact grid, with equal coordinates
            and the same dimension order for every band. Band names must be unique.

    Returns:
        Dataset with bands in input order and metadata merged through attrs models.

    Raises:
        ValueError: Inputs are empty, bands collide, or grids, coordinates, or
            dimensions differ. Align grids and select observations before merging.

    Examples:
        >>> image = merge_bands([optical[["red", "nir"]], radar[["VV", "VH"]]])
    """
    if not rasters or any(not data.data_vars for data in rasters):
        raise ValueError("merge_bands requires non-empty rasters")
    reference = rasters[0]
    grid = reference.gs.geobox
    if not isinstance(grid, GeoBox):
        raise ValueError("merge_bands requires a regular georeferenced grid")
    dimensions = next(iter(reference.data_vars.values())).dims
    names: set[str] = set()
    for data in rasters:
        if data.gs.geobox != grid:
            raise ValueError("merge_bands requires the same grid; align rasters first")
        if not data.coords.to_dataset().equals(reference.coords.to_dataset()):
            raise ValueError(
                "merge_bands requires equal coordinates; select observations explicitly"
            )
        if any(variable.dims != dimensions for variable in data.data_vars.values()):
            raise ValueError(
                "merge_bands requires equal band dimensions; broadcasting is not supported"
            )
        overlap = names.intersection(str(name) for name in data.data_vars)
        if overlap:
            raise ValueError(
                f"Duplicate bands {sorted(overlap)}; rename bands before merging"
            )
        names.update(str(name) for name in data.data_vars)
    result = xr.merge(rasters, join="exact", compat="equals", combine_attrs="drop")
    return cast("Dataset", attrs.rebase(result, attrs.merge(rasters)))
