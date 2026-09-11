"""Transformations of the data variables a raster carries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from geosave_engine.geodata.attrs import combine, stamp
from geosave_engine.geodata.core.raster import GeoRaster

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def merge(rasters: Sequence[xr.Dataset]) -> xr.Dataset:
    """Gather several rasters' data variables into one Dataset.

    Axes are required to match exactly rather than aligned, so no operand
    gains a filled slice or a promoted dtype.

    Args:
        rasters: Datasets on one exact GeoBox with identical time
            coordinates and no shared data-variable name, at least one.

    Returns:
        New Dataset carrying every operand's data variables.

    Raises:
        ValueError: `rasters` is empty, the operands sit on different grids,
            their time coordinates differ, they share a data-variable name,
            or they disagree on an attr.

    Examples:
        >>> transform.variables.merge([optical, elevation]).gs.variables
        ('red', 'nir', 'elevation')
    """
    if not rasters:
        raise ValueError("merging variables needs at least one raster")

    reference, *others = rasters
    grid = GeoRaster(reference).geobox
    times = reference.time.values if "time" in reference.coords else None
    seen = {str(name) for name in reference.data_vars}

    for position, raster in enumerate(others, start=1):
        other_grid = GeoRaster(raster).geobox
        if other_grid != grid:
            raise ValueError(
                f"raster {position} is on {other_grid} but raster 0 is on "
                f"{grid}; reproject or resample it onto one grid before merging"
            )

        other_times = raster.time.values if "time" in raster.coords else None
        drifted = (times is None) != (other_times is None)
        if not drifted and times is not None and other_times is not None:
            drifted = not np.array_equal(times, other_times)
        if drifted:
            raise ValueError(
                f"raster {position} does not carry raster 0's time coordinate; "
                f"resample or select them onto one axis before merging, because "
                f"merging does not align"
            )

        collisions = sorted(seen & {str(name) for name in raster.data_vars})
        if collisions:
            raise ValueError(
                f"raster {position} carries {collisions}, which raster 0 already "
                f"carries; rename one side before merging"
            )
        seen |= {str(name) for name in raster.data_vars}

    merged = xr.merge(rasters, join="exact")
    return stamp(merged, combine(rasters))


def rename(raster: xr.Dataset, mapping: Mapping[str, str]) -> xr.Dataset:
    """Rename data variables without changing their order or pixels.

    Args:
        raster: Dataset holding the variables to rename.
        mapping: Existing variable names mapped to replacement names.

    Returns:
        New Dataset with renamed data variables, each keeping its own attrs.

    Raises:
        KeyError: A source variable is absent.
        ValueError: A replacement is empty, names a coordinate, or creates a
            duplicate.

    Examples:
        >>> transform.variables.rename(ds, {"B04": "red"}).gs.variables
        ('red', 'B08')
    """
    missing = sorted(name for name in mapping if name not in raster.data_vars)
    if missing:
        raise KeyError(
            f"{missing} are not data variables of this raster; it carries "
            f"{sorted(str(name) for name in raster.data_vars)}"
        )

    blank = sorted(source for source, target in mapping.items() if not target.strip())
    if blank:
        raise ValueError(f"{blank} were given an empty replacement name")

    shadowed = sorted(target for target in mapping.values() if target in raster.coords)
    if shadowed:
        raise ValueError(
            f"{shadowed} already name coordinates; a data variable cannot take "
            f"a coordinate's name"
        )

    kept = {str(name) for name in raster.data_vars} - set(mapping)
    replacements = list(mapping.values())
    duplicates = sorted(
        {name for name in replacements if replacements.count(name) > 1}
        | (kept & set(replacements))
    )
    if duplicates:
        raise ValueError(
            f"{duplicates} would name more than one data variable; every name "
            f"must stay unique"
        )

    return raster.rename_vars(dict(mapping))
