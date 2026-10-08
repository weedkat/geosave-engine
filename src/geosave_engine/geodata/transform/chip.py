"""Describe and extract spatial chips from native rasters without computing pixels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

from geosave_engine.geodata.conventions import SPATIAL_DIMENSIONS

if TYPE_CHECKING:
    pass


type PaddingMode = Literal["constant", "edge", "reflect", "wrap"]


def crop(
    data: xr.Dataset | xr.DataArray | xr.DataTree,
    offset: tuple[int, int],
    shape: tuple[int, int],
    *,
    padding: Sequence[tuple[int, int]] = ((0, 0), (0, 0)),
    mode: PaddingMode = "constant",
    constant_value: int | float | None = None,
) -> xr.Dataset | xr.DataArray | xr.DataTree:
    """Read a pixel window with the same padding used by its native Tiler.

    Args:
        data: Raster or stack; leading dimensions remain intact.
        offset: Row and column offsets in the original raster, before padding.
        shape: Requested height and width.
        padding: Halo widths before and after each spatial axis.
        mode: Padding mode for halo and fringe pixels.
        constant_value: Constant fill. None uses xarray's missing-value fill.

    Returns:
        Lazy window of the same native type. Georeferenced rasters retain
        the requested grid, including padded pixels outside the parent.
    """
    if isinstance(data, xr.DataTree):
        from geosave_engine.geodata.core.stack import map_groups

        return map_groups(
            data,
            lambda raster: cast(
                "xr.Dataset",
                crop(
                    raster,
                    offset,
                    shape,
                    padding=padding,
                    mode=mode,
                    constant_value=constant_value,
                ),
            ),
        )

    dims = SPATIAL_DIMENSIONS
    grid = data.gs.geobox

    # Halo offsets are relative to the original parent, not its padded shape.
    if any(before or after for before, after in padding):
        data = _pad(data, dict(zip(dims, padding, strict=True)), mode, constant_value)
    starts = [
        start + before for start, (before, _) in zip(offset, padding, strict=True)
    ]
    data = data.isel(
        {
            dim: slice(start, start + size)
            for dim, start, size in zip(dims, starts, shape, strict=True)
        }
    )

    # Edge tiles may extend beyond the parent even when no halo was added.
    fringe = {
        dim: (0, size - data.sizes[dim]) for dim, size in zip(dims, shape, strict=True)
    }
    if any(after for _, after in fringe.values()):
        data = _pad(data, fringe, mode, constant_value)

    # Reflected/wrapped coordinate labels repeat; the output grid does not.
    if isinstance(grid, GeoBox):
        data = data.assign_coords(
            xr_coords(grid.translate_pix(offset[1], offset[0]).crop(shape), dims=dims)
        )
    return data


def _pad(
    data: xr.Dataset | xr.DataArray,
    widths: Mapping[str, tuple[int, int]],
    mode: PaddingMode,
    constant_value: int | float | None,
) -> xr.Dataset | xr.DataArray:
    """Pad pixels lazily, including reflection wider than the source."""
    # Dask truncates repeated reflection/wrapping. Pad small index arrays with
    # NumPy, then let xarray select pixels without computing the raster.
    if mode in ("reflect", "wrap"):
        return data.isel(
            {
                dim: np.pad(np.arange(data.sizes[dim]), width, mode=mode)
                for dim, width in widths.items()
            }
        )
    return data.pad(widths, mode=mode, constant_values=constant_value)
