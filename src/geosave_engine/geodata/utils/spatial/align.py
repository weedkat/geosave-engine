"""Compatibility checks for rasters about to be composed."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from geosave_engine.geodata.utils.array import same_nodata

from .geobox import geobox_matches

if TYPE_CHECKING:
    from geosave_engine.geodata.spatial import GeoRaster


def validate_rasters(
    *rasters: GeoRaster,
    grid: bool = False,
    bands: bool = False,
    times: bool = False,
    operation: str = "composition",
) -> None:
    """Require rasters to agree on the axes an operation preserves.

    Args:
        *rasters: Rasters about to be composed, first one authoritative.
        grid: Require the first raster's exact geobox.
        bands: Require the first raster's bands and order.
        times: Require the first raster's time coordinates.
        operation: Operation name used in mismatch errors.

    Raises:
        ValueError: A raster disagrees on a required axis, dtype, or nodata.
    """
    first = rasters[0]
    for raster in rasters[1:]:
        if grid and not geobox_matches(raster.data.odc.geobox, first.data.odc.geobox):
            raise ValueError(
                f"{operation} needs one exact grid; raster at {raster.stem} differs from the first. "
                "Call reproject(reference.anchor.geobox) on that raster explicitly before composing"
            )
        if bands and raster.bands != first.bands:
            raise ValueError(
                f"{operation} needs bands {list(first.bands)} in that order; raster at {raster.stem} "
                f"has {list(raster.bands)}. Select or rename bands explicitly before composing"
            )
        if times and (
            raster.has_time != first.has_time
            or (raster.has_time and not np.array_equal(raster.data.time.values, first.data.time.values))
        ):
            raise ValueError(
                f"{operation} needs identical time coordinates; raster at {raster.stem} differs from the first. "
                "Select or resample time explicitly before composing"
            )
        if raster.dtype != first.dtype:
            raise ValueError(
                f"{operation} needs dtype {first.dtype}; raster at {raster.stem} has {raster.dtype}. "
                f"Call raster.astype({str(first.dtype)!r}) explicitly before composing"
            )
        if not same_nodata(raster.nodata, first.nodata):
            raise ValueError(
                f"{operation} needs nodata {first.nodata!r}; raster at {raster.stem} declares "
                f"{raster.nodata!r}. Call raster.rebase(nodata={first.nodata!r}) explicitly"
            )
