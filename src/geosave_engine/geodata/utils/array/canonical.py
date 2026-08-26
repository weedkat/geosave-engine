"""Validate the one in-memory raster representation Spatial accepts."""
from __future__ import annotations

from typing import TypeVar

import numpy as np
import odc.geo.xr  # noqa: F401 — registers .odc accessor on xr.DataArray/Dataset
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray/Dataset
import xarray as xr

SPATIAL_DIMS: tuple[str, ...] = ("time", "band", "y", "x")
CANONICAL_DIMS: frozenset[tuple[str, ...]] = frozenset(
    {("band", "y", "x"), ("time", "band", "y", "x")}
)

_GeoObj = TypeVar("_GeoObj", xr.DataArray, xr.Dataset)


def validate_spatial(da: xr.DataArray) -> xr.DataArray:
    """Validate a canonical in-memory Spatial array.

    Args:
        da: DataArray shaped `(band, y, x)` or `(time, band, y, x)`.

    Returns:
        `da` with its grid-mapping pointer present.

    Raises:
        ValueError: Dimensions, band/time coordinates, nodata, or CRS
            violate the canonical Spatial representation.
    """
    if tuple(da.dims) not in CANONICAL_DIMS:
        raise ValueError(
            "Spatial data must have dims ('band', 'y', 'x') or "
            f"('time', 'band', 'y', 'x'); got {tuple(da.dims)}"
        )

    names = da.coords["band"].values.tolist() if "band" in da.coords else []
    if not names:
        raise ValueError("Spatial data needs a non-empty 'band' coordinate")
    if any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in names):
        raise ValueError(f"Band names must be non-empty trimmed strings, got {names!r}")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"Band names must be unique, repeated: {repeated}")

    if "time" in da.dims:
        if "time" not in da.coords:
            raise ValueError("A 'time' dimension requires a datetime64 'time' coordinate")
        if not np.issubdtype(da.coords["time"].dtype, np.datetime64):
            raise ValueError(
                f"'time' coordinate must be datetime64, got {da.coords['time'].dtype}"
            )
        values = da.coords["time"].values
        if values.size == 0:
            raise ValueError("Spatial data has an empty 'time' dimension")
        if np.isnat(values).any():
            raise ValueError("Spatial data's 'time' coordinate contains NaT")
        if np.unique(values).size != values.size:
            raise ValueError("Spatial data's 'time' coordinate contains duplicate timestamps")
        if not np.all(values[:-1] < values[1:]):
            raise ValueError("Spatial data's 'time' coordinate must be strictly increasing")

    nodata = da.rio.nodata
    if nodata is not None:
        from .nodata import cast_nodata

        cast_nodata(nodata, da.dtype)
    return ensure_crs(da)


def ensure_crs(obj: _GeoObj) -> _GeoObj:
    """Raise if obj has no CRS, else make sure its grid_mapping pointer is set.

    Args:
        obj: DataArray or Dataset to check.

    Returns:
        obj with a `spatial_ref` coord and `.encoding["grid_mapping"]`
        pointing at it. Existing DataArray grid coordinates are retained.

    Raises:
        ValueError: `obj.odc.crs` is None.
    """
    crs = obj.odc.crs
    if crs is None:
        raise ValueError("Raster has no CRS set — call .odc.assign_crs(crs) first, or pass one to GeoRaster.open")

    if isinstance(obj, xr.DataArray) and "spatial_ref" in obj.coords:
        if obj.encoding.get("grid_mapping") == "spatial_ref":
            return obj
        obj = obj.copy(deep=False)
        obj.encoding = {**obj.encoding, "grid_mapping": "spatial_ref"}
        return obj
    return obj.odc.assign_crs(crs)
