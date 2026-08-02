from __future__ import annotations

from datetime import datetime as dt
from typing import Any

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray/Dataset
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords


def np_to_da(
    geobox: GeoBox,
    array: np.ndarray,
    names: str | list[str] | None = None,
    times: list[dt] | None = None,
) -> xr.DataArray:
    """Shape a plain array into a DataArray on geobox, preserving spatial
    metadata (y, x, spatial_ref) — use this instead of bare
    ``xr.DataArray(arr, dims=["y","x"], coords={"y":..., "x":...})``, which
    silently drops the ``spatial_ref`` coordinate and breaks CRS detection.

    2D `(y, x)` is a single unnamed band. 3D `(band, y, x)` requires
    `names`. 4D `(time, band, y, x)` requires both `names` and `times`.

    Args:
        geobox: Spatial reference for the array's (y, x) axes.
        array: 2-4D array; last two axes are (y, x).
        names: Band name(s) for a 3D/4D array — a single string for one
            band, or one name per row for several.
        times: Observation datetimes for a 4D array.

    Returns:
        DataArray with dims/coords matching the array's shape.

    Raises:
        ValueError: Wrong ndim, or `names`/`times` missing or mismatched for it.
    """
    arr = np.asarray(array)
    base_coords: dict[Any, Any] = dict(xr_coords(geobox))
    if arr.ndim == 2:
        return xr.DataArray(arr, dims=("y", "x"), coords=base_coords)

    band_names = [names] if isinstance(names, str) else names
    if arr.ndim == 3:
        if band_names is None:
            raise ValueError("names is required for a 3D array (band, y, x)")
        if len(band_names) != arr.shape[0]:
            raise ValueError(f"Expected {arr.shape[0]} names, got {len(band_names)}")
        return xr.DataArray(arr, dims=("band", "y", "x"), coords={**base_coords, "band": band_names})
    if arr.ndim == 4:
        if band_names is None or times is None:
            raise ValueError("names and times are both required for a 4D array (time, band, y, x)")
        if len(band_names) != arr.shape[1]:
            raise ValueError(f"Expected {arr.shape[1]} names, got {len(band_names)}")
        if len(times) != arr.shape[0]:
            raise ValueError(f"Expected {arr.shape[0]} times, got {len(times)}")
        time_coord = [np.datetime64(t, "ns") for t in times]
        return xr.DataArray(
            arr, dims=("time", "band", "y", "x"),
            coords={**base_coords, "band": band_names, "time": time_coord},
        )
    raise ValueError(f"Expected a 2-4D array, got {arr.ndim}D")


def validate_da(da: xr.DataArray) -> xr.DataArray:
    """Check/coerce a DataArray into GeoTile's in-memory shape: one stacked
    array, dims `(band, y, x)`, `(time, band, y, x)`, `(y, x)`, or
    `(time, y, x)`, CRS set. `band` is optional — its absence means the
    array has no named bands (a single implicit one), not an error.

    Fixable (coerced, no error): dims present but out of order — transposed
    to canonical order.

    Not fixable (raises): missing `x`/`y`, missing CRS — nothing to invent
    these from.

    Args:
        da: Array to check.

    Returns:
        `da`, transposed to canonical dim order if needed.

    Raises:
        ValueError: Missing `x`/`y` dim, unexpected dims, or no CRS set
            (`da.rio.crs is None`).
    """
    dims = set(da.dims)
    if dims == {"band", "y", "x"}:
        canonical = ("band", "y", "x")
    elif dims == {"time", "band", "y", "x"}:
        canonical = ("time", "band", "y", "x")
    elif dims == {"y", "x"}:
        canonical = ("y", "x")
    elif dims == {"time", "y", "x"}:
        canonical = ("time", "y", "x")
    else:
        raise ValueError(
            f"Expected dims (band, y, x), (time, band, y, x), (y, x), or (time, y, x), got {tuple(da.dims)}"
        )
    if "band" in dims and "band" not in da.coords:
        raise ValueError("DataArray has a 'band' dim but no 'band' coordinate")
    if da.rio.crs is None:
        raise ValueError("DataArray has no CRS set")
    return da.transpose(*canonical)


def da_to_ds(da: xr.DataArray) -> xr.Dataset:
    """Dataset shape for a Zarr write: one variable per band, or a single
    unnamed `data` variable when `da` has no `band` dim (e.g. a mask/label
    tile built from a plain 2D array).

    `has_band` attr on the returned Dataset records which, so `ds_to_da`
    can invert this losslessly.

    Args:
        da: Array to split, dims `(band, y, x)`, `(y, x)`, `(time, band, y, x)`, or `(time, y, x)`.

    Returns:
        Dataset with one variable per band, or one `data` variable.
    """
    if "band" in da.dims:
        return da.to_dataset(dim="band").assign_attrs(has_band=True)
    return da.to_dataset(name="data").assign_attrs(has_band=False)


def ds_to_da(ds: xr.Dataset) -> xr.DataArray:
    """Reverse of `da_to_ds`.

    Args:
        ds: Dataset carrying a `has_band` attr from `da_to_ds`. Missing
            attr (data predating it) assumed banded.

    Returns:
        `da`, with nodata (read off the source variable(s) before `to_array`
        stacking would otherwise drop it) re-attached.
    """
    has_band = ds.attrs.get("has_band", True)
    if has_band:
        nodata = next(iter(ds.data_vars.values())).rio.nodata
        da = ds.to_array(dim="band")
        da = da.transpose("time", "band", "y", "x") if "time" in da.dims else da.transpose("band", "y", "x")
    else:
        da = next(iter(ds.data_vars.values()))
        nodata = da.rio.nodata
        da = da.transpose("time", "y", "x") if "time" in da.dims else da.transpose("y", "x")
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    return da


def validate_ds(ds: xr.Dataset) -> xr.Dataset:
    """Check/coerce a Dataset into CF disk shape: one variable per band,
    each dims `(y, x)` or `(time, y, x)`, CRS grid-mapping coordinate set.

    Fixable (coerced, no error): a variable's dims present but out of order —
    transposed to canonical order. `grid_mapping` missing/stale on a
    variable — re-stamped via `write_crs` so every store gets it
    consistently, not just ones that happened to pass through a GeoTIFF read.

    Not fixable (raises): a variable still carrying a `band` dim (means it
    was never split into one-variable-per-band — wrong shape for this
    function), missing CRS — nothing to invent these from.

    Args:
        ds: Dataset to check.

    Returns:
        `ds`, each variable transposed to its canonical dim order if needed,
        with CRS `grid_mapping` set on every variable.

    Raises:
        ValueError: A variable has a `band` dim, an unexpected dim, or the
            Dataset has no CRS set (`ds.rio.crs is None`).
    """
    if ds.rio.crs is None:
        raise ValueError("Dataset has no CRS set")
    ds = ds.rio.write_crs(ds.rio.crs)
    for name, var in ds.data_vars.items():
        dims = set(var.dims)
        if "band" in dims:
            raise ValueError(f"Variable {name!r} has a 'band' dim — split into one variable per band first")
        if dims == {"y", "x"}:
            canonical = ("y", "x")
        elif dims == {"time", "y", "x"}:
            canonical = ("time", "y", "x")
        else:
            raise ValueError(f"Variable {name!r}: expected dims (y, x) or (time, y, x), got {tuple(var.dims)}")
        ds[name] = var.transpose(*canonical)
    return ds
