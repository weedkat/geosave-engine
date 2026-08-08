from __future__ import annotations

from datetime import datetime as dt
from typing import Any, cast

import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray/Dataset
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords

BANDLESS_VAR_NAME = "data"  # da_to_ds's sentinel var name for a bandless da; reserved, can't be a real band name


def np_to_da(
    geobox: GeoBox,
    array: np.ndarray,
    names: str | list[str] | None = None,
    times: list[dt] | None = None,
) -> xr.DataArray:
    """Shape a plain array into a DataArray on geobox — preserves `spatial_ref`, bare `xr.DataArray(...)` doesn't.

    2D is one implicit band. 3D needs `names`. 4D needs `names` and `times`.

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


def default_nodata(dtype: np.dtype) -> float | int:
    """Sentinel nodata for a dtype with none declared.

    NaN for float, dtype max for unsigned int, dtype min for signed int.

    Raises:
        ValueError: `dtype` isn't float or int (e.g. bool has no safe sentinel).
    """
    if np.issubdtype(dtype, np.floating):
        return float("nan")
    if np.issubdtype(dtype, np.unsignedinteger):
        return int(np.iinfo(dtype).max)
    if np.issubdtype(dtype, np.integer):
        return int(np.iinfo(dtype).min)
    raise ValueError(f"default_nodata(): no safe sentinel for dtype {dtype}")


def validate_da(da: xr.DataArray) -> xr.DataArray:
    """Coerce da into this library's array format — CF dims/CRS, plus our band-as-coordinate extension.

    No `band` dim is valid too — means one implicit band, not an error.

    Args:
        da: Array to check.

    Returns:
        `da`, canonical dim order, CRS set.

    Raises:
        ValueError: Missing `x`/`y` dim, unexpected dims, missing `band`
            coordinate, a band literally named `"data"`, or no CRS set
            (`da.odc.crs is None`).
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
    if "band" in dims:
        if "band" not in da.coords:
            raise ValueError("DataArray has a 'band' dim but no 'band' coordinate")
        if BANDLESS_VAR_NAME in da.band.values:
            raise ValueError(f"{BANDLESS_VAR_NAME!r} is reserved for a bandless array, can't be a real band name")
    if da.odc.crs is None:
        raise ValueError("validate_da(): da has no CRS set — call da.odc.assign_crs(crs) first")
    da = da.odc.assign_crs(da.odc.crs)  # stamp grid_mapping — CRS-detectable alone doesn't mean it's declared
    return da.transpose(*canonical)


def da_to_ds(da: xr.DataArray) -> xr.Dataset:
    """Split da into this library's disk format — one variable per band, or one `data` variable if bandless.

    Args:
        da: Array to split, dims `(band, y, x)`, `(y, x)`, `(time, band, y, x)`, or `(time, y, x)`.

    Returns:
        Dataset with one variable per band, or one `data` variable.
    """
    if "band" in da.dims:
        return da.to_dataset(dim="band")
    return da.to_dataset(name=BANDLESS_VAR_NAME)


def ds_to_da(ds: xr.Dataset) -> xr.DataArray:
    """Reverse of `da_to_ds`.

    Args:
        ds: Dataset built by `da_to_ds` (var_order-restored, see `validate_ds`/`from_zarr`).

    Returns:
        `da`, with nodata reattached (`to_array` stacking drops it otherwise).
    """
    has_band = not (next(iter(ds.data_vars)) == BANDLESS_VAR_NAME and len(ds.data_vars) == 1)
    if has_band:
        nodata = next(iter(ds.data_vars.values())).rio.nodata
        da = ds[list(ds.data_vars)].to_array(dim="band")  # subset first — stray coords (e.g. var_order) don't fit "band"
        da = da.transpose("time", "band", "y", "x") if "time" in da.dims else da.transpose("band", "y", "x")
    else:
        da = next(iter(ds.data_vars.values()))
        nodata = da.rio.nodata
        da = da.transpose("time", "y", "x") if "time" in da.dims else da.transpose("y", "x")
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    return da


def validate_ds(ds: xr.Dataset) -> xr.Dataset:
    """Coerce ds into this library's disk format — CF's one-variable-per-band, plus our var_order extension.

    Args:
        ds: Dataset to check.

    Returns:
        `ds`, canonical variable order and per-variable dim order, CRS
        `grid_mapping` set on every variable.

    Raises:
        ValueError: A variable has a `band` dim, an unexpected dim, or the
            Dataset has no CRS set (`ds.odc.crs is None`).
    """
    if "var_order" in ds.coords:
        var_order: list[str] = ds.coords["var_order"].values.tolist()
        if set(var_order) == set(ds.data_vars):
            ds = cast(xr.Dataset, ds[var_order])  # restore write-time order; zarr alphabetizes on reopen

    if ds.odc.crs is None:
        raise ValueError("validate_ds(): ds has no CRS set — call ds.odc.assign_crs(crs) first")
    ds = ds.odc.assign_crs(ds.odc.crs)  # defensive backstop — validate_da should've stamped it already

    for name, var in ds.data_vars.items():
        dims = set(var.dims)
        if dims == {"y", "x"}:
            canonical = ("y", "x")
        elif dims == {"time", "y", "x"}:
            canonical = ("time", "y", "x")
        else:
            raise ValueError(f"Variable {name!r}: expected dims (y, x) or (time, y, x), got {tuple(var.dims)}")
        ds[name] = var.transpose(*canonical)

    return ds.assign_coords(var_order=("var", list(ds.data_vars)))  # refresh for the next to_zarr write
