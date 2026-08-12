"""CF-compliant NetCDF I/O for xr.Dataset.

No GeoTile dependency — GeoTile's own to_netcdf/from_netcdf call these
internally, but these operate purely on xarray objects. Same variable
layout as to_zarr (one variable per band, var_order restore) — the two
formats are interchangeable disk representations of the same Dataset.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import xarray as xr

from .geodata import validate_ds

NetcdfFormat = Literal["NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT", "NETCDF3_CLASSIC"]


def to_netcdf(
    path: str | Path,
    ds: xr.Dataset,
    group: str | None = None,
    chunk_px: int | None = 512,
    format: NetcdfFormat | None = None,
) -> Path:
    """Write a CF-compliant NetCDF store/group — same layout as to_zarr.

    Args:
        path: Output NetCDF path, must end in `.nc`.
        ds: Dataset to write, one variable per band. Any attrs already on
            it are kept.
        group: NetCDF4 group to write into; None writes the file root.
            Several groups can share one file — each written independently.
        chunk_px: Spatial (y/x) chunk side length, applied as each data
            variable's on-disk `chunksizes` encoding. `time`, if present,
            is never split. None skips chunking — library default applies.
        format: On-disk NetCDF variant. None keeps the library default (NETCDF4).

    Returns:
        The written store path.

    Raises:
        ValueError: `path` doesn't end in `.nc`, or `ds` fails `validate_ds` (see there).
    """
    path = Path(path)
    if path.suffix != ".nc":
        raise ValueError(f"Expected a .nc path, got: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = validate_ds(ds)

    encoding = None
    if chunk_px is not None:
        encoding = {}
        for name, var in ds.data_vars.items():
            chunksizes = tuple(
                min(chunk_px, var.sizes[d]) if d in ("y", "x") else var.sizes[d] for d in var.dims
            )
            # merge, don't replace — var.encoding already carries grid_mapping
            # (stamped by validate_ds's assign_crs) and must survive to disk
            encoding[name] = {**var.encoding, "chunksizes": chunksizes, "zlib": True}

    # mode="w" truncates the whole FILE on netCDF4 (unlike zarr, where "w"
    # only replaces the target group) — always "a" so sibling groups survive;
    # "a" also creates the file fresh when path doesn't exist yet.
    ds.to_netcdf(path, mode="a", group=group, format=format, engine="netcdf4", encoding=encoding)
    return path


def from_netcdf(path: str | Path, group: str | None = None) -> xr.Dataset:
    """Read a to_netcdf store/group back, reindexed to its original variable order.

    Args:
        path: Store written by `to_netcdf`.
        group: NetCDF4 group to read; None reads the file root.

    Returns:
        Dataset with `data_vars` in write order, `var_order` coord still
        attached. A store not written by `to_netcdf` (missing/stale
        `var_order` coord) warns and comes back as opened.

    Raises:
        ValueError: `path` doesn't end in `.nc`.
    """
    path = Path(path)
    if path.suffix != ".nc":
        raise ValueError(f"Expected a .nc path, got: {path}")
    ds = xr.open_dataset(path, group=group, engine="netcdf4", decode_coords="all")

    if "var_order" not in ds.coords:
        warnings.warn(f"{path} has no var_order coord — not written by to_netcdf, returning as opened")
        return ds
    var_order = ds.coords["var_order"].values.tolist()
    if set(var_order) != set(ds.data_vars):
        warnings.warn(
            f"{path} var_order {sorted(var_order)} doesn't match variables "
            f"{sorted(map(str, ds.data_vars))} — returning as opened"
        )
        return ds

    return validate_ds(ds)
