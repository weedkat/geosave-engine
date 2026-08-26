"""CF-compliant NetCDF I/O for xr.Dataset. Same layout as Zarr."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Unpack, cast

import xarray as xr
from netCDF4 import Dataset

import geopandas as gpd

from ..array import cf_to_da, progress_bar
from .options import NetcdfFormat, NetcdfOptions, check_options
from .vector import write_sidecar

# Options to_netcdf sets itself, and the named parameter that sets each.
_OWNED_NETCDF = {"path": "path", "compute": "progress"}
# Options that would break the file this writer maintains.
_BLOCKED_NETCDF = {
    "mode": "derived from group — a root write owns the file, a group write preserves siblings",
    "engine": "netcdf4 is what from_netcdf reads back; another engine differs on group support, "
              "so the file would not reopen through this library",
}

# re-exported: the writer's own format option is declared beside the rest of its config
NetcdfFormat = NetcdfFormat


def _group_exists(path: Path, group: str) -> bool:
    with Dataset(path, mode="r") as root:
        current = root
        for name in group.strip("/").split("/"):
            if name not in current.groups:
                return False
            current = current.groups[name]
    return True


def to_netcdf(
    path: str | Path,
    ds: xr.Dataset,
    vector: gpd.GeoDataFrame | None = None,
    chunk_px: int | None = 512,
    progress: bool = True,
    **options: Unpack[NetcdfOptions],
) -> Path:
    """Write a CF-compliant NetCDF store/group — same layout as `to_zarr`.

    Args:
        path: Output NetCDF path, must end in `.nc`.
        ds: Dataset already in CF form. Its `.attrs` are written as they
            stand, so unsupported values must already be encoded.
        chunk_px: Spatial (y/x) chunk side length, applied as each data
            variable's on-disk `chunksizes` encoding. `time` is never split.
            None leaves the library default.
        vector: Features to write to this file's sidecar. None removes a
            stale one, so the sidecar always matches the pixels.
        progress: Show a dask progress bar while the pixels compute.
        **options: How the file is written — group, chunking, format version,
            encoding, and anything else xarray takes. See `NetcdfOptions`.
            No-op if they're already in memory.

    Returns:
        The written store path.

    Raises:
        ValueError: `path` doesn't end in `.nc` or `group` is empty.
        TypeError: `ds` is not a Dataset.
        FileExistsError: `group` already exists. NetCDF groups cannot be
            replaced without rebuilding the whole file.
    """
    group = options.get("group")
    check_options(options, owned=_OWNED_NETCDF, blocked=_BLOCKED_NETCDF, writer="to_netcdf")
    path = Path(path)
    if path.suffix != ".nc":
        raise ValueError(f"Expected a .nc path, got: {path}")
    if group is not None and not group.strip("/"):
        raise ValueError("group must contain a name")
    if group is not None and path.exists() and _group_exists(path, group):
        raise FileExistsError(f"NetCDF group already exists: {group}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"NetCDF writer requires a CF Dataset, got {type(ds).__name__}")

    # caller encoding layers onto the writer's own: var.encoding carries grid_mapping
    supplied: Any = options.pop("encoding", None) or {}
    encoding: dict[str, Any] | None = None
    if chunk_px is not None or supplied:
        encoding = {}
        for name, var in ds.data_vars.items():
            derived: dict[str, Any] = dict(var.encoding)
            if chunk_px is not None:
                derived["chunksizes"] = tuple(
                    min(chunk_px, var.sizes[d]) if d in ("y", "x") else var.sizes[d] for d in var.dims
                )
                derived["zlib"] = True
            encoding[str(name)] = {**derived, **dict(supplied.get(str(name), {}))}

    # A root write owns the whole file; group writes preserve sibling groups.
    mode: Literal["w", "a"] = "w" if group is None else "a"
    with progress_bar(progress):
        # same widening as to_zarr: an overload cannot be matched through a TypedDict
        ds.to_netcdf(
            path, mode=mode, engine="netcdf4", encoding=encoding, compute=True,
            **cast("dict[str, Any]", options),
        )
    write_sidecar(path, vector, group)
    return path


def from_netcdf(path: str | Path, group: str | None = None) -> xr.DataArray:
    """Open a NetCDF store/group as a canonical lazy array.

    Args:
        path: File to open.
        group: NetCDF4 group to read; None reads the file root.

    Returns:
        Canonical, dask-backed Spatial array with encoded attrs unchanged.

    Raises:
        ValueError: `path` doesn't end in `.nc`.
    """
    path = Path(path)
    if path.suffix != ".nc":
        raise ValueError(f"Expected a .nc path, got: {path}")
    # mask_and_scale would swap nodata for NaN and upcast an int raster to float — rioxarray defaults it off too
    ds = xr.open_dataset(
        path, group=group, engine="netcdf4", decode_coords="all", chunks={}, mask_and_scale=False
    )
    return cf_to_da(ds)
