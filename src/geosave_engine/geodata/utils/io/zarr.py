"""CF-compliant Zarr I/O for xr.Dataset."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Unpack, cast, overload

import xarray as xr

import geopandas as gpd

if TYPE_CHECKING:
    from dask.delayed import Delayed

from ..array import cf_to_da, progress_bar
from .options import ZarrOptions, check_options
from .vector import write_sidecar

# Options to_zarr sets itself, and the named parameter that sets each.
_OWNED_ZARR = {"store": "path", "compute": "compute"}
# Options that would break the store this writer maintains.
_BLOCKED_ZARR = {
    "zarr_version": "a deprecated alias of zarr_format, which would silently disagree with it",
    "consolidated": "derived from zarr_format, so overriding it can make the store unreadable "
                    "by the very readers that format targets",
    "mode": "a stack stamps its layer order on the store root after writing each group; "
            "any mode but a fresh write can leave that root disagreeing with the groups",
    "append_dim": "the header's array spec and time span describe the pre-append array, "
                  "so appending leaves them stale against their own pixels",
    "region": "a region write needs the store to already exist at full extent, and the vector "
              "sidecar is whole-store — write regions through a method that owns both",
}


@overload
def to_zarr(
    path: str | Path,
    ds: xr.Dataset,
    vector: gpd.GeoDataFrame | None = ...,
    chunk_px: int | None = ...,
    progress: bool = ...,
    compute: Literal[True] = ...,
    **options: Unpack[ZarrOptions],
) -> Path: ...


@overload
def to_zarr(
    path: str | Path,
    ds: xr.Dataset,
    vector: gpd.GeoDataFrame | None = ...,
    chunk_px: int | None = ...,
    progress: bool = ...,
    *,
    compute: Literal[False],
    **options: Unpack[ZarrOptions],
) -> Delayed: ...


def to_zarr(
    path: str | Path,
    ds: xr.Dataset,
    vector: gpd.GeoDataFrame | None = None,
    chunk_px: int | None = 512,
    progress: bool = True,
    compute: bool = True,
    **options: Unpack[ZarrOptions],
) -> Path | Delayed:
    """Write a CF-compliant Zarr store/group, e.g.::

        <path>/[<group>/]
          B02, B03         one variable per band, (y, x) or (time, y, x)
          spatial_ref      CRS coordinate

    Args:
        path: Output Zarr store path.
        ds: Dataset already in CF form. Its `.attrs` are written as they
            stand, so unsupported values must already be encoded.
        chunk_px: Spatial (y/x) chunk side length, applied with `ds.chunk()`
            before writing. `time` is never split. None leaves ds as it is.
        vector: Features to write to this store's sidecar. None removes a
            stale one, so the sidecar always matches the pixels.
        progress: Show a dask progress bar while the pixels compute.
            No-op if they're already in memory or `compute` is False.
        compute: False defers the pixel writes and returns the pending write
            instead of the path, so several of them run under one
            `dask.compute` and share the tasks their sources have in common.
            Structure, attrs and sidecar are written either way.
        **options: How the store is written — group, chunking, spec version,
            encoding, and anything else xarray takes. See `ZarrOptions`.

    Returns:
        The written store path, or the pending pixel write when `compute`
        is False — the caller computes it to finish the store.

    Raises:
        ValueError: `path` doesn't end in `.zarr`, or an option in `options`
            is one this writer sets itself or cannot forward.
        TypeError: `ds` is not a Dataset.
    """
    # group and zarr_format are read here and still forwarded to xarray as its own
    group, zarr_format = options.get("group"), options.get("zarr_format", 3)
    check_options(options, owned=_OWNED_ZARR, blocked=_BLOCKED_ZARR, writer="to_zarr")
    
    path = Path(path)
    if path.suffix != ".zarr":
        raise ValueError(f"Expected a .zarr path, got: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Zarr writer requires a CF Dataset, got {type(ds).__name__}")

    encoding: Any = options.get("encoding")
    if chunk_px is not None:
        chunked = [name for name, spec in (encoding or {}).items() if "chunks" in spec]
        if chunked:
            raise ValueError(
                f"encoding sets chunks for {chunked}, which chunk_px already sets — "
                "pass chunk_px=None to chunk through encoding instead"
            )
        chunks = {d: chunk_px for d in ("y", "x") if d in ds.dims}
        if "time" in ds.dims:
            chunks["time"] = -1
        ds = ds.chunk(chunks)
    # to_zarr overloads on compute's literal, and no overload matches through a TypedDict spread
    forwarded = cast("dict[str, Any]", options)
    if not compute:
        deferred = ds.to_zarr(path, mode="w", consolidated=zarr_format == 2, compute=False, **forwarded)
        write_sidecar(path, vector, group)
        return deferred
    
    with progress_bar(progress):
        ds.to_zarr(path, mode="w", consolidated=zarr_format == 2, compute=True, **forwarded)
    write_sidecar(path, vector, group)
    return path


def from_zarr(path: str | Path, group: str | None = None) -> xr.DataArray:
    """Open a Zarr store/group as a canonical lazy array.

    Args:
        path: Store to open.
        group: Zarr group to read; None reads the store root.

    Returns:
        Canonical, dask-backed Spatial array with encoded attrs unchanged.

    Raises:
        ValueError: `path` doesn't end in `.zarr`.

    Examples:
        >>> from_zarr("data/train/13.0000E_52.0000N_5kmx5km_10m.zarr", group="sentinel_2_l1c/0")
        >>> from_zarr("data/train/13.0000E_52.0000N_5kmx5km_10m.zarr")  # group=None reads the root
    """
    path = Path(path)
    if path.suffix != ".zarr":
        raise ValueError(f"Expected a .zarr path, got: {path}")
    # mask_and_scale would swap nodata for NaN and upcast an int raster to float — rioxarray defaults it off too
    return cf_to_da(
        xr.open_zarr(
            path,
            group=group,
            decode_coords="all",
            mask_and_scale=False,
            consolidated=False,
        )
    )
