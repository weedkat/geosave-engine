"""CF-compliant Zarr I/O for xr.Dataset.

No GeoTile dependency — GeoTile's own to_zarr/from_zarr call these
internally, but these operate purely on xarray objects.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import xarray as xr

from .geodata import validate_ds


def to_zarr(
    path: str | Path,
    ds: xr.Dataset,
    group: str | None = None,
    chunk_px: int | None = 512,
    zarr_format: Literal[2, 3] | None = None,
) -> Path:
    """Write a CF-compliant Zarr store/group — one variable per name, e.g.::

        <path>/[<group>/]
          B02          (y, x) or (time, y, x)
          B03          (y, x) or (time, y, x)
          spatial_ref  # CRS grid-mapping coord, shared by every variable
          var_order    # records B02/B03's write order — zarr forgets it on reopen

    Same shape `odc.stac.load()` itself produces — not a format of our own.
    `var_order` comes from `validate_ds` — zarr forgets variable write
    order on reopen, this is what `from_zarr` restores it from.

    Args:
        path: Output Zarr store path.
        ds: Dataset to write, one variable per band. Any attrs already on
            it are kept.
        group: Zarr group to write into; None writes the store root. Several
            groups can share one store — each written independently, own
            attrs, own dims — nothing forces them into a common shape.
        chunk_px: Spatial (y/x) chunk side length. `time`, if present, is
            never split. None skips chunking — zarr's own default applies.
        zarr_format: On-disk spec version. None keeps the installed zarr
            package's own default. 2 for readers stuck on zarr-python <3
            (e.g. xcube-core, as of this writing).

    Returns:
        The written store path.

    Raises:
        ValueError: `path` doesn't end in `.zarr`, or `ds` fails `validate_ds` (see there).
    """
    path = Path(path)
    if path.suffix != ".zarr":
        raise ValueError(f"Expected a .zarr path, got: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = validate_ds(ds)
    if chunk_px is not None:
        chunks = {d: chunk_px for d in ("y", "x") if d in ds.dims}
        if "time" in ds.dims:
            chunks["time"] = -1
        ds = ds.chunk(chunks)
    ds.to_zarr(path, mode="w", group=group, consolidated=True, zarr_format=zarr_format)
    return path


def from_zarr(path: str | Path, group: str | None = None) -> xr.Dataset:
    """Read a to_zarr store/group back, reindexed to its original variable order.

    Args:
        path: Store written by `to_zarr`.
        group: Zarr group to read; None reads the store root.

    Returns:
        Dataset with `data_vars` in write order, `var_order` coord still
        attached. A store not written by `to_zarr` (missing/stale
        `var_order` coord) warns and comes back as opened.

    Raises:
        ValueError: `path` doesn't end in `.zarr`.

    Examples:
        >>> to_zarr("data/train/13.0000E_52.0000N_5kmx5km_10m.zarr", ds, group="sentinel_2_l1c/0")  # nested group, "/" works today
        >>> from_zarr("data/train/13.0000E_52.0000N_5kmx5km_10m.zarr", group="sentinel_2_l1c/0")  # same nested group back
        >>> from_zarr("data/train/13.0000E_52.0000N_5kmx5km_10m.zarr")  # group=None reads the store root instead
    """
    path = Path(path)
    if path.suffix != ".zarr":
        raise ValueError(f"Expected a .zarr path, got: {path}")
    ds = xr.open_zarr(path, group=group, decode_coords="all")

    if "var_order" not in ds.coords:
        warnings.warn(f"{path} has no var_order coord — not written by to_zarr, returning as opened")
        return ds
    var_order = ds.coords["var_order"].values.tolist()
    if set(var_order) != set(ds.data_vars):
        warnings.warn(
            f"{path} var_order {sorted(var_order)} doesn't match variables "
            f"{sorted(map(str, ds.data_vars))} — returning as opened"
        )
        return ds

    return validate_ds(ds)
