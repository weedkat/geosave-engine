"""CF-compliant Zarr and GeoTIFF/COG I/O for xr.Dataset/DataArray.

No GeoTile dependency — GeoTile's own to_zarr/to_geotiff/from_zarr/from_geotiff
call these internally, but these operate purely on xarray objects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray/Dataset
import xarray as xr

from .geodata import validate_da, validate_ds

# ----------------------------------------------------------------------
# Zarr I/O
# ----------------------------------------------------------------------

def to_zarr(path: str | Path, ds: xr.Dataset, group: str | None = None) -> Path:
    """Write a CF-compliant Zarr store/group — one variable per name, e.g.::

        <path>/[<group>/]
          B02          (y, x) or (time, y, x)
          B03          (y, x) or (time, y, x)
          spatial_ref  # CRS grid-mapping coord, shared by every variable

    Same shape `odc.stac.load()` itself produces — not a format of our own.
    zarr doesn't preserve a Dataset's variable order on reopen (confirmed
    empirically — `xr.open_zarr` lists variables alphabetically regardless
    of `consolidated=True`), so `var_order`/`dim_order` attrs record it;
    `from_zarr` restores it.

    Args:
        path: Output Zarr store path.
        ds: Dataset to write, one variable per band. Any attrs already on
            it are kept — `var_order`/`dim_order` are added alongside.
        group: Zarr group to write into; None writes the store root. Several
            groups can share one store — each written independently, own
            attrs, own dims — nothing forces them into a common shape.

    Returns:
        The written store path.

    Raises:
        ValueError: `ds` fails `validate_ds` (see there).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = validate_ds(ds)
    ds = ds.assign_attrs(
        var_order=list(ds.data_vars),
        dim_order={name: list(da.dims) for name, da in ds.data_vars.items()},
    )
    ds.to_zarr(path, mode="w", group=group, consolidated=True)
    return path


def from_zarr(path: str | Path, group: str | None = None) -> xr.Dataset:
    """Read a to_zarr store/group back, reindexed to its original order.

    Also restores the CRS grid-mapping coordinate — zarr round-trips it
    back as a plain data variable otherwise.

    Args:
        path: Store written by `to_zarr`.
        group: Zarr group to read; None reads the store root.

    Returns:
        Dataset with `data_vars` in `var_order`, each transposed to its own `dim_order`.

    Raises:
        ValueError: Store/group has no `var_order`/`dim_order` attrs, or
            they don't match the variables actually present — not written
            by `to_zarr`.
    """
    path = Path(path)
    ds = xr.open_zarr(path, group=group)
    grid_mappings = {
        var.attrs["grid_mapping"] for var in ds.data_vars.values() if "grid_mapping" in var.attrs
    } & set(ds.data_vars)
    if grid_mappings:
        ds = ds.set_coords(grid_mappings)

    var_order = ds.attrs.get("var_order")
    dim_order = ds.attrs.get("dim_order")
    if var_order is None or dim_order is None:
        raise ValueError(f"{path} has no var_order/dim_order attrs — not written by to_zarr")
    if set(var_order) != set(ds.data_vars):
        raise ValueError(f"{path} var_order {sorted(var_order)} doesn't match variables {sorted(ds.data_vars)}") #type: ignore

    ds = ds[var_order]
    for name in var_order:
        ds[name] = ds[name].transpose(*dim_order[name])
    return ds


# ----------------------------------------------------------------------
# GeoTIFF/COG I/O
# ----------------------------------------------------------------------

def to_geotiff(path: str | Path, da: xr.DataArray, driver: str = "GTiff", tags: dict[str, str] | None = None) -> Path:
    """Write a DataArray to GeoTIFF/COG via rioxarray, one band per array layer.

    Args:
        path: Output `.tif` path.
        da: Array to write, dims `(band, y, x)`.
        driver: GDAL driver — `"GTiff"` or `"COG"`.
        tags: GDAL string tags to attach.

    Returns:
        The written path.

    Raises:
        ValueError: `path` doesn't end in `.tif`/`.tiff`, or `da` fails
            `validate_da` (see there — note `validate_da` allows a `time`
            dim, but GeoTIFF itself has no time axis; callers with a
            multi-step `da` must select a single step before calling this).
    """
    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError(f"Expected .tif path, got: {path}")
    da = validate_da(da)
    if "time" in da.dims:
        raise ValueError(f"Expected a single-step array (no 'time' dim), got dims {tuple(da.dims)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    da.rio.to_raster(path, driver=driver, tags=tags or {})
    return path


def from_geotiff(path: str | Path, bands: tuple[str, ...] | None = None) -> xr.Dataset:
    """Read a GeoTIFF/COG as a Dataset, one variable per band.

    GeoTIFF has no native per-band name slot — variables come back named
    `band_1`, `band_2`, ... regardless of how the source array's bands were
    named at write time. Restored to their real names here from a `"bands"`
    tag (a JSON list, written by `GeoTile.to_geotiff`/`to_cog`), if present.

    Args:
        path: Path to GeoTIFF/COG file.
        bands: Variable names to select (real names if a `"bands"` tag
            restored them, else `band_1`, `band_2`, ...); None keeps all.

    Returns:
        Dataset, lazy — pixels read only when accessed.

    Raises:
        TypeError: `path` isn't a multi-variable raster (rioxarray gave back
            a bare DataArray instead of a Dataset).
    """
    path = Path(path)
    opened = rioxarray.open_rasterio(path, chunks=True, band_as_variable=True)
    if not isinstance(opened, xr.Dataset):
        raise TypeError(f"Expected a Dataset from {path}, got {type(opened).__name__}")

    raw_bands = opened.attrs.get("bands")
    band_names: list[str] | None = json.loads(raw_bands) if raw_bands else None
    if band_names and len(band_names) == len(opened.data_vars):
        opened = opened.rename(dict(zip(list(opened.data_vars), band_names)))

    if bands:
        opened = cast(xr.Dataset, opened[list(bands)])
    return opened
