"""GeoTIFF/COG I/O for xr.DataArray.

No GeoTile dependency — GeoTile's own to_geotiff/from_geotiff call these
internally, but these operate purely on xarray objects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray/Dataset
import xarray as xr

from .geodata import validate_da


def to_geotiff(
    path: str | Path,
    da: xr.DataArray,
    driver: str = "GTiff",
    tags: dict[str, str] | None = None,
    chunk_px: int | None = 512,
) -> Path:
    """Write a DataArray to GeoTIFF/COG via rioxarray, one band per array layer.

    Args:
        path: Output `.tif` path.
        da: Array to write, dims `(band, y, x)`.
        driver: GDAL driver — `"GTiff"` or `"COG"`.
        tags: GDAL string tags to attach.
        chunk_px: Tile block side length. GTiff defaults to untiled
            single-row strips otherwise; COG already self-tiles regardless.
            None skips tiling on GTiff, leaves COG's own default alone.

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

    raster_kwargs: dict[str, int | bool] = {}
    if chunk_px is not None:
        if driver == "GTiff":
            raster_kwargs = {"tiled": True, "blockxsize": chunk_px, "blockysize": chunk_px}
        elif driver == "COG":
            raster_kwargs = {"blocksize": chunk_px}

    da.rio.to_raster(path, driver=driver, tags=tags or {}, **raster_kwargs)
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
