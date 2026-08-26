"""GDAL-driver raster I/O via rioxarray — GeoTIFF/COG out, any GDAL raster in."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Unpack, cast

import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray/Dataset
import xarray as xr

import geopandas as gpd

from ..array import progress_bar, validate_spatial
from .options import GeotiffOptions, check_options
from .vector import write_sidecar

# Options to_geotiff sets itself, and the named parameter that sets each.
_OWNED_GEOTIFF = {
    "raster_path": "path", "compute": "progress",
    "tiled": "chunk_px", "blockxsize": "chunk_px", "blockysize": "chunk_px", "blocksize": "chunk_px",
}
# Options that would break the raster this writer maintains.
_BLOCKED_GEOTIFF = {
    "dtype": "casting on write leaves the file disagreeing with the array spec and nodata "
             "stamped from the pixels — call astype(), which moves the sentinel too",
    "recalc_transform": "the transform comes from the anchor's geobox and must not be re-derived",
}

# CF flag_* trio (see cf_flag_attrs) — GDAL tags would stringify its list values via Python repr, not JSON.
_CF_ONLY_ATTRS = frozenset({"flag_values", "flag_meanings", "flag_colors"})

# GDAL's own per-band name slot. rioxarray writes it to band descriptions and reads it back.
BAND_NAME_ATTR = "long_name"


def to_geotiff(
    path: str | Path,
    da: xr.DataArray,
    vector: gpd.GeoDataFrame | None = None,
    chunk_px: int | None = 512,
    progress: bool = True,
    **options: Unpack[GeotiffOptions],
) -> Path:
    """Write a DataArray to GeoTIFF/COG, one file band per array band.

    Band names go to GDAL band descriptions, so `from_rasterio` reads
    them back. CF-only `flag_*` attrs are dropped.

    Args:
        path: Output `.tif` path.
        da: Canonical array with dimensions `(band, y, x)`. Its `.attrs`
            land as GDAL tags as they stand, so anything but a string must
            already be encoded (see `spatial.attrs.encode_attrs`).
        driver: GDAL driver — `"GTiff"` or `"COG"`.
        chunk_px: Tile block side length. GTiff defaults to untiled
            single-row strips otherwise; COG already self-tiles regardless.
            None skips tiling on GTiff, leaves COG's own default alone.
        chunk_px: On-disk block side length, applied as the driver's own
            block keys. None leaves the driver's default.
        vector: Features to write to this file's sidecar. None removes a
            stale one, so the sidecar always matches the pixels.
        progress: Show a dask progress bar while da's pixels compute.
            No-op if da is already in memory.

    Returns:
        The written path.

    Raises:
        ValueError: `path` doesn't end in `.tif`/`.tiff`, `da` has a
            `time` dim (GeoTIFF has no time axis — select one step
            first), or `da` violates the canonical array contract.
    """
    driver: str = options.pop("driver", "GTiff")
    supplied_tags = options.pop("tags", {})
    check_options(options, owned=_OWNED_GEOTIFF, blocked=_BLOCKED_GEOTIFF, writer="to_geotiff")
    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError(f"Expected .tif path, got: {path}")
    da = validate_spatial(da)
    if "time" in da.dims:
        raise ValueError(f"Expected a single-step array (no 'time' dim), got dims {tuple(da.dims)}")
    path.parent.mkdir(parents=True, exist_ok=True)

    raster_kwargs: dict[str, Any] = {}
    if chunk_px is not None:
        if driver == "GTiff":
            raster_kwargs = {"tiled": True, "blockxsize": chunk_px, "blockysize": chunk_px}
        elif driver == "COG":
            raster_kwargs = {"blocksize": chunk_px}
    raster_kwargs |= cast("dict[str, Any]", options)

    # band names into the slot rioxarray turns into GDAL band descriptions
    names = tuple(str(band) for band in da.band.values)

    # GDAL tags would stringify a flag_* list through Python repr, not JSON
    da = da.copy(deep=False)
    # caller tags sit under the writer's own, so band names survive whatever else is stamped
    da.attrs = (
        dict(supplied_tags)
        | {key: value for key, value in da.attrs.items() if key not in _CF_ONLY_ATTRS}
        | {BAND_NAME_ATTR: names}
    )

    with progress_bar(progress):
        da.rio.to_raster(path, driver=driver, **raster_kwargs)
    write_sidecar(path, vector)
    return path


def from_rasterio(path: str | Path) -> xr.DataArray:
    """Read any GDAL-driver raster in GDAL form. Lazy — pixels read only when accessed.

    Band descriptions become the `band` coord; without a complete unique
    set, all bands use GDAL's `1..N` indices as strings.

    Args:
        path: GeoTIFF, COG, PNG, JPEG — anything GDAL opens. A file with
            no georeferencing (bare PNG/JPEG, no world file) comes back
            with no CRS and an identity transform, and needs
            `GeoRaster.open`'s `anchor=` to become a raster.
    Returns:
        Canonical `(band, y, x)` DataArray. Its `.attrs` are the GDAL tags
        as they sit on disk.

    Raises:
        ValueError: `path` holds subdatasets (NetCDF/HDF through GDAL) —
            open the subdataset directly instead.
    """
    opened = rioxarray.open_rasterio(path, chunks=True)
    if not isinstance(opened, xr.DataArray):
        raise ValueError(f"{path} holds subdatasets — open the one you want directly, not the container")

    da = opened.copy(deep=False)
    da.attrs = {key: value for key, value in da.attrs.items() if key != BAND_NAME_ATTR}

    # GDAL band descriptions come back as long_name — one string for a single band, a tuple otherwise
    described = opened.attrs.get(BAND_NAME_ATTR)
    names = [described] if isinstance(described, str) else list(described or [])
    normalized = [str(name).strip() for name in names if name is not None]
    has_descriptions = (
        len(normalized) == da.sizes.get("band", 0)
        and all(normalized)
        and len(set(normalized)) == len(normalized)
    )
    if has_descriptions:
        da = da.assign_coords(band=normalized)
    else:
        da = da.assign_coords(band=[str(value) for value in da.band.values])

    return validate_spatial(da)
