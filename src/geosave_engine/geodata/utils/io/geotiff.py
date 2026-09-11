"""Write one banded DataArray as a COG or a plain GeoTIFF.

Tags come from the array's own attrs, so state them with
`rebase(array, GeoTIFFTags(...))` before writing. Reading is `gdal.read`, which
opens these files and every other raster GDAL supports.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import rioxarray  # noqa: F401  — registers the .rio accessor

from geosave_engine.geodata.attrs import GeoTIFFTags, read, rebase

from geosave_engine.geodata.core.array import BAND_DIMENSION

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

    import numpy as np
    import xarray as xr

type CreationOptionValue = str | int | float | bool

_FILE_SUFFIXES = (".tif", ".tiff")
_BAND_DESCRIPTIONS = "long_name"
_TIME_COORDINATE = "time"


class GeoTIFFWriteOptions(TypedDict, total=False):
    """Optional rioxarray and GDAL behavior shared by both TIFF writers."""

    dtype: str | np.dtype[Any] | None
    windowed: bool
    lock: Any
    compress: str
    predictor: int
    zlevel: int
    num_threads: int | Literal["ALL_CPUS"]
    bigtiff: Literal["YES", "NO", "IF_NEEDED", "IF_SAFER"]
    interleave: Literal["PIXEL", "BAND"]
    sparse_ok: bool
    creation_options: dict[str, CreationOptionValue]


class COGWriteOptions(GeoTIFFWriteOptions, total=False):
    """Optional behavior supported by GDAL's COG driver."""

    blocksize: int
    overview_resampling: str
    overview_count: int


class GTiffWriteOptions(GeoTIFFWriteOptions, total=False):
    """Optional behavior supported by GDAL's GTiff driver."""

    tiled: bool
    blockxsize: int
    blockysize: int
    copy_src_overviews: bool
    photometric: str
    nbits: int


def write_cog(
    array: xr.DataArray,
    path: str | PathLike[str],
    *,
    overwrite: bool = False,
    **options: Unpack[COGWriteOptions],
) -> Path:
    """Write one array as a Cloud Optimized GeoTIFF.

    The `band` coordinate's labels become GDAL band descriptions, and a
    scalar `time` coordinate becomes ``TIFFTAG_DATETIME``.

    Args:
        array: Array shaped `(band, y, x)`.
        path: Output path ending in ``.tif`` or ``.tiff``.
        overwrite: Replace an existing file when true.
        **options: COG creation options.

    Returns:
        The written path.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, the array carries a `time` dimension
            rather than a scalar coordinate, or a scalar `time` carries
            sub-second precision the tag cannot hold.

    Examples:
        >>> write_cog(array, "scene.tif", compress="DEFLATE")
        PosixPath('scene.tif')
    """
    return _write(array, path, "COG", overwrite=overwrite, options=options)


def write_gtiff(
    array: xr.DataArray,
    path: str | PathLike[str],
    *,
    overwrite: bool = False,
    **options: Unpack[GTiffWriteOptions],
) -> Path:
    """Write one array as a plain GeoTIFF.

    Reach for `write_cog` unless a consumer needs a striped or otherwise
    non-COG file. Coordinates map to tags exactly as `write_cog` maps them.

    Args:
        array: Array shaped `(band, y, x)`.
        path: Output path ending in ``.tif`` or ``.tiff``.
        overwrite: Replace an existing file when true.
        **options: GTiff creation options.

    Returns:
        The written path.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, the array carries a `time` dimension
            rather than a scalar coordinate, or a scalar `time` carries
            sub-second precision the tag cannot hold.

    Examples:
        >>> write_gtiff(array, "scene.tif", tiled=True, blockxsize=512)
        PosixPath('scene.tif')
    """
    return _write(array, path, "GTiff", overwrite=overwrite, options=options)


def _write(
    array: xr.DataArray,
    path: str | PathLike[str],
    driver: Literal["COG", "GTiff"],
    *,
    overwrite: bool,
    options: Mapping[str, Any],
) -> Path:
    """Encode one array through the GDAL driver `driver` names.

    Args:
        array: Array shaped `(band, y, x)`.
        path: Output path ending in ``.tif`` or ``.tiff``.
        driver: GDAL driver creating the file.
        overwrite: Replace an existing file when true.
        options: Creation options the driver supports.

    Returns:
        The written path.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix or the `time` axis is invalid.
    """
    target = Path(path)
    if target.suffix not in _FILE_SUFFIXES:
        raise ValueError(
            f"destination {target.name!r} must end in one of {list(_FILE_SUFFIXES)}"
        )
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")

    # One file holds one grid, so an instant travels in the datetime tag.
    if _TIME_COORDINATE in array.dims:
        raise ValueError(
            f"one GeoTIFF holds one grid, but this array spans "
            f"{array.sizes[_TIME_COORDINATE]} instants; select one with "
            f".sel(time=...) or resample the time axis away first"
        )
    # A scalar time coordinate is the instant, so it becomes the datetime tag.
    payload = array
    if _TIME_COORDINATE in payload.coords:
        instant = GeoTIFFTags.model_validate(
            {"TIFFTAG_DATETIME": payload[_TIME_COORDINATE].values}
        )
        payload = rebase(payload, instant).drop_vars(_TIME_COORDINATE)

    carried = read(payload).root.get(GeoTIFFTags)
    stated = carried if isinstance(carried, GeoTIFFTags) else GeoTIFFTags()

    # GDAL reads band descriptions off this one attr, so band labels ride there.
    if BAND_DIMENSION in payload.dims and BAND_DIMENSION in payload.coords:
        payload = payload.assign_attrs(
            {
                _BAND_DESCRIPTIONS: tuple(
                    str(label) for label in payload[BAND_DIMENSION].values
                )
            }
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    payload.rio.to_raster(target, driver=driver, tags=stated.to_attrs(), **options)
    return target
