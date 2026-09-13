"""Write a Dataset as a COG or a plain GeoTIFF, one band per variable.

Tags come from the Dataset's own attrs, so write them with
`rebase(ds, GeoTIFFTags(...))` before writing. Each variable's name travels in
its band's own metadata, which `gdal.read` reads back.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import rasterio
import rioxarray  # noqa: F401  — registers the .rio accessor
from rasterio.enums import ColorInterp

# rasterio ships this as a compiled module, which no type checker can resolve.
from rasterio.shutil import copy  # type: ignore[import-untyped]  # ty: ignore[unresolved-import]

from geosave_engine.geodata.attrs import GDALVariable, GeoTIFFTags, read, rebase

from geosave_engine.geodata.core.convention import TIME_COORDINATE

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

    from geosave_engine.geodata import Dataset

type CreationOptionValue = str | int | float | bool

_FILE_SUFFIXES = (".tif", ".tiff")


class GeoTIFFWriteOptions(TypedDict, total=False):
    """GDAL creation options both TIFF drivers support."""

    compress: str
    predictor: int
    zlevel: int
    num_threads: int | Literal["ALL_CPUS"]
    bigtiff: Literal["YES", "NO", "IF_NEEDED", "IF_SAFER"]
    interleave: Literal["PIXEL", "BAND"]
    sparse_ok: bool
    creation_options: dict[str, CreationOptionValue]


class COGWriteOptions(GeoTIFFWriteOptions, total=False):
    """Creation options GDAL's COG driver supports."""

    blocksize: int
    overview_resampling: str
    overview_count: int


class GTiffWriteOptions(GeoTIFFWriteOptions, total=False):
    """Creation options GDAL's GTiff driver supports."""

    tiled: bool
    blockxsize: int
    blockysize: int
    copy_src_overviews: bool
    photometric: str
    nbits: int


def write_cog(
    ds: Dataset,
    path: str | PathLike[str],
    *,
    overwrite: bool = False,
    **options: Unpack[COGWriteOptions],
) -> Path:
    """Write one Dataset as a Cloud Optimized GeoTIFF, a band per variable.

    Each variable names itself in its band's metadata, and a scalar `time`
    coordinate becomes ``TIFFTAG_DATETIME``.

    Args:
        ds: Cube of `(y, x)` variables sharing one grid.
        path: Output path ending in ``.tif`` or ``.tiff``.
        overwrite: Replace an existing file when true.
        **options: COG creation options.

    Returns:
        The written path.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, the cube carries a `time` dimension
            rather than a scalar coordinate, or a scalar `time` carries
            sub-second precision the tag cannot hold.

    Examples:
        >>> write_cog(ds, "scene.tif", compress="DEFLATE")
        PosixPath('scene.tif')
    """
    return _write(ds, path, "COG", overwrite=overwrite, options=options)


def write_gtiff(
    ds: Dataset,
    path: str | PathLike[str],
    *,
    overwrite: bool = False,
    **options: Unpack[GTiffWriteOptions],
) -> Path:
    """Write one Dataset as a plain GeoTIFF, a band per variable.

    Reach for `write_cog` unless a consumer needs a striped or otherwise
    non-COG file. Coordinates map to tags exactly as `write_cog` maps them.

    Args:
        ds: Cube of `(y, x)` variables sharing one grid.
        path: Output path ending in ``.tif`` or ``.tiff``.
        overwrite: Replace an existing file when true.
        **options: GTiff creation options.

    Returns:
        The written path.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix is wrong, the cube carries a `time` dimension
            rather than a scalar coordinate, or a scalar `time` carries
            sub-second precision the tag cannot hold.

    Examples:
        >>> write_gtiff(ds, "scene.tif", tiled=True, blockxsize=512)
        PosixPath('scene.tif')
    """
    return _write(ds, path, "GTiff", overwrite=overwrite, options=options)


def _write(
    ds: Dataset,
    path: str | PathLike[str],
    driver: Literal["COG", "GTiff"],
    *,
    overwrite: bool,
    options: Mapping[str, Any],
) -> Path:
    """Encode one cube through the GDAL driver `driver` names.

    Args:
        ds: Cube of `(y, x)` variables sharing one grid.
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
    if TIME_COORDINATE in ds.dims:
        raise ValueError(
            f"one GeoTIFF holds one grid, but this cube spans "
            f"{ds.sizes[TIME_COORDINATE]} instants; select one with "
            f".sel(time=...) or resample the time axis away first"
        )
    # A scalar time coordinate is the instant, so it becomes the datetime tag.
    cube = ds
    if TIME_COORDINATE in cube.coords:
        instant = GeoTIFFTags.model_validate(
            {"TIFFTAG_DATETIME": cube[TIME_COORDINATE].values}
        )
        cube = rebase(cube, instant).drop_vars(TIME_COORDINATE)

    header = read(cube)
    tags = header.root.get(GeoTIFFTags) or GeoTIFFTags()
    colorinterp = [
        (header.data_vars[str(name)].get(GDALVariable) or GDALVariable()).colorinterp
        for name in cube.data_vars
    ]

    # A band names itself in a tag, but its interpretation has a slot of GDAL's own.
    for name in cube.data_vars:
        cube = rebase(
            cube,
            GDALVariable(variable_name=str(name), colorinterp=None),
            target=str(name),
        )

    # rioxarray computes a chunked cube whole unless told to walk its windows.
    streaming: Mapping[str, Any] = {"windowed": True} if cube.chunks else {}

    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as workspace:
        # A COG is copied from a source raster, which its own options create.
        if driver == "COG":
            source = Path(workspace) / target.name
            source_options: Mapping[str, Any] = {"tiled": True}
        else:
            source = target
            source_options = options

        cube.rio.to_raster(
            source,
            driver="GTiff",
            tags=tags.to_attrs(),
            **streaming,
            **source_options,
        )
        # GDAL honours this as a property of an open dataset, not as an option.
        if any(colorinterp):
            with rasterio.open(source, "r+") as dst:
                dst.colorinterp = [
                    ColorInterp[band or ColorInterp.undefined.name]
                    for band in colorinterp
                ]

        if driver == "COG":
            copy(source, target, driver="COG", **options)
    return target
