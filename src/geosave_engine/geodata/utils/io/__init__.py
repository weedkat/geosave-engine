"""One module per file format, each reading and writing that format alone.

Every module has a `read`; a writable format also has a `write`, except
GeoTIFF, whose two drivers are `write_cog` and `write_gtiff`. Reach for them
module-qualified, so the format is named once:

Examples:
    >>> from geosave_engine.geodata.utils import io
    >>> raster = io.zarr.read("scene.zarr", mask_and_scale=True)
    >>> io.geotiff.write_cog(raster, "scene.tif")
    >>> io.read("scene.zarr")  # by suffix, when the format is not known

The fluent `to_cog`, `to_zarr`, and `to_netcdf` names belong to the `gs`
accessors, not here.
"""

from __future__ import annotations

from pathlib import PurePath
from typing import TYPE_CHECKING, Any

from geosave_engine.geodata.core.vector import GeoVector

from . import gdal, geojson, geopackage, geoparquet, geotiff, layout, netcdf, zarr
from .gdal import RasterioOpenOptions
from .geojson import GeoJSONOpenOptions
from .geopackage import GeoPackageOpenOptions
from .geoparquet import GeoParquetOpenOptions
from .geotiff import (
    COGWriteOptions,
    GeoTIFFTags,
    GeoTIFFWriteOptions,
    GTiffWriteOptions,
)
from .layout import FlatLayout, Layout, NestedLayout, SAFELayout
from .netcdf import NetCDFOpenOptions, NetCDFWriteOptions
from .zarr import ZarrOpenOptions, ZarrWriteOptions

if TYPE_CHECKING:
    from os import PathLike

    from geosave_engine.geodata import Dataset, DataTree

# Stores holding named variables, and the only formats holding groups.
_DATASET_SUFFIXES = (".zarr", ".nc", ".nc4", ".cdf")

# Files holding one banded array, which GDAL reads.
_ARRAY_SUFFIXES = (".tif", ".tiff", ".jp2", ".png")

_VECTOR_SUFFIXES = (".geojson", ".json", ".gpkg", ".parquet", ".geoparquet")

__all__ = [
    "COGWriteOptions",
    "FlatLayout",
    "GTiffWriteOptions",
    "GeoJSONOpenOptions",
    "GeoPackageOpenOptions",
    "GeoParquetOpenOptions",
    "GeoTIFFTags",
    "GeoTIFFWriteOptions",
    "Layout",
    "NestedLayout",
    "NetCDFOpenOptions",
    "NetCDFWriteOptions",
    "RasterioOpenOptions",
    "SAFELayout",
    "ZarrOpenOptions",
    "ZarrWriteOptions",
    "gdal",
    "geojson",
    "geopackage",
    "geoparquet",
    "geotiff",
    "layout",
    "netcdf",
    "read",
    "zarr",
]


def read(
    source: str | PathLike[str],
    *,
    stack: bool = False,
    **options: Any,
) -> Dataset | DataTree | GeoVector:
    """Read a path through the reader its suffix names.

    Forwards `options` untouched. Each format module types its own options as
    keywords, so reach for `zarr.read`, `netcdf.read`, or `gdal.read` when
    configuring a read.

    Args:
        source: Local path or URI ending in a recognised suffix.
        stack: Read a multi-group Zarr or NetCDF store as a raster stack.
            Only these two formats hold one.
        **options: Forwarded to the reader the suffix selects.

    Returns:
        `Dataset` for a raster file or store, `DataTree` when `stack` is set,
        and `GeoVector` for a vector file.

    Raises:
        ValueError: The suffix names no supported format, or `stack` is set for
            a format that holds no groups.

    Examples:
        >>> read("scene.zarr").gs.variables
        ("red", "nir")
        >>> read("scene.tif").gs.variables
        ('B04', 'B08')
        >>> read("plantations.geojson").crs.to_epsg()
        4326
        >>> read("training.zarr", stack=True).gs.groups
        ('sentinel-2-l2a', 'dem')
    """
    suffix = PurePath(str(source)).suffix.lower()

    if stack and suffix not in _DATASET_SUFFIXES:
        raise ValueError(
            f"only {_DATASET_SUFFIXES} hold groups, so {source} cannot be read "
            f"as a stack; drop stack=True"
        )

    if suffix == ".zarr":
        if stack:
            return zarr.read_stack(source, **options)
        return zarr.read(source, **options)

    if suffix in (".nc", ".nc4", ".cdf"):
        if stack:
            return netcdf.read_stack(source, **options)
        return netcdf.read(source, **options)

    if suffix in _ARRAY_SUFFIXES:
        return gdal.read(source, **options)

    if suffix in _VECTOR_SUFFIXES:
        if suffix in (".geojson", ".json"):
            return GeoVector(geojson.read(source, **options))
        if suffix == ".gpkg":
            return GeoVector(geopackage.read(source, **options))
        return GeoVector(geoparquet.read(source, **options))

    raise ValueError(
        f"{source} ends in {suffix!r}, which names no supported format; read a "
        f"store ending in {_DATASET_SUFFIXES}, a raster ending in "
        f"{_ARRAY_SUFFIXES}, or a vector ending in {_VECTOR_SUFFIXES}"
    )
