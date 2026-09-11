"""One entry point that reads a path through its format's own reader."""

from __future__ import annotations

from pathlib import PurePath
from typing import TYPE_CHECKING, Any, cast

from geosave_engine.geodata.core.vector import GeoVector

from . import geojson, geopackage, geoparquet, netcdf, zarr

if TYPE_CHECKING:
    from os import PathLike

    from geosave_engine.geodata import Dataset, DataTree

_RASTER_SUFFIXES = (".zarr", ".nc", ".nc4", ".cdf")
_ARRAY_SUFFIXES = (".tif", ".tiff", ".jp2", ".png")
_VECTOR_SUFFIXES = (".geojson", ".json", ".gpkg", ".parquet", ".geoparquet")


def read(
    source: str | PathLike[str],
    *,
    stack: bool = False,
    **options: Any,
) -> Dataset | DataTree | GeoVector:
    """Read a raster or vector path through the reader its suffix names.

    Forwards `options` untouched. The format modules state their own options as
    typed keywords, so reach for `zarr.read`, `netcdf.read`, or `gdal.read`
    when configuring a read.

    Args:
        source: Local path or URI ending in a recognised suffix.
        stack: Read a multi-group Zarr or NetCDF store as a raster stack.
            Only these two formats hold one.
        **options: Forwarded to the reader the suffix selects.

    Returns:
        `Dataset` for a raster store, `DataTree` when `stack` is set, and
        `GeoVector` for a vector file.

    Raises:
        ValueError: The suffix names no supported format, names a single
            raster file `gdal.read` reads, or `stack` is set for a format that
            holds no groups.

    Examples:
        >>> read("scene.zarr").gs.variables
        ("red", "nir")
        >>> read("plantations.geojson").crs.to_epsg()
        4326
        >>> read("training.zarr", stack=True).gs.groups
        ('sentinel-2-l2a', 'dem')
    """
    suffix = PurePath(str(source)).suffix.lower()

    if suffix == ".zarr":
        if stack:
            return cast("DataTree", zarr.read_stack(source, **options))
        return cast("Dataset", zarr.read(source, **options))

    if suffix in (".nc", ".nc4", ".cdf"):
        if stack:
            return cast("DataTree", netcdf.read_stack(source, **options))
        return cast("Dataset", netcdf.read(source, **options))

    if suffix in _ARRAY_SUFFIXES:
        raise ValueError(
            f"{source} is one banded array rather than a raster Dataset; read "
            f"it with gdal.read, then .to_dataset(dim='band') if you need a "
            f"Dataset"
        )

    if suffix in _VECTOR_SUFFIXES:
        if stack:
            raise ValueError(
                f"{source} is a vector file, which holds no raster groups; "
                f"drop stack=True"
            )
        if suffix in (".geojson", ".json"):
            return GeoVector(geojson.read(source, **options))
        if suffix == ".gpkg":
            return GeoVector(geopackage.read(source, **options))
        return GeoVector(geoparquet.read(source, **options))

    raise ValueError(
        f"{source} ends in {suffix!r}, which names no supported format; read a "
        f"raster ending in {_RASTER_SUFFIXES} or a vector ending in "
        f"{_VECTOR_SUFFIXES}"
    )
