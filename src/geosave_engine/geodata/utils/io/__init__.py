"""One module per file format, each reading and writing that format alone.

Every module has a `read`; a writable format also has a `write`, except
GeoTIFF, whose two drivers are `write_cog` and `write_gtiff`. Reach for them
module-qualified, so the format is named once:

Examples:
    >>> from geosave_engine.geodata.utils import io
    >>> raster = io.zarr.read("scene.zarr", mask_and_scale=True)
    >>> io.geotiff.write_cog(raster, "scene.tif")
    >>> io.read_raster("scene.zarr")  # by suffix, when the format is not known

The fluent `to_cog`, `to_zarr`, and `to_netcdf` names belong to the `gs`
accessors, not here.
"""

from __future__ import annotations

from os import PathLike
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, cast

from geosave_engine.geodata.core.vector import GeoVector

from . import (
    gdal,
    geojson,
    geopackage,
    geoparquet,
    geotiff,
    layout,
    netcdf,
    safe,
    zarr,
)
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
from .layout import LAYOUTS, LeafPath, read_tree, write_tree
from .netcdf import NetCDFOpenOptions, NetCDFWriteOptions
from .storage import StorageOptions, filesystem_path, resolve_asset_path
from .zarr import ZarrOpenOptions, ZarrWriteOptions

if TYPE_CHECKING:
    from geosave_engine.geodata import Dataset, DataTree

_NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf")

# Stores holding named variables, and the only formats holding groups.
_DATASET_SUFFIXES = (".zarr", *_NETCDF_SUFFIXES)

# Files holding one banded array, which GDAL reads.
_ARRAY_SUFFIXES = (".tif", ".tiff", ".jp2", ".png")

_VECTOR_SUFFIXES = (".geojson", ".json", ".gpkg", ".parquet", ".geoparquet")

# Vendor product directories, whose leaves name themselves by path.
_PRODUCT_SUFFIXES = (".safe",)

__all__ = [
    "COGWriteOptions",
    "GTiffWriteOptions",
    "GeoJSONOpenOptions",
    "GeoPackageOpenOptions",
    "GeoParquetOpenOptions",
    "GeoTIFFTags",
    "GeoTIFFWriteOptions",
    "NetCDFOpenOptions",
    "NetCDFWriteOptions",
    "RasterioOpenOptions",
    "ZarrOpenOptions",
    "ZarrWriteOptions",
    "gdal",
    "geojson",
    "geopackage",
    "geoparquet",
    "geotiff",
    "LAYOUTS",
    "LeafPath",
    "layout",
    "read_tree",
    "write_tree",
    "netcdf",
    "read_raster",
    "read_stack",
    "read_vector",
    "safe",
    "zarr",
]


def read_raster(source: str | PathLike[str], **options: Any) -> Dataset:
    """Read a raster file or store through the reader its suffix names.

    Forwards `options` untouched. Each format module types its own options as
    keywords, so reach for `zarr.read`, `netcdf.read`, or `gdal.read` when
    configuring a read.

    Args:
        source: Local path or URI ending in a recognised raster suffix.
        **options: Forwarded to the reader the suffix selects.

    Returns:
        Dataset holding one variable per band or stored variable.

    Raises:
        ValueError: The suffix names a vector format or no supported format.

    Examples:
        >>> read_raster("scene.zarr").gs.variables
        ('red', 'nir')
        >>> read_raster("scene.tif").gs.variables
        ('B04', 'B08')
    """
    suffix = PurePath(str(source)).suffix.lower()

    if suffix == ".zarr":
        return zarr.read(source, **options)

    if suffix in _NETCDF_SUFFIXES:
        return netcdf.read(source, **options)

    if suffix in _ARRAY_SUFFIXES:
        return gdal.read(source, **options)

    if suffix in _PRODUCT_SUFFIXES:
        return safe.read(source, **options)

    if suffix in _VECTOR_SUFFIXES:
        raise ValueError(
            f"{source} holds vector features rather than pixels; read it with "
            f"read_vector"
        )

    raise ValueError(
        f"{source} ends in {suffix!r}, which names no supported raster format; "
        f"read a store ending in {_DATASET_SUFFIXES} or a raster ending in "
        f"{_ARRAY_SUFFIXES}"
    )


def read_stack(source: str | PathLike[str], **options: Any) -> DataTree:
    """Read a multi-group store as a raster stack.

    Only Zarr and NetCDF hold groups. A single-group store reads through
    `read_raster` instead.

    Args:
        source: Local path or URI ending in a recognised store suffix.
        **options: Forwarded to `zarr.read_stack` or `netcdf.read_stack`.

    Returns:
        DataTree holding one group per stacked raster.

    Raises:
        ValueError: The suffix names a format that holds no groups.

    Examples:
        >>> read_stack("training.zarr").gs.groups
        ('sentinel-2-l2a', 'dem')
    """
    suffix = PurePath(str(source)).suffix.lower()

    if suffix == ".zarr":
        return zarr.read_stack(source, **options)

    if suffix in _NETCDF_SUFFIXES:
        return netcdf.read_stack(source, **options)

    if suffix in _PRODUCT_SUFFIXES:
        return safe.read_stack(source, **options)

    raise ValueError(
        f"{source} ends in {suffix!r}, and only {_DATASET_SUFFIXES} and "
        f"{_PRODUCT_SUFFIXES} hold groups; read it with read_raster"
    )


def read_vector(source: str | PathLike[str], **options: Any) -> GeoVector:
    """Read a vector file through the reader its suffix names.

    GeoParquet accepts local paths or fsspec URLs. Relative asset pointers in
    its optional ``path`` column are expanded into directly usable paths or
    URLs. GeoJSON and GeoPackage remain local-only.

    Args:
        source: Local path, or fsspec URL for GeoParquet, ending in a
            recognised vector suffix.
        **options: Forwarded to the reader the suffix selects.

    Returns:
        GeoVector holding the file's features in one CRS.

    Raises:
        ValueError: The suffix names a raster format or no supported format.

    Examples:
        >>> read_vector("plantations.geojson").crs.to_epsg()
        4326
        >>> catalog = read_vector(
        ...     "hf://buckets/fatmur/test/catalog.parquet",
        ...     storage_options={"token": token},
        ... )
        >>> catalog.query(prediction).gdf.iloc[0]["path"]
        'hf://buckets/fatmur/test/rasters/prediction.zarr'
    """
    suffix = PurePath(str(source)).suffix.lower()

    if suffix in (".geojson", ".json"):
        return GeoVector(geojson.read(source, **options))

    if suffix == ".gpkg":
        return GeoVector(geopackage.read(source, **options))

    if suffix in (".parquet", ".geoparquet"):
        frame = geoparquet.read(source, **options)
        if "path" in frame:
            storage_options = cast(
                "StorageOptions | None", options.get("storage_options")
            )
            filesystem, catalog_path = filesystem_path(
                source, storage_options=storage_options
            )
            values = frame["path"].dropna()
            invalid = [
                value
                for value in values
                if not isinstance(value, (str, PathLike))
            ]
            if invalid:
                value = invalid[0]
                raise TypeError(
                    "asset path must be string or path-like, got "
                    f"{type(value).__name__}"
                )
            frame.loc[values.index, "path"] = [
                resolve_asset_path(
                    value,
                    filesystem=filesystem,
                    catalog_path=catalog_path,
                )
                for value in values
            ]
        return GeoVector(frame)

    if suffix in _DATASET_SUFFIXES or suffix in _ARRAY_SUFFIXES:
        raise ValueError(
            f"{source} holds pixels rather than vector features; read it with "
            f"read_raster"
        )

    raise ValueError(
        f"{source} ends in {suffix!r}, which names no supported vector format; "
        f"read a vector ending in {_VECTOR_SUFFIXES}"
    )
