"""One module per file format, each reading and writing that format alone.

Every module has a `read`; a writable format also has a `write`, except
GeoTIFF, whose two drivers are `write_cog` and `write_gtiff`. Reach for them
module-qualified, so the format is named once:

Examples:
    >>> from geosave_engine.geodata import io
    >>> raster = io.zarr.read("scene.zarr", mask_and_scale=True)
    >>> io.geotiff.write_cog(raster, "scene.tif")
    >>> io.read_raster("scene.zarr")  # by suffix, when the format is not known

The fluent `to_cog`, `to_zarr`, and `to_netcdf` names belong to the `gs`
accessors, not here.
"""

from __future__ import annotations

from contextlib import ExitStack
from os import PathLike
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, cast

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
from .storage import StorageOptions, filesystem_path
from .zarr import ZarrOpenOptions, ZarrWriteOptions

if TYPE_CHECKING:
    from geosave_engine.geodata import Dataset, DataTree, GeoDataFrame

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
        source: Local path or URI ending in a recognised raster suffix, or
            a directory `write_tree` wrote.
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

    # A tree of COG leaves is named by its directory, whatever that is called.
    stores = (".zarr", *_PRODUCT_SUFFIXES)
    if suffix not in stores and Path(str(source)).is_dir():
        return layout.read_tree(source, **options)

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
    """Read a multi-group store, or a directory of layers, as a raster stack.

    Zarr and NetCDF hold groups in one store. A directory holds one raster per
    layer, each a file, a `.zarr` store, or a tree of leaves, and its groups
    are named by those entries. A single-group store reads through
    `read_raster` instead.

    Args:
        source: Local path or URI ending in a recognised store suffix, or a
            local directory of layers.
        **options: Forwarded to the store's reader, or to `read_raster` for
            each layer of a directory.

    Returns:
        DataTree holding one group per stacked raster, a directory's in name
        order.

    Raises:
        ValueError: The suffix names a format that holds no groups, or the
            directory holds no raster or two entries naming one layer.

    Examples:
        >>> read_stack("training.zarr").gs.groups
        ('sentinel-2-l2a', 'dem')
        >>> read_stack("prepared/s1").gs.groups
        ('label', 'sentinel_2_l2a')
    """
    suffix = PurePath(str(source)).suffix.lower()

    # A directory of one raster per layer is what `stack.gs.to_cog` writes.
    stores = (".zarr", *_PRODUCT_SUFFIXES)
    if suffix not in stores and Path(str(source)).is_dir():
        from geosave_engine.geodata.core.stack import stack

        layers = _ARRAY_SUFFIXES + _DATASET_SUFFIXES
        entries = sorted(
            entry
            for entry in Path(str(source)).iterdir()
            if not entry.name.startswith(".")
            and (entry.is_dir() or entry.suffix.lower() in layers)
        )
        if not entries:
            raise ValueError(f"{source} holds no raster to read as a layer")
        stems = [entry.stem for entry in entries]
        repeats = sorted({stem for stem in stems if stems.count(stem) > 1})
        if repeats:
            raise ValueError(
                f"{source} holds several entries naming the layers {repeats}; a "
                f"layer is one file, store, or folder"
            )
        with ExitStack() as opened:
            rasters = {}
            for entry in entries:
                raster = read_raster(entry, **options)
                opened.callback(raster.close)
                rasters[entry.stem] = raster
            tree = stack(rasters)
            tree.set_close(opened.pop_all().close)
            return tree

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


def read_vector(source: str | PathLike[str], **options: Any) -> GeoDataFrame:
    """Read a vector file through the reader its suffix names.

    GeoParquet accepts local paths or fsspec URLs. In a STAC table, which is
    one carrying ``assets``, relative asset hrefs are expanded into directly
    usable paths or URLs. GeoJSON and GeoPackage remain local-only.

    Args:
        source: Local path, or fsspec URL for GeoParquet, ending in a
            recognised vector suffix.
        **options: Forwarded to the reader the suffix selects.

    Returns:
        GeoDataFrame holding the file's features in one CRS, read through `gs`.

    Raises:
        ValueError: The suffix names a raster format or no supported format.

    Examples:
        >>> read_vector("plantations.geojson").gs.crs.to_epsg()
        4326
        >>> catalog = read_vector(
        ...     "hf://buckets/fatmur/test/catalog.parquet",
        ...     storage_options={"token": token},
        ... )
        >>> catalog.gs.query(prediction).iloc[0]["assets"]["prediction"]["href"]
        'hf://buckets/fatmur/test/rasters/prediction.zarr'
    """
    suffix = PurePath(str(source)).suffix.lower()

    if suffix in (".geojson", ".json"):
        return cast("GeoDataFrame", geojson.read(source, **options))

    if suffix == ".gpkg":
        return cast("GeoDataFrame", geopackage.read(source, **options))

    if suffix in (".parquet", ".geoparquet"):
        frame = geoparquet.read(source, **options)
        if geoparquet.is_stac(frame):
            storage_options = cast(
                "StorageOptions | None", options.get("storage_options")
            )
            filesystem, catalog_path = filesystem_path(
                source, storage_options=storage_options
            )
            frame = geoparquet.resolved_assets(
                frame, filesystem=filesystem, catalog_path=catalog_path
            )
        return cast("GeoDataFrame", frame)

    if suffix in _DATASET_SUFFIXES or suffix in _ARRAY_SUFFIXES:
        raise ValueError(
            f"{source} holds pixels rather than vector features; read it with "
            f"read_raster"
        )

    raise ValueError(
        f"{source} ends in {suffix!r}, which names no supported vector format; "
        f"read a vector ending in {_VECTOR_SUFFIXES}"
    )
