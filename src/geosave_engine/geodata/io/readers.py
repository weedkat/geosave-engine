"""Read native rasters, stacks, and vectors through their format modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from os import PathLike
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, cast

import xarray as xr

from geosave_engine.geodata.attrs import merge, rebase
from geosave_engine.geodata.conventions import TIME_COORDINATE

from .raster import gdal, netcdf, safe, zarr
from .vector import geojson, geopackage, geoparquet
from .storage import absolute_location

if TYPE_CHECKING:
    from geosave_engine.geodata import Dataset, DataTree, GeoDataFrame

type RasterSource = str | PathLike[str] | Sequence[str | PathLike[str]]

_NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf")

# Stores holding named variables.
_DATASET_SUFFIXES = (".zarr", *_NETCDF_SUFFIXES)

# Files holding one banded array, which GDAL reads.
_ARRAY_SUFFIXES = (".tif", ".tiff", ".jp2", ".png")

_VECTOR_SUFFIXES = (".geojson", ".json", ".gpkg", ".parquet", ".geoparquet")

# Vendor product directories, whose leaves name themselves by path.
_PRODUCT_SUFFIXES = (".safe",)


def read_raster(source: RasterSource, **options: Any) -> Dataset:
    """Read a raster file or store through the reader its suffix names.

    Several sources, or a folder of `.tif` files, read as one raster: variables
    merge in source order and dated files join along `time`, always a dimension
    then. `options` are forwarded untouched; each format module types its own.

    Args:
        source: Local path or URI ending in a recognised raster suffix, a
            folder of COGs, or several of these holding one raster.
        **options: Forwarded to the reader each suffix selects.

    Returns:
        Dataset holding one variable per band or stored variable.

    Raises:
        ValueError: The suffix names a vector format or no supported format, no
            source is given, a folder holds no `.tif`, the sources sit on
            different grids, or two of them hold one variable at one instant.

    Examples:
        >>> read_raster("scene.zarr").gs.variables
        ('red', 'nir')
        >>> read_raster("scene.tif").gs.variables
        ('B04', 'B08')
    """
    if not isinstance(source, (str, PathLike)):
        sources = list(source)
        if not sources:
            raise ValueError("read_raster needs at least one source")
        with ExitStack() as opened:
            rasters: list[xr.Dataset] = []
            for path in sources:
                raster = read_raster(path, **options)
                opened.callback(raster.close)
                rasters.append(raster)
            cube = _combine_rasters(rasters)
            cube.set_close(opened.pop_all().close)
            return cast("Dataset", cube)

    suffix = PurePath(str(source)).suffix.lower()

    # A folder of COGs is named by its directory, whatever that is called.
    stores = (".zarr", *_PRODUCT_SUFFIXES)
    if suffix not in stores and Path(str(source)).is_dir():
        paths = sorted(Path(str(source)).rglob("*.tif"))
        if not paths:
            raise ValueError(f"{source} holds no .tif file to read")
        cube = read_raster(paths, **options)
        cube.encoding["source"] = absolute_location(source)
        return cube

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


def read_stack(
    source: str | PathLike[str] | Mapping[str, RasterSource], **options: Any
) -> DataTree:
    """Read a directory of rasters, or named rasters, as a raster stack.

    A directory holds one raster per group, each a file, a store, or a folder
    of COGs, named by its stem. Named sources read one group each through
    `read_raster`.

    Args:
        source: Local directory of groups, or group names mapped to what
            `read_raster` reads, as a stack writer returns them.
        **options: Forwarded to `read_raster` for each group.

    Returns:
        DataTree holding one group per stacked raster: a mapping's in its own
        order, a directory's in name order.

    Raises:
        ValueError: The source is one raster file or store, or the directory
            holds no raster or two entries naming one group.

    Examples:
        >>> read_stack(sample.gs.to_zarr("prepared/s1")).gs.groups
        ('sentinel_2_l2a', 'label')
        >>> read_stack("prepared/s1").gs.groups
        ('label', 'sentinel_2_l2a')
    """
    if isinstance(source, Mapping):
        from geosave_engine.geodata.core.stack import stack

        with ExitStack() as opened:
            rasters = {}
            for name, paths in source.items():
                rasters[name] = read_raster(paths, **options)
                opened.callback(rasters[name].close)
            tree = stack(rasters)
            tree.set_close(opened.pop_all().close)
            return tree

    suffix = PurePath(str(source)).suffix.lower()

    # A directory of one raster per group is what `stack.gs.to_cog` writes.
    stores = (".zarr", *_PRODUCT_SUFFIXES)
    if suffix not in stores and Path(str(source)).is_dir():
        # A group is a folder of COGs, or one raster file or store.
        group_suffixes = _ARRAY_SUFFIXES + _DATASET_SUFFIXES
        entries = []
        for entry in sorted(Path(str(source)).iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir() or entry.suffix.lower() in group_suffixes:
                entries.append(entry)
        if not entries:
            raise ValueError(f"{source} holds no raster to read as a group")

        # A group is named by its stem, so `dem.tif` beside `dem.zarr` is ambiguous.
        groups: dict[str, Path] = {}
        repeats = set()
        for entry in entries:
            if entry.stem in groups:
                repeats.add(entry.stem)
            groups[entry.stem] = entry
        if repeats:
            raise ValueError(
                f"{source} holds several entries naming the groups {sorted(repeats)}; "
                f"a group is one file, store, or folder"
            )
        return read_stack(groups, **options)

    if suffix in _PRODUCT_SUFFIXES:
        return safe.read_stack(source, **options)
    raise ValueError(
        f"{source} ends in {suffix!r}, which holds one raster; read it with "
        f"read_raster, or read the folder holding one raster per group"
    )


def read_vector(source: str | PathLike[str], **options: Any) -> GeoDataFrame:
    """Read a vector file through the reader its suffix names.

    GeoParquet accepts local paths or fsspec URLs; GeoJSON and GeoPackage are
    local only. An item table is read with `stac.table.read`, which also makes
    its asset hrefs openable.

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
    """
    suffix = PurePath(str(source)).suffix.lower()

    if suffix in (".geojson", ".json"):
        return cast("GeoDataFrame", geojson.read(source, **options))

    if suffix == ".gpkg":
        return cast("GeoDataFrame", geopackage.read(source, **options))

    if suffix in (".parquet", ".geoparquet"):
        return cast("GeoDataFrame", geoparquet.read(source, **options))

    if suffix in _DATASET_SUFFIXES or suffix in _ARRAY_SUFFIXES:
        raise ValueError(
            f"{source} holds pixels rather than vector features; read it with "
            f"read_raster"
        )

    raise ValueError(
        f"{source} ends in {suffix!r}, which names no supported vector format; "
        f"read a vector ending in {_VECTOR_SUFFIXES}"
    )


def _combine_rasters(sources: Sequence[xr.Dataset]) -> Dataset:
    """Combine opened Datasets on one grid, refusing repeated variable/time planes.

    Scalar acquisition coordinates become time dimensions. Metadata follows
    attrs model merge rules. Source lifetime remains the caller's responsibility.
    """
    if not sources:
        raise ValueError("read_raster needs at least one source")
    rasters = []
    for raster in sources:
        if TIME_COORDINATE in raster.coords and TIME_COORDINATE not in raster.dims:
            raster = raster.expand_dims(TIME_COORDINATE)
        rasters.append(raster)
    grid = rasters[0].gs.geobox
    if any(raster.gs.geobox != grid for raster in rasters[1:]):
        raise ValueError(
            "the sources sit on different grids; read each raster on its "
            "own, or reproject them onto one grid first"
        )
    # xarray lets one source win a repeated plane without reading pixels.
    planes = set()
    for raster in rasters:
        for name in raster.data_vars:
            for instant in raster.indexes.get(TIME_COORDINATE, [None]):
                plane = (name, instant)
                if plane in planes:
                    raise ValueError(
                        "two sources hold one variable at one instant; read the "
                        "files of one raster, each instant once"
                    )
                planes.add(plane)
    # Attrs drop here and are rebased below, where each model rules on its own.
    cube = xr.combine_by_coords(
        rasters, compat="no_conflicts", join="exact", combine_attrs="drop"
    )
    cube = rebase(cube, merge(rasters))
    return cast("Dataset", cube)
