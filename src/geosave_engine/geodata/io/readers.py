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

from . import (
    gdal,
    geojson,
    geopackage,
    geoparquet,
    netcdf,
    safe,
    zarr,
)
from .storage import absolute_location

if TYPE_CHECKING:
    from geosave_engine.geodata import Dataset, DataTree, GeoDataFrame

type RasterSource = str | PathLike[str] | Sequence[str | PathLike[str]]

_NETCDF_SUFFIXES = (".nc", ".nc4", ".cdf")

# Stores holding named variables, and the only formats holding groups.
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
                # A file dates itself with a scalar; several files join along the axis.
                if (
                    TIME_COORDINATE in raster.coords
                    and TIME_COORDINATE not in raster.dims
                ):
                    raster = raster.expand_dims(TIME_COORDINATE)
                rasters.append(raster)
            grid = rasters[0].gs.geobox
            if any(raster.gs.geobox != grid for raster in rasters[1:]):
                raise ValueError(
                    "the sources sit on different grids; read each raster on its "
                    "own, or reproject them onto one grid first"
                )
            # xarray lets one source win a repeated plane without reading pixels.
            planes = [
                (name, instant)
                for raster in rasters
                for name in raster.data_vars
                for instant in raster.indexes.get(TIME_COORDINATE, [None])
            ]
            if len(set(planes)) != len(planes):
                raise ValueError(
                    "two sources hold one variable at one instant; read the "
                    "files of one raster, each instant once"
                )
            # Attrs drop here and are rebased below, where each model rules on its own.
            cube = xr.combine_by_coords(
                rasters, compat="no_conflicts", join="exact", combine_attrs="drop"
            )
            cube = rebase(cube, merge(rasters))
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
    """Read a multi-group store, or a directory of layers, as a raster stack.

    Zarr and NetCDF hold groups in one store. A directory holds one raster per
    layer, each a file, a `.zarr` store, or a folder of COGs, named by its stem.
    Named sources read one group each through `read_raster`.

    Args:
        source: Local path or URI ending in a recognised store suffix, a
            local directory of layers, or group names mapped to what
            `read_raster` reads.
        **options: Forwarded to the store's reader, or to `read_raster` for
            each layer of a directory or named source.

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

    # A directory of one raster per layer is what `stack.gs.to_cog` writes.
    stores = (".zarr", *_PRODUCT_SUFFIXES)
    if suffix not in stores and Path(str(source)).is_dir():
        # A layer is a folder of COGs, or one raster file or store.
        layer_suffixes = _ARRAY_SUFFIXES + _DATASET_SUFFIXES
        entries = []
        for entry in sorted(Path(str(source)).iterdir()):
            if entry.name.startswith("."):
                continue
            if entry.is_dir() or entry.suffix.lower() in layer_suffixes:
                entries.append(entry)
        if not entries:
            raise ValueError(f"{source} holds no raster to read as a layer")

        # A layer is named by its stem, so `dem.tif` beside `dem.zarr` is ambiguous.
        layers: dict[str, Path] = {}
        repeats = set()
        for entry in entries:
            if entry.stem in layers:
                repeats.add(entry.stem)
            layers[entry.stem] = entry
        if repeats:
            raise ValueError(
                f"{source} holds several entries naming the layers {sorted(repeats)}; "
                f"a layer is one file, store, or folder"
            )
        return read_stack(layers, **options)

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

    GeoParquet accepts local paths or fsspec URLs. Tables carrying ``assets``
    have relative hrefs expanded into directly usable paths or URLs. GeoJSON and GeoPackage remain local-only.

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
        if "assets" in frame:
            frame = geoparquet.absolute_hrefs(frame, absolute_location(source))
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
