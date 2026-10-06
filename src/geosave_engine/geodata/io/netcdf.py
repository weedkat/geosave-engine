"""Open and write raster Datasets and raster-stack DataTrees as netCDF."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast, overload

import xarray as xr
from dask.delayed import delayed

from .storage import (
    StorageOptions,
    absolute_location,
    is_local,
    local_path,
    local_target,
)

if TYPE_CHECKING:
    from os import PathLike

    from dask.delayed import Delayed
    from xarray.coding.times import CFDatetimeCoder, CFTimedeltaCoder

    from geosave_engine.geodata import Dataset, DataTree

_FILE_SUFFIXES = (".nc", ".nc4", ".cdf")

type NetCDFEngine = Literal["netcdf4", "h5netcdf"]
type NetCDFChunkSpec = (
    int | tuple[int, ...] | Mapping[str, int] | Literal["auto"] | None
)
type NetCDFFormat = Literal[
    "NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT", "NETCDF3_CLASSIC"
]


class NetCDFOpenOptions(TypedDict, total=False):
    """Optional xarray behavior supported when opening netCDF."""

    cache: bool | None
    decode_cf: bool | None
    decode_times: bool | CFDatetimeCoder | Mapping[str, bool | CFDatetimeCoder] | None
    decode_timedelta: (
        bool | CFTimedeltaCoder | Mapping[str, bool | CFTimedeltaCoder] | None
    )
    use_cftime: bool | Mapping[str, bool] | None
    concat_characters: bool | Mapping[str, bool] | None
    drop_variables: str | Sequence[str]
    create_default_indexes: bool
    inline_array: bool
    chunked_array_type: str | None
    from_array_kwargs: dict[str, Any] | None
    backend_kwargs: dict[str, Any] | None


class NetCDFWriteOptions(TypedDict, total=False):
    """Optional xarray behavior supported when writing netCDF."""

    format: NetCDFFormat | None
    group: str | None
    encoding: Mapping[Hashable, Mapping[str, Any]] | Mapping[str, Any]
    unlimited_dims: Sequence[Hashable]
    invalid_netcdf: bool
    auto_complex: bool | None
    write_inherited_coords: bool


def read(
    source: str | PathLike[str],
    *,
    engine: NetCDFEngine = "netcdf4",
    group: str | None = None,
    chunks: NetCDFChunkSpec = None,
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[NetCDFOpenOptions],
) -> Dataset:
    """Open one netCDF group as a raster Dataset.

    Args:
        source: Local netCDF path or URI.
        engine: Xarray netCDF backend.
        group: Group to open, or None for the root.
        chunks: Chunk configuration for the opened arrays.
        mask_and_scale: Decode CF packing to physical values instead of
            returning stored digital numbers.
        storage_options: Accepted for a uniform call; a URL is refused.
        **open_options: Supported `xarray.open_dataset` options.

    Returns:
        Raster Dataset as stored.

    Raises:
        ValueError: The file cannot be read, or `source` is a URL.
    """
    if not is_local(source):
        raise ValueError(
            f"{source} is remote, and NetCDF reads only local files; download "
            f"it first, or store the raster as Zarr"
        )
    if group is not None:
        tree = read_stack(
            source,
            engine=engine,
            chunks=chunks,
            mask_and_scale=mask_and_scale,
            **open_options,
        )
        try:
            node = cast("xr.DataTree", tree[group])
            cube = node.to_dataset(inherit="all_coords")
        except BaseException:
            tree.close()
            raise
        cube.set_close(tree.close)
        return cast("Dataset", cube)

    cube = xr.open_dataset(
        source,
        engine=engine,
        # A grid mapping variable is a coordinate; the CF default leaves it a data variable.
        decode_coords="all",
        group=group,
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        **open_options,
    )
    cube.encoding["source"] = absolute_location(source)
    return cast("Dataset", cube)


def read_stack(
    source: str | PathLike[str],
    *,
    engine: NetCDFEngine = "netcdf4",
    chunks: NetCDFChunkSpec = None,
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[NetCDFOpenOptions],
) -> DataTree:
    """Open a netCDF hierarchy as a raster-stack DataTree.

    Args:
        source: Local netCDF path or URI.
        engine: Xarray netCDF backend.
        chunks: Chunk configuration for the opened arrays.
        mask_and_scale: Decode CF packing to physical values instead of
            returning stored digital numbers.
        storage_options: Accepted for a uniform call; a URL is refused.
        **open_options: Supported `xarray.open_datatree` options.

    Returns:
        DataTree whose every leaf is a raster Dataset.

    Raises:
        ValueError: The file cannot be read, or `source` is a URL.
    """
    if not is_local(source):
        raise ValueError(
            f"{source} is remote, and NetCDF reads only local files; download "
            f"it first, or store the raster as Zarr"
        )
    stack = xr.open_datatree(
        source,
        engine=engine,
        # A grid mapping variable is a coordinate; the CF default leaves it a data variable.
        decode_coords="all",
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        **open_options,
    )
    stack.encoding["source"] = absolute_location(source)
    for node in stack.subtree:
        if node is not stack:
            node.encoding["source"] = stack.encoding["source"]
            node.encoding["group"] = node.path.lstrip("/")
    return cast("DataTree", stack)


@overload
def write(
    raster_or_stack: xr.Dataset | xr.DataTree,
    destination: str | PathLike[str],
    *,
    compute: Literal[True] = True,
    engine: NetCDFEngine = "netcdf4",
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[NetCDFWriteOptions],
) -> Path | str: ...


@overload
def write(
    raster_or_stack: xr.Dataset | xr.DataTree,
    destination: str | PathLike[str],
    *,
    compute: Literal[False],
    engine: NetCDFEngine = "netcdf4",
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[NetCDFWriteOptions],
) -> Delayed: ...


def write(
    raster_or_stack: xr.Dataset | xr.DataTree,
    destination: str | PathLike[str],
    *,
    compute: bool = True,
    engine: NetCDFEngine = "netcdf4",
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[NetCDFWriteOptions],
) -> Path | str | Delayed:
    """Write a raster or raster stack to netCDF.

    Args:
        raster_or_stack: Raster Dataset, or raster-stack DataTree.
        destination: Output path or fsspec URL ending in `.nc`, `.nc4`, or
            `.cdf`. A URL is uploaded once the file is written.
        compute: False returns a delayed write instead of writing now. A URL
            needs true.
        engine: Xarray netCDF backend.
        overwrite: Replace an existing destination when true.
        storage_options: Options for the filesystem a URL names.
        **write_options: Supported xarray netCDF write options.

    Returns:
        Destination path, or a delayed task returning it when `compute=False`.

    Raises:
        FileExistsError: The destination exists and overwrite is false.
        TypeError: The input is neither a Dataset nor a DataTree.
        ValueError: The destination path is invalid, or a URL is asked to
            defer its write.
    """
    if not isinstance(raster_or_stack, xr.Dataset | xr.DataTree):
        raise TypeError(
            f"raster_or_stack must be an xarray Dataset or DataTree, got "
            f"{type(raster_or_stack).__name__}"
        )

    name = PurePosixPath(str(destination)).name
    if PurePosixPath(name).suffix not in _FILE_SUFFIXES:
        raise ValueError(
            f"destination {name!r} must end in one of {list(_FILE_SUFFIXES)}"
        )
    local = is_local(destination)
    options: dict[str, Any] = dict(write_options)
    if not compute:
        if not local:
            raise ValueError(
                f"{destination} is remote, and a NetCDF file is uploaded once "
                f"written; pass compute=True"
            )
        path = local_path(destination)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        # Passing the literal lets xarray's overloads say what each call returns.
        written = raster_or_stack.to_netcdf(
            path, mode="w", engine=engine, compute=False, **options
        )
        return delayed(lambda _: path)(written)
    with local_target(
        destination, overwrite=overwrite, storage_options=storage_options
    ) as path:
        raster_or_stack.to_netcdf(
            path, mode="w", engine=engine, compute=True, **options
        )

    # A local write answers with a path; an upload answers with the URL it went to.
    if local:
        return local_path(destination)
    return str(destination)
