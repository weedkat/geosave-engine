"""Open and write raster Datasets and raster-stack DataTrees as Zarr."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, Unpack, Literal, cast, overload

import xarray as xr
from xarray.core.types import T_Chunks

from geosave_engine.geodata.attrs import (
    Nodata,
    ZarrOrder,
    create_header,
    rebase,
)


if TYPE_CHECKING:
    from os import PathLike

    from dask.delayed import Delayed

    from geosave_engine.geodata import Dataset, DataTree

type ZarrChunkSpec = T_Chunks

_STORE_SUFFIX = ".zarr"
_STORE_ZARR_FORMAT = 3


class ZarrOpenOptions(TypedDict, total=False):
    """Optional xarray behavior supported when opening Zarr.

    CF decoding options are absent because disabling them drops `spatial_ref`,
    the time axis, or the spatial index.
    """

    synchronizer: Any
    drop_variables: str | Sequence[str]
    consolidated: bool | None
    overwrite_encoded_chunks: bool
    chunk_store: Any
    storage_options: Mapping[str, Any]
    decode_timedelta: bool | None
    use_zarr_fill_value_as_mask: bool | None
    chunked_array_type: str | None
    from_array_kwargs: dict[str, Any] | None


class ZarrWriteOptions(TypedDict, total=False):
    """Optional xarray behavior supported when writing Zarr.

    Layout options are absent on purpose: `group`, `zarr_format`, and
    `write_inherited_coords` decide the store's shape, which the store version
    fixes. `region` is absent because this writer fixes `mode`, which it forbids.
    """

    encoding: Mapping[Hashable, Mapping[str, Any]] | Mapping[str, Any]
    # compute: bool
    consolidated: bool | None
    safe_chunks: bool
    align_chunks: bool
    write_empty_chunks: bool | None
    chunkmanager_store_kwargs: dict[str, Any] | None
    synchronizer: Any


def read(
    source: str | PathLike[str],
    *,
    group: str | None = None,
    chunks: ZarrChunkSpec = None,
    mask_and_scale: bool = False,
    **open_options: Unpack[ZarrOpenOptions],
) -> Dataset:
    """Open one Zarr group of a GeoSave store as a raster Dataset.

    Args:
        source: Local Zarr path or URI.
        group: Group to open, or None for the root.
        chunks: Chunk configuration for the opened arrays.
        mask_and_scale: Decode CF packing to physical values instead of
            returning stored digital numbers.
        **open_options: Supported `xarray.open_zarr` options.

    Returns:
        Raster Dataset carrying stored digital numbers, or physical values
        when `mask_and_scale` asked for them. Variables come back in the order
        they were written, which `ZarrOrder` records.

    Raises:
        ValueError: The store cannot be read.

    Examples:
        >>> raster = read("scene.zarr")
        >>> raster.red.dtype
        dtype('uint16')
    """
    options: dict[str, Any] = dict(open_options)
    options.setdefault("consolidated", False)
    # A grid mapping variable is a coordinate; the CF default leaves it a data variable.
    options.setdefault("decode_coords", "all")
    opened = xr.open_dataset(
        source,
        engine="zarr",
        group=group,
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        **options,
    )
    try:
        cube = _in_written_order(opened)
    except BaseException:
        opened.close()
        raise
    if cube is not opened:
        cube.set_close(opened.close)
    return cast("Dataset", cube)


def _recording_order(ds: xr.Dataset) -> xr.Dataset:
    """Write the Dataset's own variable order into its attrs.

    Args:
        ds: Raster about to be written, or a node of a stack about to be.

    Returns:
        The same raster carrying `ZarrOrder`, or `ds` itself where it holds no
        variables to order, as a stack's root does.
    """
    if not ds.data_vars:
        return ds
    names = tuple(str(name) for name in ds.data_vars)
    return rebase(ds, ZarrOrder(zarr_variable_order=names))


def _in_written_order(ds: xr.Dataset) -> xr.Dataset:
    """Put the variables back in the order the store was written in.

    A Zarr group lists its members in no order, so the written one is read from
    `ZarrOrder`. Variables it does not name trail the ones it does, which is
    where a variable written by something else ends up.

    Args:
        ds: Dataset as the store listed it.

    Returns:
        Dataset holding the same variables, ordered as written, or `ds` itself
        where the store names no order.
    """
    order = create_header(ds).root.get(ZarrOrder)
    if order is None or order.zarr_variable_order is None:
        return ds
    named = [name for name in order.zarr_variable_order if name in ds.data_vars]
    trailing = [name for name in ds.data_vars if name not in named]
    return ds[[*named, *trailing]]


def read_stack(
    source: str | PathLike[str],
    *,
    chunks: ZarrChunkSpec = None,
    mask_and_scale: bool = False,
    **open_options: Unpack[ZarrOpenOptions],
) -> DataTree:
    """Open a Zarr hierarchy as a raster-stack DataTree.

    Args:
        source: Local Zarr path or URI.
        chunks: Chunk configuration for the opened arrays.
        mask_and_scale: Decode CF packing to physical values instead of
            returning stored digital numbers.
        **open_options: Supported `xarray.open_datatree` options.

    Returns:
        DataTree whose every leaf is a raster Dataset.

    Raises:
        ValueError: The store cannot be read.

    Examples:
        >>> read_stack("scene.zarr").gs.groups
        ('dem', 'sentinel-2-l2a')
    """
    options: dict[str, Any] = dict(open_options)
    options.setdefault("consolidated", False)
    # A grid mapping variable is a coordinate; the CF default leaves it a data variable.
    options.setdefault("decode_coords", "all")
    opened = xr.open_datatree(
        source,
        engine="zarr",
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        **options,
    )
    try:
        stack = opened.map_over_datasets(_in_written_order)
    except BaseException:
        opened.close()
        raise
    if stack is not opened:
        stack.set_close(opened.close)
    return cast("DataTree", stack)


@overload
def write(
    raster_or_stack: xr.Dataset | xr.DataTree,
    destination: str | PathLike[str],
    *,
    compute: Literal[True] = True,
    overwrite: bool = False,
    **write_options: Unpack[ZarrWriteOptions],
) -> Path: ...


@overload
def write(
    raster_or_stack: xr.Dataset | xr.DataTree,
    destination: str | PathLike[str],
    *,
    compute: Literal[False],
    overwrite: bool = False,
    **write_options: Unpack[ZarrWriteOptions],
) -> Delayed: ...


def write(
    raster_or_stack: xr.Dataset | xr.DataTree,
    destination: str | PathLike[str],
    *,
    compute: bool = True,
    overwrite: bool = False,
    **write_options: Unpack[ZarrWriteOptions],
) -> Path | Delayed:
    """Write a raster or raster stack to a Zarr store.

    A Zarr group records no member order, so a Dataset's variable order
    travels in `ZarrOrder` for `read` to restore.

    Args:
        raster_or_stack: Raster Dataset, or raster-stack DataTree.
        destination: Output path ending in `.zarr`.
        compute: False returns a delayed write instead of writing now.
        overwrite: Replace an existing destination when true.
        **write_options: Supported xarray Zarr write options.

    Returns:
        Destination path, or xarray's delayed write when `compute=False`.

    Raises:
        FileExistsError: The destination exists and overwrite is false.
        TypeError: `raster_or_stack` is neither a Dataset nor a DataTree.
        ValueError: The destination does not end in `.zarr`.

    Examples:
        >>> write(raster, "scene.zarr")
        PosixPath('scene.zarr')
    """
    if not isinstance(raster_or_stack, xr.Dataset | xr.DataTree):
        raise TypeError(
            f"raster_or_stack must be an xarray Dataset or DataTree, got "
            f"{type(raster_or_stack).__name__}"
        )

    path = Path(destination)
    if path.suffix != _STORE_SUFFIX:
        raise ValueError(f"destination {path.name!r} must end in {_STORE_SUFFIX!r}")

    # A Zarr group records no member order, so every node's own travels in attrs.
    if isinstance(raster_or_stack, xr.Dataset):
        raster_or_stack = _recording_order(raster_or_stack)
    else:
        raster_or_stack = raster_or_stack.map_over_datasets(_recording_order)

    options: dict[str, Any] = dict(write_options)
    # A Zarr array holds its own fill value, which is what a GDAL reader masks on.
    options["encoding"] = {
        **fill_encoding(raster_or_stack),
        **dict(options.get("encoding") or {}),
    }
    options.setdefault("consolidated", False)
    mode: Literal["w", "w-"] = "w" if overwrite else "w-"
    # Passing the literal lets xarray's overloads say what each call returns.
    if not compute:
        return raster_or_stack.to_zarr(
            path,
            mode=mode,
            zarr_format=_STORE_ZARR_FORMAT,
            compute=False,
            **options,
        )
    raster_or_stack.to_zarr(
        path, mode=mode, zarr_format=_STORE_ZARR_FORMAT, compute=True, **options
    )
    return path


def fill_encoding(
    raster_or_stack: xr.Dataset | xr.DataTree,
) -> dict[str, dict[str, Any]]:
    """Say which fill value each variable's Zarr array should hold.

    A Zarr array carries a fill value of its own, defaulting to zero, and that
    is what a GDAL reader masks on — the CF `_FillValue` attr rides alongside
    for a CF reader but is not read as the array's own.

    Args:
        raster_or_stack: Raster Dataset, or raster-stack DataTree.

    Returns:
        {
            "<variable name>": {"fill_value": the value it calls absent},
        }
        Variables declaring none are absent, leaving Zarr its own default. A
        DataTree nests each group's variables under the group's path. Existing
        CF grid-mapping references accompany each generated fill encoding.

    Examples:
        >>> fill_encoding(raster)
        {'lc': {'fill_value': 255}}
    """
    if isinstance(raster_or_stack, xr.DataTree):
        grouped: dict[str, dict[str, Any]] = {}
        for node in raster_or_stack.subtree:
            fills = fill_encoding(node.dataset)
            if fills:
                grouped[node.path] = fills
        return grouped

    declared: dict[str, dict[str, Any]] = {}
    for name, namespace in create_header(raster_or_stack).data_vars.items():
        nodata = namespace.get(Nodata)
        if nodata is not None and nodata.fill_value is not None:
            # xarray consumes coordinates before backend options, but grid_mapping after.
            encoding = raster_or_stack[name].encoding
            declared[name] = {"fill_value": nodata.fill_value}
            if "grid_mapping" in encoding:
                declared[name]["grid_mapping"] = encoding["grid_mapping"]
    return declared
