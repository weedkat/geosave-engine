"""Open and write raster Datasets and raster-stack DataTrees as Zarr."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, Unpack, Literal, cast, overload

import xarray as xr
from xarray.core.types import T_Chunks


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
        Profile raster Dataset, carrying stored digital numbers unless
        `mask_and_scale` asked for physical values.

    Raises:
        ValueError: The store cannot be read, carries no GeoSave signature, or
            declares an incompatible store version.

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
    return cast("Dataset", opened)


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
        ValueError: The store cannot be read, carries no GeoSave signature, or
            declares an incompatible store version.

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
    return cast("DataTree", opened)


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
    """Write a raster to a GeoSave Zarr store.

    Args:
        raster_or_stack: Profile raster Dataset.
        destination: Output path ending in `.zarr`.
        overwrite: Replace an existing destination when true.
        **write_options: Supported xarray Zarr write options.

    Returns:
        Destination path, or xarray's delayed write when `compute=False`.

    Raises:
        FileExistsError: The destination exists and overwrite is false.
        NotImplementedError: `raster_or_stack` is a DataTree.
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

    options: dict[str, Any] = dict(write_options)
    options.setdefault("consolidated", False)
    written = raster_or_stack.to_zarr(
        path,
        mode="w" if overwrite else "w-",
        zarr_format=_STORE_ZARR_FORMAT,
        compute=compute,
        **options,
    )
    return path if compute else written
