"""Open and write raster Datasets as Zarr stores."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, TypedDict, Unpack, Literal, cast, overload

import xarray as xr
import zarr
from dask.delayed import delayed
from xarray.core.types import T_Chunks

from geosave_engine.geodata.attrs import Nodata
from geosave_engine.geodata.conventions import to_yx
from ..storage import absolute_location, is_local, local_path


if TYPE_CHECKING:
    from os import PathLike

    from dask.delayed import Delayed

    from geosave_engine.geodata import Dataset

type ZarrChunkSpec = T_Chunks

_STORE_SUFFIX = ".zarr"
_STORE_ZARR_FORMAT = 3
_VARIABLE_ORDER_ATTR = "zarr_variable_order"


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

    Layout options are absent on purpose: a store holds one raster at its
    root, in the Zarr format this writer fixes. `region` is absent because this writer fixes `mode`, which it forbids.
    """

    encoding: Mapping[Hashable, Mapping[str, Any]] | Mapping[str, Any]
    consolidated: bool | None
    safe_chunks: bool
    align_chunks: bool
    write_empty_chunks: bool | None
    chunkmanager_store_kwargs: dict[str, Any] | None
    synchronizer: Any
    storage_options: Mapping[str, Any] | None


def read(
    source: str | PathLike[str],
    *,
    chunks: ZarrChunkSpec = None,
    mask_and_scale: bool = False,
    **open_options: Unpack[ZarrOpenOptions],
) -> Dataset:
    """Open a Zarr store as a raster Dataset.

    Args:
        source: Local Zarr path or URI.
        chunks: Chunk configuration for the opened arrays.
        mask_and_scale: Decode CF packing to physical values instead of
            returning stored digital numbers.
        **open_options: Supported `xarray.open_zarr` options.

    Returns:
        Raster Dataset carrying stored digital numbers, or physical values
        when `mask_and_scale` asked for them. Variables come back in the order
        they were written, recorded in the store's attrs.

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
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        **options,
    )
    try:
        cube = to_yx(_in_written_order(opened))
        group = zarr.open_group(
            source, mode="r", storage_options=options.get("storage_options")
        )
        cube.encoding["zarr"] = {
            "zarr_format": group.metadata.zarr_format,
            "node_type": group.metadata.node_type,
            "consolidated": group.metadata.consolidated_metadata is not None,
        }
    except BaseException:
        opened.close()
        raise
    if cube is not opened:
        cube.set_close(opened.close)
    cube.encoding["source"] = absolute_location(source)
    return cast("Dataset", cube)


def _recording_order(ds: xr.Dataset) -> xr.Dataset:
    """Write the Dataset's own variable order into its attrs.

    Args:
        ds: Raster about to be written.

    Returns:
        Raster with its variable order recorded in storage attrs, or `ds`
        itself where it holds no variables to order.
    """
    if not ds.data_vars:
        return ds
    names = [str(name) for name in ds.data_vars]
    return ds.assign_attrs({_VARIABLE_ORDER_ATTR: names})


def _in_written_order(ds: xr.Dataset) -> xr.Dataset:
    """Put the variables back in the order the store was written in.

    A Zarr group lists its members in no order, so the written one is read from
    storage attrs. Variables it does not name trail the ones it does, which is
    where a variable written by something else ends up.

    Args:
        ds: Dataset as the store listed it.

    Returns:
        Dataset holding the same variables, ordered as written, or `ds` itself
        where the store names no order.
    """
    order = ds.attrs.get(_VARIABLE_ORDER_ATTR)
    if order is None:
        return ds
    named = [name for name in order if name in ds.data_vars]
    trailing = [name for name in ds.data_vars if name not in named]
    return ds[[*named, *trailing]]


@overload
def write(
    raster: xr.Dataset,
    destination: str | PathLike[str],
    *,
    compute: Literal[True] = True,
    overwrite: bool = False,
    **write_options: Unpack[ZarrWriteOptions],
) -> Path | str: ...


@overload
def write(
    raster: xr.Dataset,
    destination: str | PathLike[str],
    *,
    compute: Literal[False],
    overwrite: bool = False,
    **write_options: Unpack[ZarrWriteOptions],
) -> Delayed: ...


def write(
    raster: xr.Dataset,
    destination: str | PathLike[str],
    *,
    compute: bool = True,
    overwrite: bool = False,
    **write_options: Unpack[ZarrWriteOptions],
) -> Path | str | Delayed:
    """Write a raster to a Zarr store.

    A Zarr group records no member order, so a Dataset's variable order
    travels in storage attrs for `read` to restore.

    Args:
        raster: Raster Dataset.
        destination: Output path or fsspec URL ending in `.zarr`.
        compute: False returns a delayed write instead of writing now.
        overwrite: Replace an existing destination when true.
        **write_options: Supported xarray Zarr write options.

    Returns:
        Destination path, or a delayed task returning it when `compute=False`.

    Raises:
        FileExistsError: The destination exists and overwrite is false.
        TypeError: `raster` is not a Dataset.
        ValueError: The destination does not end in `.zarr`.

    Examples:
        >>> write(raster, "scene.zarr")
        PosixPath('scene.zarr')
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            f"raster must be an xarray Dataset, got {type(raster).__name__}; "
            f"write a stack with its own gs.to_zarr"
        )

    name = PurePosixPath(str(destination)).name
    if PurePosixPath(name).suffix != _STORE_SUFFIX:
        raise ValueError(f"destination {name!r} must end in {_STORE_SUFFIX!r}")
    # A URL passes to xarray as written; `Path` would fold its `scheme://`.
    if is_local(destination):
        path: Path | str = local_path(destination)
    else:
        path = str(destination)

    # A Zarr group records no member order, so the raster's own travels in attrs.
    raster = _recording_order(raster)

    options: dict[str, Any] = dict(write_options)
    # A Zarr array holds its own fill value, which is what a GDAL reader masks on.
    options["encoding"] = {
        **fill_encoding(raster),
        **dict(options.get("encoding") or {}),
    }
    options.setdefault("consolidated", False)
    mode: Literal["w", "w-"] = "w" if overwrite else "w-"
    # Passing the literal lets xarray's overloads say what each call returns.
    if not compute:
        written = raster.to_zarr(
            path,
            mode=mode,
            zarr_format=_STORE_ZARR_FORMAT,
            compute=False,
            **options,
        )
        return delayed(lambda _: path)(written)
    raster.to_zarr(
        path, mode=mode, zarr_format=_STORE_ZARR_FORMAT, compute=True, **options
    )
    return path


def fill_encoding(raster: xr.Dataset) -> dict[str, dict[str, Any]]:
    """Say which fill value each variable's Zarr array should hold.

    A Zarr array carries a fill value of its own, defaulting to zero, and that
    is what a GDAL reader masks on — the CF `_FillValue` attr rides alongside
    for a CF reader but is not read as the array's own.

    Args:
        raster: Raster Dataset about to be written.

    Returns:
        {
            "<variable name>": {"fill_value": the value it calls absent},
        }
        Variables declaring none are absent, leaving Zarr its own default.
        Existing CF grid-mapping references accompany each generated fill
        encoding.

    Examples:
        >>> fill_encoding(raster)
        {'lc': {'fill_value': 255}}
    """
    declared: dict[str, dict[str, Any]] = {}
    for name, variable in raster.data_vars.items():
        nodata = Nodata.from_attrs(variable.attrs)
        if nodata is not None and nodata.fill_value is not None:
            # xarray consumes coordinates before backend options, but grid_mapping after.
            encoding = variable.encoding
            declared[str(name)] = {"fill_value": nodata.fill_value}
            if "grid_mapping" in encoding:
                declared[str(name)]["grid_mapping"] = encoding["grid_mapping"]
    return declared
