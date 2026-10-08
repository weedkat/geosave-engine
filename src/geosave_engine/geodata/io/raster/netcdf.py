"""Open and write raster Datasets as netCDF files."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast, overload

import xarray as xr
from dask.delayed import delayed

from geosave_engine.geodata.conventions import to_yx

from .. import storage
from ..storage import StorageOptions, absolute_location

if TYPE_CHECKING:
    from os import PathLike

    from dask.delayed import Delayed
    from xarray.coding.times import CFDatetimeCoder, CFTimedeltaCoder

    from geosave_engine.geodata import Dataset

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
    encoding: Mapping[Hashable, Mapping[str, Any]] | Mapping[str, Any]
    unlimited_dims: Sequence[Hashable]
    invalid_netcdf: bool
    auto_complex: bool | None


def read(
    source: str | PathLike[str],
    *,
    engine: NetCDFEngine = "netcdf4",
    chunks: NetCDFChunkSpec = None,
    mask_and_scale: bool = False,
    storage_options: StorageOptions | None = None,
    **open_options: Unpack[NetCDFOpenOptions],
) -> Dataset:
    """Open a netCDF file as a raster Dataset.

    Args:
        source: Local netCDF path or URI.
        engine: Xarray netCDF backend.
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
    if not storage.is_local(source):
        raise ValueError(
            f"{source} is remote, and NetCDF reads only local files; download "
            f"it first, or store the raster as Zarr"
        )
    opened = xr.open_dataset(
        source,
        engine=engine,
        # A grid mapping variable is a coordinate; the CF default leaves it a data variable.
        decode_coords="all",
        chunks=chunks,
        mask_and_scale=mask_and_scale,
        **open_options,
    )
    cube = to_yx(opened)
    if cube is not opened:
        cube.set_close(opened.close)
    cube.encoding["source"] = absolute_location(source)
    return cast("Dataset", cube)


@overload
def write(
    raster: xr.Dataset,
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
    raster: xr.Dataset,
    destination: str | PathLike[str],
    *,
    compute: Literal[False],
    engine: NetCDFEngine = "netcdf4",
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[NetCDFWriteOptions],
) -> Delayed: ...


def write(
    raster: xr.Dataset,
    destination: str | PathLike[str],
    *,
    compute: bool = True,
    engine: NetCDFEngine = "netcdf4",
    overwrite: bool = False,
    storage_options: StorageOptions | None = None,
    **write_options: Unpack[NetCDFWriteOptions],
) -> Path | str | Delayed:
    """Write a raster to a netCDF file.

    Args:
        raster: Raster Dataset.
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
        TypeError: `raster` is not a Dataset.
        ValueError: The destination path is invalid, or a URL is asked to
            defer its write.
    """
    if not isinstance(raster, xr.Dataset):
        raise TypeError(
            f"raster must be an xarray Dataset, got {type(raster).__name__}; "
            f"write a stack with its own gs.to_netcdf"
        )

    options: dict[str, Any] = dict(write_options)
    # A deferred write cannot be staged, so it goes to a local path directly.
    if not compute:
        if not storage.is_local(destination):
            raise ValueError(
                f"{destination} is remote, and a NetCDF file is uploaded once "
                f"written; pass compute=True"
            )
        path = storage.local_path(destination)
        if path.suffix not in _FILE_SUFFIXES:
            raise ValueError(
                f"destination {path.name!r} must end in one of {list(_FILE_SUFFIXES)}"
            )
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        # Passing the literal lets xarray's overloads say what each call returns.
        written = raster.to_netcdf(
            path, mode="w", engine=engine, compute=False, **options
        )
        return delayed(lambda _: path)(written)

    def save(target: Path) -> None:
        raster.to_netcdf(target, mode="w", engine=engine, compute=True, **options)

    return storage.write(
        destination,
        save,
        suffixes=_FILE_SUFFIXES,
        overwrite=overwrite,
        storage_options=storage_options,
    )
