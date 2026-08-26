"""Store configuration each writer accepts, and the options it refuses.

Each TypedDict mirrors its underlying function's own signature, so every key
reaches that function as given. Anything this library invents — the sidecar,
the progress bar, `chunk_px` — stays a named parameter instead.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict

NetcdfFormat = Literal["NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT", "NETCDF3_CLASSIC"]


class ZarrOptions(TypedDict, total=False):
    """How `to_zarr` writes a store, and what it forwards to xarray.

    Every key is an `xarray.Dataset.to_zarr` parameter and reaches it as given.
    `group` and `zarr_format` are read on the way past.

    Args:
        group: Group to write into; None writes the store root. Group writes
            preserve sibling groups, so several rasters share one store.
        zarr_format: On-disk spec version. Use 2 for pre-Zarr-3 readers.
        encoding: Per-variable compressors, filters and dtype. A variable's
            `chunks` key is rejected while `chunk_px` sets chunking.
        storage_options: Forwarded to the store's filesystem.
        write_empty_chunks: False skips chunks that are entirely fill value —
            a real saving on sparse masks.
        safe_chunks: Whether xarray checks Dask chunks against store chunks.
        align_chunks: Whether xarray rechunks to the store's own chunking.
        synchronizer: Zarr synchronizer for concurrent writes.
        chunk_store: Separate store for chunks, metadata staying in `store`.
        chunkmanager_store_kwargs: Forwarded to the chunk manager.
    """

    group: NotRequired[str | None]
    zarr_format: NotRequired[Literal[2, 3]]
    encoding: NotRequired[Mapping[str, Mapping[str, Any]]]
    storage_options: NotRequired[dict[str, Any]]
    write_empty_chunks: NotRequired[bool]
    safe_chunks: NotRequired[bool]
    align_chunks: NotRequired[bool]
    synchronizer: NotRequired[Any]
    chunk_store: NotRequired[Any]
    chunkmanager_store_kwargs: NotRequired[dict[str, Any]]


class NetcdfOptions(TypedDict, total=False):
    """How `to_netcdf` writes a file, and what it forwards to xarray.

    Every key is an `xarray.Dataset.to_netcdf` parameter and reaches it as
    given. `group` is read on the way past.

    Args:
        group: Group to write into; None writes the file root. Group writes
            preserve sibling groups.
        format: NetCDF format version. None uses the engine's default.
        encoding: Per-variable encoding, merged **onto** the writer's own so
            `grid_mapping` survives — see `to_netcdf`. A variable absent here
            keeps the writer's.
        unlimited_dims: Dimensions written as unlimited.
        invalid_netcdf: True lets the engine write values NetCDF4 cannot hold.
        auto_complex: True writes complex values through a compound type.
    """

    group: NotRequired[str | None]
    format: NotRequired[NetcdfFormat | None]
    encoding: NotRequired[Mapping[str, Mapping[str, Any]]]
    unlimited_dims: NotRequired[list[str]]
    invalid_netcdf: NotRequired[bool]
    auto_complex: NotRequired[bool]


class GeotiffOptions(TypedDict, total=False):
    """How `to_geotiff` writes a raster, and what it forwards to rioxarray.

    Every key reaches `rioxarray.rio.to_raster` as given — the GDAL creation
    options through its own `**profile_kwargs`. `driver` and `tags` are read
    on the way past.

    Args:
        driver: GDAL driver — `"GTiff"` or `"COG"`.
        compress: GDAL compression, e.g. `"ZSTD"`, `"DEFLATE"`, `"LZW"`.
            Unset writes uncompressed, roughly twice the size.
        predictor: Compression predictor — 2 for integers, 3 for floats.
        zlevel: DEFLATE level.
        num_threads: Compression threads, or `"ALL_CPUS"`.
        bigtiff: `"YES"`, `"NO"` or `"IF_SAFER"`, for files past 4 GB.
        interleave: `"PIXEL"` or `"BAND"`.
        windowed: True writes window by window, bounding memory.
        lock: Write lock for a Dask-backed array.
        tags: GDAL metadata tags, merged **under** the writer's own so band
            names survive — see `to_geotiff`.
    """

    driver: NotRequired[str]
    compress: NotRequired[str]
    predictor: NotRequired[int]
    zlevel: NotRequired[int]
    num_threads: NotRequired[int | str]
    bigtiff: NotRequired[str]
    interleave: NotRequired[str]
    windowed: NotRequired[bool]
    lock: NotRequired[Any]
    tags: NotRequired[dict[str, str]]


def check_options(
    options: Mapping[str, Any],
    *,
    owned: Mapping[str, str],
    blocked: Mapping[str, str],
    writer: str,
) -> None:
    """Reject options this writer sets itself or cannot honour.

    Args:
        options: Store options a caller passed, minus the ones the writer
            has already read.
        owned: Option name mapped to the named parameter that sets it.
        blocked: Option name mapped to why this writer cannot forward it.
        writer: Writer name, for the error message.

    Raises:
        ValueError: An option is one this writer sets, or one that would
            break the representation it maintains.
    """
    for name in sorted(options):
        if name in owned:
            raise ValueError(f"{writer}() sets {name!r} itself — pass {owned[name]} instead")
        if name in blocked:
            raise ValueError(f"{writer}() cannot forward {name!r}: {blocked[name]}")
