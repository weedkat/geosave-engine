"""Open and write GeoParquet vector files."""

from __future__ import annotations

import os
from contextlib import suppress
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypedDict, Unpack, cast

import geopandas as gpd
from fsspec.core import split_protocol

from .storage import (
    StorageOptions,
    filesystem_path,
    is_local_filesystem,
    stored_asset_path,
)

_FILE_SUFFIXES = (".parquet", ".geoparquet")

class GeoParquetOpenOptions(TypedDict, total=False):
    """Optional GeoPandas behavior supported when opening GeoParquet."""

    columns: Sequence[str] | None
    bbox: tuple[float, float, float, float] | None
    storage_options: StorageOptions | None
    to_pandas_kwargs: Mapping[str, object] | None
    parquet_options: dict[str, object]


def read(
    source: str | PathLike[str],
    **open_options: Unpack[GeoParquetOpenOptions],
) -> gpd.GeoDataFrame:
    """Open one local or fsspec-backed GeoParquet file.

    Args:
        source: Local path or fsspec URL to GeoParquet.
        **open_options: Supported GeoPandas and Parquet read options.

    Returns:
        GeoDataFrame on the CRS the file names.

    Raises:
        TypeError: An option is unsupported or supplied twice.
        ValueError: A direct option is repeated in `parquet_options`.
    """
    storage_options = open_options.pop("storage_options", None)
    parquet_options = dict(open_options.pop("parquet_options", {}))
    filesystem, source_path = filesystem_path(source, storage_options)
    return gpd.read_parquet(
        source_path,
        filesystem=filesystem,
        **open_options,
        **parquet_options,
    )


class GeoParquetWriteOptions(TypedDict, total=False):
    """Optional GeoPandas behavior supported when writing GeoParquet."""

    index: bool | None
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None
    geometry_encoding: Literal["WKB", "geoarrow"]
    write_covering_bbox: bool
    schema_version: Literal["0.1.0", "0.4.0", "1.0.0-beta.1", "1.0.0", "1.1.0"] | None
    storage_options: StorageOptions | None
    parquet_options: dict[str, object]


def write(
    gdf: gpd.GeoDataFrame,
    path: str | PathLike[str],
    *,
    overwrite: bool = False,
    **write_options: Unpack[GeoParquetWriteOptions],
) -> Path | str:
    """Write one GeoDataFrame to local or fsspec-backed GeoParquet.

    GeoParquet records the frame's own CRS, so nothing is reprojected.

    Args:
        gdf: Frame to write.
        path: Output path or URL ending in `.parquet` or `.geoparquet`.
        overwrite: Replace an existing file when true.
        **write_options: Supported GeoPandas and Parquet write options.

    Returns:
        Local writes return a path; URL writes return the supplied URL.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        TypeError: An option is unsupported or supplied twice.
        ValueError: The suffix is wrong, or a direct option is repeated in
            `parquet_options`.
    """
    storage_options = write_options.pop("storage_options", None)
    protocol, _ = split_protocol(str(path))
    filesystem, target = filesystem_path(path, storage_options)
    suffix = PurePosixPath(target).suffix.lower()
    if suffix not in _FILE_SUFFIXES:
        raise ValueError(
            f"destination {PurePosixPath(target).name!r} must end in one of "
            f"{list(_FILE_SUFFIXES)}"
        )

    frame = gdf
    if "path" in gdf:
        frame = gdf.copy()
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
            stored_asset_path(
                value,
                filesystem=filesystem,
                catalog_path=target,
            )
            for value in values
        ]

    parquet_options = dict(write_options.pop("parquet_options", {}))
    # PyArrow accepts engine-specific keywords that GeoPandas cannot type.
    parquet_kwargs = cast("dict[str, Any]", parquet_options)
    if is_local_filesystem(filesystem):
        local_target = Path(target)
        if local_target.exists() and not overwrite:
            raise FileExistsError(
                f"{local_target} exists; pass overwrite=True to replace it"
            )
        staged = local_target.with_name(
            f".{local_target.stem}.staging{local_target.suffix}"
        )
        try:
            frame.to_parquet(staged, **write_options, **parquet_kwargs)
            os.replace(staged, local_target)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        return Path(target) if protocol is not None else Path(path)

    existed = filesystem.exists(target)
    if existed and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    try:
        frame.to_parquet(
            target,
            filesystem=filesystem,
            **write_options,
            **parquet_kwargs,
        )
    except BaseException:
        if not existed:
            # Cleanup is best-effort and must not replace the write error.
            with suppress(Exception):
                filesystem.rm(target)
        raise
    return str(path)
