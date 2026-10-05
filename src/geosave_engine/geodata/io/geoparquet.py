"""Open and write GeoParquet vector files."""

from __future__ import annotations

import os
from contextlib import suppress
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fsspec.core import split_protocol

from .storage import (
    StorageOptions,
    filesystem_path,
    is_local_filesystem,
    resolve_asset_path,
    stored_asset_path,
)

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem

_FILE_SUFFIXES = (".parquet", ".geoparquet")

# The file key the stac-geoparquet specification names itself by.
_STAC_KEY = b"stac-geoparquet"
_STAC_VALUE = b'{"version": "1.0.0"}'


def is_stac(frame: gpd.GeoDataFrame) -> bool:
    """Report whether a table is a STAC table, which is one carrying assets."""
    return "assets" in frame


def stored_assets(
    frame: gpd.GeoDataFrame, *, filesystem: AbstractFileSystem, catalog_path: str
) -> gpd.GeoDataFrame:
    """Check a STAC table and shorten its asset hrefs for storage.

    Args:
        frame: Table carrying `assets`.
        filesystem: Filesystem the table is written to.
        catalog_path: Protocol-free path of the table on that filesystem.

    Returns:
        Copy whose hrefs below the table's folder are relative to it.

    Raises:
        TypeError: An href is neither a string nor path-like.
        ValueError: `id` is absent, null, or repeated, or the table is not in
            longitude/latitude.
    """
    if "id" not in frame:
        raise ValueError("a STAC table needs an 'id' column naming each item")
    ids = cast("pd.Series", frame["id"])
    if ids.isna().any():
        raise ValueError("STAC item ids must not be null")
    duplicates = sorted(set(ids[ids.duplicated()]))
    if duplicates:
        raise ValueError(f"STAC item ids must be unique, got {duplicates}")
    if frame.crs != "EPSG:4326":
        raise ValueError(
            f"a STAC table states footprints in EPSG:4326, got {frame.crs}; "
            "call to_crs('EPSG:4326') first"
        )

    # A cell is one row's assets, or null where the row has none.
    def shorten(cell: object) -> object:
        if not isinstance(cell, Mapping):
            return cell
        assets = {}
        for name, asset in cell.items():
            href = asset["href"]
            if not isinstance(href, (str, PathLike)):
                raise TypeError(
                    f"asset href must be string or path-like, got {type(href).__name__}"
                )
            assets[name] = {
                **asset,
                "href": stored_asset_path(
                    href, filesystem=filesystem, catalog_path=catalog_path
                ),
            }
        return assets

    return cast(
        "gpd.GeoDataFrame",
        frame.assign(assets=[shorten(cell) for cell in frame["assets"]]),
    )


def resolved_assets(
    frame: gpd.GeoDataFrame, *, filesystem: AbstractFileSystem, catalog_path: str
) -> gpd.GeoDataFrame:
    """Expand a stored STAC table's hrefs into directly openable ones.

    Args:
        frame: Table as read, carrying `assets`.
        filesystem: Filesystem the table was read from.
        catalog_path: Protocol-free path of the table on that filesystem.

    Returns:
        Table whose rows hold only the assets they have, each with an href
        usable as written.
    """

    def expand(cell: object) -> object:
        if not isinstance(cell, Mapping):
            return cell
        assets = {}
        for name, asset in cell.items():
            # Parquet stores one struct for every row, null where a row has none.
            if asset is None:
                continue
            fields = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in asset.items()
                if value is not None
            }
            fields["href"] = str(
                resolve_asset_path(
                    fields["href"], filesystem=filesystem, catalog_path=catalog_path
                )
            )
            assets[name] = fields
        return assets

    return cast(
        "gpd.GeoDataFrame",
        frame.assign(assets=[expand(cell) for cell in frame["assets"]]),
    )


def _stamp_stac(
    path: str | Path,
    *,
    filesystem: AbstractFileSystem | None = None,
    **parquet_options: Any,
) -> None:
    """Rewrite a table with the stac-geoparquet file key beside its own."""
    table = pq.read_table(path, filesystem=filesystem)
    metadata = {**(table.schema.metadata or {}), _STAC_KEY: _STAC_VALUE}
    pq.write_table(
        table.replace_schema_metadata(metadata),
        path,
        filesystem=filesystem,
        **parquet_options,
    )


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

    Nothing is reprojected. A table carrying `assets` is written as a STAC
    table: hrefs below the file are stored relative to it, and the bounding
    box and the `stac-geoparquet` file key are added.

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
        ValueError: The suffix is wrong, a direct option is repeated in
            `parquet_options`, or a STAC table has a null or repeated `id` or
            is not in longitude/latitude.
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
    stac = is_stac(gdf)
    if stac:
        frame = stored_assets(gdf, filesystem=filesystem, catalog_path=target)
        write_options["write_covering_bbox"] = True

    parquet_options = dict(write_options.pop("parquet_options", {}))
    # PyArrow accepts engine-specific keywords that GeoPandas cannot type.
    parquet_kwargs = cast("dict[str, Any]", parquet_options)
    stamp_options = {**parquet_kwargs}
    if "compression" in write_options:
        stamp_options["compression"] = write_options["compression"]
    if is_local_filesystem(filesystem):
        local_target = Path(target)
        if local_target.exists() and not overwrite:
            raise FileExistsError(
                f"{local_target} exists; pass overwrite=True to replace it"
            )
        draft = local_target.with_name(
            f".{local_target.stem}.staging{local_target.suffix}"
        )
        try:
            frame.to_parquet(draft, **write_options, **parquet_kwargs)
            if stac:
                _stamp_stac(draft, **stamp_options)
            os.replace(draft, local_target)
        except BaseException:
            draft.unlink(missing_ok=True)
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
        if stac:
            _stamp_stac(target, filesystem=filesystem, **stamp_options)
    except BaseException:
        if not existed:
            # Cleanup is best-effort and must not replace the write error.
            with suppress(Exception):
                filesystem.rm(target)
        raise
    return str(path)
