"""Open and write GeoParquet vector files."""

from __future__ import annotations

import os
from contextlib import suppress
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypedDict, Unpack, cast

import geopandas as gpd
import numpy as np
from stac_geoparquet.arrow import to_parquet
from fsspec.core import split_protocol
from pystac.utils import make_absolute_href, make_relative_href

from .storage import (
    StorageOptions,
    absolute_location,
    filesystem_path,
    is_local_filesystem,
)

_FILE_SUFFIXES = (".parquet", ".geoparquet")


def relative_hrefs(frame: gpd.GeoDataFrame, table: str) -> gpd.GeoDataFrame:
    """Rewrite asset hrefs relative to the table that stores them.

    Args:
        frame: Table carrying `assets` with absolute hrefs.
        table: Absolute path or URL of the table.

    Returns:
        Copy whose hrefs on the table's own filesystem are relative to it.

    Raises:
        TypeError: An href is neither a string nor path-like.
    """
    assets = []
    for cell in frame["assets"]:
        # A cell is one row's assets, or null where the row has none.
        if not isinstance(cell, Mapping):
            assets.append(cell)
            continue
        row = {}
        for key, asset in cell.items():
            if asset is None:
                continue
            href = asset["href"]
            if not isinstance(href, (str, PathLike)):
                raise TypeError(
                    f"asset href must be string or path-like, got {type(href).__name__}"
                )
            row[key] = {**asset, "href": make_relative_href(str(href), table)}
        assets.append(row)
    return cast("gpd.GeoDataFrame", frame.assign(assets=assets))


def absolute_hrefs(frame: gpd.GeoDataFrame, table: str) -> gpd.GeoDataFrame:
    """Rewrite the hrefs a table stores into directly openable ones.

    Args:
        frame: Table as read, carrying `assets`.
        table: Absolute path or URL the table was read from.

    Returns:
        Table whose rows hold only the assets they have, each with an
        absolute href.
    """
    assets = []
    for cell in frame["assets"]:
        if not isinstance(cell, Mapping):
            assets.append(cell)
            continue
        row = {}
        for key, asset in cell.items():
            # Parquet stores one struct for every row, null where a row has none.
            if asset is None:
                continue
            fields = {
                name: value.tolist() if isinstance(value, np.ndarray) else value
                for name, value in asset.items()
                if value is not None
            }
            # An href stored whole, as a URL or a rooted path, names its file already.
            href = str(fields["href"])
            protocol, _ = split_protocol(href)
            if protocol is None and not href.startswith("/"):
                href = make_absolute_href(href, table)
            row[key] = {**fields, "href": href}
        assets.append(row)
    return cast("gpd.GeoDataFrame", frame.assign(assets=assets))


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

    A table of STAC Items is written as STAC GeoParquet, stating each row's
    `bbox` from the geometry written; GeoPandas geometry options then do not
    apply. Asset hrefs on the table's filesystem are stored relative to it.

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
            `parquet_options`.
    """
    storage_options = write_options.pop("storage_options", None)
    filesystem, target = filesystem_path(path, storage_options)
    target_name = PurePosixPath(target).name
    if PurePosixPath(target).suffix.lower() not in _FILE_SUFFIXES:
        raise ValueError(
            f"destination {target_name!r} must end in one of {list(_FILE_SUFFIXES)}"
        )
    if not overwrite and filesystem.exists(target):
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")

    # Assets beside the table are stored relative to it, so the two move together.
    frame = gdf
    if "assets" in gdf:
        frame = relative_hrefs(gdf, absolute_location(path))

    # PyArrow accepts engine-specific keywords that GeoPandas cannot type.
    parquet_options = cast(
        "dict[str, Any]", dict(write_options.pop("parquet_options", {}))
    )

    def save(destination: str | Path, **location: Any) -> None:
        # A table of STAC Items carries `stac_version`; anything else is plain vectors.
        if "stac_version" not in frame:
            frame.to_parquet(
                destination, **location, **write_options, **parquet_options
            )
            return

        # GeoPandas drops the covering bbox on read, so every write states the current one.
        bounds = frame.geometry.bounds
        bounds.columns = ["xmin", "ymin", "xmax", "ymax"]
        items = frame.assign(bbox=bounds.to_dict("records"))
        table = items.to_arrow(index=write_options.get("index"))
        serializer_options = {
            name: value for name, value in write_options.items() if name != "index"
        }
        to_parquet(
            table, destination, **location, **serializer_options, **parquet_options
        )

    # A local table is written beside its destination, then moved over it whole.
    if is_local_filesystem(filesystem):
        destination = Path(target)
        staging = destination.with_name(
            f".{destination.stem}.staging{destination.suffix}"
        )
        try:
            save(staging)
            os.replace(staging, destination)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
        return destination

    # A remote table is streamed to its URL, and a failed first write is removed.
    existed = filesystem.exists(target)
    try:
        save(target, filesystem=filesystem)
    except BaseException:
        if not existed:
            # Cleanup is best-effort and must not replace the write error.
            with suppress(Exception):
                filesystem.rm(target)
        raise
    return str(path)
