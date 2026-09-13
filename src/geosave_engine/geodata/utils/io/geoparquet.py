"""Open and write GeoParquet vector files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import geopandas as gpd

from pathlib import Path

if TYPE_CHECKING:
    from os import PathLike

_FILE_SUFFIXES = (".parquet", ".geoparquet")


class GeoParquetOpenOptions(TypedDict, total=False):
    """Optional GeoPandas behavior supported when opening GeoParquet."""

    columns: Sequence[str] | None
    bbox: tuple[float, float, float, float] | None
    storage_options: Mapping[str, Any] | None
    to_pandas_kwargs: Mapping[str, Any] | None
    parquet_options: dict[str, Any]


def read(
    source: str | PathLike[str],
    **open_options: Unpack[GeoParquetOpenOptions],
) -> gpd.GeoDataFrame:
    """Open one GeoParquet file as a GeoDataFrame.

    Args:
        source: Local GeoParquet path or URI.
        **open_options: Supported GeoPandas and Parquet read options.

    Returns:
        GeoDataFrame on the CRS the file names.

    Raises:
        TypeError: An option is unsupported or supplied twice.
        ValueError: A direct option is repeated in `parquet_options`.
    """
    parquet_options = dict(open_options.pop("parquet_options", {}))
    return gpd.read_parquet(source, **open_options, **parquet_options)


class GeoParquetWriteOptions(TypedDict, total=False):
    """Optional GeoPandas behavior supported when writing GeoParquet."""

    index: bool | None
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None
    geometry_encoding: Literal["WKB", "geoarrow"]
    write_covering_bbox: bool
    schema_version: Literal["0.1.0", "0.4.0", "1.0.0-beta.1", "1.0.0", "1.1.0"] | None
    parquet_options: dict[str, Any]


def write(
    gdf: gpd.GeoDataFrame,
    path: str | PathLike[str],
    *,
    overwrite: bool = False,
    **write_options: Unpack[GeoParquetWriteOptions],
) -> Path:
    """Write one GeoDataFrame as a GeoParquet file.

    GeoParquet records the frame's own CRS, so nothing is reprojected.

    Args:
        gdf: Frame to write.
        path: Output path ending in `.parquet` or `.geoparquet`.
        overwrite: Replace an existing file when true.
        **write_options: Supported GeoPandas and Parquet write options.

    Returns:
        The written path.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        TypeError: An option is unsupported or supplied twice.
        ValueError: The suffix is wrong, or a direct option is repeated in
            `parquet_options`.
    """
    target = Path(path)
    if target.suffix not in _FILE_SUFFIXES:
        raise ValueError(
            f"destination {target.name!r} must end in one of {list(_FILE_SUFFIXES)}"
        )
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")

    parquet_options = dict(write_options.pop("parquet_options", {}))
    gdf.to_parquet(target, **write_options, **parquet_options)  # type: ignore
    # GeoPandas accepts None for no compression.
    return target
