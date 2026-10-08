"""Open and write GeoParquet vector files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypedDict, Unpack, cast

import geopandas as gpd

from .. import storage
from ..storage import StorageOptions

FILE_SUFFIXES = (".parquet", ".geoparquet")


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
    filesystem, source_path = storage.filesystem_path(source, storage_options)
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
    # PyArrow accepts engine-specific keywords that GeoPandas cannot type.
    parquet_options = cast(
        "dict[str, Any]", dict(write_options.pop("parquet_options", {}))
    )

    def save(target: Path) -> None:
        gdf.to_parquet(target, **write_options, **parquet_options)

    return storage.write(
        path,
        save,
        suffixes=FILE_SUFFIXES,
        overwrite=overwrite,
        storage_options=storage_options,
    )
