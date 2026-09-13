"""Open and write GeoJSON vector files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import geopandas as gpd

from pathlib import Path

if TYPE_CHECKING:
    from os import PathLike

_FILE_SUFFIXES = (".geojson", ".json")


class GeoJSONOpenOptions(TypedDict, total=False):
    """Optional GeoPandas behavior supported when opening GeoJSON."""

    columns: list[str] | None
    bbox: tuple[float, float, float, float] | None
    mask: Any
    rows: int | slice | None
    driver_options: dict[str, Any]


def read(
    source: str | PathLike[str],
    *,
    engine: Literal["fiona", "pyogrio"] = "pyogrio",
    **open_options: Unpack[GeoJSONOpenOptions],
) -> gpd.GeoDataFrame:
    """Open one GeoJSON file as a GeoDataFrame.

    Args:
        source: Local GeoJSON path or URI.
        engine: Concrete GeoPandas file-reading engine.
        **open_options: Supported GeoPandas and file-driver read options.

    Returns:
        GeoDataFrame on the CRS the file names.

    Raises:
        TypeError: An option is unsupported or supplied twice.
        ValueError: A driver option selects an engine or disables geometry.
    """
    driver_options = dict(open_options.pop("driver_options", {}))
    if "engine" in driver_options:
        raise ValueError("driver_options must not contain engine; use engine=")
    if "ignore_geometry" in driver_options:
        raise ValueError("driver_options must not contain ignore_geometry")
    return gpd.read_file(
        source,
        engine=engine,
        ignore_geometry=False,
        **open_options,
        **driver_options,
    )


class GeoJSONWriteOptions(TypedDict, total=False):
    """Optional GeoPandas behavior supported when writing GeoJSON."""

    index: bool | None
    promote_to_multi: bool | None
    nan_as_null: bool
    metadata: dict[str, str] | None
    driver_options: dict[str, Any]


def write(
    gdf: gpd.GeoDataFrame,
    path: str | PathLike[str],
    *,
    engine: Literal["fiona", "pyogrio"] = "pyogrio",
    overwrite: bool = False,
    **write_options: Unpack[GeoJSONWriteOptions],
) -> Path:
    """Write one GeoDataFrame as a GeoJSON file.

    GeoJSON states coordinates in WGS84, so a projected frame is reprojected
    on the way out rather than written in its own CRS.

    Args:
        gdf: Frame to write.
        path: Output path ending in `.geojson` or `.json`.
        engine: Concrete GeoPandas file-writing engine.
        overwrite: Replace an existing file when true.
        **write_options: Supported GeoPandas and file-driver write options.

    Returns:
        The written path.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        TypeError: An option is unsupported or supplied twice.
        ValueError: The suffix is wrong, or an option conflicts with `engine`.
    """
    target = _checked_target(path, overwrite)
    driver_options = dict(write_options.pop("driver_options", {}))
    if "engine" in driver_options:
        raise ValueError("driver_options must not contain engine; use engine=")
    gdf.to_crs("EPSG:4326").to_file(
        target,
        driver="GeoJSON",
        engine=engine,
        **write_options,
        **driver_options,
    )
    return target


def _checked_target(path: str | PathLike[str], overwrite: bool) -> Path:
    """Resolve an output path, refusing a wrong suffix or an existing file.

    Args:
        path: Output path.
        overwrite: Replace an existing file when true.

    Returns:
        The resolved path.

    Raises:
        FileExistsError: The path exists and `overwrite` is false.
        ValueError: The suffix names no GeoJSON file.
    """
    target = Path(path)
    if target.suffix not in _FILE_SUFFIXES:
        raise ValueError(
            f"destination {target.name!r} must end in one of {list(_FILE_SUFFIXES)}"
        )
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
    return target
