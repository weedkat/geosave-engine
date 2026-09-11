"""Open and write GeoPackage vector layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import geopandas as gpd

from pathlib import Path

if TYPE_CHECKING:
    from os import PathLike

_FILE_SUFFIXES = (".gpkg",)


class GeoPackageOpenOptions(TypedDict, total=False):
    """Optional GeoPandas behavior supported when opening a GeoPackage."""

    columns: list[str] | None
    bbox: tuple[float, float, float, float] | None
    mask: Any
    rows: int | slice | None
    driver_options: dict[str, Any]


def read(
    source: str | PathLike[str],
    *,
    layer: str | None = None,
    engine: Literal["fiona", "pyogrio"] = "pyogrio",
    **open_options: Unpack[GeoPackageOpenOptions],
) -> gpd.GeoDataFrame:
    """Open one GeoPackage layer as a GeoDataFrame.

    Args:
        source: Local GeoPackage path or URI.
        layer: Layer name. None selects the layer only when exactly one exists.
        engine: Concrete GeoPandas file-reading engine.
        **open_options: Supported GeoPandas and file-driver read options.

    Returns:
        GeoDataFrame on the CRS declared by the selected layer.

    Raises:
        TypeError: An option is unsupported or supplied twice.
        ValueError: `layer` is ambiguous, or a driver option selects an engine
            or disables geometry.
    """
    selected_layer = layer
    if selected_layer is None:
        layer_names = [str(name) for name in gpd.list_layers(source)["name"]]
        if len(layer_names) != 1:
            raise ValueError(
                f"GeoPackage {Path(str(source)).name!r} contains layers "
                f"{layer_names}; select one with layer="
            )
        selected_layer = layer_names[0]

    driver_options = dict(open_options.pop("driver_options", {}))
    if "engine" in driver_options:
        raise ValueError("driver_options must not contain engine; use engine=")
    if "ignore_geometry" in driver_options:
        raise ValueError("driver_options must not contain ignore_geometry")
    return gpd.read_file(
        source,
        layer=selected_layer,
        engine=engine,
        ignore_geometry=False,
        **open_options,
        **driver_options,
    )


class GeoPackageWriteOptions(TypedDict, total=False):
    """Optional GeoPandas behavior supported when writing a GeoPackage."""

    index: bool | None
    promote_to_multi: bool | None
    nan_as_null: bool
    metadata: dict[str, str] | None
    driver_options: dict[str, Any]


def write(
    gdf: gpd.GeoDataFrame,
    path: str | PathLike[str],
    *,
    layer: str | None = None,
    engine: Literal["fiona", "pyogrio"] = "pyogrio",
    overwrite: bool = False,
    **write_options: Unpack[GeoPackageWriteOptions],
) -> Path:
    """Write one GeoDataFrame as a layer in a GeoPackage.

    A GeoPackage holds the frame in its own CRS, so nothing is reprojected.

    Args:
        gdf: Frame to write.
        path: Output path ending in `.gpkg`.
        layer: Layer name. None names the layer after the file stem.
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
    target = Path(path)
    if target.suffix not in _FILE_SUFFIXES:
        raise ValueError(
            f"destination {target.name!r} must end in one of {list(_FILE_SUFFIXES)}"
        )
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} exists; pass overwrite=True to replace it")
        target.unlink()

    driver_options = dict(write_options.pop("driver_options", {}))
    if "engine" in driver_options:
        raise ValueError("driver_options must not contain engine; use engine=")
    gdf.to_file(
        target,
        layer=layer if layer is not None else target.stem,
        driver="GPKG",
        engine=engine,
        **write_options,
        **driver_options,
    )
    return target
