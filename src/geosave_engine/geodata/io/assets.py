"""Open named raster pointers and publish their GeoVector catalog."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from dask.delayed import Delayed, delayed

from .storage import absolute_location

if TYPE_CHECKING:
    from geosave_engine.geodata import DataTree
    from geosave_engine.geodata.stac.item import Asset


def normalize(
    assets: Mapping[str, Asset] | str | PathLike[str],
) -> dict[str, dict[str, Any]]:
    """Resolve constructor paths once, preserving each asset's fields."""
    if isinstance(assets, (str, PathLike)):
        assets = {PurePosixPath(str(assets)).stem: assets}
    result = {}
    for name, asset in assets.items():
        fields: dict[str, Any] = (
            dict(asset) if isinstance(asset, Mapping) else {"href": asset}
        )
        if "href" not in fields:
            raise ValueError(f"asset {name!r} states no href")
        fields["href"] = absolute_location(fields["href"])
        result[name] = fields
    return result


def read(
    assets: Mapping[str, Mapping[str, Any] | None],
    *,
    layers: Collection[str] | str | None = None,
) -> DataTree:
    """Open selected raster assets lazily, including named groups in a store."""
    from geosave_engine.geodata.core.stack import stack
    from . import read_raster

    available = {name: asset for name, asset in assets.items() if asset is not None}
    names = (
        list(available)
        if layers is None
        else ([layers] if isinstance(layers, str) else list(layers))
    )
    if missing := [name for name in names if name not in available]:
        raise KeyError(
            f"the row has no {missing} layer; its layers are {list(available)}"
        )
    if not names:
        raise KeyError("the row names no assets")
    rasters = {}
    try:
        for name in names:
            asset = available[name]
            options = (
                {"group": asset["group"]} if asset.get("group") is not None else {}
            )
            rasters[name] = read_raster(asset["href"], chunks="auto", **options)
        tree = stack(rasters)
        for name, raster in rasters.items():
            tree.children[name].set_close(raster.close)
        return tree
    except BaseException:
        for raster in rasters.values():
            raster.close()
        raise


def write_catalog(
    saved: Path | Delayed,
    destination: str | PathLike[str] | None,
    *,
    id: str | None = None,
    stack: bool = False,
    overwrite: bool = False,
) -> Path | Delayed:
    """Publish a saved raster's record after its pixels finish writing."""
    if destination is None:
        return saved
    if isinstance(saved, Delayed):
        return delayed(write_catalog)(
            saved, destination, id=id, stack=stack, overwrite=overwrite
        )

    from geosave_engine.geodata.stac.item import file_assets, record
    from . import read_raster, read_stack

    data = read_stack(saved) if stack else read_raster(saved)
    try:
        catalog = record(data, assets=file_assets(data), id=id)
        catalog.gs.to_geoparquet(destination, overwrite=overwrite)
    finally:
        data.close()
    return saved
