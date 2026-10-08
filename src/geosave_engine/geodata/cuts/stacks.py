"""One whole window per saved sample, read off an item table."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import geopandas as gpd
import pandas as pd
import pystac
from affine import Affine
from odc.geo.geobox import GeoBox
from pystac.extensions.datacube import DatacubeExtension
from pystac.extensions.projection import ProjectionExtension

from geosave_engine.geodata.conventions import TIME_COORDINATE
from geosave_engine.geodata.stac import table
from geosave_engine.geodata.stac.extensions import GeosaveExtension
from geosave_engine.geodata.stac.extensions.geosave import STACK_PROP

if TYPE_CHECKING:
    from geosave_engine.geodata import GeoDataFrame


def stacks(items: gpd.GeoDataFrame) -> GeoDataFrame:
    """List each saved sample as one window covering all of it.

    Args:
        items: Item table whose rows share `geosave:stack` per sample, as
            `create_stack_items` builds them, one collection per group.

    Returns:
        One row per sample, in the order samples first appear:

            id, stack     the sample's name
            parent        None; a cut names the window it came from here
            times         {group: [time labels]}, None for a timeless group
            start_datetime, end_datetime   what those labels span, or None
            crs, transform                 the window's grid
            row_off, col_off, height, width   its pixels on the sample's grid
            geometry      its footprint in WGS84

    Raises:
        ValueError: The table has no `geosave:stack` column, a row states no
            grid, or the groups of one sample sit on different grids.

    Examples:
        >>> stacks(items)[["id", "height", "width"]].values.tolist()
        [['s0', 512, 512], ['s1', 512, 512]]
    """
    if STACK_PROP not in items:
        raise ValueError(
            f"the item table has no {STACK_PROP!r} column, so its rows name no "
            f"samples; build it with stac.create_stack_items"
        )

    samples: dict[str, list[pystac.Item]] = {}
    for item in table.to_items(items):
        samples.setdefault(cast("str", GeosaveExtension.ext(item).stack), []).append(
            item
        )

    rows = []
    for name, members in samples.items():
        grids: dict[str, GeoBox] = {}
        labels: dict[str, list[pd.Timestamp]] = {}
        for item in members:
            group = cast("str", item.collection_id)
            for asset in item.assets.values():
                grids[group] = _grid(asset)
                labels.setdefault(group, []).extend(_times(item, asset))

        grid = next(iter(grids.values()))
        apart = sorted(group for group, other in grids.items() if other != grid)
        if apart:
            raise ValueError(
                f"sample {name!r} holds {apart} on a different grid from "
                f"{next(iter(grids))!r}, so one window would not name the same "
                f"pixels in each; write the sample as one stack"
            )

        times = {
            group: [stamp.isoformat() for stamp in sorted(set(stamps))] or None
            for group, stamps in labels.items()
        }
        dated = [stamp for stamps in labels.values() for stamp in stamps]
        rows.append(
            {
                "id": name,
                "parent": None,
                "stack": name,
                "times": times,
                "start_datetime": min(dated) if dated else None,
                "end_datetime": max(dated) if dated else None,
                "crs": str(grid.crs),
                "transform": list(grid.transform)[:6],
                "row_off": 0,
                "col_off": 0,
                "height": grid.shape[0],
                "width": grid.shape[1],
                "geometry": grid.extent.to_crs("EPSG:4326").geom,
            }
        )
    return cast(
        "GeoDataFrame", gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    )


def _grid(asset: pystac.Asset) -> GeoBox:
    """Return the grid an asset's Projection fields state."""
    stated = ProjectionExtension.ext(asset)
    if stated.shape is None or stated.transform is None:
        raise ValueError(f"{asset.href} states no grid to cut windows on")
    crs = stated.code if stated.code is not None else stated.wkt2
    return GeoBox(tuple(stated.shape), Affine(*stated.transform[:6]), crs)


def _times(item: pystac.Item, asset: pystac.Asset) -> list[pd.Timestamp]:
    """Return the time labels an asset's Datacube fields state, naive UTC."""
    if not DatacubeExtension.has_extension(item):
        return []
    if "cube:dimensions" not in asset.extra_fields:
        return []
    time = DatacubeExtension.ext(asset).dimensions.get(TIME_COORDINATE)
    if time is None:
        return []
    return list(pd.to_datetime(time.values, utc=True).tz_localize(None))
