"""Build native STAC Items from named assets."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime as DateTime

import pystac
from affine import Affine
from odc.geo.geobox import GeoBox
from pystac.extensions.eo import EOExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterExtension
from shapely.geometry import mapping



def from_assets(
    assets: Mapping[str, pystac.Asset],
    *,
    id: str,
    datetime: DateTime | None = None,
    collection: str | None = None,
) -> pystac.Item:
    """Build one Item from named assets.

    Args:
        assets: Assets from `stac.asset.from_path`, keyed as the Item names them.
        id: Item identity.
        datetime: Item instant. None takes the span its dated assets cover.
        collection: Name shared by every Item of one raster.

    Returns:
        Item over the first asset's grid, dated by one instant or by a start
        and end, holding copies of the assets.

    Raises:
        ValueError: No asset is dated and `datetime` is None.

    Examples:
        >>> from_assets({"label": asset.from_path("s0/label.tif")}, id="s0").id
        's0'
    """
    # The Item's footprint is the first asset's grid, in WGS84 as STAC states it.
    first_key = next(iter(assets))
    projection = ProjectionExtension.ext(assets[first_key])
    shape = projection.shape
    transform = projection.transform
    if shape is None or transform is None:
        raise ValueError(
            f"asset {first_key!r} states no grid, so the Item has no footprint; "
            f"build assets with stac.asset.from_raster or stac.asset.from_path"
        )
    height, width = shape
    geobox = GeoBox((height, width), Affine(*transform[:6]), projection.crs_string)
    footprint = geobox.extent.to_crs("EPSG:4326").geom

    # The Item covers `datetime`, or else everything its dated assets cover.
    if datetime is not None:
        start = datetime
        end = datetime
    else:
        starts = []
        ends = []
        for asset in assets.values():
            times = asset.common_metadata
            if times.start_datetime is not None and times.end_datetime is not None:
                starts.append(times.start_datetime)
                ends.append(times.end_datetime)
        if not starts:
            raise ValueError(
                f"none of the assets {list(assets)} is dated, and a STAC Item needs a "
                f"time; pass datetime="
            )
        start = min(starts)
        end = max(ends)

    # STAC dates an Item by one instant, or by a start and an end, never both.
    if start == end:
        instant = start
        range_start = None
        range_end = None
    else:
        instant = None
        range_start = start
        range_end = end

    item = pystac.Item(
        id,
        mapping(footprint),
        list(footprint.bounds),
        instant,
        {},
        start_datetime=range_start,
        end_datetime=range_end,
        collection=collection,
    )
    for extension in (ProjectionExtension, RasterExtension, EOExtension):
        extension.add_to(item)
    for key, asset in assets.items():
        item.add_asset(key, asset.clone())
    return item
