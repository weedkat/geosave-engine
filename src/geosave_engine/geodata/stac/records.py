"""Parse STAC items into the records a raster carries. See parse_item."""
from __future__ import annotations

from datetime import datetime as dt
from datetime import timedelta
from typing import TYPE_CHECKING, Literal, Sequence

from geosave_engine.geodata.errors import AnchorFetchError
from geosave_engine.geodata.extensions import StacItemRecord, StacItems
from geosave_engine.geodata.utils.datetime import DateRange, naive_utc

if TYPE_CHECKING:
    import pystac

# Extension properties read unless a source names its own — set without declaring the extension URI.
DEFAULT_PROPERTIES = (
    "eo:cloud_cover",
    "eo:snow_cover",
    "view:sun_azimuth",
    "view:sun_elevation",
    "view:off_nadir",
    "sat:orbit_state",
    "sat:relative_orbit",
    "sat:absolute_orbit",
)


def item_datetime(item: pystac.Item) -> dt:
    """Timestamp an item is dated by — its own, else the start or end of its range.

    A collection publishing a validity range rather than an instant, such as
    an annual land-cover map, leaves `datetime` null and dates the item
    through `start_datetime`/`end_datetime`.

    Args:
        item: Item to date.

    Returns:
        Timestamp as naive UTC.

    Raises:
        AnchorFetchError: Item declares no datetime, start_datetime or end_datetime.
    """
    common = item.common_metadata
    timestamp = item.datetime or common.start_datetime or common.end_datetime
    if timestamp is None:
        raise AnchorFetchError(f"STAC item {item.id!r} has no datetime, start_datetime or end_datetime")
    return naive_utc(timestamp)


def parse_item(item: pystac.Item, properties: Sequence[str] | None = None) -> StacItemRecord:
    """Parse one STAC item into a record.

    Args:
        item: Item to parse.
        properties: Extension property keys to carry, keyed as STAC
            publishes them. A key the item doesn't have is skipped. None
            reads `DEFAULT_PROPERTIES`.

    Returns:
        Parsed record — identity, every common metadata field the item
        declares, and the extension properties asked for.

    Raises:
        AnchorFetchError: Item declares no datetime, start_datetime or end_datetime.
    """
    common = item.common_metadata
    wanted = DEFAULT_PROPERTIES if properties is None else properties
    providers = common.providers
    return StacItemRecord(
        id=item.id,
        datetime=item_datetime(item),
        collection=item.collection_id,
        title=common.title,
        description=common.description,
        start_datetime=common.start_datetime,
        end_datetime=common.end_datetime,
        created=common.created,
        updated=common.updated,
        platform=common.platform,
        instruments=_as_tuple(common.instruments),
        constellation=common.constellation,
        mission=common.mission,
        gsd=common.gsd,
        license=common.license,
        providers=None if providers is None else tuple(provider.to_dict() for provider in providers),
        keywords=_as_tuple(common.keywords),
        roles=_as_tuple(common.roles),
        properties={key: item.properties[key] for key in wanted if key in item.properties},
    )


def parse_items(items: Sequence[pystac.Item], properties: Sequence[str] | None = None) -> StacItems:
    """Parse a whole search result into the extension a raster carries.

    Args:
        items: Items to parse, in any order.
        properties: Extension property keys to carry in each record. None
            reads `DEFAULT_PROPERTIES`.

    Returns:
        StacItems over every item.

    Raises:
        AnchorFetchError: An item declares no usable datetime.
    """
    return StacItems(items=tuple(parse_item(item, properties) for item in items))


def select_release(
    items: Sequence[pystac.Item],
    release: Literal["latest", "nearest"],
    window: DateRange | None,
) -> list[pystac.Item]:
    """Keep only the items sharing one release's publication date.

    Every tile of one release carries the same timestamp, so the whole
    footprint survives while the other releases drop out.

    Args:
        items: Matched items, in any order.
        release: "latest" takes the newest date, "nearest" the date closest
            to `window` — landing inside it counts as no distance at all.
        window: Inclusive span to measure "nearest" against. None is only
            valid for "latest".

    Returns:
        Items dated exactly on the chosen release, in the order given.

    Raises:
        AnchorFetchError: An item declares no usable datetime.
        ValueError: `release` is "nearest" and `window` is None.

    Examples:
        >>> select_release(items, "nearest", (dt(2021, 6, 1), dt(2021, 6, 30)))
    """
    dated = {item.id: item_datetime(item) for item in items}
    if release == "latest":
        chosen = max(dated.values())
    elif window is None:
        raise ValueError("release='nearest' needs a dated anchor to measure against")
    else:
        start, end = window
        chosen = min(dated.values(), key=lambda when: max(start - when, when - end, timedelta(0)))
    return [item for item in items if dated[item.id] == chosen]


def _as_tuple(values: Sequence[str] | None) -> tuple[str, ...] | None:
    """One optional string list as a tuple, so a record stays frozen.

    Args:
        values: Strings the item declared, or None.

    Returns:
        Tuple of the same strings, or None.
    """
    return None if values is None else tuple(values)
