"""Per-acquisition STAC metadata carried with a raster."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime as dt
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pystac.extensions.eo import BANDS_PROP as EO_BANDS
from pystac.extensions.raster import BANDS_PROP as RASTER_BANDS

from geosave_engine.geodata.attrs.model import AttrsModel
from geosave_engine.geodata.utils.datetime import naive_utc

if TYPE_CHECKING:
    import pystac

# STAC 1.1 core field replacing eo:bands/raster:bands; pystac has no constant for it.
_CORE_BANDS = "bands"
_BAND_LISTINGS = (RASTER_BANDS, EO_BANDS, _CORE_BANDS)

# pystac publishes extra_fields untyped, so a band listing is parsed on the way in.
_BAND_LISTING = TypeAdapter(list[dict[str, Any]])

# What to do with a field the items of one load describe differently.
type AssetConflict = Literal["drop", "reject"]


def read_asset_fields(asset: pystac.Asset) -> dict[str, object]:
    """Read one asset's fields, merging the band description it nests.

    A field may sit on the asset, such as `gsd`, or one level down inside a
    band list, such as `common_name` under `eo:bands`.

    Args:
        asset: Asset to read.

    Returns:
        {
            "<field>": value,
        }
        Band lists themselves are dropped once merged.

    Raises:
        ValidationError: A band listing is not a list of JSON objects.

    Examples:
        >>> read_asset_fields(item.assets["B04"])["common_name"]
        'red'
    """
    source: Mapping[str, Any] = asset.extra_fields or {}

    found: dict[str, Any] = {}
    for key, value in source.items():
        if key not in _BAND_LISTINGS:
            found[key] = value

    for listing in _BAND_LISTINGS:
        published = source.get(listing)
        if published is None:
            continue
        bands = _BAND_LISTING.validate_python(published)
        # One asset holds one variable here, so only its first band is relevant.
        if bands:
            found.update(bands[0])
    return found


def shared_asset_fields(
    items: Sequence[pystac.Item],
    asset: str,
    fields: Mapping[str, str],
    *,
    on_conflict: AssetConflict,
) -> dict[str, object]:
    """Read the fields every item describes one asset with identically.

    A load merges these items into one variable, and that variable carries one
    attrs mapping, so a field only holds of it where every item carries the
    asset, states the field non-null, and states the same value.

    Args:
        items: Items making up one load.
        asset: Asset name to read, which names the variable it loads into.
        fields: STAC asset field names mapped to the model field each fills.
        on_conflict: `"drop"` leaves out a field the items state differently;
            `"reject"` raises instead, for a field that decides what the pixel
            values mean.

    Returns:
        {
            "<model field>": the one value every item states,
        }
        Fields the items do not all state identically are absent.

    Raises:
        ValueError: `on_conflict` is `"reject"` and the items state one of
            `fields` differently.

    Examples:
        >>> shared_asset_fields(items, "B04", {"unit": "units"}, on_conflict="drop")
        {'units': '1'}
    """
    published = [
        read_asset_fields(found) for item in items if (found := item.assets.get(asset))
    ]
    if len(published) != len(items):
        return {}

    shared: dict[str, object] = {}
    for key, field in fields.items():
        stated = [entry.get(key) for entry in published]
        if any(value is None for value in stated):
            continue
        if any(value != stated[0] for value in stated[1:]):
            if on_conflict == "drop":
                continue
            raise ValueError(
                f"STAC items disagree on {key!r} for asset {asset!r}: "
                f"{_disagreement(items, stated)}. They load into one {asset!r} "
                f"variable carrying one {key!r}, so one item's pixels would "
                f"decode wrong. Narrow the search so every item states the same "
                f"{key!r}, or override it through stac_cfg."
            )
        shared[field] = stated[0]
    return shared


def _disagreement(items: Sequence[pystac.Item], stated: Sequence[object]) -> str:
    """Name an item behind each distinct value of one disputed field.

    Args:
        items: Items making up one load.
        stated: The value each item published, in the same order.

    Returns:
        One `'<item id>' states <value>` phrase per distinct value, in
        first-stated order.
    """
    example: dict[str, tuple[str, object]] = {}
    for item, value in zip(items, stated, strict=True):
        example.setdefault(repr(value), (item.id, value))
    return ", ".join(
        f"{item_id!r} states {value!r}" for item_id, value in example.values()
    )


class StacItem(BaseModel):
    """One STAC item as it was loaded.

    Args:
        id: Item ID.
        datetime: The item's own instant, naive UTC.
        properties: Item properties captured, keyed as STAC publishes them.
        assets: Asset name mapped to that asset's captured fields.

    Examples:
        >>> row.properties["eo:cloud_cover"]
        4.2
    """

    model_config = ConfigDict(frozen=True)

    id: str
    datetime: dt
    properties: dict[str, Any] = Field(default_factory=dict)
    assets: dict[str, dict[str, Any]] = Field(default_factory=dict)


class StacMetadata(AttrsModel):
    """Metadata of the STAC items a raster was loaded from.

    One row per item, each keeping its own identity and instant rather than an
    index into the loaded time axis, so a join accumulates provenance instead
    of realigning it.

    Args:
        stac_items: One row per item, in the order loaded.
        stac_groupby: How the items were grouped onto the time axis, as
            `odc.stac.load` was asked to group them.

    Examples:
        >>> ds.gs.attrs.root.get(StacMetadata).properties()
        ('eo:cloud_cover', 'platform')
    """

    NAME: ClassVar[str] = "stac"

    stac_items: tuple[StacItem, ...] = ()
    stac_groupby: str | None = None

    @classmethod
    def from_items(
        cls,
        items: Sequence[pystac.Item],
        *,
        groupby: str,
        item_properties: Sequence[str] | None = (),
        asset_fields: Sequence[str] | None = None,
    ) -> Self:
        """Record the items one load was built from, one row each.

        Args:
            items: Items making up one load, in search order.
            groupby: Grouping mode `odc.stac.load` was asked to apply, recorded
                so a reader can tell why several items share a time step.
            item_properties: Item property names to record. Empty records
                identity only; None records every property.
            asset_fields: Asset field names to record. Empty records none; None
                records every field an asset declares.

        Returns:
            Model holding one row per item, in search order.

        Raises:
            ValueError: An item declares neither `datetime` nor a datetime
                range, so it states no instant to record.

        Examples:
            >>> stac = StacMetadata.from_items(matched, groupby="solar_day")
            >>> len(stac.stac_items), stac.stac_groupby
            (4, 'solar_day')
        """
        import odc.stac

        rows = tuple(
            StacItem(
                id=item.id,
                datetime=naive_utc(entry.nominal_datetime),
                properties=_select(item.properties, item_properties),
                assets=_item_assets(item, asset_fields),
            )
            for entry, item in zip(odc.stac.parse_items(items), items, strict=True)
        )
        return cls(stac_groupby=groupby, stac_items=rows)

    def properties(self) -> tuple[str, ...]:
        """Name the item properties the rows carry.

        Returns:
            Property names across every row, sorted.
        """
        return tuple(sorted({key for row in self.stac_items for key in row.properties}))

    @classmethod
    def combine(cls, sides: Sequence[AttrsModel | None]) -> tuple[Self, set[str]]:
        """Combine STAC metadata: union `stac_items`, drop a mixed grouping.

        Items are provenance, so a joined result was loaded from all of them. A
        grouping the sides do not share describes neither, so it is dropped
        rather than refused.

        Args:
            sides: This model as each side of the join stated it, in call
                order, at least one, None where a side carries no STAC
                metadata.

        Returns:
            Metadata carrying every side's items once, the first occurrence
            of an item id winning, and the attr keys it could not keep.

        Raises:
            TypeError: A side holds a different model.
            ValueError: `sides` is empty.
        """
        grouped, dropped = cls._combine_fields(sides, must_agree=())
        items: dict[str, StacItem] = {}
        # _combine_fields has already refused any side that is not this model.
        for side in sides:
            if not isinstance(side, cls):
                continue
            for item in side.stac_items:
                items.setdefault(item.id, item)
        combined = cls(
            stac_items=tuple(items.values()),
            stac_groupby=grouped.stac_groupby,
        )
        # Items accumulate rather than drop, so differing ones are not a loss.
        return combined, dropped - {cls.attr_keys["stac_items"]}


def _item_assets(
    item: pystac.Item, asset_fields: Sequence[str] | None
) -> dict[str, dict[str, object]]:
    """Read each asset's captured fields, keyed by asset name.

    Args:
        item: Item whose assets to read.
        asset_fields: Field names to capture. Empty captures none; None captures
            every field an asset declares.

    Returns:
        {
            "B03": {captured fields},
        }
        Assets that would contribute no field are omitted.
    """
    captured: dict[str, dict[str, object]] = {}
    for name, asset in item.assets.items():
        selected = _select(read_asset_fields(asset), asset_fields)
        if selected:
            captured[name] = selected
    return captured


def _select(
    source: Mapping[str, object], names: Sequence[str] | None
) -> dict[str, object]:
    """Keep the named keys of a mapping, None keeping every key."""
    if names is None:
        return dict(source)
    return {key: source[key] for key in names if key in source}
