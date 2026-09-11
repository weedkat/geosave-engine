"""Stamp odc-stac loads with the semantics their STAC items publish."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from geosave_engine.geodata.attrs import (
    ACDD,
    CFVariable,
    Packing,
    StacMetadata,
    rebase,
)

if TYPE_CHECKING:
    import pystac
    import xarray as xr

# Grouping modes a source may ask odc-stac for.
type StacGroupby = Literal["solar_day", "id", "time"]


def stamp_stac(
    ds: xr.Dataset,
    items: Sequence[pystac.Item],
    collection: pystac.Collection,
    *,
    groupby: StacGroupby,
    item_properties: Sequence[str] | None = (),
    asset_fields: Sequence[str] | None = None,
) -> xr.Dataset:
    """Stamp a STAC load with the provenance of the items it came from.

    Collection metadata becomes `ACDD`, matched items become `StacMetadata` rows,
    and each variable receives `CFVariable` and `Packing` fields its asset declares
    consistently. Missing or conflicting descriptive fields are not stamped.

    Args:
        ds: Dataset `odc.stac.load` produced from `items`.
        items: Items making up the load, in search order.
        collection: Collection the items belong to.
        groupby: Grouping mode passed to `odc.stac.load`.
        item_properties: Item property names to record. Empty records identity
            only; None records every property.
        asset_fields: Asset field names to record. Empty records none; None
            records every field an asset declares.

    Returns:
        New Dataset carrying the STAC-derived attrs, same pixels.

    Raises:
        ValueError: `items` is empty, an item declares no usable timestamp, or
            items disagree on one asset's packing.

    Examples:
        >>> stamped = stamp_stac(ds, matched, collection, groupby="solar_day")
        >>> stamped.gs.attrs.root.get(StacMetadata).stac_groupby
        'solar_day'
    """
    if not items:
        raise ValueError("STAC stamping needs at least one item")

    stamped = ds.copy(deep=False)
    rebase(
        stamped,
        ACDD.from_collection(collection),
        StacMetadata.from_items(
            items,
            groupby=groupby,
            item_properties=item_properties,
            asset_fields=asset_fields,
        ),
        inplace=True,
    )

    for variable in map(str, stamped.data_vars):
        rebase(
            stamped,
            CFVariable.from_stac_asset(items, variable),
            Packing.from_stac_asset(items, variable),
            target=variable,
            inplace=True,
        )
    return stamped
