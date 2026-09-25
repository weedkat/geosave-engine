"""Create an attrs header from one STAC load.

STAC names its fields its own way — `unit`, `nodata`, `scale` — so this is
where those names become the attr keys GeoSave writes. No attrs model reaches
back into STAC.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import TypeAdapter
from pystac.extensions.eo import BANDS_PROP as EO_BANDS
from pystac.extensions.raster import BANDS_PROP as RASTER_BANDS

from geosave_engine.geodata.utils.datetime import naive_utc

from ..header import AttrsHeader

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pystac
    import xarray as xr

type StacGroupby = Literal["solar_day", "id", "time"]

_CORE_BANDS = "bands"
_BAND_LISTINGS = (RASTER_BANDS, EO_BANDS, _CORE_BANDS)
_BAND_LISTING = TypeAdapter(list[dict[str, Any]])

type AssetConflict = Literal["drop", "reject"]

_LABELLING = {"unit": "units", "description": "long_name"}
_PACKING = {"scale": "scale_factor", "offset": "add_offset"}


def create_header(
    items: Sequence[pystac.Item],
    collection: pystac.Collection,
    loaded: xr.Dataset,
    *,
    groupby: str,
    item_properties: Sequence[str] | None = (),
    asset_fields: Sequence[str] | None = None,
    stac_cfg: dict[str, Any] | None = None,
) -> AttrsHeader:
    """Create the attrs header described by one STAC load.

    The collection describes the whole raster and the items record where it
    came from, so both land on the root; each asset describes the variable it
    loads into.

    Args:
        items: Items making up one load, in search order.
        collection: Collection the items belong to.
        loaded: Output Dataset whose loader attrs describe effective pixels.
        groupby: Grouping mode `odc.stac.load` was asked to apply.
        item_properties: Item property names to record. Empty records identity
            only; None records every property.
        asset_fields: Asset field names to record. Empty records none; None
            records every field an asset publishes.
        stac_cfg: The same ODC conversion settings used for loading.

    Returns:
        Header to rebase onto the loaded Dataset.

    Raises:
        ValueError: `items` is empty, an item publishes no instant to record,
            or the items publish different values for a field decoding pixels.

    Examples:
        >>> header = create_header(matched, collection, loaded, groupby="solar_day")
        >>> header.data_vars["B04"].to_attrs()
        {'units': '1', 'long_name': 'Red', '_FillValue': 0, 'nodata': 0,
         'scale_factor': 0.0001, 'add_offset': -0.1}
    """
    if not items:
        raise ValueError("creating a STAC header needs at least one item")

    import odc.stac

    parsed = list(odc.stac.parse_items(items, cfg=stac_cfg))
    providers = collection.providers or []
    rows = [
        {
            "id": item.id,
            "datetime": naive_utc(entry.nominal_datetime),
            "properties": _selected(item.properties, item_properties),
            "assets": _item_assets(item, asset_fields),
        }
        for entry, item in zip(parsed, items, strict=True)
    ]
    root: dict[str, Any] = {
        "id": collection.id,
        "title": collection.title,
        "summary": collection.description,
        "keywords": ", ".join(collection.keywords) if collection.keywords else None,
        "institution": providers[0].name if providers else None,
        "license": collection.license,
        "stac_groupby": groupby,
        "stac_items": rows,
    }

    variables = {}
    for name, variable in loaded.data_vars.items():
        published = []
        for item, entry in zip(items, parsed, strict=True):
            asset_name, band_index = entry.collection.band_key(str(name))
            asset = item.assets.get(asset_name)
            published.append(
                read_asset_fields(asset, band_index=band_index) if asset else {}
            )
        variables[str(name)] = {
            **_shared_fields(
                items, str(name), published, _LABELLING, on_conflict="drop"
            ),
            **_shared_fields(
                items, str(name), published, _PACKING, on_conflict="reject"
            ),
            **variable.attrs,
        }
    return AttrsHeader.from_attrs(
        root={
            **loaded.attrs,
            **{key: value for key, value in root.items() if value is not None},
        },
        data_vars=variables,
    )


def read_asset_fields(asset: pystac.Asset, *, band_index: int = 1) -> dict[str, object]:
    """Read one asset's fields, merging the band description it nests.

    Args:
        asset: Asset to read.
        band_index: One-based source band index.

    Returns:
        Asset fields with nested band fields merged and band listings removed.

    Raises:
        ValidationError: A band listing is not a list of JSON objects.
    """
    source: Mapping[str, Any] = asset.extra_fields or {}
    found = {key: value for key, value in source.items() if key not in _BAND_LISTINGS}

    for listing in _BAND_LISTINGS:
        published = source.get(listing)
        if published is None:
            continue
        bands = _BAND_LISTING.validate_python(published)
        if len(bands) >= band_index:
            found.update(bands[band_index - 1])
    return found


def _shared_fields(
    items: Sequence[pystac.Item],
    variable: str,
    published: Sequence[Mapping[str, object]],
    fields: Mapping[str, str],
    *,
    on_conflict: AssetConflict,
) -> dict[str, object]:
    """Keep fields shared by the resolved source bands of one output variable."""
    shared: dict[str, object] = {}
    for key, attr_key in fields.items():
        per_item = [entry.get(key) for entry in published]
        if any(value is None for value in per_item):
            continue
        if any(value != per_item[0] for value in per_item[1:]):
            if on_conflict == "drop":
                continue
            raise ValueError(
                f"STAC items disagree on {key!r} for variable {variable!r}: "
                f"{_disagreement(items, per_item)}. They load into one {variable!r} "
                f"variable carrying one {key!r}, so one item's pixels would "
                f"decode wrong. Narrow the search so every item publishes the same "
                f"{key!r}."
            )
        shared[attr_key] = per_item[0]
    return shared


def _item_assets(
    item: pystac.Item, asset_fields: Sequence[str] | None
) -> dict[str, dict[str, object]]:
    """Read each asset's captured fields, keyed by asset name."""
    captured: dict[str, dict[str, object]] = {}
    for name, asset in item.assets.items():
        selected = _selected(read_asset_fields(asset), asset_fields)
        if selected:
            captured[name] = selected
    return captured


def _selected(
    source: Mapping[str, object], names: Sequence[str] | None
) -> dict[str, object]:
    """Keep the named keys of a mapping, None keeping every key."""
    if names is None:
        return dict(source)
    return {key: source[key] for key in names if key in source}


def _disagreement(items: Sequence[pystac.Item], per_item: Sequence[object]) -> str:
    """Name an item behind each distinct value of one disputed field."""
    example: dict[str, tuple[str, object]] = {}
    for item, value in zip(items, per_item, strict=True):
        example.setdefault(repr(value), (item.id, value))
    return ", ".join(
        f"{item_id!r} publishes {value!r}" for item_id, value in example.values()
    )
