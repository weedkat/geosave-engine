"""Create an attrs header from one STAC load, and name a variable's attrs for STAC.

STAC names its fields its own way — `unit`, `scale` — so this is where those
names and the attr keys GeoSave writes are exchanged. No attrs model reaches
back into STAC.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Required, TypedDict

import numpy as np
from pydantic import TypeAdapter
from pystac.extensions.eo import BANDS_PROP as EO_BANDS
from pystac.extensions.raster import BANDS_PROP as RASTER_BANDS

from geosave_engine import __path__ as _package_paths
from geosave_engine.geodata.warnings import DroppedAttrsWarning
from geosave_engine.geodata.utils.datetime import naive_utc

from ..header import AttrsHeader
from ..model import common_attrs

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pystac
    import xarray as xr

_CORE_BANDS = "bands"
_BAND_LISTINGS = (RASTER_BANDS, EO_BANDS, _CORE_BANDS)
_BAND_LISTING = TypeAdapter(list[dict[str, Any]])

# STAC asset field mapped to the attr key that carries it on a loaded variable.
_ATTR_KEYS = {
    "unit": "units",
    "description": "long_name",
    "scale": "scale_factor",
    "offset": "add_offset",
}


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
    loads into, where a field the items publish differently is dropped.

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
        ValueError: `items` is empty, or an item publishes no instant to record.

    Warns:
        DroppedAttrsWarning: The items publish a field differently, or only
            some of them publish it.

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
            fields = read_asset_fields(asset, band_index=band_index) if asset else {}
            published.append(
                {
                    _ATTR_KEYS[key]: value
                    for key, value in fields.items()
                    if key in _ATTR_KEYS
                }
            )
        agreed = common_attrs(published)
        disputed = sorted(set().union(*published) - agreed.keys())
        if disputed:
            values = {
                key: {
                    item.id: item_attrs.get(key)
                    for item, item_attrs in zip(items, published, strict=True)
                }
                for key in disputed
            }
            warnings.warn(
                f"STAC items publish {disputed} differently for {str(name)!r}, so "
                f"the loaded variable carries none of them: {values}",
                DroppedAttrsWarning,
                skip_file_prefixes=tuple(_package_paths),
            )
        # The loader's own attrs describe the pixels it actually produced.
        variables[str(name)] = {**agreed, **variable.attrs}
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


class BandFields(TypedDict, total=False):
    """One saved variable in the names STAC gives a band."""

    name: Required[str]
    data_type: Required[str]
    nodata: float | int
    unit: str
    description: str
    scale: float
    offset: float


def band_fields(variable: xr.DataArray) -> BandFields:
    """Describe one saved variable in the names STAC gives a band.

    Args:
        variable: Variable as its file stores it.

    Returns:
        {
            "name": variable name,
            "data_type": stored dtype name,
            "nodata": fill value, where the variable declares one,
            "unit" | "description" | "scale" | "offset": the attr `_ATTR_KEYS`
                maps to that field, where the variable carries it,
        }
    """
    attrs = variable.attrs
    fields: BandFields = {
        "name": str(variable.name),
        "data_type": variable.dtype.name,
    }

    # JSON holds native numbers, and a file hands back numpy ones.
    nodata = attrs.get("_FillValue")
    if nodata is not None:
        fields["nodata"] = nodata.item() if isinstance(nodata, np.generic) else nodata

    unit = attrs.get("units")
    if unit is not None:
        fields["unit"] = str(unit)
    description = attrs.get("long_name")
    if description is not None:
        fields["description"] = str(description)
    scale = attrs.get("scale_factor")
    if scale is not None:
        fields["scale"] = float(scale)
    offset = attrs.get("add_offset")
    if offset is not None:
        fields["offset"] = float(offset)
    return fields
