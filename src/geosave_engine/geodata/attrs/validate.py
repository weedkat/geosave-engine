"""Shared conversions for values read from flat xarray attrs."""

from __future__ import annotations

import orjson


def parse_collection_text(value: object) -> object:
    """Decode JSON text while leaving native values and other text unchanged.

    GDAL metadata tags are text-only, so lists and mappings written there come
    back as JSON text. Zarr and netCDF return those same values as native Python
    collections. Concrete attrs fields use this before their normal validation
    so both representations produce the same field value.

    Args:
        value: Stored attr value.

    Returns:
        Decoded JSON when `value` is valid JSON text; otherwise `value` itself.
    """
    if not isinstance(value, str):
        return value
    try:
        return orjson.loads(value)
    except orjson.JSONDecodeError:
        return value
