"""Sentinel-2 Level-2A ingestor (Surface reflectance)."""
from __future__ import annotations

import numpy as np
import pystac
import xarray as xr

from geosave_engine.geodata.core.constants import SENTINEL2_L2A
from geosave_engine.geodata.core.errors import IngestionError, IngestionErrorReason
from geosave_engine.geodata.core.ingestion import BaseIngestor


def _extract_scale_offset_from_item(item: pystac.Item) -> tuple[float, float]:
    """Read (scale, offset) from STAC item asset metadata.

    Tries ``raster:scale`` / ``raster:offset`` top-level asset fields first,
    then falls back to ``raster:bands[0].scale`` / ``raster:bands[0].offset``.
    Works across CDSE, Element84, and Planetary Computer layouts.

    Raises:
        IngestionError: with reason ``METADATA_MISSING`` when no asset carries a
            usable (non-zero) scale value.
    """
    for asset in item.assets.values():
        extra = asset.extra_fields
        for scale_key, offset_key in [("raster:scale", "raster:offset"), (None, None)]:
            if scale_key:
                scale  = extra.get(scale_key)
                offset = extra.get(offset_key)
            else:
                rb = extra.get("raster:bands")
                if not (rb and isinstance(rb, list)):
                    continue
                scale  = rb[0].get("scale")
                offset = rb[0].get("offset")
            if scale is not None and offset is not None:
                s = float(scale)
                if s != 0.0:
                    return s, float(offset)
    raise IngestionError(
        IngestionErrorReason.METADATA_MISSING,
        f"item {item.id!r} has no usable raster scale/offset metadata",
    )


class Sentinel2L2AIngestor(BaseIngestor):
    """Sentinel-2 Level-2A ingestor (Surface reflectance).

    Applies a single global scale/offset read from the first STAC item to all
    requested bands. Override ``_apply_radiometry`` in a subclass when
    per-band handling or SCL exclusion is needed.
    """

    COLLECTION = SENTINEL2_L2A

    def _apply_radiometry(
        self,
        data_array: xr.DataArray,
        items: list[pystac.Item],
    ) -> xr.DataArray:
        scale, offset = _extract_scale_offset_from_item(items[0])
        return data_array.astype(np.float32) * np.float32(scale) + np.float32(offset)
