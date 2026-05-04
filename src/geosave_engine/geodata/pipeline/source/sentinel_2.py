import pystac
import xarray as xr
from dataclasses import dataclass, field, replace

from geosave_engine.geodata.pipeline.anchor import Anchor
from geosave_engine.geodata.stac.query import Sentinel2Query, Sentinel2L2AQuery, Sentinel2L1CQuery

from .base import BaseSource


def _extract_scale_offset(item: pystac.Item) -> tuple[float, float]:
    """Extract radiometric scale and offset from a STAC item's raster:bands metadata."""
    for asset in item.assets.values():
        bands = asset.extra_fields.get("raster:bands", [])
        if bands:
            band = bands[0]
            return float(band.get("scale", 1.0)), float(band.get("offset", 0.0))
    raise ValueError(
        f"Cannot extract scale/offset from STAC item '{item.id}': "
        "no 'raster:bands' found in any asset"
    )


@dataclass(frozen=True, kw_only=True)
class Sentinel2(BaseSource):
    """Shared Sentinel-2 query logic. Subclasses implement _apply_processing."""

    query: Sentinel2Query

    def _build_query(self, anchor: Anchor) -> Sentinel2Query:
        start = anchor.datetime - self.time_range
        end = anchor.datetime + self.time_range
        return replace(self.query, bbox=anchor.bbox, datetime=(start, end))


@dataclass(frozen=True, kw_only=True)
class Sentinel2L2A(Sentinel2):
    """Sentinel-2 L2A (surface reflectance). Scales DN to reflectance via item metadata."""

    query: Sentinel2L2AQuery = field(default_factory=Sentinel2L2AQuery)

    def _apply_processing(self, ds: xr.Dataset, items: list[pystac.Item]) -> xr.Dataset:
        scale, offset = _extract_scale_offset(items[0])
        return ds * scale + offset


@dataclass(frozen=True, kw_only=True)
class Sentinel2L1C(Sentinel2):
    """Sentinel-2 L1C (top-of-atmosphere radiance). No radiometric correction applied yet."""

    query: Sentinel2L1CQuery = field(default_factory=Sentinel2L1CQuery)

    def _apply_processing(self, ds: xr.Dataset, items: list[pystac.Item]) -> xr.Dataset:
        scale, offset = _extract_scale_offset(items[0])
        return ds * scale + offset
