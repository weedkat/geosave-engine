import pystac
import xarray as xr
from dataclasses import dataclass, field

from geosave_engine.geodata.stac.query import Sentinel2L2AQuery, Sentinel2L1CQuery
from geosave_engine.utils.geodata import extract_raster_scale_offset

from .base import BaseSource


@dataclass(frozen=True, kw_only=True)
class Sentinel2L2A(BaseSource):
    """Sentinel-2 L2A (surface reflectance). Scales DN to reflectance via item metadata."""

    query: Sentinel2L2AQuery = field(default_factory=Sentinel2L2AQuery)

    def _apply_processing(self, ds: xr.Dataset, items: list[pystac.Item]) -> xr.Dataset:
        scale, offset = extract_raster_scale_offset(items[0])
        return ds * scale + offset


@dataclass(frozen=True, kw_only=True)
class Sentinel2L1C(BaseSource):
    """Sentinel-2 L1C (top-of-atmosphere radiance). No radiometric correction applied yet."""

    query: Sentinel2L1CQuery = field(default_factory=Sentinel2L1CQuery)

    def _apply_processing(self, ds: xr.Dataset, items: list[pystac.Item]) -> xr.Dataset:
        scale, offset = extract_raster_scale_offset(items[0])
        return ds * scale + offset
