from __future__ import annotations

import pystac
import xarray as xr
from dataclasses import dataclass
from typing import TypeVar

from geosave_engine.geodata.stac.query import StacQuery
from geosave_engine.utils.geodata import extract_raster_scale_offset
from geosave_engine.utils.stac_query import CQL2

from .base import BaseSource, BaseSourceSeries, _SourceBase

T = TypeVar("T", bound="_SourceBase")


def preprocess_sentinel2(ds: xr.Dataset, items: list[pystac.Item]) -> xr.Dataset:
    """Apply radiometric scaling to Sentinel-2 data using item metadata.

    Scaling broadcasts over every band variable in the Dataset.
    """
    scale, offset = extract_raster_scale_offset(items[0])
    return ds * scale + offset


class _Sentinel2Mixin:
    """Cloud cover filtering and sorting shared by all Sentinel-2 source types."""

    def max_cloud_cover(self: T, max_cover: float) -> T:
        """Add a cloud cover filter to the query."""
        self.query.with_filter(CQL2.lte("eo:cloud_cover", max_cover))
        return self

    def sortby_cloud_cover(self: T, direction: str = "asc") -> T:
        """Sort results by cloud cover ascending to prioritize clearer scenes."""
        self.query.with_sortby("eo:cloud_cover", direction)
        return self


@dataclass
class Sentinel2(_Sentinel2Mixin, BaseSource):
    """Base class for single-scene Sentinel-2 sources. Applies radiometric preprocessing by default."""

    def __post_init__(self) -> None:
        if self.preprocess is None:
            self.preprocess = preprocess_sentinel2
        super().__post_init__()


@dataclass
class Sentinel2L2A(Sentinel2):
    """Sentinel-2 L2A (surface reflectance). Scales DN to reflectance via item metadata."""

    def _default_query(self) -> StacQuery:
        return StacQuery(collections=["sentinel-2-l2a"])


@dataclass
class Sentinel2L1C(Sentinel2):
    """Sentinel-2 L1C (top-of-atmosphere radiance). No atmospheric correction applied."""

    def _default_query(self) -> StacQuery:
        return StacQuery(collections=["sentinel-2-l1c"])


@dataclass
class Sentinel2Series(_Sentinel2Mixin, BaseSourceSeries):
    """Base class for time-series Sentinel-2 sources. Applies radiometric preprocessing by default."""

    def __post_init__(self) -> None:
        if self.preprocess is None:
            self.preprocess = preprocess_sentinel2
        super().__post_init__()


@dataclass
class Sentinel2L2ASeries(Sentinel2Series):
    """Sentinel-2 L2A time series. Returns ``n_steps`` composited or per-scene GeoTiles."""

    def _default_query(self) -> StacQuery:
        return StacQuery(collections=["sentinel-2-l2a"])


@dataclass
class Sentinel2L1CSeries(Sentinel2Series):
    """Sentinel-2 L1C time series. Returns ``n_steps`` composited or per-scene GeoTiles."""

    def _default_query(self) -> StacQuery:
        return StacQuery(collections=["sentinel-2-l1c"])
