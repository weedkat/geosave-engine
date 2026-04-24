"""Sentinel-2 STAC query: CQL2 filter builders shared across providers."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from geosave_engine.geodata.core.constants import CQL2Key
from geosave_engine.geodata.core.query import BaseStacQuery
from geosave_engine.utils.geodata.cql2 import CQL2


@dataclass(frozen=True)
class Sentinel2Query(BaseStacQuery):
    """A STAC query specialised for Sentinel-2 data.

    Inherits all core STAC parameters from ``BaseStacQuery``. The builder
    methods populate the ``filter`` field via CQL2 expressions so callers
    never need to write raw CQL2-JSON. Every condition lives in ``filter``,
    so a ``Sentinel2Query`` can be passed anywhere a ``BaseStacQuery`` is
    expected.

    Filter keys come from the STAC item-property namespace and work against
    any provider that indexes them (CDSE, Element84, Planetary Computer).
    Filters on properties a provider does not index simply match nothing.

    Example::

        from geosave_engine.geodata.stac_client import CdseClient
        from geosave_engine.geodata.stac_query.sentinel2 import Sentinel2Query

        client = CdseClient()
        query = (
            Sentinel2Query(collections=["sentinel-2-l1c"])
            .max_cloud_cover(20)
            .max_no_data_pixel_pct(5)
        )
        items = client.search(query)
    """

    def with_filter(self, expr: dict[str, Any]) -> Sentinel2Query:
        merged = CQL2.and_(self.filter, expr) if self.filter is not None else expr
        return dataclasses.replace(self, filter=merged)

    def max_cloud_cover(self, max_pct: float) -> Sentinel2Query:
        """Filter items where ``eo:cloud_cover`` < ``max_pct``."""
        return self.with_filter(CQL2.lt(CQL2Key.CLOUD_COVER, max_pct))

    def max_no_data_pixel_pct(self, max_pct: float) -> Sentinel2Query:
        """Filter items where ``s2:nodata_pixel_percentage`` ≤ ``max_pct``."""
        return self.with_filter(CQL2.lte(CQL2Key.NODATA_PIXEL_PCT, max_pct))

    def platform(self, name: str) -> Sentinel2Query:
        """Filter by platform name, e.g. ``"sentinel-2a"`` or ``"sentinel-2b"``."""
        return self.with_filter(CQL2.eq(CQL2Key.PLATFORM, name))

    def orbit_state(self, state: str) -> Sentinel2Query:
        """Filter by orbit state, e.g. ``"ascending"`` or ``"descending"``."""
        return self.with_filter(CQL2.eq(CQL2Key.ORBIT_STATE, state.lower()))

    def processing_baseline(self, baseline: str) -> Sentinel2Query:
        """Filter by ``s2:processing_baseline``, e.g. ``"05.09"``."""
        return self.with_filter(CQL2.eq(CQL2Key.PROCESSING_BASELINE, baseline))
