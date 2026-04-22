from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from geosave_engine.geodata.stac_query.base import StacQuery
from geosave_engine.utils.cql2 import CQL2


@dataclass(frozen=True)
class Sentinel2Query(StacQuery):
    """A STAC query specialised for Sentinel-2 data.

    Inherits all core STAC parameters from `StacQuery`.  The builder methods
    here populate the `filter` field via CQL2 expressions so callers never
    need to write raw CQL2-JSON.  Because every Sentinel-2 specific condition
    lives in `filter`, this class is fully recognised as a `StacQuery` and
    can be passed anywhere a `StacQuery` is expected.

    Example::

        from geosave_engine.stac_query import CdseClient, Sentinel2Query

        client = CdseClient()
        query = (
            Sentinel2Query(collections=["SENTINEL-2"])
            .max_cloud_cover(20)
            .max_no_data_pixel_pct(5)
        )
        for item in client.search(query):
            print(item.id, item.properties.get("eo:cloud_cover"))
    """

    def with_filter(self, expr: dict[str, Any]) -> Sentinel2Query:
        merged = CQL2.and_(self.filter, expr) if self.filter is not None else expr
        return dataclasses.replace(self, filter=merged)

    def max_cloud_cover(self, max_pct: float) -> Sentinel2Query:
        """Filter items where `eo:cloud_cover` < `max_pct`."""
        return self.with_filter(CQL2.lt("eo:cloud_cover", max_pct))

    def max_no_data_pixel_pct(self, max_pct: float) -> Sentinel2Query:
        """Filter items where `s2:nodata_pixel_percentage` ≤ `max_pct`."""
        return self.with_filter(CQL2.lte("s2:nodata_pixel_percentage", max_pct))

    def platform(self, name: str) -> Sentinel2Query:
        """Filter by platform name, e.g. `"sentinel-2a"` or `"sentinel-2b"`."""
        return self.with_filter(CQL2.eq("platform", name))

    def orbit_state(self, state: str) -> Sentinel2Query:
        """Filter by orbit state, e.g. `"ascending"` or `"descending"`."""
        state = state.lower()
        return self.with_filter(CQL2.eq("sat:orbit_state", state))

    def processing_baseline(self, baseline: str) -> Sentinel2Query:
        """Filter by `s2:processing_baseline`, e.g. `"05.09"`."""
        return self.with_filter(CQL2.eq("s2:processing_baseline", baseline))
