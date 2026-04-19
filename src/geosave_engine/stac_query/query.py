from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from geosave_engine.utils.cql2 import CQL2


@dataclass(frozen=True)
class StacQuery:
    """A typed STAC Item Search request.

    Fields map directly to `pystac_client.Client.search()` parameters.
    Serialize with `to_search_params()` and unpack into the client::

        client.search(**query.to_search_params())

    datetime accepts any form that pystac_client handles natively: an RFC 3339
    string, a simple date string, a "/" separated range, a `datetime.datetime`
    instance, or a `(start, end)` tuple.

    filter holds a CQL2-JSON dict built via `CQL2` helpers from
    `geosave_engine.utils.cql2`.  `with_filter` accumulates conditions with AND;
    each call returns a new frozen instance.  `filter_lang="cql2-json"` is
    injected automatically when filter is set.

    fields limits which item properties are returned, keeping payloads small
    (e.g. `["id", "properties.datetime", "properties.eo:cloud_cover"]`).
    """

    collections: list[str]
    ids: list[str] | None = None
    bbox: tuple[float, float, float, float] | None = None
    intersects: dict[str, Any] | None = None
    datetime: datetime | str | tuple[datetime, datetime] | None = None
    max_items: int | None = None
    limit: int | None = None
    filter: dict[str, Any] | None = None
    fields: list[str] | None = None
    sortby: str | list[str] | None = None

    def with_filter(self, expr: dict[str, Any]) -> StacQuery:
        """Return a copy with `expr` AND-ed into the current filter."""
        merged = CQL2.and_(self.filter, expr) if self.filter is not None else expr
        return dataclasses.replace(self, filter=merged)

    def to_search_params(self) -> dict[str, Any]:
        """Serialize to keyword arguments for `pystac_client.Client.search()`."""
        params: dict[str, Any] = {"collections": self.collections}
        if self.ids is not None:
            params["ids"] = self.ids
        if self.bbox is not None:
            params["bbox"] = list(self.bbox)
        if self.intersects is not None:
            params["intersects"] = self.intersects
        if self.datetime is not None:
            params["datetime"] = self.datetime
        if self.max_items is not None:
            params["max_items"] = self.max_items
        if self.limit is not None:
            params["limit"] = self.limit
        if self.filter is not None:
            params["filter"] = self.filter
            params["filter_lang"] = "cql2-json"
        if self.fields is not None:
            params["fields"] = self.fields
        if self.sortby is not None:
            params["sortby"] = self.sortby
        return params


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
