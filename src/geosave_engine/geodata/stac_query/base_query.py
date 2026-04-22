from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from geosave_engine.utils.cql2 import CQL2

@dataclass(frozen=True)
class BaseStacQuery:
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
    query: dict[str, Any] | None = None
    filter: dict[str, Any] | None = None
    fields: list[str] | None = None
    sortby: str | list[str] | None = None

    def with_filter(self, expr: dict[str, Any]) -> BaseStacQuery:
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
        if self.query is not None:
            params["query"] = self.query
        if self.filter is not None:
            params["filter"] = self.filter
            params["filter_lang"] = "cql2-json"
        if self.fields is not None:
            params["fields"] = self.fields
        if self.sortby is not None:
            params["sortby"] = self.sortby
        return params
