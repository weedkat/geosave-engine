from typing import Any, Literal, TypedDict
from dataclasses import dataclass
from datetime import datetime

from geosave_engine.utils.crs import validate_bbox


class SortEntry(TypedDict):
    """One STAC sortby entry, e.g. {"field": "datetime", "direction": "desc"}."""

    field: str
    direction: Literal["asc", "desc"]


class FilterEntry(TypedDict):
    """One CQL2-JSON expression node: {"op": ..., "args": [...]}.

    `args`' shape depends on `op`:
        comparisons (`lt`/`lte`/`gt`/`gte`/`eq`/`neq`/`like`/`in`):
            `[{"property": name}, value]`
        `and`/`or`: two or more nested `FilterEntry`
        `not`: exactly one nested `FilterEntry`

    Examples:
        >>> FilterEntry(op="<", args=[{"property": "eo:cloud_cover"}, 20])
        >>> FilterEntry(op="and", args=[
        ...     {"op": "<", "args": [{"property": "eo:cloud_cover"}, 10]},
        ...     {"op": "=", "args": [{"property": "s2:nodata_pixel_percentage"}, 0]},
        ... ])
    """

    op: str
    args: list[Any]


@dataclass
class StacQuery:
    """STAC search parameters for a single catalog query.

    Args:
        collections: STAC collection IDs to search.
        ids: Filter by specific item IDs.
        bbox: WGS84 bounding box ``(min_lon, min_lat, max_lon, max_lat)``.
        intersects: GeoJSON geometry to spatially intersect.
        datetime: Single datetime, ISO string, or ``(start, end)`` tuple.
        max_items: Client-side max items to return.
        limit: Server-side page size hint.
        query: Legacy STAC query extension filters.
        filter: CQL2-JSON filter expression — see `FilterEntry`. Passed
            straight to pystac-client.
        fields: Item fields to include or exclude.
        sortby: Sort order — list of ``{"field": ..., "direction": "asc"|"desc"}``.
    """

    collections: list[str]
    ids: list[str] | None = None
    bbox: tuple[float, float, float, float] | None = None
    intersects: dict[str, Any] | None = None
    datetime: datetime | str | tuple[datetime, datetime] | None = None
    max_items: int | None = None
    limit: int | None = None
    query: dict[str, Any] | None = None
    filter: FilterEntry | None = None
    fields: list[str] | None = None
    sortby: list[SortEntry] | None = None

    def __post_init__(self):
        """Validate bbox is valid WGS84."""
        validate_bbox(self.bbox)

    def to_search_params(self) -> dict[str, Any]:
        """Build pystac-client search kwargs. Strips ``None`` values.

        Returns:
            Dict of search params ready to pass to ``Client.search(**params)``.
        """
        params = {
            "collections": self.collections,
            "ids": self.ids,
            "bbox": self.bbox,
            "intersects": self.intersects,
            "datetime": self.datetime,
            "max_items": self.max_items,
            "limit": self.limit,
            "query": self.query,
            "filter": self.filter,
            "filter_lang": "cql2-json" if self.filter is not None else None,
            "fields": self.fields,
            "sortby": self.sortby,
        }
        return {k: v for k, v in params.items() if v is not None}
