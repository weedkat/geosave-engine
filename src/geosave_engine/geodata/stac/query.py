from typing import Any
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

from geosave_engine.geodata.utils.stac_query import CQL2
from geosave_engine.geodata.utils.crs import validate_bbox

T = TypeVar("T", bound="StacQuery")
SortBy = list[dict[str, str]] | dict[str, str] | str

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
        filter: CQL2-JSON filter expression; build with ``CQL2`` helpers.
        fields: Item fields to include or exclude.
        sortby: Sort order — string, dict, or list of ``{"field": ..., "direction": ...}``.
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
    sortby: SortBy | None = None

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

    def with_filter(self: T, expr: dict[str, Any]) -> None:
        """Merge CQL2 filter expression into existing filter.

        Wraps both expressions in ``and`` if filter already exists.

        Args:
            expr: CQL2-JSON filter dict; use ``CQL2`` helpers to build.
                Example: ``{"op": "<=", "args": [{"property": "eo:cloud_cover"}, 0.1]}``.
        """
        if self.filter is None:
            self.filter = expr
        else:
            self.filter = CQL2.and_(self.filter, expr)

    def with_sortby(self: T, field: str, direction: str = "asc") -> None:
        """Append a sort field to existing sort order.

        Args:
            field: STAC item property name to sort by.
            direction: ``'asc'`` or ``'desc'``.
        """
        new_sort = {"field": field, "direction": direction}
        if self.sortby is None:
            self.sortby = [new_sort]
        elif isinstance(self.sortby, list):
            self.sortby.append(new_sort)
        elif isinstance(self.sortby, dict):
            self.sortby = [self.sortby, new_sort]
        elif isinstance(self.sortby, str):
            parsed_sort = parse_sortby(self.sortby)
            self.sortby = [parsed_sort, new_sort]


def parse_sortby(sortby: str) -> dict[str, str]:
    """Convert pystac-client sort string to STAC sort dict."""
    direction = "asc"

    if sortby.startswith("-"):
        direction = "desc"
        field = sortby[1:]
    elif sortby.startswith("+"):
        field = sortby[1:]
    else:
        field = sortby

    if not field:
        raise ValueError("sortby field cannot be empty")

    return {
        "field": field,
        "direction": direction,
    }