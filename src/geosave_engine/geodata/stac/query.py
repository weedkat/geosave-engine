from typing import Any
import dataclasses
from dataclasses import dataclass
from datetime import datetime as dt
from typing import TypeVar

from cql2 import Expr

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
        filter: CQL2-JSON filter expression; build with ``with_filter`` from CQL2 text.
        fields: Item fields to include or exclude.
        sortby: Sort order — string, dict, or list of ``{"field": ..., "direction": ...}``.
    """

    collections: list[str]
    ids: list[str] | None = None
    bbox: tuple[float, float, float, float] | None = None
    intersects: dict[str, Any] | None = None
    datetime: dt | str | tuple[dt, dt] | None = None
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

    def with_filter(self: T, expr: str, inplace: bool = False) -> T:
        """Merge CQL2 text filter expression into existing filter.

        Wraps both expressions in ``and`` if filter already exists.

        Args:
            expr: CQL2 text filter expression.
            inplace: Mutate self and return it. Default leaves self untouched,
                returns a copy with the merged filter instead.
        
        Example:
            >>> query = StacQuery(collections=["sentinel-2-l2a"])
            >>> query = query.with_filter("eo:cloud_cover <= 10")
        """
        parsed = Expr(expr).to_json()

        if inplace:
            self.filter = parsed
            return self
        return dataclasses.replace(self, filter=parsed)

    def with_sortby(self: T, field: str, direction: str = "asc", inplace: bool = False) -> T:
        """Append a sort field to existing sort order.

        Args:
            field: STAC item property name to sort by.
            direction: ``'asc'`` or ``'desc'``.
            inplace: Mutate self and return it. Default leaves self untouched,
                returns a copy with the appended sort field instead.
        """
        new_sort = {"field": field, "direction": direction}
        if self.sortby is None:
            new_sortby = [new_sort]
        elif isinstance(self.sortby, list):
            new_sortby = [*self.sortby, new_sort]
        elif isinstance(self.sortby, dict):
            new_sortby = [self.sortby, new_sort]
        else:
            new_sortby = [parse_sortby(self.sortby), new_sort]

        if inplace:
            self.sortby = new_sortby
            return self
        return dataclasses.replace(self, sortby=new_sortby)

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