from typing import Any
from dataclasses import dataclass, replace, field
from datetime import datetime
from typing import Protocol, runtime_checkable, TypeVar

from geosave_engine.utils.cql2 import CQL2
from geosave_engine.utils.geom import validate_bbox

T = TypeVar("T", bound="StacQuery")


@runtime_checkable
class BaseQuery(Protocol):
    def to_search_params(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class StacQuery:
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

    def __post_init__(self):
        """Strict validation for WGS84 logic."""
        validate_bbox(self.bbox)

    def to_search_params(self) -> dict[str, Any]:
        """Convert the query dataclass to a dictionary of search parameters."""
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
        # Remove None values to avoid passing them to the STAC client
        return {k: v for k, v in params.items() if v is not None}

    def with_filter(self: T, expr: dict[str, Any]) -> T:
        """Chainable filter merger that preserves existing filters."""
        if self.filter is None:
            new_filter = expr
        else:
            new_filter = CQL2.and_(self.filter, expr)
            
        return replace(self, filter=new_filter)


@dataclass(frozen=True)
class Sentinel2Query(StacQuery):
    def max_cloud_cover(self: T, max_cover: float) -> T:
        """Add a cloud cover filter to the query."""
        cloud_filter = CQL2.lte("eo:cloud_cover", max_cover)
        return self.with_filter(cloud_filter)

@dataclass(frozen=True)
class Sentinel2L2AQuery(Sentinel2Query):
    collections: list[str] = field(default_factory=lambda: ["sentinel-2-l2a"])


@dataclass(frozen=True)
class Sentinel2L1CQuery(Sentinel2Query):
    collections: list[str] = field(default_factory=lambda: ["sentinel-2-l1c"])