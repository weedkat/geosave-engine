"""Primitive per-run STAC query and raster loading parameters."""

from typing import Any, Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    PositiveInt,
    TypeAdapter,
    field_validator,
    model_validator,
)

from geosave_engine.geodata.stac.query import StacQuery
from geosave_engine.geodata.stac.source import StacSourceConfig

from .base import ConfigModel


class SortConfig(ConfigModel):
    """One STAC sort field and direction."""

    field: str
    direction: Literal["asc", "desc"]


class QueryConfig(ConfigModel):
    """Deployment-safe fields accepted by one native STAC query."""

    bbox: tuple[float, float, float, float] | None = None
    intersects: dict[str, JsonValue] | None = None
    datetime: str | tuple[str, str] | None = None
    filter: dict[str, JsonValue] | None = None
    max_items: PositiveInt | None = None
    limit: PositiveInt | None = None
    ids: tuple[str, ...] | None = None
    sortby: tuple[SortConfig, ...] | None = None

    def to_query(self, collection: str) -> StacQuery:
        """Build the native query with model-owned collection identity."""
        values = self.model_dump(exclude_none=True)
        if "sortby" in values:
            values["sortby"] = [dict(item) for item in values["sortby"]]
        return StacQuery(collections=[collection], **values)

    @model_validator(mode="after")
    def validate_native_query(self) -> Self:
        """Apply native bounding-box, limit, and identifier validation."""
        self.to_query("validation")
        return self


class SourceConfig(ConfigModel):
    """Query and pixel-loading parameters for one named model source."""

    query: QueryConfig = Field(default_factory=QueryConfig)
    load: StacSourceConfig = Field(default_factory=StacSourceConfig)

    @field_validator("load", mode="before")
    @classmethod
    def validate_primitive_load(cls, value: Any) -> Any:
        """Reject callables and non-finite values in deployment input."""
        if isinstance(value, StacSourceConfig):
            return value
        return TypeAdapter(
            dict[str, JsonValue], config=ConfigDict(allow_inf_nan=False)
        ).validate_python(value, strict=True)
