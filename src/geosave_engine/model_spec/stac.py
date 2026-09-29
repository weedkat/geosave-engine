"""Model-owned STAC search and raster loading recipe."""

from __future__ import annotations

from functools import cache
from typing import Any, Annotated, Literal, Self

from pydantic import (
    ConfigDict,
    Field,
    HttpUrl,
    JsonValue,
    PositiveInt,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)
from pystac_client.exceptions import APIError
from requests.exceptions import RequestException
import xarray as xr

from geosave_engine.geodata.core import GeoAnchor
from geosave_engine.geodata.stac.client import StacClient, StacProvider
from geosave_engine.geodata.stac.query import StacQuery
from geosave_engine.geodata.stac.source import StacSourceConfig

from .base import SpecModel, Text


class SortConfig(SpecModel):
    """Declare one STAC sort field and direction."""

    field: Text
    direction: Literal["asc", "desc"]


class QueryConfig(SpecModel):
    """Declare the STAC item search owned by a model raster."""

    bbox: tuple[float, float, float, float] | None = None
    intersects: dict[str, JsonValue] | None = None
    datetime: str | tuple[str, str] | None = None
    filter: dict[str, JsonValue] | None = None
    max_items: PositiveInt | None = None
    limit: PositiveInt | None = None
    ids: tuple[Text, ...] | None = None
    sortby: tuple[SortConfig, ...] | None = None

    def to_query(self, collection: str) -> StacQuery:
        """Build the native query with its collection identity."""
        values = self.model_dump(exclude_none=True)
        if "sortby" in values:
            values["sortby"] = [dict(item) for item in values["sortby"]]
        return StacQuery(collections=[collection], **values)

    @model_validator(mode="after")
    def _validate_query(self) -> Self:
        if self.bbox is not None and self.intersects is not None:
            raise ValueError("bbox and intersects are mutually exclusive")
        self.to_query("validation")
        return self


class StacRecipe(SpecModel):
    """Describe how to find and load one raster from STAC.

    Args:
        collection: STAC collection identifier.
        endpoints: Built-in provider names or HTTP(S) catalog endpoints in
            fallback order.
        query: Item search parameters.
        load: Native raster loading parameters.
    """

    collection: Text
    endpoints: Annotated[tuple[StacProvider | HttpUrl, ...], Field(min_length=1)]
    query: QueryConfig = Field(default_factory=QueryConfig)
    load: StacSourceConfig = Field(default_factory=StacSourceConfig)

    @field_serializer("endpoints")
    def _serialize_endpoints(self, endpoints: tuple[HttpUrl, ...]) -> tuple[str, ...]:
        return tuple(str(url) for url in endpoints)

    @field_validator("load", mode="before")
    @classmethod
    def _validate_primitive_load(cls, value: Any) -> Any:
        if isinstance(value, StacSourceConfig):
            value = value.model_dump()

        def lists(item: Any) -> Any:
            if isinstance(item, tuple):
                return [lists(member) for member in item]
            if isinstance(item, dict):
                return {key: lists(member) for key, member in item.items()}
            return item

        return TypeAdapter(
            dict[str, JsonValue], config=ConfigDict(allow_inf_nan=False)
        ).validate_python(lists(value), strict=True)

    @field_serializer("load")
    def _serialize_load(self, load: StacSourceConfig) -> dict[str, JsonValue]:
        """Serialize native load settings as portable primitive values."""
        return load.model_dump(mode="json")

    @model_validator(mode="after")
    def _validate_endpoints(self) -> Self:
        endpoints = tuple(map(str, self.endpoints))
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("endpoints must not contain duplicate URLs")
        return self

    def load_raster(self, anchor: GeoAnchor, /) -> xr.Dataset:
        """Load a lazy raster on an anchor from the first available endpoint."""
        endpoints = tuple(str(endpoint) for endpoint in self.endpoints)
        client = _open_client(self.collection, endpoints)
        source = client.source(self.collection)
        source.query = self.query.to_query(self.collection)
        source.config = self.load
        return source.load(anchor)


def _endpoint_unavailable(error: Exception) -> bool:
    """Return whether an endpoint failure is safe to retry elsewhere."""
    if isinstance(error, APIError):
        status = getattr(error, "status_code", None)
        if status is not None:
            return status == 404 or status >= 500
        cause = error.__cause__ or error.__context__
        seen = {id(error)}
        while cause is not None and id(cause) not in seen:
            if isinstance(cause, RequestException) and not isinstance(
                cause, ValueError
            ):
                return True
            seen.add(id(cause))
            cause = cause.__cause__ or cause.__context__
        return False
    return isinstance(error, RequestException) and not isinstance(error, ValueError)


@cache
def _open_client(collection: str, endpoints: tuple[str, ...]) -> StacClient:
    """Open and cache the first catalog publishing a collection."""
    failures = []
    last_error: Exception | None = None
    missing = f"collection {collection!r} not found on this STAC endpoint"
    for endpoint in endpoints:
        try:
            client = StacClient.open(endpoint)
            try:
                metadata = client.collection(collection)
            except ValueError as error:
                if missing not in str(error):
                    raise
                metadata = None
        except (RequestException, APIError) as error:
            if not _endpoint_unavailable(error):
                raise
            last_error = error
        else:
            if metadata is not None and metadata.id == collection:
                return client
            last_error = LookupError(f"Collection {collection!r} not found")
        status = getattr(last_error, "status_code", None)
        cause = f"HTTP {status}: {last_error}" if status is not None else str(last_error)
        failures.append(f"{endpoint}: {cause}")
    raise ConnectionError(
        "STAC endpoints unavailable: " + "; ".join(failures)
    ) from last_error
