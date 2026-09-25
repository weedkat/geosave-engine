"""Decode primitive flow settings into native geodata objects inside workers."""

from collections.abc import Mapping
from dataclasses import fields
from pathlib import PurePath
from typing import Any

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError
from pystac_client import Client
from pystac_client.exceptions import APIError
from requests.exceptions import RequestException

from geosave_engine.geodata.core.anchor import GeoAnchor
from geosave_engine.geodata.stac.client import StacClient
from geosave_engine.geodata.stac.query import StacQuery
from geosave_engine.geodata.stac.source import StacSource, StacSourceConfig
from geosave_engine.geodata.utils import io

from .spec import RasterRequirement


def open_anchor(settings: Mapping[str, Any]) -> GeoAnchor:
    """Build an anchor from named point coordinates or a GeoJSON file.

    Args:
        settings: Primitive settings. `type: coordinates` requires named
            latitude, longitude, shape and resolution. `type: geojson` requires
            path and exactly one of resolution or shape. Both accept crs and
            timespan; GeoJSON also accepts pad. Shape is an integer or [height,
            width]. GeoJSON positions use their standard longitude/latitude order.

    Returns:
        Native anchor with the requested spatial and temporal coverage.

    Raises:
        ValueError: The selector, fields or grid settings are invalid.
        OSError: The GeoJSON file cannot be read.
    """
    # Check the fields required by this kind of region.
    config = _mapping(settings, label="Anchor")
    kind = config.pop("type", None)
    common = {"shape", "resolution", "crs", "timespan"}
    if kind == "coordinates":
        _fields(
            config,
            allowed=common | {"latitude", "longitude"},
            required={"latitude", "longitude", "shape", "resolution"},
            label="Coordinate anchor",
        )
    elif kind == "geojson":
        _fields(
            config,
            allowed=common | {"path", "pad"},
            required={"path"},
            label="GeoJSON anchor",
        )
        if (config.get("resolution") is None) == (config.get("shape") is None):
            raise ValueError(
                "GeoJSON anchor requires exactly one of resolution or shape"
            )
    else:
        raise ValueError("Anchor type must be 'coordinates' or 'geojson'")
    _normalize_grid(config)

    # Build the requested spatial and temporal region.
    if kind == "coordinates":
        return GeoAnchor.from_coordinates(**config)
    path = config.pop("path")
    if not isinstance(path, str) or PurePath(path).suffix.lower() not in (
        ".geojson",
        ".json",
    ):
        raise ValueError("GeoJSON anchor path must name a .geojson or .json file")
    return GeoAnchor.from_geometry(io.read_vector(path).footprint, **config)


def open_sources(
    settings: Mapping[str, Any], requirements: Mapping[str, RasterRequirement]
) -> dict[str, StacSource]:
    """Open native STAC sources from primitive settings inside the worker.

    Args:
        settings: Source names mapped to optional native StacQuery fields under
            query and StacSourceConfig fields under load. Collection identity
            belongs to requirements. Query filter uses native CQL2 JSON.
        requirements: Matching model sources with collection and ordered endpoints.

    Returns:
        Named STAC sources with native query and loader defaults preserved.

    Raises:
        ValueError: Fields or collection bindings are invalid.
        TypeError: Native query or load settings have unsupported values.
        ConnectionError: Every endpoint is unavailable or lacks the collection.

    Only transport failures, HTTP 404/5xx and missing collections try the next
    endpoint. Authentication, document and configuration errors propagate.
    Search and lazy asset reads happen later, without endpoint failover.
    """
    settings = _mapping(settings, label="Sources")
    if missing := requirements.keys() - settings.keys():
        raise ValueError(f"Source bindings are missing: {sorted(missing)}")
    if extra := settings.keys() - requirements.keys():
        raise ValueError(f"Unknown source bindings: {sorted(extra)}")
    # Validate every source before opening any remote catalogue.
    configured = {
        name: _source_settings(name, settings[name], requirement)
        for name, requirement in requirements.items()
    }

    # Return ready-to-use STAC sources with their query and load settings.
    sources = {}
    for name, (requirement, query, load) in configured.items():
        source = StacSource(
            _open_client(requirement), collection=requirement.collection
        )
        source.query = query
        source.config = load
        sources[name] = source
    return sources


def _normalize_grid(config: dict[str, Any]) -> None:
    """Validate shape and dates, replacing YAML lists with native tuples."""
    # Accept a square size or an explicit height and width.
    if config.get("shape") is not None:
        shape = config["shape"]
        dimensions = shape if isinstance(shape, list) else [shape, shape]
        if len(dimensions) != 2 or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in dimensions
        ):
            raise ValueError(
                "Anchor shape must be a positive integer or [height, width]"
            )
        if isinstance(shape, list):
            config["shape"] = tuple(shape)
    # Accept one date or a start/end date range.
    if config.get("timespan") is not None:
        timespan = config["timespan"]
        if (
            isinstance(timespan, list)
            and len(timespan) == 2
            and all(isinstance(value, str) for value in timespan)
        ):
            config["timespan"] = tuple(timespan)
        elif not isinstance(timespan, str):
            raise ValueError("Anchor timespan must be a date string or [start, end]")


def _source_settings(
    name: str, supplied: Mapping[str, Any], requirement: RasterRequirement
) -> tuple[RasterRequirement, StacQuery, StacSourceConfig]:
    """Validate one model binding and its primitive query/load configuration."""
    requirement = RasterRequirement.model_validate(requirement)
    if requirement.collection is None or requirement.endpoints is None:
        raise ValueError(f"Source {name!r} needs collection and endpoints to acquire")
    config = _mapping(supplied, label=f"Source {name!r}")
    _fields(
        config,
        allowed={"query", "load"},
        required=set(),
        label=f"Source {name!r}",
    )
    query = _mapping(config.get("query", {}), label=f"Source {name!r} query")
    load = _mapping(config.get("load", {}), label=f"Source {name!r} load")
    _fields(
        query,
        allowed={field.name for field in fields(StacQuery)} - {"collections"},
        required=set(),
        label=f"Source {name!r} query",
    )
    return (
        requirement,
        TypeAdapter(StacQuery).validate_python(
            {"collections": [requirement.collection], **query}
        ),
        StacSourceConfig.model_validate(load),
    )


def _open_client(requirement: RasterRequirement) -> StacClient:
    """Probe endpoint roots and the required collection in declared order."""
    failures = []
    last_error = None
    for endpoint in requirement.endpoints:
        url = str(endpoint)
        try:
            client = Client.open(url)
            try:
                collection = client.get_collection(requirement.collection)
            except KeyError as error:
                # PySTAC uses this specific KeyError for a missing static child;
                # parser KeyErrors for missing document fields must propagate.
                if error.args != (
                    f"Collection {requirement.collection} not found on catalog",
                ):
                    raise
                collection = None
        except (RequestException, APIError) as error:
            if not _unavailable(error):
                raise
            last_error = error
        else:
            if collection is not None:
                return StacClient(client)
            last_error = LookupError(f"Collection {requirement.collection!r} not found")
        status = getattr(last_error, "status_code", None)
        cause = (
            f"HTTP {status}: {last_error}" if status is not None else str(last_error)
        )
        failures.append(f"{url}: {cause}")
    raise ConnectionError(
        "STAC endpoints unavailable: " + "; ".join(failures)
    ) from last_error


def _unavailable(error: Exception) -> bool:
    """Recognize availability errors without masking malformed documents."""
    if isinstance(error, APIError):
        status = getattr(error, "status_code", None)
        if status is not None:
            return status == 404 or status >= 500
        # StacApiIO wraps transport errors as status-less APIError exceptions.
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


def _mapping(settings: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        return TypeAdapter(
            dict[str, JsonValue], config=ConfigDict(allow_inf_nan=False)
        ).validate_python(settings, strict=True)
    except ValidationError as error:
        raise ValueError(
            f"{label} settings must map string keys to finite primitive values"
        ) from error


def _fields(
    settings: Mapping[str, Any], *, allowed: set[str], required: set[str], label: str
) -> None:
    if extra := settings.keys() - allowed:
        raise ValueError(f"{label} has unknown fields: {sorted(extra)}")
    if missing := required - settings.keys():
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")
