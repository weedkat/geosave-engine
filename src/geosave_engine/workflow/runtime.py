"""Decode primitive flow settings into native geodata objects inside workers."""

from collections.abc import Mapping
from pathlib import PurePath
from typing import Any

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError
from pystac_client import Client

from geosave_engine.geodata.core.anchor import GeoAnchor
from geosave_engine.geodata.stac.client import StacClient
from geosave_engine.geodata.stac.query import StacQuery
from geosave_engine.geodata.stac.source import StacSource, StacSourceConfig
from geosave_engine.geodata.utils import io


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
    if kind == "coordinates":
        return GeoAnchor.from_coordinates(**config)
    path = config.pop("path")
    if not isinstance(path, str) or PurePath(path).suffix.lower() not in (
        ".geojson",
        ".json",
    ):
        raise ValueError("GeoJSON anchor path must name a .geojson or .json file")
    return GeoAnchor.from_geometry(io.read_vector(path).footprint, **config)


def open_sources(settings: Mapping[str, Any]) -> dict[str, StacSource]:
    """Open native STAC sources from primitive settings inside the worker.

    Args:
        settings: Source names mapped to url, collection, optional native
            StacQuery fields under query, and StacSourceConfig fields under load.
            Query collections defaults to the outer collection and must agree
            if supplied. Query filter uses native CQL2 JSON.

    Returns:
        Named STAC sources with native query and loader defaults preserved.

    Raises:
        ValueError: Fields or collection bindings are invalid.
        TypeError: Native query or load settings have unsupported values.
    """
    settings = _mapping(settings, label="Sources")
    configured = {}
    for name, supplied in settings.items():
        config = _mapping(supplied, label=f"Source {name!r}")
        _fields(
            config,
            allowed={"url", "collection", "query", "load"},
            required={"url", "collection"},
            label=f"Source {name!r}",
        )
        for field in ("url", "collection"):
            if not isinstance(config[field], str) or not config[field]:
                raise ValueError(f"Source {name!r} {field} must be a nonempty string")
        collection = config["collection"]
        query = _mapping(config.get("query", {}), label=f"Source {name!r} query")
        if query.pop("collections", [collection]) != [collection]:
            raise ValueError(
                f"Source {name!r} query collections must match {collection!r}"
            )
        configured[name] = (
            config["url"],
            collection,
            StacQuery(collections=[collection], **query),
            StacSourceConfig.model_validate(config.get("load", {})),
        )
    sources = {}
    for name, (url, collection, query, load) in configured.items():
        source = StacSource(StacClient(Client.open(url)), collection=collection)
        source.query = query
        source.config = load
        sources[name] = source
    return sources


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
