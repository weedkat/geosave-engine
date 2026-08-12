from __future__ import annotations

from abc import abstractmethod
from itertools import chain, islice
from pathlib import Path
from typing import Annotated, Any, Iterator, Literal

from pydantic import BaseModel, Field, TypeAdapter, field_validator

from geosave_engine.geodata.spatial import AnchorDatetime, GeoAnchor
from geosave_engine.geodata.utils import chunk_geotile


class AnchorSource(BaseModel):
    """Base for all anchor source specs.

    Subclasses declare a ``type`` discriminator literal and implement
    ``_iter_anchors`` to yield ready-to-ingest anchors lazily, one at a
    time. Callers who need the same anchors more than once (e.g. two
    passes over one directory's worth) materialize with
    ``list(source.to_anchors())`` themselves, at the point that actually
    needs it.

    Args:
        datetime: Acquisition datetime or (start, end) date range.
        resolution: Pixel size in meters.
        crs: Target projected CRS for the anchor's raster grid. Defaults to
            each anchor's own local UTM/UPS zone — set this to force every
            anchor onto one shared grid instead (e.g. anchors from separate
            requests that need to align/mosaic together, or an AOI wide
            enough to span more than one UTM zone).
        tile_size_px: Grid the source's own area into anchors this many
            pixels per side (square). Keeps every yielded anchor at a
            consistent, model-input-sized footprint regardless of the
            source's own extent or resolution.
    """

    datetime: AnchorDatetime
    resolution: float = 10.0
    crs: str | None = None
    tile_size_px: int = 500

    @field_validator("datetime", mode="before")
    @classmethod
    def _coerce_datetime(cls, v: Any) -> Any:
        if isinstance(v, list) and len(v) == 2:
            return tuple(v)
        return v

    def to_anchors(self, limit: int | None = None) -> Iterator[GeoAnchor]:
        # Applied once here via islice, not per-subclass, so no subclass
        # needs its own count-tracker that can drift out of sync.
        return islice(self._iter_anchors(), limit)

    @abstractmethod
    def _iter_anchors(self) -> Iterator[GeoAnchor]: ...


class GeoJSONSource(AnchorSource):
    """Ingest anchors from every feature in a GeoJSON file, or every file in a directory.

    Args:
        src: Path to a GeoJSON FeatureCollection/Feature/raw geometry, or a
            directory containing ``*.geojson``/``*.json`` files (each parsed
            the same way, results concatenated).
    """

    src: Path
    type: Literal["geojson"] = "geojson"

    def _iter_anchors(self) -> Iterator[GeoAnchor]:
        if self.src.is_dir():
            files = sorted(self.src.rglob("*.geojson")) + sorted(self.src.rglob("*.json"))
        else:
            files = [self.src]
        # chain.from_iterable stitches each file's own lazy from_geojson()
        # generator into one stream, instead of a manual nested for loop.
        per_file = (GeoAnchor.from_geojson(f, datetime=self.datetime, resolution=self.resolution, crs=self.crs)
                    for f in files)
        for full in chain.from_iterable(per_file):
            yield from chunk_geotile(full, self.tile_size_px)


class CoordinateSource(AnchorSource):
    """Ingest anchors covering a WGS84 coordinate's surrounding area.

    Args:
        lat: Center latitude in WGS84 degrees.
        lon: Center longitude in WGS84 degrees.
        area_m: Total AOI extent around the point, in meters (square).
    """

    type: Literal["coordinate"] = "coordinate"
    lat: float
    lon: float
    area_m: float

    def _iter_anchors(self) -> Iterator[GeoAnchor]:
        full = GeoAnchor.from_coordinate(
            self.lat,
            self.lon,
            datetime=self.datetime,
            size_m=self.area_m,
            resolution=self.resolution,
            crs=self.crs,
        )
        yield from chunk_geotile(full, self.tile_size_px)


class PolygonSource(AnchorSource):
    """Ingest anchors covering a GeoJSON polygon geometry dict.

    Args:
        geom: GeoJSON geometry dict with WGS84 lon/lat coordinates.
    """

    type: Literal["polygon"] = "polygon"
    geom: dict[str, Any]

    def _iter_anchors(self) -> Iterator[GeoAnchor]:
        full = GeoAnchor.from_polygon(
            self.geom,
            datetime=self.datetime,
            resolution=self.resolution,
            crs=self.crs,
        )
        yield from chunk_geotile(full, self.tile_size_px)


AnyAnchorSource = Annotated[
    GeoJSONSource | CoordinateSource | PolygonSource,
    Field(discriminator="type"),
]

_source_adapter: TypeAdapter[AnyAnchorSource] = TypeAdapter(AnyAnchorSource)


def source_from_dict(data: dict) -> AnyAnchorSource:
    """Parse a source config dict into a typed AnyAnchorSource.

    Meant for turning an inbound request (e.g. an API call's JSON body)
    into an anchor source — the AOI shapes (coordinate/polygon/geojson)
    a caller would actually supply at request time.

    Args:
        data: Dict with a ``"type"`` discriminator key.
            Example: ``{"type": "coordinate", "lat": -6.2, "lon": 106.8,
            "datetime": "2024-01-01", "area_m": 5120}``.

    Returns:
        Typed source instance (CoordinateSource, PolygonSource, or GeoJSONSource).

    Raises:
        ValidationError: If ``data`` is missing required fields or has unknown type.
    """
    return _source_adapter.validate_python(data)

def anchor_from_dict(data: dict) -> Iterator[GeoAnchor]:
    """Parse a source config dict into its anchors.

    Args:
        data: Dict with a ``"type"`` discriminator key.
            Example: ``{"type": "coordinate", "lat": -6.2, "lon": 106.8,
            "datetime": "2024-01-01", "area_m": 5120}``.

    Returns:
        GeoAnchors, lazily.

    Raises:
        ValidationError: If ``data`` is missing required fields or has unknown type.
    """
    return source_from_dict(data).to_anchors()
