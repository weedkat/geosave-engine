"""Turn a plain JSON-safe dict into a GeoAnchor — GeoJSON-like, but for anchors.

No I/O. Each spec mirrors one of GeoAnchor's own from_bbox/from_coordinate/
from_geometry constructors field for field. `spec_from_dict` parses a dict
keyed by a `"type"` discriminator.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, field_validator

from geosave_engine.geodata.spatial.anchor import GeoAnchor
from geosave_engine.geodata.utils.datetime import AnchorDatetime, parse_daterange


class AnchorSpec(BaseModel):
    """Shared fields for an area-of-interest specification.

    Args:
        timespan: Acquisition datetime, date range, or None for a timeless anchor.
    """

    timespan: AnchorDatetime | None = None

    @field_validator("timespan", mode="before")
    @classmethod
    def _coerce_timespan(cls, v: Any) -> Any:
        """Reuse parse_daterange so a malformed date fails here, not three calls later inside to_anchor()."""
        return None if v is None else parse_daterange(v)

    @abstractmethod
    def to_anchor(self) -> GeoAnchor:
        """Build this spec's one GeoAnchor."""
        ...


class BboxSpec(AnchorSpec):
    """Mirrors GeoAnchor.from_bbox — bbox numbers used as-is, no reprojection.

    Args:
        bbox: (minx, miny, maxx, maxy) in `crs`.
        resolution: Pixel size in `crs` units.
        crs: CRS the bbox numbers are given in — same field name and
            default as `from_bbox`'s own `crs`.
    """

    type: Literal["bbox"] = "bbox"
    bbox: tuple[float, float, float, float]
    resolution: float
    crs: str = "EPSG:4326"

    def to_anchor(self) -> GeoAnchor:
        return GeoAnchor.from_bbox(self.bbox, timespan=self.timespan, crs=self.crs, resolution=self.resolution)


class CoordinateSpec(AnchorSpec):
    """Mirrors GeoAnchor.from_coordinate — WGS84 center + size, reprojected to crs or local UTM/UPS.

    Args:
        lat: Center latitude in WGS84 degrees.
        lon: Center longitude in WGS84 degrees.
        size_m: Tile size in meters. Single number = square; (w, h) = rectangle.
        resolution: Pixel size in meters.
        crs: Target projected CRS. None defaults to local UTM/UPS zone.
    """

    type: Literal["coordinate"] = "coordinate"
    lat: float
    lon: float
    size_m: float | tuple[float, float]
    resolution: float = 10.0
    crs: str | None = None

    def to_anchor(self) -> GeoAnchor:
        return GeoAnchor.from_coordinate(
            self.lat,
            self.lon,
            timespan=self.timespan,
            size_m=self.size_m,
            resolution=self.resolution,
            crs=self.crs,
        )


class GeometrySpec(AnchorSpec):
    """Mirrors GeoAnchor.from_geometry — geobox molds to the shape's bbox, reprojected to crs or local UTM/UPS.

    Args:
        geometry: GeoJSON geometry dict, or WKT string, WGS84 lon/lat coordinates.
        resolution: Pixel size in meters.
        crs: Target projected CRS. None defaults to local UTM/UPS zone.
    """

    type: Literal["geometry"] = "geometry"
    geometry: dict[str, Any] | str
    resolution: float = 10.0
    crs: str | None = None

    def to_anchor(self) -> GeoAnchor:
        return GeoAnchor.from_geometry(
            self.geometry, timespan=self.timespan, resolution=self.resolution, crs=self.crs
        )


AnySpec = Annotated[
    BboxSpec | CoordinateSpec | GeometrySpec,
    Field(discriminator="type"),
]

_spec_adapter: TypeAdapter[AnySpec] = TypeAdapter(AnySpec)


def spec_from_dict(data: dict) -> AnySpec:
    """Parse an AOI request dict into its typed AnchorSpec.

    Args:
        data: Dict with a `"type"` discriminator key.
            Example: `{"type": "coordinate", "lat": -6.2, "lon": 106.8,
            "timespan": "2024-01-01", "size_m": 5120}`.

    Returns:
        Typed spec instance (BboxSpec, CoordinateSpec, or GeometrySpec).

    Raises:
        ValidationError: `data` is missing required fields or has an unknown type.
    """
    return _spec_adapter.validate_python(data)
