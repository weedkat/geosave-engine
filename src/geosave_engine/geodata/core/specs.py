from __future__ import annotations

from abc import abstractmethod
from datetime import datetime as dt
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .geotile import GeoTile


class IngestSource(BaseModel):
    """Base for all ingest source specs.

    Subclasses declare a ``type`` discriminator literal and implement
    ``to_anchors`` to return ready-to-ingest GeoTile anchors.
    """

    @abstractmethod
    def to_anchors(self, limit: int | None = None) -> list[GeoTile]: ...


class ZarrSource(IngestSource):
    """Ingest from a zarr store or directory of zarr stores.

    Args:
        src: Single ``.zarr`` store path or directory containing ``*.zarr`` stores.
    """

    type: Literal["zarr"] = "zarr"
    src: Path

    def to_anchors(self, limit: int | None = None) -> list[GeoTile]:
        if self.src.suffix == ".zarr":
            paths = [self.src]
        else:
            paths = sorted(self.src.rglob("*.zarr"))
        return [GeoTile.from_zarr(s) for s in paths[:limit]]


class GeotiffSource(IngestSource):
    """Ingest from a GeoTIFF file or directory of GeoTIFFs.

    Args:
        src: Single ``.tif`` / ``.tiff`` file or directory containing them.
    """

    type: Literal["geotiff"] = "geotiff"
    src: Path

    def to_anchors(self, limit: int | None = None) -> list[GeoTile]:
        if self.src.is_dir():
            files = sorted(self.src.rglob("*.tif")) + sorted(self.src.rglob("*.tiff"))
        else:
            files = [self.src]
        return [GeoTile.from_geotiff(f) for f in files[:limit]]


class GeoJSONSource(IngestSource):
    """Ingest one anchor per feature in a GeoJSON file.

    Args:
        src: Path to a GeoJSON FeatureCollection, Feature, or raw geometry.
        datetime: Acquisition datetime applied to all features.
        resolution: Pixel size in meters. Overrides pipeline default if set.
        crs: Target projected CRS. Defaults to local UTM/UPS per feature.
    """

    type: Literal["geojson"] = "geojson"
    src: Path
    datetime: str | dt
    resolution: float = 10.0
    crs: str | None = None

    def to_anchors(self, limit: int | None = None) -> list[GeoTile]:
        return GeoTile.from_geojson(self.src, datetime=self.datetime, resolution=self.resolution, crs=self.crs)[:limit]


class CoordinateSource(IngestSource):
    """Ingest one anchor centered on a WGS84 coordinate.

    Args:
        lat: Center latitude in WGS84 degrees.
        lon: Center longitude in WGS84 degrees.
        datetime: Acquisition datetime.
        size_m: Tile size in meters (square).
        resolution: Pixel size in meters. Overrides pipeline default if set.
        crs: Target projected CRS. Defaults to local UTM/UPS.
    """

    type: Literal["coordinate"] = "coordinate"
    lat: float
    lon: float
    datetime: str | dt
    size_m: float
    resolution: float = 10.0
    crs: str | None = None

    def to_anchors(self, limit: int | None = None) -> list[GeoTile]:  # noqa: ARG002
        return [GeoTile.from_coordinate(
            self.lat, self.lon,
            datetime=self.datetime,
            size_m=self.size_m,
            resolution=self.resolution,
            crs=self.crs,
        )]


class PolygonSource(IngestSource):
    """Ingest one anchor from a GeoJSON polygon geometry dict.

    Args:
        geom: GeoJSON geometry dict with WGS84 lon/lat coordinates.
        datetime: Acquisition datetime.
        resolution: Pixel size in meters. Overrides pipeline default if set.
        crs: Target projected CRS. Defaults to local UTM/UPS.
    """

    type: Literal["polygon"] = "polygon"
    geom: dict[str, Any]
    datetime: str | dt
    resolution: float = 10.0
    crs: str | None = None

    def to_anchors(self, limit: int | None = None) -> list[GeoTile]:  # noqa: ARG002
        return [GeoTile.from_polygon(
            self.geom,
            datetime=self.datetime,
            resolution=self.resolution,
            crs=self.crs,
        )]


AnyIngestSource = Annotated[
    ZarrSource | GeotiffSource | GeoJSONSource | CoordinateSource | PolygonSource,
    Field(discriminator="type"),
]
