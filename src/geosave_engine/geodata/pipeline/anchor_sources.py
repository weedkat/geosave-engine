from __future__ import annotations

from abc import abstractmethod
from itertools import chain, islice
from pathlib import Path
from typing import Annotated, Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from geosave_engine.geodata.tile import AnchorDatetime, GeoAnchor, GeoTile
from geosave_engine.utils import chunk_geotile


class AnchorSource(BaseModel):
    """Base for all anchor source specs.

    Subclasses declare a ``type`` discriminator literal and implement
    ``_iter_anchors`` to yield ready-to-ingest anchors lazily — one at a
    time, not a pre-built list — with no limit-handling of its own;
    ``to_anchors`` applies ``limit`` once here, via ``itertools.islice``,
    so it's a single, centrally-correct implementation instead of five
    hand-rolled count-trackers that can each drift out of sync (as one
    already did: opening one file past the limit before noticing it should
    stop). ``islice`` on a lazy generator only pulls as many items as asked
    for, so anchors past ``limit`` are never touched either way.

    Callers who need the same anchors more than once (e.g. two passes over
    one directory's worth) materialize with ``list(source.to_anchors())``
    themselves, at the point that actually needs it. `ZarrSource`/
    `GeotiffSource` open real files, so theirs already carry data (`GeoTile`,
    not bare `GeoAnchor`) — `GeoPipeline.fetch` skips real I/O for those.
    """

    def to_anchors(self, limit: int | None = None) -> Iterator[GeoAnchor]:
        return islice(self._iter_anchors(), limit)

    @abstractmethod
    def _iter_anchors(self) -> Iterator[GeoAnchor]: ...


class ZarrSource(AnchorSource):
    """Ingest from a zarr store or directory of zarr stores.

    Args:
        src: Single ``.zarr`` store path or directory containing ``*.zarr`` stores.
        tile_size_m: Split each store into a grid of anchors this size
            (meters, square), instead of one anchor per store. Bounds
            per-anchor memory for a large store — opened lazily, so
            rendering a sub-tile only reads the window it covers, not the
            whole store. Uses each store's own resolution. None keeps one
            anchor per store.
    """

    src: Path
    type: Literal["zarr"] = "zarr"
    tile_size_m: float | None = None

    def _iter_anchors(self) -> Iterator[GeoTile]:
        if self.src.suffix == ".zarr":
            paths = [self.src]
        else:
            paths = sorted(self.src.rglob("*.zarr"))
        for path in paths:
            full = GeoTile.from_zarr(path)
            if self.tile_size_m is None:
                yield full
            else:
                yield from chunk_geotile(full, self.tile_size_m, full.resolution)


class GeotiffSource(AnchorSource):
    """Ingest from a GeoTIFF file or directory of GeoTIFFs.

    Datetime comes from ``GeoTile.from_geotiff``'s own filename-suffix
    convention (``-YYYYMMDD`` or ``-YYYYMMDD-YYYYMMDD``) — this source
    doesn't do any date handling of its own, same as ``ZarrSource`` reading
    a store's own attrs.

    Args:
        src: Single ``.tif`` / ``.tiff`` file or directory containing them.
        tile_size_m: Split each file into a grid of anchors this size
            (meters, square), instead of one anchor per file. Bounds
            per-anchor memory for a large raster — opened lazily, so
            rendering a sub-tile only reads the window it covers, not the
            whole file. Uses each file's own resolution. None keeps one
            anchor per file.
    """

    model_config = ConfigDict(extra="forbid")

    src: Path
    type: Literal["geotiff"] = "geotiff"
    tile_size_m: float | None = None

    def _iter_anchors(self) -> Iterator[GeoTile]:
        if self.src.is_dir():
            files = sorted(self.src.rglob("*.tif")) + sorted(self.src.rglob("*.tiff"))
        else:
            files = [self.src]
        for f in files:
            full = GeoTile.from_geotiff(f)
            if self.tile_size_m is None:
                yield full
            else:
                yield from chunk_geotile(full, self.tile_size_m, full.resolution)


class GeoJSONSource(AnchorSource):
    """Ingest one anchor per feature in a GeoJSON file, or every file in a directory.

    Args:
        src: Path to a GeoJSON FeatureCollection/Feature/raw geometry, or a
            directory containing ``*.geojson``/``*.json`` files (each parsed
            the same way, results concatenated) — same single-file-or-directory
            convention as GeotiffSource/ZarrSource.
        datetime: Acquisition datetime or (start, end) date range applied to all features.
        resolution: Pixel size in meters. Overrides pipeline default if set.
        crs: Target projected CRS. Defaults to local UTM/UPS per feature.
        tile_size_m: Split each feature into a grid of anchors this size (meters,
            square), instead of one anchor per feature. Bounds per-anchor memory
            for large features. None keeps one anchor per feature.
    """

    src: Path
    type: Literal["geojson"] = "geojson"
    datetime: AnchorDatetime
    resolution: float = 10.0
    crs: str | None = None
    tile_size_m: float | None = None

    @field_validator("datetime", mode="before")
    @classmethod
    def _coerce_datetime(cls, v: Any) -> Any:
        if isinstance(v, list) and len(v) == 2:
            return tuple(v)
        return v

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
            if self.tile_size_m is None:
                yield full
            else:
                yield from chunk_geotile(full, self.tile_size_m, self.resolution)


class CoordinateSource(AnchorSource):
    """Ingest one anchor centered on a WGS84 coordinate.

    Args:
        lat: Center latitude in WGS84 degrees.
        lon: Center longitude in WGS84 degrees.
        datetime: Acquisition datetime or (start, end) date range.
        size_m: Tile size in meters (square).
        resolution: Pixel size in meters. Overrides pipeline default if set.
        crs: Target projected CRS. Defaults to local UTM/UPS.
    """

    type: Literal["coordinate"] = "coordinate"
    lat: float
    lon: float
    datetime: AnchorDatetime
    size_m: float
    resolution: float = 10.0
    crs: str | None = None

    @field_validator("datetime", mode="before")
    @classmethod
    def _coerce_datetime(cls, v: Any) -> Any:
        if isinstance(v, list) and len(v) == 2:
            return tuple(v)
        return v

    def _iter_anchors(self) -> Iterator[GeoAnchor]:
        yield GeoAnchor.from_coordinate(
            self.lat,
            self.lon,
            datetime=self.datetime,
            size_m=self.size_m,
            resolution=self.resolution,
            crs=self.crs,
        )


class PolygonSource(AnchorSource):
    """Ingest one anchor from a GeoJSON polygon geometry dict.

    Args:
        geom: GeoJSON geometry dict with WGS84 lon/lat coordinates.
        datetime: Acquisition datetime or (start, end) date range.
        resolution: Pixel size in meters. Overrides pipeline default if set.
        crs: Target projected CRS. Defaults to local UTM/UPS.
        tile_size_m: Split the polygon into a grid of anchors this size (meters,
            square), instead of one anchor covering the whole polygon. Bounds
            per-anchor memory for large AOIs. None keeps one anchor.
    """

    type: Literal["polygon"] = "polygon"
    geom: dict[str, Any]
    datetime: AnchorDatetime
    resolution: float = 10.0
    crs: str | None = None
    tile_size_m: float | None = None

    @field_validator("datetime", mode="before")
    @classmethod
    def _coerce_datetime(cls, v: Any) -> Any:
        if isinstance(v, list) and len(v) == 2:
            return tuple(v)
        return v

    def _iter_anchors(self) -> Iterator[GeoAnchor]:
        full = GeoAnchor.from_polygon(
            self.geom,
            datetime=self.datetime,
            resolution=self.resolution,
            crs=self.crs,
        )
        if self.tile_size_m is None:
            yield full
        else:
            yield from chunk_geotile(full, self.tile_size_m, self.resolution)


AnyAnchorSource = Annotated[
    ZarrSource | GeotiffSource | GeoJSONSource | CoordinateSource | PolygonSource,
    Field(discriminator="type"),
]

_source_adapter: TypeAdapter[AnyAnchorSource] = TypeAdapter(AnyAnchorSource)


def source_from_dict(data: dict) -> AnyAnchorSource:
    """Parse a source config dict into a typed AnyAnchorSource.

    Args:
        data: Dict with a ``"type"`` discriminator key.
            Example: ``{"type": "geotiff", "src": "data/labels/"}``.

    Returns:
        Typed source instance (GeotiffSource, GeoJSONSource, etc.).

    Raises:
        ValidationError: If ``data`` is missing required fields or has unknown type.
    """
    return _source_adapter.validate_python(data)

def geotile_from_dict(data: dict) -> Iterator[GeoAnchor]:
    """Parse a source config dict into its anchors.

    Args:
        data: Dict with a ``"type"`` discriminator key.
            Example: ``{"type": "geotiff", "src": "data/labels/"}``.

    Returns:
        Anchors, lazily — GeoTile for sources that already carry data
        (Zarr/GeoTIFF), bare GeoAnchor otherwise.

    Raises:
        ValidationError: If ``data`` is missing required fields or has unknown type.
    """
    return source_from_dict(data).to_anchors()