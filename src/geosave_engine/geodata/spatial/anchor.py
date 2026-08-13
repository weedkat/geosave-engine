from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Self, cast

import json
import numpy as np
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from pyproj import CRS, Transformer
from pystac import Item

from geosave_engine.geodata.utils.crs import calculate_crs, validate_coordinate
from geosave_engine.geodata.utils.datetime import AnchorDatetime, DateRange, format_stem_dates, parse_daterange
from geosave_engine.geodata.utils.geodata import np_to_da, validate_da
from geosave_engine.geodata.utils.geolocator import Place
from geosave_engine.utils.colorize import Palette

if TYPE_CHECKING:
    from .tile import GeoTile


class GeoTag(BaseModel):
    """Everything a GeoAnchor/GeoTile carries besides geobox/pixels.

    One validated unit instead of separate `datetime`/`polygon`/rendering-hint
    fields — round-trips through GeoTIFF tags/Zarr attrs as one JSON blob.
    Any other keyword accepted too (`extra="allow"`) and stored as-is —
    user metadata, arbitrary keys, no separate `metadata` dict to nest under.

    Args:
        datetime: Anchor datetime, normalized to an inclusive (start, end)
            range same as `parse_daterange`.
        polygon: Exact AOI footprint, if narrower than the geobox's bbox.
        bands: Band names, in order. GeoTile keeps this synced to its own
            data — not meant to be set by hand. None for a single unnamed band.
        rgb_bands: Which 3 band names count as R/G/B. `plot()` rendering
            hint — otherwise ambiguous from a tile's shape/dtype alone.
        class_map: `{pixel value: class name}` for a categorical tile.
        color_map: `{pixel value: hex or RGB}` for a categorical tile.
        stac: STAC Items this tile's data was sourced from, if any.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    datetime: AnchorDatetime
    polygon: Geometry | None = None
    bands: tuple[str, ...] | None = None
    rgb_bands: tuple[str, str, str] | None = None
    class_map: dict[int, str] | None = None
    color_map: Palette | None = None
    stac: list[Item] = Field(default_factory=list)

    @field_validator("datetime", mode="before")
    @classmethod
    def _parse_datetime(cls, v: Any) -> Any:
        return parse_daterange(v)

    @field_serializer("datetime")
    def _dump_datetime(self, v: DateRange) -> tuple[str, str]:
        start, end = v
        return start.isoformat(timespec="microseconds"), end.isoformat(timespec="microseconds")

    @field_serializer("polygon")
    def _dump_polygon(self, v: Geometry | None) -> dict[str, Any] | None:
        return {"geojson": dict(v.__geo_interface__), "polygon_crs": str(v.crs)} if v is not None else None

    @field_validator("polygon", mode="before")
    @classmethod
    def _load_polygon(cls, v: Any) -> Any:
        return Geometry(v["geojson"], crs=v["polygon_crs"]) if isinstance(v, dict) else v

    @field_serializer("stac")
    def _dump_stac(self, v: list[Item]) -> list[dict[str, Any]]:
        return [item.to_dict() for item in v]

    @field_validator("stac", mode="before")
    @classmethod
    def _load_stac(cls, v: Any) -> Any:
        return [Item.from_dict(item) if isinstance(item, dict) else item for item in v]


def _geobox_to_dict(geobox: GeoBox) -> dict[str, Any]:
    """Plain, JSON-safe geobox dict — shape/affine/crs rebuild it exactly.

    Handles rotation/skew (unlike a bbox+resolution reduction); geobox.crs
    is used directly here, not GeoAnchor.crs's EPSG-only shortcut, since
    that drops non-EPSG CRSs. Also carries a precomputed WGS84 "centroid"
    (lon, lat) — reprojecting a GeoBox is real work (~10ms, a fresh pyproj
    transform every call, measured), so it's done once here on the live
    object, not left for every downstream reader (e.g. a model's hot
    forward loop) to rebuild a GeoBox from the dict and redo it per call.
    """
    affine = geobox.affine
    crs = geobox.crs
    left, bottom, right, top = geobox.geographic_extent.boundingbox.bbox
    return {
        "shape": tuple(geobox.shape),
        "affine": (affine.a, affine.b, affine.c, affine.d, affine.e, affine.f),
        "crs": str(crs) if crs is not None else None,
        "centroid": ((left + right) / 2, (bottom + top) / 2),
    }


def _format_meters(value: float) -> str:
    """Compact size string — cm below 1m, m below 1km, km at or above."""
    if value < 1:
        return f"{int(value * 100)}cm"
    if value < 1000:
        return f"{int(value)}m"
    return f"{value / 1000:g}km"


@dataclass(frozen=True, kw_only=True)
class GeoAnchor:
    """Where + when, no pixel data. A `GeoTile` always has data — a
    data-less reference is a `GeoAnchor`, never a `GeoTile` with `data=None`.

    Datetime always normalizes to an inclusive (start, end) range via
    `parse_daterange` — a reduced-precision string covers its whole stated
    period; an exact instant is a (dt, dt) pair of the same value.

    Examples:
        >>> anchor = GeoAnchor.from_coordinate(52.0, 13.0, datetime="2024-01-15", size_m=5000)
        >>> anchor.stem
        '13.0000E_52.0000N_5kmx5km_20240115_10m'
    """

    geobox: GeoBox
    geotag: GeoTag = field(compare=False)

    def __repr__(self) -> str:
        when = str(self.start) if self.start == self.end else f"{self.start} – {self.end}"
        bbox = tuple(round(v, 2) for v in self.bbox)
        return (
            f"{type(self).__name__}\n"
            f"  bbox:     {bbox}\n"
            f"  crs:      {self.crs!r}\n"
            f"  datetime: {when}\n"
            f"  metadata: {self.metadata}"
        )

    # ------------------------------------------------------------------
    # Tag passthroughs
    # ------------------------------------------------------------------

    @property
    def datetime(self) -> DateRange:
        # GeoTag.datetime is typed AnchorDatetime (constructor accepts a raw
        # string too) but its own validator always resolves it to a (dt, dt)
        # pair before storage — this cast asserts that once, here.
        return cast(DateRange, self.geotag.datetime)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.geotag.model_extra or {}

    @property
    def polygon(self) -> Geometry | None:
        return self.geotag.polygon

    @property
    def rgb_bands(self) -> tuple[str, str, str] | None:
        return self.geotag.rgb_bands

    @property
    def class_map(self) -> dict[int, str] | None:
        return self.geotag.class_map

    @property
    def color_map(self) -> Palette | None:
        return self.geotag.color_map

    @property
    def stac(self) -> list[Item]:
        return self.geotag.stac

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def resolution(self) -> float:
        return self.geobox.affine.a

    @property
    def affine(self) -> Affine:
        return self.geobox.affine

    @property
    def crs(self) -> str | None:
        crs = self.geobox.crs
        if crs is None:
            return None
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else None

    @property
    def width(self) -> int:
        return self.geobox.width

    @property
    def height(self) -> int:
        return self.geobox.height

    @property
    def area_m2(self) -> float:
        """Polygon area if stored, else pixel-grid area. Both in native CRS units² (m² for projected CRS)."""
        if self.polygon is not None:
            return self.polygon.geom.area
        return self.width * self.height * (self.resolution ** 2)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box in native CRS: (minx, miny, maxx, maxy)."""
        return self.geobox.boundingbox.bbox

    @property
    def wgs84_bbox(self) -> tuple[float, float, float, float]:
        """Bounding box in WGS84: (minlon, minlat, maxlon, maxlat)."""
        return self.geobox.geographic_extent.boundingbox.bbox

    @property
    def centroid(self) -> tuple[float, float]:
        """Centroid in WGS84: (lon, lat)."""
        left, bottom, right, top = self.geobox.geographic_extent.boundingbox.bbox
        return ((left + right) / 2, (bottom + top) / 2)

    @property
    def coordinate_str(self) -> str:
        """Human-readable WGS84 centroid, e.g. '45.1549°N, 15.0020°E'."""
        lon, lat = self.centroid
        return f"{abs(lat):.4f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.4f}°{'E' if lon >= 0 else 'W'}"

    @property
    def start(self) -> dt:
        """Range start. For a resolved (non-range) anchor this equals `end`."""
        return self.datetime[0]

    @property
    def end(self) -> dt:
        """Range end. For a resolved (non-range) anchor this equals `start`."""
        return self.datetime[1]

    @property
    def stem(self) -> str:
        """Deterministic filename stem — same anchor always gets the same stem.

        Derived from centroid + extent + datetime range + resolution, not
        list position, so it's stable across runs even if a source's anchor
        list changes shape (files added/removed/reordered) — two anchors at
        the same place/size/time/resolution always produce the same stem.
        Extent is `WxH`, distinguishable on sight from the bare resolution
        value after it — same convention as `1920x1080`, not an EO-specific one.

        Examples:
            >>> anchor.stem
            '13.0000E_52.0000N_5kmx5km_20240115_10m'
        """
        lon, lat = self.centroid
        lon_str = f"{abs(lon):.4f}{'E' if lon >= 0 else 'W'}"
        lat_str = f"{abs(lat):.4f}{'N' if lat >= 0 else 'S'}"
        minx, miny, maxx, maxy = self.bbox
        extent_str = f"{_format_meters(maxx - minx)}x{_format_meters(maxy - miny)}"
        date_token = format_stem_dates((self.start, self.end))
        res_str = _format_meters(self.resolution)
        return f"{lon_str}_{lat_str}_{extent_str}_{date_token}_{res_str}"

    @property
    def bbox_polygon(self) -> Geometry:
        return self.geobox.boundingbox.polygon

    @property
    def geojson(self) -> dict:
        """WGS84 footprint as a GeoJSON Polygon dict."""
        if self.polygon is not None:
            return self.polygon.to_crs("EPSG:4326").geojson()
        return self.bbox_polygon.geojson()

    @property
    def location(self) -> dict[str, Any]:
        """Reverse-geocode centroid. Returns {} on failure."""
        lon, lat = self.centroid
        place = Place.from_coordinate(lat, lon)
        return place.to_dict() if place is not None else {}

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def rebase(
        self,
        *,
        geobox: GeoBox | None = None,
        datetime: AnchorDatetime | None = None,
        polygon: Geometry | None = None,
        rgb_bands: tuple[str, str, str] | None = None,
        class_map: dict[int, str] | None = None,
        color_map: Palette | None = None,
        stac: list[Item] | None = None,
        **metadata: Any,
    ) -> Self:
        """Return new instance rebased onto given geobox/geotag, sharing any data reference.

        Pure — no pixels read/copied, no re-fetch, no re-search. Omitted args
        keep their current value and replace outright, except `**metadata`
        which merges into existing custom keys, overwriting clashing ones.
        `stac` replaces outright too — `GeoTile.rebase` overrides this with
        append-dedupe-by-id instead.

        Args:
            geobox: New geobox.
            datetime: ISO/compact string, or (start, end) pair of either.
            polygon: New footprint polygon.
            rgb_bands: New RGB band names.
            class_map: New `{pixel value: class name}` map.
            color_map: New `{pixel value: hex or RGB}` map.
            stac: New STAC Items, replacing the current list outright.
            **metadata: Custom key-value pairs merged into existing metadata.
        """
        changes: dict[str, Any] = {}
        if geobox is not None:
            changes["geobox"] = geobox
        if datetime is not None or polygon is not None or rgb_bands is not None or class_map is not None \
                or color_map is not None or stac is not None or metadata:
            # Rebuilt via GeoTag's own constructor, not model_copy — model_copy
            # skips validation, and `datetime` needs its string/tuple coercion.
            changes["geotag"] = GeoTag(
                datetime=datetime if datetime is not None else self.geotag.datetime,
                polygon=polygon if polygon is not None else self.geotag.polygon,
                rgb_bands=rgb_bands if rgb_bands is not None else self.geotag.rgb_bands,
                class_map=class_map if class_map is not None else self.geotag.class_map,
                color_map=color_map if color_map is not None else self.geotag.color_map,
                stac=stac if stac is not None else self.geotag.stac,
                **{**self.metadata, **metadata},
            )
        return dataclasses.replace(self, **changes)

    def to_geotile(
        self,
        data: xr.DataArray | np.ndarray,
        names: str | list[str] | None = None,
        times: list[dt] | None = None,
    ) -> "GeoTile":
        """Attach pixel data, turning this anchor into a GeoTile.

        An `xr.DataArray` already carries its own dims/coords and is
        attached as-is (after `validate_da`). A plain array is shaped from
        this anchor's own geobox instead: 2D `(y, x)` is a single unnamed
        band, 3D `(band, y, x)` requires `names`, 4D `(time, band, y, x)`
        requires both `names` and `times`.

        Args:
            data: Pixel data — a DataArray, or a 2-4D array to shape from this anchor's geobox.
            names: Band name(s) for a 3D/4D array — a single string for one
                band, or one name per row for several. Ignored for a DataArray.
            times: Observation datetimes for a 4D array. Ignored otherwise.

        Raises:
            ValueError: `names`/`times` missing or mismatched for the
                array's dimensionality (see `validate_da` for DataArray-specific errors).
        """
        from .tile import GeoTile

        da = data if isinstance(data, xr.DataArray) else np_to_da(self.geobox, data, names, times)
        da = validate_da(da)
        return GeoTile(geobox=self.geobox, data=da, geotag=self.geotag)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain, JSON-safe dict — lossless round-trip via from_dict.

        Returns:
            {
                "geobox": {
                    "shape": (width, height), "affine": (a, b, c, d, e, f),
                    "crs": CRS string or None, "centroid": (lon, lat),
                },
                "geotag": geotag as a JSON-safe dict,
            }
        """
        return {
            "geobox": _geobox_to_dict(self.geobox),
            "geotag": self.geotag.model_dump(mode="json", exclude_none=True),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeoAnchor":
        """Rebuild a GeoAnchor from to_dict()'s output.

        Args:
            data: Dict as returned by to_dict().

        Returns:
            Equivalent GeoAnchor.
        """
        geobox = data["geobox"]
        return cls(
            geobox=GeoBox(shape=tuple(geobox["shape"]), affine=Affine(*geobox["affine"]), crs=geobox["crs"]),
            geotag=GeoTag.model_validate(data["geotag"]),
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        datetime: AnchorDatetime,
        crs: str = "EPSG:4326",
        resolution: float = 10.0,
    ) -> "GeoAnchor":
        """Create a GeoAnchor from a bounding box and resolution.

        Args:
            bbox: (minx, miny, maxx, maxy) in the given CRS.
            crs: Coordinate reference system string (e.g. "EPSG:4326").
            resolution: Pixel size in CRS units.
            datetime: Anchor datetime or (start, end) date range for this anchor.
        """
        return cls(
            geobox=GeoBox.from_bbox(bbox, crs=crs, resolution=resolution, anchor="edge"),
            geotag=GeoTag(datetime=datetime),
        )

    @classmethod
    def from_coordinate(
        cls,
        latitude: float,
        longitude: float,
        *,
        datetime: AnchorDatetime,
        size_m: float | tuple[float, float],
        resolution: float = 10.0,
        crs: str | None = None,
    ) -> "GeoAnchor":
        """Create a GeoAnchor centered on a WGS84 coordinate.

        Args:
            latitude: Center latitude in WGS84 degrees.
            longitude: Center longitude in WGS84 degrees.
            size_m: Tile size in meters. Single number = square; ``(w, h)`` = rectangle.
            resolution: Pixel size in meters.
            datetime: Anchor datetime or (start, end) date range for this anchor.
            crs: Target projected CRS. Defaults to the local UTM/UPS zone.
        """
        latitude, longitude = validate_coordinate(latitude, longitude)
        if resolution <= 0:
            raise ValueError(f"Resolution must be positive, got {resolution}")
        width_m, height_m = (
            (size_m, size_m) if isinstance(size_m, (int, float)) else size_m
        )
        if width_m <= 0 or height_m <= 0:
            raise ValueError(f"Tile size must be positive, got {size_m}")
        target_crs = (
            CRS.from_user_input(crs) if crs is not None else calculate_crs(latitude, longitude)
        )
        if target_crs.is_geographic:
            raise ValueError(
                "from_coordinate requires a projected CRS; omit crs to use local UTM/UPS"
            )
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        cx, cy = transformer.transform(longitude, latitude)
        bbox = (cx - width_m / 2, cy - height_m / 2, cx + width_m / 2, cy + height_m / 2)
        return cls(
            geobox=GeoBox.from_bbox(
                bbox, crs=target_crs.to_string(), resolution=resolution, tight=True
            ),
            geotag=GeoTag(datetime=datetime),
        )

    @classmethod
    def from_polygon(
        cls,
        polygon: dict,
        datetime: AnchorDatetime,
        resolution: float = 10.0,
        crs: str | None = None,
    ) -> "GeoAnchor":
        """Create a GeoAnchor from a GeoJSON polygon geometry dict.

        Coordinates must be in WGS84 lon/lat. Resolution is in meters;
        the anchor is projected into local UTM/UPS unless ``crs`` is specified.

        Args:
            polygon: GeoJSON geometry dict with WGS84 lon/lat coordinates.
            resolution: Pixel size in meters.
            datetime: Anchor datetime or (start, end) date range for this anchor.
            crs: Target projected CRS. Defaults to local UTM/UPS zone.
        """
        geom = Geometry(polygon, crs="EPSG:4326")
        bbox = geom.boundingbox
        if crs is None:
            lon = (bbox.left + bbox.right) / 2
            lat = (bbox.bottom + bbox.top) / 2
            target_crs = calculate_crs(lat, lon)
        else:
            target_crs = CRS.from_user_input(crs)
        if target_crs.is_geographic:
            raise ValueError(
                "from_polygon requires a projected CRS; omit crs to use local UTM/UPS"
            )
        projected_geom = geom.to_crs(target_crs)
        return cls(
            geobox=GeoBox.from_geopolygon(projected_geom, resolution=resolution, anchor="edge"),
            geotag=GeoTag(datetime=datetime, polygon=projected_geom),
        )

    @classmethod
    def from_geojson(
        cls,
        path: str | Path,
        datetime: AnchorDatetime,
        resolution: float = 10.0,
        crs: str | None = None,
    ) -> "Iterator[GeoAnchor]":
        """Yield one GeoAnchor per feature/geometry in a GeoJSON file, lazily.

        Reading and parsing the file itself is unavoidably eager (one
        `json.load`, not a streaming parser) — what's deferred is the actual
        per-feature work (`from_polygon`'s reprojection/GeoBox build), so a
        caller only paying for the first few features of a large file (e.g.
        via `itertools.islice`) doesn't pay for the rest.

        Args:
            path: Path to a GeoJSON FeatureCollection, Feature, or raw geometry.
            resolution: Pixel size in meters.
            datetime: Anchor datetime or (start, end) date range applied to all features.
            crs: Target projected CRS. Defaults to local UTM/UPS per feature centroid.
        """
        with open(path) as f:
            geojson = json.load(f)
        if geojson.get("type") == "FeatureCollection":
            geometries = [feat["geometry"] for feat in geojson["features"]]
        elif geojson.get("type") == "Feature":
            geometries = [geojson["geometry"]]
        else:
            geometries = [geojson]
        for geom in geometries:
            yield cls.from_polygon(geom, resolution=resolution, datetime=datetime, crs=crs)
