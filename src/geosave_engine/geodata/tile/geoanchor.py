from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Self, cast

import json
import numpy as np
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.geo.xr import xr_coords
from pyproj import CRS, Transformer

from geosave_engine.utils.crs import calculate_crs, validate_coordinate
from geosave_engine.utils.datetime import DateRange, parse_datetime_range
from geosave_engine.utils.geolocator import Place

if TYPE_CHECKING:
    from .geotile import GeoTile

AnchorDatetime = dt | str | tuple[str, str] | DateRange


@dataclass(frozen=True, kw_only=True)
class GeoAnchor:
    """Where + when, no pixel data.

    What ingest sources produce (`to_anchors()`), what `GeoPipeline.fetch`
    takes in. A `GeoTile` always has data — a data-less reference is a
    `GeoAnchor`, never a `GeoTile` with `data=None`.

    Datetime always normalizes to an inclusive (start, end) range. Reduced-
    precision strings cover their whole stated period; datetime objects are
    exact instants.
    """

    geobox: GeoBox
    datetime: AnchorDatetime
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)
    polygon: Geometry | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "datetime", parse_datetime_range(self.datetime))

    def _repr_fields(self) -> str:
        """Shared field list for `__repr__` — subclasses append their own (e.g. bands, shape)."""
        when = str(self.start) if self.start == self.end else f"{self.start}–{self.end}"
        return f"bbox={self.bbox}, crs={self.crs!r}, datetime={when}, metadata={self.metadata}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._repr_fields()})"

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
        return cast(DateRange, self.datetime)[0]

    @property
    def end(self) -> dt:
        """Range end. For a resolved (non-range) anchor this equals `start`."""
        return cast(DateRange, self.datetime)[1]

    @property
    def stem(self) -> str:
        """Deterministic filename stem — same anchor always gets the same stem.

        Derived from centroid + datetime range + resolution, not list position, so
        it's stable across runs even if a source's anchor list changes shape
        (files added/removed/reordered) — two anchors at the same place/time/
        resolution always produce the same stem.

        Examples:
            >>> anchor.stem
            '13.000000_52.000000_20240101T000000_20240101T235959.999999_10m'
        """
        lon, lat = self.centroid
        res = self.resolution
        res_str = f"{int(res * 100)}cm" if res < 1 else f"{int(res)}m"

        def format_datetime(value: dt) -> str:
            result = value.strftime("%Y%m%dT%H%M%S")
            if value.microsecond:
                result += f".{value.microsecond:06d}"
            if value.utcoffset() is not None:
                result += value.strftime("%z")
            return result

        start = format_datetime(self.start)
        end = format_datetime(self.end)
        return f"{lon:.6f}_{lat:.6f}_{start}_{end}_{res_str}"

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

    def with_geobox(self, geobox: GeoBox) -> Self:
        """Return new instance rebased onto geobox, sharing any data reference.

        Pure geometry — no pixels are read or copied.
        """
        return dataclasses.replace(self, geobox=geobox)

    def with_metadata(self, extra: Mapping[str, Any], replace: bool = False) -> Self:
        """Merge key-value pairs into metadata field.

        Args:
            extra: Key-value pairs to merge.
            replace: If True, overwrite existing keys instead of raising.

        Raises:
            ValueError: If any key in extra already exists and replace is False.
        """
        if not replace:
            clash = set(extra) & set(self.metadata)
            if clash:
                raise ValueError(
                    f"metadata keys already present: {sorted(clash)}; pass replace=True to overwrite"
                )
        return dataclasses.replace(self, metadata={**self.metadata, **extra})

    def with_data(self, data: xr.DataArray) -> "GeoTile":
        """Attach pixel data, turning this anchor into a GeoTile.

        Args:
            data: Band values shaped (band, y, x) or (time, band, y, x) with a "band" coordinate.

        Raises:
            TypeError: If data is not an xr.DataArray.
        """
        if not isinstance(data, xr.DataArray):
            raise TypeError(f"with_data expects an xr.DataArray, got {type(data).__name__}")
        from .geotile import GeoTile

        return GeoTile(
            geobox=self.geobox, datetime=self.datetime, data=data,
            metadata=self.metadata, polygon=self.polygon,
        )

    def with_np(
        self,
        array: np.ndarray,
        names: list[str],
        times: list[dt] | None = None,
    ) -> "GeoTile":
        """Build DataArray from numpy array on this anchor's geobox and attach it.

        Accepts (y, x), (band, y, x), or (time, band, y, x). Spatial coords from geobox.

        Args:
            array: Pixel array; last two axes are (y, x).
            names: Band names in order.
            times: Observation datetimes; required for 4D array.
        """
        arr = np.asarray(array)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        base_coords: dict[Any, Any] = dict(xr_coords(self.geobox))
        if arr.ndim == 3:
            if len(names) != arr.shape[0]:
                raise ValueError(f"Expected {arr.shape[0]} names, got {len(names)}")
            da = xr.DataArray(arr, dims=("band", "y", "x"), coords={**base_coords, "band": names})
        elif arr.ndim == 4:
            if len(names) != arr.shape[1]:
                raise ValueError(f"Expected {arr.shape[1]} names, got {len(names)}")
            if times is None or len(times) != arr.shape[0]:
                got = 0 if times is None else len(times)
                raise ValueError(f"Expected {arr.shape[0]} times, got {got}")
            time_coord = [np.datetime64(t, "ns") for t in times]
            da = xr.DataArray(
                arr, dims=("time", "band", "y", "x"),
                coords={**base_coords, "band": names, "time": time_coord},
            )
        else:
            raise ValueError(f"with_np expects a 2-4D array, got {arr.ndim}D")
        return self.with_data(da)

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
            datetime=datetime,
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
        validate_coordinate(latitude, longitude)
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
            datetime=datetime,
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
            datetime=datetime,
            polygon=projected_geom,
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
