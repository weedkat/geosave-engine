"""GeoAnchor: where and when, no pixels. See GeoAnchor for details."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime as dt
from math import hypot, isclose
from typing import TYPE_CHECKING, Any

import numpy as np
import odc.geo.xr  # noqa: F401 — registers .odc accessor on xr.DataArray
import rioxarray  # noqa: F401 — registers .rio accessor on xr.DataArray
import xarray as xr
from affine import Affine  # type: ignore[import-untyped]
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from pyproj import CRS, Transformer

from geosave_engine.geodata.extensions import GeoExtension, TimeSpan, TimeSpec, span_from_times
from geosave_engine.geodata.utils.array import bind_pixels, cast_nodata, validate_spatial
from geosave_engine.geodata.utils.datetime import (
    AnchorDatetime,
    DateRange,
    format_stem_dates,
    naive_utc,
)
from geosave_engine.geodata.utils.spatial.crs import calculate_crs, linear_unit_factors, validate_coordinate
from geosave_engine.geodata.utils.spatial.geobox import geobox_matches
from geosave_engine.geodata.utils.spatial.geolocator import Place
from geosave_engine.geodata.utils.spatial.geometry import SomeGeometry
from geosave_engine.utils.fn import UNSET, Unset

from .header import GeoHeader, decode_attrs, encode_attrs
from .vector import GeoVector

if TYPE_CHECKING:
    from .raster import GeoRaster
    from .tile import GeoTile

WGS84 = "EPSG:4326"


@dataclass(frozen=True, kw_only=True, eq=False)
class GeoAnchor:
    """Everything a raster carries except its pixels — grid, time span, features, header.

    Compares by object identity, so spell out what two anchors must share,
    e.g. `geobox_matches(a.geobox, b.geobox)`. Put pixels on it with
    `to_geotile`/`to_raster`; change any field with `rebase`.

    Args:
        geobox: Spatial extent + resolution + CRS.
        vector: Features over this extent, or None. Must share `geobox`'s CRS.
        header: Time span, tags, extensions, tiling and timespec. Empty on
            an anchor built as a plain window spec. Change any of it
            through `rebase` rather than by hand.

    Examples:
        >>> anchor = GeoAnchor.from_coordinate(52.0, 13.0, timespan="2024-01-15", size_m=5000)
        >>> anchor.stem
        '13.0000E_52.0000N_5kmx5km_20240115_10m'
    """

    geobox: GeoBox
    vector: GeoVector | None = None
    header: GeoHeader = field(default_factory=GeoHeader)

    def __post_init__(self) -> None:
        """Check `geobox`, `vector` and `header` still describe one another.

        Raises:
            ValueError: `geobox` has no CRS, `vector`'s CRS differs from
                it, not one of its features touches this extent, or
                `header.tiling`'s `tile_shape` isn't this geobox's own
                `(height, width)`.
        """
        if self.geobox.crs is None:
            raise ValueError(
                "GeoAnchor needs a CRS — build the geobox with one, or stamp it on with "
                "geobox.to_crs(crs) / data.odc.assign_crs(crs) first"
            )
        self._validate_vector()
        tiling = self.header.tiling
        if tiling is not None and tuple(tiling.tile_shape) != self.shape:
            raise ValueError(
                f"tiling stamp is for a {tuple(tiling.tile_shape)} window, this geobox is "
                f"{self.shape} — clear it with tiling=None"
            )

    def _validate_vector(self) -> None:
        """Check `vector` shares this geobox's CRS and its bounding box reaches this extent.

        Bounding boxes only, so features whose box overlaps but whose
        geometry doesn't still pass, and a feature may overhang the extent —
        one crossing a tile edge is kept whole.

        Raises:
            ValueError: `vector`'s CRS differs from `geobox`'s, or its
                bounding box lies fully outside this extent.
        """
        if self.vector is None:
            return

        # compared this way round — odc's CRS knows how to read geopandas', not the reverse
        vector_crs = self.vector.gdf.crs
        if self.geobox.crs != vector_crs:
            named = vector_crs.to_string() if vector_crs is not None else None
            raise ValueError(f"vector's CRS {named!r} doesn't match this grid's {str(self.geobox.crs)!r}")

        minx, miny, maxx, maxy = self.vector.gdf.total_bounds
        left, bottom, right, top = self.bounds
        if minx > right or maxx < left or miny > top or maxy < bottom:
            raise ValueError(
                f"this vector spans {(minx, miny, maxx, maxy)}, nowhere near the extent {self.bounds}"
            )

    def _validate_time(self, data: xr.DataArray) -> None:
        """Check this anchor's recorded span still covers an array's own time labels.

        Args:
            data: Pixel array in GDAL form, already normalized.

        Raises:
            ValueError: this span doesn't reach as far as the buckets
                `data`'s own time labels stand for.
        """
        span = self.timespan
        if span is None or "time" not in data.dims:
            return

        # a recorded span stands for whole buckets, so it has to reach at least as far as the labels do
        start, end = span_from_times(data.time.values, self.header.timespec)
        if naive_utc(span[0]) > start or naive_utc(span[1]) < end:
            raise ValueError(
                f"recorded time span {span[0]}–{span[1]} doesn't cover this data's own {start}–{end}"
            )

    def __repr__(self) -> str:
        if self.timespan is None:
            when = "timeless"
        elif self.start == self.end:
            when = str(self.start)
        else:
            when = f"{self.start} – {self.end}"

        bounds = tuple(round(value, 2) for value in self.bounds)
        return (
            f"{type(self).__name__}\n"
            f"  bounds: {bounds}\n"
            f"  crs:    {self.crs!r}\n"
            f"  time:   {when}"
        )

    # --- Grid ---

    @property
    def resolution(self) -> float:
        """Pixel size in CRS units — one number, so the grid must be square.

        Raises:
            ValueError: pixels aren't square. Read `resolution_xy` for the
                pair; `stem` assumes one number.
        """
        x_size, y_size = self.resolution_xy
        if not isclose(x_size, y_size):
            raise ValueError(f"geobox pixels aren't square ({x_size} x {y_size}) — read resolution_xy instead")
        return x_size

    @property
    def resolution_xy(self) -> tuple[float, float]:
        """Pixel size as `(x, y)` in CRS units, both positive. Correct for a non-square grid.

        Returns:
            `(x, y)` pixel sizes. A north-up grid steps down in y, so the
            sign is dropped — these are sizes, not signed steps.
        """
        x_size, y_size = self.geobox.resolution.xy
        return abs(x_size), abs(y_size)

    @property
    def affine(self) -> Affine:
        return self.geobox.affine

    @property
    def crs(self) -> str:
        """This anchor's CRS — `"EPSG:<code>"`, or its full WKT for a grid with no EPSG code.

        Returns:
            CRS string, which `CRS.from_user_input` reads either way.

        Raises:
            ValueError: `geobox` lost its CRS after this anchor was built —
                only reachable by replacing the geobox in place.
        """
        crs = self.geobox.crs
        if crs is None:
            raise ValueError("GeoAnchor's geobox lost its CRS")
        epsg = crs.to_epsg()
        return f"EPSG:{epsg}" if epsg else crs.to_wkt()

    @property
    def width(self) -> int:
        return self.geobox.width

    @property
    def height(self) -> int:
        return self.geobox.height

    @property
    def shape(self) -> tuple[int, int]:
        """Pixel grid as `(height, width)` — numpy/xarray order."""
        return self.height, self.width

    @property
    def area_m2(self) -> float:
        """Pixel-grid area in m². Correct for non-square pixels.

        Raises:
            ValueError: this anchor is on a geographic CRS, where the
                grid's own units are degrees, not meters.
        """
        crs = self.geobox.crs
        if crs is None or crs.geographic:
            raise ValueError(f"area_m2 needs a projected CRS, this anchor is on {crs!r}")
        x_to_m, y_to_m = linear_unit_factors(crs)
        pixel_area = abs(self.affine.determinant) * x_to_m * y_to_m
        return self.width * self.height * pixel_area

    # --- Extent ---

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Bounding box in this anchor's own CRS: `(minx, miny, maxx, maxy)`. For grid math."""
        return self.geobox.boundingbox.bbox

    @property
    def extent(self) -> Geometry:
        """Pixel-grid footprint as a polygon in this anchor's own CRS."""
        return self.geobox.extent

    @property
    def geographic_bounds(self) -> tuple[float, float, float, float]:
        """Bounding box in WGS84: `(minlon, minlat, maxlon, maxlat)`. What a STAC search wants."""
        return self.geobox.geographic_extent.boundingbox.bbox

    @property
    def geographic_centroid(self) -> tuple[float, float]:
        """Centroid in WGS84 as `(lon, lat)`. Used by `stem` and `location`."""
        lon, lat = self.extent.centroid.to_crs(WGS84).coords[0]
        return float(lon), float(lat)

    @property
    def geojson(self) -> dict[str, Any]:
        """Footprint as a GeoJSON geometry dict, WGS84. For a STAC item's own geometry, or a map."""
        return self.extent.to_crs(WGS84, wrapdateline=True).json

    @property
    def location(self) -> Place | None:
        """Reverse-geocoded place for this anchor's centroid, or None if nothing resolves.

        Hits Nominatim over the network on a coordinate it hasn't seen
        (results are cached per ~1km). Nominatim allows one request a
        second, so don't touch this in a loop over many anchors.
        """
        lon, lat = self.geographic_centroid
        return Place.from_coordinate(lat, lon)

    # --- Time ---

    @property
    def timespan(self) -> DateRange | None:
        """Recorded `(start, end)` span, paired off `header`. None for an anchor with no time at all."""
        return self.header.timespan

    @property
    def start(self) -> dt | None:
        """Range start. Equals `end` for a single instant, None for a timeless anchor."""
        return self.header.start

    @property
    def end(self) -> dt | None:
        """Range end. Equals `start` for a single instant, None for a timeless anchor."""
        return self.header.end

    def rebase(
        self,
        *,
        geobox: GeoBox | Unset = UNSET,
        timespan: AnchorDatetime | None | Unset = UNSET,
        vector: GeoVector | None | Unset = UNSET,
        **extensions: GeoExtension | Mapping[str, Any] | None,
    ) -> GeoAnchor:
        """New anchor with any of its fields changed.

        The `GeoRaster.rebase` counterpart for an anchor holding no pixels.
        `timespec` is unreachable here: it records a bucketing that
        happened to real data, so only `resample_time` creates one and
        `concat` carries it.

        Args:
            geobox: New grid. Omitted keeps the current one. A `tiling`
                stamp left on while the shape changes is rejected.
            timespan: Datetime string, `(start, end)` pair, or None to clear
                the span. Omitted keeps the current one.
            vector: New features over this extent, or None to clear.
            **extensions: `namespace=value` for any registered extension,
                e.g. `render={"class_map": {0: "bg", 1: "palm"}}`,
                `tags={"source": "survey"}`, `tiling=None`. A dict merges
                onto that namespace's current fields; a built instance
                replaces it whole; `None` drops the namespace. A namespace
                named `timespan` is unreachable here — it binds to this
                method's own param.

        Returns:
            New GeoAnchor.

        Raises:
            ValueError: `timespan` is a string that can't be parsed, a field
                dict fails its extension's validation, `vector`'s CRS
                differs from the result's geobox, `tiling`'s `tile_shape`
                isn't that geobox's own, or a namespace declares
                `SETTABLE = False`.
            UnknownExtensionError: a keyword names an unregistered namespace.

        Examples:
            >>> anchor.rebase(timespan="2024-01-15")
            >>> anchor.rebase(tiling=None, render={"class_map": {0: "bg", 1: "palm"}})
        """
        merged: dict[str, Any] = dict(extensions)
        if not isinstance(timespan, Unset):
            merged[TimeSpan.NAMESPACE] = None if timespan is None else TimeSpan.from_input(timespan)
        header = self.header.rebase(**merged) if merged else self.header

        moved: dict[str, Any] = {"header": header}
        if not isinstance(geobox, Unset):
            moved["geobox"] = geobox
        if not isinstance(vector, Unset):
            moved["vector"] = vector
        return replace(self, **moved)

    def _with_timespec(self, timespec: TimeSpec | None) -> GeoAnchor:
        """New anchor recording how an axis was bucketed — for the operations that did the bucketing.

        The sanctioned bypass around `TimeSpec.SETTABLE = False`: edits
        `header.extensions` directly rather than going through the guarded
        `rebase()` merge loop.

        Args:
            timespec: Resolved bucket grid, or None to clear one.

        Returns:
            New GeoAnchor, everything else untouched.
        """
        extensions = dict(self.header.extensions)
        if timespec is None:
            extensions.pop(TimeSpec.NAMESPACE, None)
        else:
            extensions[TimeSpec.NAMESPACE] = timespec
        return replace(self, header=replace(self.header, extensions=extensions))

    # --- Naming ---

    @property
    def stem(self) -> str:
        """Deterministic filename stem — same anchor always gets the same stem.

        Built from centroid, extent, time range and resolution, never list
        position. Degrees on a geographic CRS, meters on a projected one; a
        non-square grid names both pixel sides, a timeless anchor no date.

        Examples:
            >>> anchor.stem
            '13.0000E_52.0000N_5kmx5km_20240115_10m'
        """
        lon, lat = self.geographic_centroid
        lon_str = f"{abs(lon):.4f}{'E' if lon >= 0 else 'W'}"
        lat_str = f"{abs(lat):.4f}{'N' if lat >= 0 else 'S'}"

        geographic = self.geobox.crs is None or self.geobox.crs.geographic
        x_size, y_size = self.resolution_xy
        if geographic:
            width, height = self.width * x_size, self.height * y_size
        else:
            x_to_m, y_to_m = linear_unit_factors(self.geobox.crs)
            affine = self.affine
            x_size = hypot(affine.a * x_to_m, affine.d * y_to_m)
            y_size = hypot(affine.b * x_to_m, affine.e * y_to_m)
            width, height = self.width * x_size, self.height * y_size
        extent_str = f"{self._format_size(width, geographic)}x{self._format_size(height, geographic)}"

        # a non-square grid gets both pixel sides, so the stem still names the grid it came off
        resolution_str = self._format_size(x_size, geographic)
        if not isclose(x_size, y_size):
            resolution_str = f"{resolution_str}x{self._format_size(y_size, geographic)}"

        if self.timespan is None:
            return f"{lon_str}_{lat_str}_{extent_str}_{resolution_str}"
        return f"{lon_str}_{lat_str}_{extent_str}_{format_stem_dates(self.timespan)}_{resolution_str}"

    @staticmethod
    def _format_size(value: float, geographic: bool) -> str:
        """Compact size token for a filename stem.

        Args:
            value: Length in the CRS's own units.
            geographic: Whether those units are degrees rather than meters.

        Returns:
            `"0.1deg"` on a geographic CRS; otherwise cm below 1m, m below
            1km, km at or above.
        """
        if geographic:
            return f"{value:g}deg"
        if value < 1:
            return f"{int(value * 100)}cm"
        if value < 1000:
            return f"{int(value)}m"
        return f"{value / 1000:g}km"

    # --- Spatial tests ---

    def intersects(self, other: GeoAnchor) -> bool:
        """Whether the two extents overlap at all. Reprojects `other` onto this CRS first.

        Args:
            other: Anchor to test against.

        Returns:
            True if the extents share any area.
        """
        return self.extent.intersects(self._extent_of(other))

    def contains(self, other: GeoAnchor) -> bool:
        """Whether `other`'s extent sits entirely inside this one. Reprojects `other` onto this CRS first.

        Args:
            other: Anchor to test against.

        Returns:
            True if this extent covers all of other's.
        """
        return self.extent.contains(self._extent_of(other))

    def _extent_of(self, other: GeoAnchor) -> Geometry:
        """`other`'s extent on this anchor's CRS, ready to compare against `self.extent`.

        Args:
            other: Anchor whose extent is wanted.

        Returns:
            other's extent, reprojected only when the two CRSs differ.
        """
        if other.geobox.crs == self.geobox.crs:
            return other.extent
        return other.extent.to_crs(self.crs)

    # --- Builders ---

    def pad(self, pad_px: int) -> GeoAnchor:
        """Expand geobox by pad_px on every side.

        Args:
            pad_px: Pixels the geobox grows by on each side.

        Returns:
            New GeoAnchor, same time and vector, on the grown grid.
            `tiling` is cleared — it described the old window.
        """
        return self.rebase(geobox=self.geobox.pad(pad_px), tiling=None)

    def to_crs(self, crs: str, resolution: float | None = None) -> GeoAnchor:
        """Put this anchor on a different CRS. No pixels involved.

        Args:
            crs: Target CRS, e.g. `"EPSG:4326"`.
            resolution: Pixel size in the target CRS's units. None lets
                odc.geo pick one keeping roughly the same pixel count.

        Returns:
            New GeoAnchor on `crs`, `vector` reprojected alongside it.
            `tiling` is cleared — it described the old grid.
        """
        if resolution is None:
            geobox = self.geobox.to_crs(crs)
        else:
            # odc's own to_crs only takes a resolution *strategy*, so build the grid from the reprojected extent
            geobox = GeoBox.from_bbox(self.extent.to_crs(crs).boundingbox.bbox, crs=crs, resolution=resolution, anchor="edge")
        return self.rebase(
            geobox=geobox,
            vector=self.vector.to_crs(crs) if self.vector is not None else None,
            tiling=None,
        )

    # --- Attaching pixels ---

    def to_geotile(
        self,
        data: xr.DataArray | np.ndarray,
        *,
        bands: Sequence[str] | None = None,
        times: Sequence[dt] | None = None,
        header: GeoHeader | None = None,
        nodata: float | int | None = None,
    ) -> GeoTile:
        """Attach pixels to this anchor.

        Args:
            data: Canonical DataArray on this geobox, or a 2-4D NumPy
                array. Two-dimensional input becomes one named band.
            bands: One name per raw-array band. Required for NumPy input;
                invalid alongside a DataArray.
            times: Observation datetimes for raw 4D input. Invalid
                alongside a DataArray or lower-dimensional input.
            header: Tags, extensions, tiling and timespec for the result.
                None starts empty. An unset time span takes this anchor's
                own `timespan`.
            nodata: Sentinel to declare on the result — what raw NumPy
                pixels lose. None keeps whatever `data` already carries.

        Returns:
            GeoTile carrying `data`, `header`, and this anchor's `vector`.
            A recorded time span is kept as it stands and must cover
            `data`'s own labels; only an undated result reads its span off
            those labels.

        Raises:
            ValueError: Pixel dimensions or coordinates violate the
                canonical Spatial representation, the grid differs from
                this anchor, its recorded span excludes pixel times, or
                `nodata` can't be represented by the pixel dtype.

        Examples:
            >>> tile.anchor.to_geotile(prediction, bands=["class"])
        """
        from .tile import GeoTile

        pixels, anchor = self._bind_pixels(data, bands, times, header, nodata)
        return GeoTile(data=pixels, anchor=anchor)

    def to_raster(
        self,
        data: xr.DataArray | np.ndarray,
        *,
        bands: Sequence[str] | None = None,
        times: Sequence[dt] | None = None,
        header: GeoHeader | None = None,
        nodata: float | int | None = None,
    ) -> GeoRaster:
        """Attach pixels to this anchor as a surface — same rules as `to_geotile`.

        Args:
            data: Canonical DataArray on this geobox, or a 2-4D NumPy array.
            bands: One name per raw-array band. Required for NumPy input.
            times: Observation datetimes for raw 4D input.
            header: Tags, extensions, tiling and timespec for the result.
                None starts empty. An unset time span takes this anchor's
                own `timespan`.
            nodata: Sentinel to declare on the result — what raw NumPy
                pixels lose. None keeps whatever `data` already carries.

        Returns:
            GeoRaster carrying `data`, `header`, and this anchor's `vector`.
            A recorded time span is kept as it stands and must cover
            `data`'s own labels; only an undated result reads its span off
            those labels.

        Raises:
            ValueError: `bands`/`times` missing or mismatched for the
                array's shape, a DataArray's own geobox doesn't match this
                one, the recorded span doesn't cover `data`'s own labels,
                or `nodata` can't be represented by the pixel dtype.
        """
        from .raster import GeoRaster

        pixels, anchor = self._bind_pixels(data, bands, times, header, nodata)
        return GeoRaster(data=pixels, anchor=anchor)

    @staticmethod
    def _name_band(data: xr.DataArray, bands: Sequence[str]) -> xr.DataArray:
        """Give a bandless DataArray its one band, keeping it lazy.

        What a derived layer needs: an index or mask computed off selected
        bands comes back with no `band` dim, and often a stale scalar `band`
        coord left over by `sel`.

        Args:
            data: Array shaped `(y, x)` or `(time, y, x)`, no `band` dim.
            bands: Exactly one name for the band it becomes.

        Returns:
            Same pixels shaped `(band, y, x)` or `(time, band, y, x)`.

        Raises:
            ValueError: `data` already has a `band` dim, or `bands` doesn't
                hold exactly one name.
        """
        if "band" in data.dims:
            raise ValueError("bands= names a bandless DataArray's one band; this one already names its own")
        if len(bands) != 1:
            raise ValueError(f"a bandless DataArray becomes exactly one band, got {len(bands)} names")

        named = data.drop_vars("band", errors="ignore").expand_dims(band=list(bands))
        return named.transpose("time", "band", "y", "x") if "time" in named.dims else named

    def _bind_pixels(
        self,
        data: xr.DataArray | np.ndarray,
        bands: Sequence[str] | None,
        times: Sequence[dt] | None,
        header: GeoHeader | None,
        nodata: float | int | None,
    ) -> tuple[xr.DataArray, GeoAnchor]:
        """Put an array on this anchor's grid, ready for a GeoTile or GeoRaster.

        Args:
            data: Canonical DataArray, or raw 2-4D NumPy pixels.
            bands: Required names for raw pixels; None for a DataArray.
            times: Required timestamps for raw 4D pixels; None otherwise.
            header: Header for the result, or None to keep this anchor's own.
            nodata: Sentinel to declare on the result, or None to keep
                whatever `data` carries.

        Returns:
            `(data in GDAL form, anchor for the result)`. The anchor's span
            is `header`'s own, this anchor's window when `header` names none,
            or None — which leaves GeoTile/GeoRaster to read it off
            `data`'s time labels.

        Raises:
            ValueError: `bands`/`times` missing or mismatched for the
                array's shape, a DataArray's own geobox doesn't match
                this one, or `nodata` can't be represented by the pixel dtype.
        """
        if isinstance(data, xr.DataArray):
            if times is not None:
                raise ValueError("times belong to raw NumPy pixels; a DataArray owns its time coordinate")
            if bands is not None:
                data = self._name_band(data, bands)
            data = validate_spatial(data)
            if not geobox_matches(data.odc.geobox, self.geobox):
                raise ValueError(f"data's geobox {data.odc.geobox!r} doesn't match this anchor's {self.geobox!r}")
        else:
            if bands is None:
                raise ValueError("Raw NumPy pixels require explicit bands, e.g. bands=('class',)")
            data = bind_pixels(self.geobox, data, bands=bands, times=times)

        # own name: rioxarray is untyped, so assigning back to `data` rewidens that parameter's union
        pixels: xr.DataArray = data
        if nodata is not None:
            pixels = pixels.rio.write_nodata(cast_nodata(nodata, pixels.dtype))

        resolved = header if header is not None else self.header
        anchor = replace(self, header=resolved)

        # a header naming no span of its own falls back to this anchor's window
        if anchor.timespan is None and self.timespan is not None:
            anchor = anchor.rebase(timespan=self.timespan)
        return pixels, anchor

    # --- Constructors ---

    @classmethod
    def from_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        *,
        resolution: float,
        timespan: AnchorDatetime | None = None,
        crs: str = WGS84,
    ) -> GeoAnchor:
        """Anchor from a bounding box and resolution.

        Args:
            bbox: (minx, miny, maxx, maxy) in the given CRS.
            timespan: Anchor datetime or (start, end) range. None for a timeless anchor.
            crs: Coordinate reference system string, e.g. `"EPSG:4326"`.
            resolution: Pixel size in CRS units.

        Raises:
            ValueError: `resolution` isn't positive, or `bbox`'s min isn't
                below its max on either axis — an antimeridian-crossing
                bbox included, which has to be split into two first.
        """
        if resolution <= 0:
            raise ValueError(f"Resolution must be positive, got {resolution}")
        minx, miny, maxx, maxy = bbox
        if minx >= maxx or miny >= maxy:
            raise ValueError(f"bbox must have min < max on both axes, got {bbox}")
        geobox = GeoBox.from_bbox(bbox, crs=crs, resolution=resolution, anchor="edge")
        return cls(geobox=geobox).rebase(timespan=timespan)

    @classmethod
    def from_coordinate(
        cls,
        latitude: float,
        longitude: float,
        *,
        timespan: AnchorDatetime | None = None,
        size_m: float | tuple[float, float],
        resolution: float = 10.0,
        crs: str | None = None,
    ) -> GeoAnchor:
        """Anchor centered on a WGS84 coordinate.

        Args:
            latitude: Center latitude in WGS84 degrees.
            longitude: Center longitude in WGS84 degrees.
            timespan: Anchor datetime or (start, end) range. None for a timeless anchor.
            size_m: Extent in meters. One number = square, `(w, h)` = rectangle.
            resolution: Pixel size in meters.
            crs: Target projected CRS. None uses the local UTM/UPS zone.

        Raises:
            ValueError: coordinate is out of range, `resolution` or
                `size_m` isn't positive, or `crs` is geographic rather
                than projected.
        """
        latitude, longitude = validate_coordinate(latitude, longitude)
        if resolution <= 0:
            raise ValueError(f"Resolution must be positive, got {resolution}")
        width_m, height_m = (size_m, size_m) if isinstance(size_m, (int, float)) else size_m
        if width_m <= 0 or height_m <= 0:
            raise ValueError(f"Tile size must be positive, got {size_m}")

        target_crs = CRS.from_user_input(crs) if crs is not None else calculate_crs(latitude, longitude)
        if target_crs.is_geographic:
            raise ValueError("from_coordinate requires a projected CRS; omit crs to use local UTM/UPS")
        x_to_m, y_to_m = linear_unit_factors(target_crs)
        if not isclose(x_to_m, y_to_m):
            raise ValueError(f"from_coordinate needs equal x/y linear units, got factors {(x_to_m, y_to_m)}")
        cx, cy = Transformer.from_crs(WGS84, target_crs, always_xy=True).transform(longitude, latitude)
        width_units, height_units = width_m / x_to_m, height_m / y_to_m
        bbox = (cx - width_units / 2, cy - height_units / 2, cx + width_units / 2, cy + height_units / 2)
        return cls(
            geobox=GeoBox.from_bbox(
                bbox,
                crs=target_crs.to_string(),
                resolution=resolution / x_to_m,
                tight=True,
            ),
        ).rebase(timespan=timespan)

    @classmethod
    def from_vector(
        cls,
        vector: GeoVector,
        timespan: AnchorDatetime | None = None,
        resolution: float = 10.0,
        crs: str | None = None,
    ) -> GeoAnchor:
        """Anchor whose geobox molds to a vector's own extent, that vector attached.

        Args:
            vector: Features the anchor should cover, on any CRS.
            timespan: Anchor datetime or (start, end) range. None for a timeless anchor.
            resolution: Pixel size in meters.
            crs: Target projected CRS. None uses the UTM/UPS zone local to
                the vector's own centroid.

        Returns:
            GeoAnchor covering `vector`'s bounding box, with `vector`
            reprojected onto the target CRS and carried as its own.

        Raises:
            ValueError: `resolution` isn't positive, `crs` is geographic,
                or its x/y axes use different linear units.
        """
        if resolution <= 0:
            raise ValueError(f"Resolution must be positive, got {resolution}")

        # the local UTM zone is picked off the vector's own centre, so it needs WGS84 degrees
        left, bottom, right, top = vector.to_crs(WGS84).gdf.total_bounds
        if crs is None:
            target_crs = calculate_crs((bottom + top) / 2, (left + right) / 2)
        else:
            target_crs = CRS.from_user_input(crs)
        if target_crs.is_geographic:
            raise ValueError("from_vector requires a projected CRS; omit crs to use local UTM/UPS")
        x_to_m, y_to_m = linear_unit_factors(target_crs)
        if not isclose(x_to_m, y_to_m):
            raise ValueError(f"from_vector needs equal x/y linear units, got factors {(x_to_m, y_to_m)}")

        projected = vector.to_crs(target_crs.to_string())
        minx, miny, maxx, maxy = (float(bound) for bound in projected.gdf.total_bounds)
        return cls(
            geobox=GeoBox.from_bbox(
                (minx, miny, maxx, maxy),
                crs=target_crs.to_string(),
                resolution=resolution / x_to_m,
                anchor="edge",
            ),
            vector=projected,
        ).rebase(timespan=timespan)

    @classmethod
    def from_geometry(
        cls,
        geometry: SomeGeometry,
        timespan: AnchorDatetime | None = None,
        resolution: float = 10.0,
        crs: str | None = None,
    ) -> GeoAnchor:
        """Anchor whose geobox molds to one geometry's bounding box, that geometry kept as `vector`.

        A point degenerates to a single pixel, a line to its extent's
        rectangle. Use `from_vector` when the shape carries properties too.

        Args:
            geometry: GeoJSON geometry dict, or WKT string (e.g.
                `"POINT (13.0 52.0)"`), WGS84 lon/lat coordinates.
            timespan: Anchor datetime or (start, end) range. None for a timeless anchor.
            resolution: Pixel size in meters.
            crs: Target projected CRS. None uses the local UTM/UPS zone.

        Raises:
            ValueError: `geometry` is WKT that can't be parsed, is empty,
                or `crs` is geographic rather than projected.
        """
        return cls.from_vector(
            GeoVector.from_geometry(geometry, crs=WGS84),
            timespan=timespan,
            resolution=resolution,
            crs=crs,
        )


def encode_anchor(anchor: GeoAnchor) -> dict[str, Any]:
    """One anchor as a JSON-safe dict — grid and header, no vector.

    The counterpart of `encode_attrs` for a whole anchor: it adds the grid a
    raw array loses, so `decode_anchor` rebuilds the exact geobox rather than
    an approximation of it.

    Args:
        anchor: Anchor to encode.

    Returns:
        {
            "shape": [height, width],
            "affine": [a, b, c, d, e, f],
            "crs": "<CRS string>",
            "header": {"<namespace>": {...}},
        }

    Examples:
        >>> encode_anchor(anchor)["crs"]
        'EPSG:32633'
    """
    return {
        "shape": list(anchor.shape),
        "affine": list(anchor.affine)[:6],
        "crs": anchor.crs,
        "header": encode_attrs({}, anchor.header, "json"),
    }


def decode_anchor(data: Mapping[str, Any]) -> GeoAnchor:
    """Rebuild the anchor `encode_anchor` wrote.

    Args:
        data: What `encode_anchor` returned.

    Returns:
        GeoAnchor on the exact geobox it was encoded from, carrying its own
        header. Its `vector` is None — features never ride in the dict.

    Raises:
        KeyError: A grid field is missing.
        ValueError: `shape` isn't two numbers or `affine` isn't six.
    """
    shape = tuple(data["shape"])
    affine = tuple(data["affine"])
    if len(shape) != 2:
        raise ValueError(f"shape must be (height, width), got {list(shape)}")
    if len(affine) != 6:
        raise ValueError(f"affine must be 6 coefficients, got {len(affine)}")
    _, header = decode_attrs(data.get("header", {}))
    return GeoAnchor(geobox=GeoBox(shape, Affine(*affine), data["crs"]), header=header)
