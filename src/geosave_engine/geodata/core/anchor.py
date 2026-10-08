"""Exact spatial and temporal coverage independent of raster values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self, get_args

from odc.geo.crs import CRS as OdcCRS
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox, point

from geosave_engine.geodata.utils.datetime import (
    AnchorDatetime,
    DateRange,
    parse_daterange,
)
from geosave_engine.geodata.utils.crs import (
    select_grid_crs,
    validate_wgs84_bbox,
    validate_wgs84_coordinate,
)
from geosave_engine.geodata.utils.geometry import SomeGeometry

# Where pixel edges sit relative to the CRS origin, spelled as odc accepts it.
type GridAnchor = Literal["edge", "centre", "floating"]

if TYPE_CHECKING:
    from odc.geo import Resolution
    from pyproj import CRS

    from geosave_engine.geodata.utils.geolocator import Place


@dataclass(frozen=True, eq=False)
class GeoAnchor:
    """Describe exactly where and when geospatial data exists.

    Args:
        geobox: Exact pixel grid, including CRS, transform, bounds, and shape.
        timespan: Inclusive temporal coverage, or None for timeless data.
    """

    geobox: GeoBox
    timespan: DateRange | None = None

    @property
    def crs(self) -> OdcCRS:
        """Read the coordinate reference system the grid is placed in.

        Returns:
            CRS the geobox names.

        Raises:
            ValueError: The geobox carries no CRS, so it is indexed in pixels
                rather than placed on the ground.
        """
        if self.geobox.crs is None:
            raise ValueError(
                "geobox carries no CRS, so the anchor is not placed on the ground"
            )
        return self.geobox.crs

    @property
    def resolution(self) -> Resolution:
        """Read the pixel size in CRS units.

        Returns:
            Signed resolution along x and y.
        """
        return self.geobox.resolution

    @property
    def lonlat(self) -> tuple[float, float]:
        """Return the grid centroid in WGS84 longitude/latitude order.

        Returns:
            Longitude and latitude in degrees.
        """
        longitude, latitude = self.geobox.extent.centroid.to_crs("EPSG:4326").coords[0]
        return float(longitude), float(latitude)

    def locate(self) -> Place | None:
        """Reverse geocode the grid centroid, asking Nominatim over the network.

        Returns:
            Resolved place, or None when Nominatim has no result or is
            unavailable.
        """
        from geosave_engine.geodata.utils.geolocator import Place

        longitude, latitude = self.lonlat
        return Place.from_coordinate(latitude, longitude)

    def format(self, template: str) -> str:
        """Spell this anchor through a `str.format` template.

        Each field takes its own format spec, so a date field reads `strftime`
        directives and a number reads a precision.

        Args:
            template: Template naming any of `lon` and `lat` (WGS84 centroid,
                degrees), `start` and `end` (timespan edges), `res` (pixel
                size along x), `width` and `height` (ground extent) — the last
                three in grid CRS units — and `epsg`.

        Returns:
            The template with every field filled in.

        Raises:
            ValueError: The template names a field this anchor does not offer,
                as `start` on a timeless one.

        Examples:
            >>> anchor = GeoAnchor.from_coordinates(
            ...     52, 13, 500, 10, timespan="2025-06-01/2025-06-11"
            ... )
            >>> anchor.format("{lat:.2f}N_{lon:.2f}E_{start:%Y%m%d}_{res:g}m")
            '52.00N_13.00E_20250601_10m'
            >>> anchor.format("{start:%Y}/{start:%m}/{epsg}")
            '2025/06/32633'
        """
        longitude, latitude = self.lonlat
        bounds = self.geobox.boundingbox
        fields: dict[str, object] = {
            "lon": longitude,
            "lat": latitude,
            "res": abs(self.resolution.x),
            "width": bounds.span_x,
            "height": bounds.span_y,
            "epsg": self.crs.epsg,
        }
        if self.timespan is not None:
            fields["start"], fields["end"] = self.timespan
        try:
            return template.format(**fields)
        except KeyError as error:
            raise ValueError(
                f"anchor has no {error.args[0]!r} to format; it offers {sorted(fields)}"
            ) from None

    @classmethod
    def from_bbox(
        cls,
        bbox: BoundingBox | tuple[float, float, float, float],
        *,
        resolution: float | None = None,
        shape: int | tuple[int, int] | None = None,
        pad: float = 0.0,
        crs: str | CRS | OdcCRS | None = None,
        anchor: GridAnchor = "edge",
        timespan: AnchorDatetime | None = None,
    ) -> Self:
        """Build an anchor on a grid covering a bounding box.

        The grid always covers the padded box. Size it by `resolution` and the
        pixel count follows, or by `shape` and the pixel size follows.

        Args:
            bbox: Bounds as `(left, bottom, right, top)` in WGS84 degrees, or
                an ODC BoundingBox carrying its own CRS.
            resolution: Pixel size in grid CRS units.
            shape: Pixels to divide the box into — one value spans the longest
                axis at square pixels, or `(height, width)` spans both.
            pad: Grid CRS units added to every side of the box.
            crs: CRS to place the grid in. None keeps a projected source CRS
                or selects local UTM/UPS for a geographic one.
            anchor: Where pixel edges fall relative to the CRS origin.
                `"edge"` puts them on multiples of `resolution`, so grids built
                from different extents share one lattice and compose.
                `"floating"` fits the extent exactly instead, at a phase
                nothing else shares.
            timespan: Temporal coverage, as `parse_daterange` accepts it. None
                leaves the anchor timeless.

        Returns:
            Anchor whose geobox covers the padded box.

        Raises:
            ValueError: Neither or both of `resolution` and `shape` are given,
                a measurement is not positive, or `anchor` is not a known
                spelling.

        Examples:
            >>> GeoAnchor.from_bbox((112.10, -8.10, 112.20, -8.00), resolution=10).geobox.crs
            CRS('EPSG:32749')
        """
        if (resolution is None) == (shape is None):
            raise ValueError(
                "give exactly one of resolution= or shape= to size the grid"
            )
        if resolution is not None and resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        if pad < 0:
            raise ValueError(f"pad must be non-negative, got {pad}")
        if anchor not in get_args(GridAnchor.__value__):
            raise ValueError(
                f"anchor must be one of {get_args(GridAnchor.__value__)}, got {anchor!r}"
            )

        bounds = bbox if isinstance(bbox, BoundingBox) else BoundingBox(*bbox, crs=None)
        if bounds.crs is None:
            # Bare numbers are lon/lat, as odc assumes; say so before the grid maths does.
            edges = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            validate_wgs84_bbox(edges)
            bounds = BoundingBox(*edges, crs="EPSG:4326")
        grid_crs = select_grid_crs(bounds, crs)
        return cls(
            GeoBox.from_bbox(
                bounds.to_crs(grid_crs).buffered(pad),
                grid_crs,
                resolution=resolution,
                shape=shape,
                anchor=anchor,
            ),
            timespan=None if timespan is None else parse_daterange(timespan),
        )

    @classmethod
    def from_geometry(
        cls,
        geometry: SomeGeometry,
        *,
        resolution: float | None = None,
        shape: int | tuple[int, int] | None = None,
        pad: float = 0.0,
        crs: str | CRS | OdcCRS | None = None,
        anchor: GridAnchor = "edge",
        timespan: AnchorDatetime | None = None,
    ) -> Self:
        """Build an anchor on a grid covering one geometry.

        The grid always covers the geometry. Size it by `resolution` and the
        pixel count follows, or by `shape` and the pixel size follows. A
        geographic geometry is placed on its local UTM or UPS CRS.

        Args:
            geometry: GeoJSON mapping, WKT string, Shapely geometry, or ODC
                geometry.
            resolution: Pixel size in grid CRS units.
            shape: Pixels to divide the extent into — one value spans the
                longest axis at square pixels, or `(height, width)` spans both.
            pad: Grid CRS units added to every side of the geometry extent.
            crs: CRS to place the grid in. None keeps a projected source CRS
                or selects local UTM/UPS for a geographic one.
            anchor: Where pixel edges fall relative to the CRS origin.
                `"edge"` puts them on multiples of `resolution`, so grids built
                from different extents share one lattice and compose.
                `"floating"` fits the extent exactly instead, at a phase
                nothing else shares.
            timespan: Temporal coverage, as `parse_daterange` accepts it. None
                leaves the anchor timeless.

        Returns:
            Anchor whose geobox covers the padded extent.

        Raises:
            ValueError: The geometry is empty or invalid, neither or both of
                `resolution` and `shape` are given, or a measurement is not
                positive.

        Examples:
            >>> GeoAnchor.from_geometry(field, resolution=10).geobox.shape
            Shape2d(x=335, y=1108)
            >>> GeoAnchor.from_geometry(field, shape=512).geobox.resolution
            Resolution(x=21.6149, y=-21.6149)
        """
        from .vector import GeoVector

        # Reproject before bounding, so the box fits no looser than the CRS change forces.
        footprint = GeoVector.from_geometry(geometry).gs.footprint
        grid_crs = select_grid_crs(footprint, crs)
        return cls.from_bbox(
            footprint.to_crs(grid_crs).boundingbox,
            resolution=resolution,
            shape=shape,
            pad=pad,
            crs=grid_crs,
            anchor=anchor,
            timespan=timespan,
        )

    @classmethod
    def from_coordinates(
        cls,
        latitude: float,
        longitude: float,
        shape: int | tuple[int, int],
        resolution: float,
        *,
        crs: str | CRS | OdcCRS | None = None,
        anchor: GridAnchor = "edge",
        timespan: AnchorDatetime | None = None,
    ) -> Self:
        """Build an anchor on a grid centred on one WGS84 coordinate.

        A coordinate bounds no area, so it fixes only the centre and `shape`
        gives the grid its extent.

        Args:
            latitude: WGS84 latitude of the grid centre, in degrees.
            longitude: WGS84 longitude of the grid centre, in degrees, wrapped
                into [-180, 180).
            shape: Grid size in pixels — one value for a square, or
                `(height, width)`.
            resolution: Square pixel size in grid CRS units.
            crs: CRS to place the grid in. None picks the local UTM or UPS
                zone, which is metre-based.
            anchor: Where pixel edges fall relative to the CRS origin.
                `"edge"` puts them on multiples of `resolution`, so grids built
                from different extents share one lattice and compose.
                `"floating"` fits the extent exactly instead, at a phase
                nothing else shares.
            timespan: Temporal coverage, as `parse_daterange` accepts it. None
                leaves the anchor timeless.

        Returns:
            Anchor whose geobox is centred on the coordinate at exactly
            `resolution` and the requested size.

        Raises:
            ValueError: Latitude is out of range, or a measurement is not
                positive.

        Examples:
            >>> GeoAnchor.from_coordinates(-8.05, 112.15, 512, 10).geobox.shape
            Shape2d(x=512, y=512)
            >>> anchor = GeoAnchor.from_coordinates(
            ...     -8.05, 112.15, 512, 10, timespan="2024-01-01/2024-06-30"
            ... )
        """
        latitude, longitude = validate_wgs84_coordinate(latitude, longitude)
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")

        centre = point(longitude, latitude, "EPSG:4326")
        grid_crs = select_grid_crs(centre, crs)
        centre_x, centre_y = centre.to_crs(grid_crs).coords[0]

        height, width = (shape, shape) if isinstance(shape, int) else shape
        if height < 1 or width < 1:
            raise ValueError(
                f"shape must be positive in both axes, got {(height, width)}"
            )
        # Sizing the box to the shape makes odc read exactly `resolution` back off it.
        half_x, half_y = width * resolution / 2, height * resolution / 2
        return cls.from_bbox(
            BoundingBox(
                centre_x - half_x,
                centre_y - half_y,
                centre_x + half_x,
                centre_y + half_y,
                crs=grid_crs,
            ),
            shape=(height, width),
            crs=grid_crs,
            anchor=anchor,
            timespan=timespan,
        )
