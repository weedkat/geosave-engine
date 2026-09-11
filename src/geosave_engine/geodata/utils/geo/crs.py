"""Coordinate validation and projected CRS selection."""

from odc.geo import SomeCRS
from odc.geo.crs import CRS as OdcCRS
from odc.geo.geom import BoundingBox, Geometry

MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0
UPS_NORTH_EPSG = 5041
UPS_SOUTH_EPSG = 5042

# Short filename tokens for the non-metre linear units EPSG actually uses.
UNIT_SUFFIXES = {
    "US survey foot": "ft",
    "foot": "ft",
    "British foot (Sears 1922)": "ft",
    "link": "li",
    "chain": "ch",
}


def format_ground_size(value: float, unit: str) -> str:
    """Render a ground distance in its CRS unit as a filename-safe token.

    Metres scale to the nearest of centimetres, metres, or kilometres. Every
    other unit keeps its own magnitude and takes a short suffix.

    Args:
        value: Non-negative distance, in `unit`.
        unit: Axis unit name as a CRS reports it, e.g. `"metre"`.

    Returns:
        Token such as `"10m"`, `"5km"`, or `"0.0001deg"`.

    Examples:
        >>> format_ground_size(5000.0, "metre")
        '5km'
        >>> format_ground_size(0.0001, "degree")
        '0.0001deg'
    """
    if unit.startswith("degree"):
        return f"{value:g}deg"
    if unit != "metre":
        return f"{value:g}{UNIT_SUFFIXES.get(unit, unit.replace(' ', '-'))}"
    if value < 1:
        return f"{value * 100:g}cm"
    if value < 1000:
        return f"{value:g}m"
    return f"{value / 1000:g}km"


def validate_wgs84_bbox(bbox: tuple[float, float, float, float] | None) -> None:
    """Check WGS84 bounds and axis order.

    Args:
        bbox: `(min_longitude, min_latitude, max_longitude, max_latitude)`, or
            None.

    Raises:
        ValueError: A coordinate is outside WGS84 bounds or the minimum
            latitude exceeds the maximum.
    """
    if bbox is None:
        return
    minx, miny, maxx, maxy = bbox
    if not all(MIN_LATITUDE <= value <= MAX_LATITUDE for value in (miny, maxy)):
        raise ValueError(f"Latitude out of WGS84 range: {miny}, {maxy}")
    if not all(MIN_LONGITUDE <= value <= MAX_LONGITUDE for value in (minx, maxx)):
        raise ValueError(f"Longitude out of WGS84 range: {minx}, {maxx}")
    if miny > maxy:
        raise ValueError(
            f"Latitude miny ({miny}) cannot be greater than maxy ({maxy})."
        )


def validate_wgs84_coordinate(
    latitude: float,
    longitude: float,
) -> tuple[float, float]:
    """Validate a WGS84 coordinate, wrapping longitude into [-180, 180).

    Latitude has no analogous wrap (no meaningful fix past a pole) — still raises.

    Args:
        latitude: Latitude in WGS84 degrees.
        longitude: Longitude in WGS84 degrees.

    Returns:
        Latitude and wrapped longitude.

    Raises:
        ValueError: Latitude is outside [-90, 90].
    """
    if not MIN_LATITUDE <= latitude <= MAX_LATITUDE:
        raise ValueError(
            f"Latitude must be between {MIN_LATITUDE} and {MAX_LATITUDE} degrees, "
            f"got {latitude}"
        )
    span = MAX_LONGITUDE - MIN_LONGITUDE
    return latitude, ((longitude - MIN_LONGITUDE) % span) + MIN_LONGITUDE


def select_grid_crs(
    footprint: Geometry | BoundingBox,
    to_crs: SomeCRS | None = None,
) -> OdcCRS:
    """Select the CRS a footprint's pixel grid is built in.

    Judging the whole footprint rather than a point keeps an extent that
    straddles the 84-degree UTM limit in the UTM zone covering most of it.

    Args:
        footprint: Geometry or bounding box carrying its own CRS.
        to_crs: Explicit target. None keeps a projected footprint CRS, and
            sends a geographic one to its local UTM zone, or to UPS at a pole.

    Returns:
        CRS to build the geobox in.

    Raises:
        ValueError: The footprint declares no CRS.

    Examples:
        >>> select_grid_crs(plots.footprint)
        CRS('EPSG:32749')
    """
    if to_crs is not None:
        return OdcCRS(to_crs)
    if footprint.crs is None:
        raise ValueError("footprint declares no CRS, so its bounds place no grid")
    if not footprint.crs.geographic:
        return footprint.crs

    bounds = footprint if isinstance(footprint, BoundingBox) else footprint.boundingbox
    try:
        return OdcCRS.utm(bounds)
    except ValueError:
        latitude = (bounds.bottom + bounds.top) / 2
        pole = UPS_NORTH_EPSG if latitude >= 0 else UPS_SOUTH_EPSG
        return OdcCRS(f"EPSG:{pole}")
