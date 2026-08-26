from odc.geo.crs import CRS as OdcCRS
from pyproj import CRS

MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0
UPS_NORTH_EPSG = 5041
UPS_SOUTH_EPSG = 5042


def linear_unit_factors(crs: str | CRS | OdcCRS | None) -> tuple[float, float]:
    """Conversion factors from a projected CRS's x/y units to metres.

    Args:
        crs: Projected coordinate reference system.

    Returns:
        Metres per x unit and metres per y unit.

    Raises:
        ValueError: The CRS is geographic or does not declare two linear axes.
    """
    if crs is None:
        raise ValueError("CRS cannot be None")

    resolved = crs.proj if isinstance(crs, OdcCRS) else CRS.from_user_input(crs)
    if not resolved.is_projected:
        raise ValueError(f"linear units need a projected CRS, got {resolved.to_string()!r}")
    axes = resolved.axis_info[:2]
    if len(axes) != 2 or any(axis.unit_conversion_factor is None for axis in axes):
        raise ValueError(f"CRS {resolved.to_string()!r} does not declare two linear-axis units")
    return float(axes[0].unit_conversion_factor), float(axes[1].unit_conversion_factor)


def validate_bbox(bbox: tuple[float, float, float, float] | None) -> None:
    """Validate bbox for WGS84 compliance and axis order."""
    if bbox is None:
        return
    minx, miny, maxx, maxy = bbox
    if not (-90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0):
        raise ValueError(f"Latitude out of WGS84 range: {miny}, {maxy}")
    if not (-180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0):
        raise ValueError(f"Longitude out of WGS84 range: {minx}, {maxx}")
    if miny > maxy:
        raise ValueError(f"Latitude miny ({miny}) cannot be greater than maxy ({maxy}).")


def validate_coordinate(latitude: float, longitude: float) -> tuple[float, float]:
    """Validate a WGS84 coordinate, wrapping longitude into [-180, 180).

    Latitude has no analogous wrap (no meaningful fix past a pole) — still raises.

    Returns:
        (latitude, wrapped_longitude).

    Raises:
        ValueError: If latitude is out of [-90, 90].
    """
    if not MIN_LATITUDE <= latitude <= MAX_LATITUDE:
        raise ValueError(f"Latitude must be between -90 and 90 degrees, got {latitude}")
    wrapped_longitude = ((longitude + 180.0) % 360.0) - 180.0
    return latitude, wrapped_longitude


def calculate_crs(latitude: float, longitude: float) -> CRS:
    """Return local projected CRS with meter units for a WGS84 coordinate (UTM or UPS).

    UTM covers roughly -80 to 84 degrees latitude. Coordinates outside
    that range use polar stereographic (UPS).
    """
    try:
        return OdcCRS.utm(longitude, latitude).proj
    except ValueError:
        return CRS.from_epsg(UPS_NORTH_EPSG if latitude >= 0 else UPS_SOUTH_EPSG)
