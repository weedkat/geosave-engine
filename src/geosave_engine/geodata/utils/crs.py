from pyproj import CRS

MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0
MIN_UTM_LATITUDE = -80.0
MAX_UTM_LATITUDE = 84.0
UTM_ZONE_WIDTH_DEGREES = 6.0
UTM_NORTH_BASE_EPSG = 32600
UTM_SOUTH_BASE_EPSG = 32700
UPS_NORTH_EPSG = 5041
UPS_SOUTH_EPSG = 5042


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


def validate_coordinate(latitude: float, longitude: float) -> None:
    """Validate a WGS84 coordinate."""
    if not MIN_LATITUDE <= latitude <= MAX_LATITUDE:
        raise ValueError(f"Latitude must be between -90 and 90 degrees, got {latitude}")
    if not MIN_LONGITUDE <= longitude <= MAX_LONGITUDE:
        raise ValueError(f"Longitude must be between -180 and 180 degrees, got {longitude}")


def calculate_crs(latitude: float, longitude: float) -> CRS:
    """Return local projected CRS with meter units for a WGS84 coordinate (UTM or UPS)."""
    if latitude > MAX_UTM_LATITUDE:
        return CRS.from_epsg(UPS_NORTH_EPSG)
    if latitude < MIN_UTM_LATITUDE:
        return CRS.from_epsg(UPS_SOUTH_EPSG)

    zone = min(int((longitude + 180.0) // UTM_ZONE_WIDTH_DEGREES) + 1, 60)
    epsg = UTM_NORTH_BASE_EPSG + zone if latitude >= 0 else UTM_SOUTH_BASE_EPSG + zone
    return CRS.from_epsg(epsg)
