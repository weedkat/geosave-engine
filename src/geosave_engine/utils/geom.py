def validate_bbox(bbox: tuple[float, float, float, float] | None) -> None:
    """Validate the bbox for WGS84 compliance and axis order."""
    if bbox is None:
        return
    minx, miny, maxx, maxy = bbox
    # 1. Strict Latitude Check (Always -90 to 90)
    if not (-90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0):
        raise ValueError(f"Latitude out of WGS84 range: {miny}, {maxy}")
    # 2. Longitude range check
    if not (-180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0):
        raise ValueError(f"Longitude out of WGS84 range: {minx}, {maxx}")
    # 3. Axis Order Check (min must be less than max, unless crossing Antimeridian)
    # If miny > maxy, they definitely swapped Lat/Lon
    if miny > maxy:
        raise ValueError(f"Latitude miny ({miny}) cannot be greater than maxy ({maxy}).")