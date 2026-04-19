from __future__ import annotations

import pyproj
import shapely
import shapely.geometry


def wkt_to_geojson(wkt: str, source_crs: str) -> dict:
    """Reproject a WKT geometry from `source_crs` to WGS84 and return GeoJSON.

    `source_crs` can be any string accepted by `pyproj.CRS.from_user_input`,
    e.g. ``"EPSG:32605"`` or an authority:code pair.  The result is suitable
    for the ``intersects`` parameter of a STAC search.
    """
    geom = shapely.from_wkt(wkt)
    transformer = pyproj.Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    reprojected = shapely.transform(geom, transformer.transform, interleaved=False)
    return shapely.geometry.mapping(reprojected)
