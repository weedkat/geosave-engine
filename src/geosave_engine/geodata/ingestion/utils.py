"""Sentinel-2 L1C data service."""
from __future__ import annotations

import pyproj
import pystac
import shapely
import shapely.geometry
import shapely.ops
from geosave_engine.geodata.stac_query.base_query import BaseStacQuery


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _normalize_epsg(code: str | int) -> str:
    s = str(code).strip()
    if s.upper().startswith("EPSG:"):
        return s.upper()
    if s.isdigit():
        return f"EPSG:{s}"
    return s


def aoi_from_query(query: BaseStacQuery) -> shapely.Geometry | None:
    """Extract AOI geometry from query bbox or intersects (both WGS-84)."""
    if query.bbox is not None:
        return shapely.geometry.box(*query.bbox)
    if query.intersects is not None:
        return shapely.geometry.shape(query.intersects)
    return None


def is_wgs84(geom: shapely.Geometry) -> bool:
    """Heuristic: True when coordinates fall within WGS-84 degree range."""
    minx, miny, maxx, maxy = geom.bounds
    return -180.0 <= minx and maxx <= 180.0 and -90.0 <= miny and maxy <= 90.0


def to_utm_bounds(
    aoi_wgs84: shapely.Geometry,
    epsg: int,
) -> tuple[float, float, float, float]:
    """Reproject a WGS-84 AOI into the given EPSG and return its bounding box."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    return shapely.ops.transform(transformer.transform, aoi_wgs84).bounds


def radiometry_from_item(item: pystac.Item) -> tuple[float, float]:
    """Return (scale, offset) to convert DN to TOA reflectance: refl = DN * scale + offset.

    Checks top-level raster:scale/offset (CDSE) and raster:bands[0] (standard).

    Raises:
        ValueError: if no usable radiometric metadata is found on any asset.
    """
    for asset in item.assets.values():
        extra = getattr(asset, "extra_fields", {}) or {}
        for scale_key, offset_key in [("raster:scale", "raster:offset"), (None, None)]:
            if scale_key:
                scale  = extra.get(scale_key)
                offset = extra.get(offset_key)
            else:
                rb = extra.get("raster:bands")
                if not (rb and isinstance(rb, list) and rb):
                    continue
                scale  = rb[0].get("scale")
                offset = rb[0].get("offset")
            if scale is not None and offset is not None:
                s = float(scale)
                if s != 0.0:
                    return s, float(offset)
    raise ValueError(f"item {item.id!r} has no usable raster scale/offset metadata")
