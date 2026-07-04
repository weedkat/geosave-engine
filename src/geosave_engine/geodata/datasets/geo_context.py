from __future__ import annotations

from typing import Any, Callable

from geosave_engine.geodata.core import GeoTile


def _ref_dt(tile: GeoTile) -> Any:
    dt = tile.datetime
    return dt[0] if isinstance(dt, tuple) else dt


GEO_CONTEXT_EXTRACTORS: dict[str, Callable[[GeoTile], Any]] = {
    "crs":           lambda t: t.crs,
    "transform":     lambda t: t.affine,
    "coordinate":    lambda t: t.centroid,
    "time":          lambda t: _ref_dt(t).timetuple().tm_yday,
    "datetime":      lambda t: _ref_dt(t).isoformat(),
    "bbox_wgs84":    lambda t: list(t.wgs84_bbox),
    "stac_item_ids": lambda t: [i.id for i in t.stac],
}
