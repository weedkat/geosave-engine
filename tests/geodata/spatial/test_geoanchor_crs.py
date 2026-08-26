"""Unit tests for GeoAnchor.crs's EPSG-vs-WKT fallback.

No network — pure geometry. Before this, a real but non-EPSG-registered
CRS (a custom/local projection) made .crs return None, indistinguishable
from "no CRS at all".
"""
from __future__ import annotations

from geosave_engine.geodata.spatial import GeoAnchor

BBOX = (0, 0, 800, 800)
CUSTOM_LCC = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs"


class TestCrsEpsgVsWktFallback:
    def test_epsg_registered_crs_returns_short_form(self):
        anchor = GeoAnchor.from_bbox(
            (500000, 5000000, 500080, 5000080), crs="EPSG:32633", resolution=10, timespan="2020-01-01"
        )
        assert anchor.crs == "EPSG:32633"

    def test_non_epsg_crs_returns_wkt_not_none(self):
        anchor = GeoAnchor.from_bbox(BBOX, crs=CUSTOM_LCC, resolution=10, timespan="2020-01-01")
        assert anchor.crs is not None
        assert anchor.crs.startswith("PROJCRS[") or anchor.crs.startswith("PROJCS[")

