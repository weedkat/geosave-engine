"""Unit tests for anchor source configuration contracts."""
from __future__ import annotations

import json
from datetime import datetime

from geosave_engine.geodata.pipeline import (
    CoordinateSource,
    GeoJSONSource,
    PolygonSource,
)


class TestToAnchorsIsLazy:
    """to_anchors() yields lazily, one anchor at a time — not a pre-built list.

    Every subclass shares this contract (AnchorSource.to_anchors -> Iterator),
    so each is checked for both "really an iterator" and that limit= genuinely
    stops before doing unnecessary chunking work — not just after building
    everything then slicing.
    """

    def test_coordinate_source_returns_iterator_yielding_one(self):
        d = datetime(2023, 2, 1)
        source = CoordinateSource(lat=52.0, lon=13.0, datetime=(d, d), area_m=320)
        result = source.to_anchors()
        assert iter(result) is result
        assert len(list(result)) == 1

    def test_geojson_source_returns_iterator_not_list(self, tmp_path):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [13.0, 52.0]}, "properties": {}},
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [14.0, 53.0]}, "properties": {}},
            ],
        }
        (tmp_path / "aoi.geojson").write_text(json.dumps(geojson))
        d = datetime(2023, 2, 1)
        source = GeoJSONSource(src=tmp_path / "aoi.geojson", datetime=(d, d))
        result = source.to_anchors()
        assert not isinstance(result, list)
        assert iter(result) is result
        assert len(list(result)) == 2

    def test_polygon_source_limit_truncates_chunked_anchors(self):
        big_square = {
            "type": "Polygon",
            "coordinates": [[[13.0, 52.0], [13.0, 52.02], [13.02, 52.02], [13.02, 52.0], [13.0, 52.0]]],
        }
        d = datetime(2023, 2, 1)
        source = PolygonSource(geom=big_square, datetime=(d, d), tile_size_px=50)
        result = source.to_anchors(limit=2)
        assert iter(result) is result
        assert len(list(result)) == 2
