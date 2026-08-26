"""Unit tests for GeoAnchor.from_bbox's own input validation.

No network — pure geometry. Before this, a swapped or antimeridian-crossing
bbox, or a zero/negative resolution, reached odc.geo's internals raw and
crashed on a bare `assert`, no message.
"""
from __future__ import annotations

import pytest

from geosave_engine.geodata.spatial import GeoAnchor

UTM = "EPSG:32633"
BBOX = (500000, 5000000, 500080, 5000080)


class TestFromBboxValidation:
    def test_swapped_min_max_raises(self):
        with pytest.raises(ValueError, match="min < max"):
            GeoAnchor.from_bbox((500080, 5000080, 500000, 5000000), crs=UTM, resolution=10, timespan="2020-01-01")

    def test_antimeridian_crossing_raises(self):
        with pytest.raises(ValueError, match="min < max"):
            GeoAnchor.from_bbox((179.5, -1, -179.5, 1), crs="EPSG:4326", resolution=0.01, timespan="2020-01-01")

    def test_zero_resolution_raises(self):
        with pytest.raises(ValueError, match="Resolution must be positive"):
            GeoAnchor.from_bbox(BBOX, crs=UTM, resolution=0, timespan="2020-01-01")

    def test_negative_resolution_raises(self):
        with pytest.raises(ValueError, match="Resolution must be positive"):
            GeoAnchor.from_bbox(BBOX, crs=UTM, resolution=-10, timespan="2020-01-01")

    def test_valid_bbox_still_works(self):
        anchor = GeoAnchor.from_bbox(BBOX, crs=UTM, resolution=10, timespan="2020-01-01")
        assert anchor.width == 8 and anchor.height == 8
