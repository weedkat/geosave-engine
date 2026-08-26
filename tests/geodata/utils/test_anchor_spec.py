"""Unit tests for anchor_spec — AOI request dict -> typed spec -> GeoAnchor.

No network — pure validation + dispatch. Each spec's to_anchor() mirrors
exactly one GeoAnchor.from_* constructor; these tests check the mirroring
is faithful (same geobox/crs/timespan a direct constructor call would give),
not GeoAnchor's own constructor behavior (covered by its own test files).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from geosave_engine.geodata.spatial import GeoAnchor
from geosave_engine.geodata.utils.spatial.anchor_spec import (
    BboxSpec,
    CoordinateSpec,
    GeometrySpec,
    spec_from_dict,
)

UTM = "EPSG:32633"
BBOX = (500000, 5000000, 500320, 5000320)


class TestBboxSpec:
    def test_to_anchor_mirrors_from_bbox(self):
        spec = BboxSpec(bbox=BBOX, crs=UTM, resolution=10, timespan="2020-01-01")
        anchor = spec.to_anchor()
        assert anchor.geobox == GeoAnchor.from_bbox(BBOX, crs=UTM, resolution=10, timespan="2020-01-01").geobox

    def test_crs_used_as_is_no_reprojection(self):
        spec = BboxSpec(bbox=BBOX, crs=UTM, resolution=10, timespan="2020-01-01")
        assert spec.to_anchor().crs == UTM

    def test_default_crs_is_wgs84(self):
        spec = BboxSpec(bbox=(10.0, 45.0, 10.01, 45.01), resolution=10, timespan="2020-01-01")
        assert spec.crs == "EPSG:4326"


class TestCoordinateSpec:
    def test_to_anchor_mirrors_from_coordinate(self):
        spec = CoordinateSpec(lat=52.0, lon=13.0, size_m=320, resolution=10, timespan="2020-01-01")
        anchor = spec.to_anchor()
        expected = GeoAnchor.from_coordinate(52.0, 13.0, timespan="2020-01-01", size_m=320, resolution=10)
        assert anchor.geobox == expected.geobox

    def test_explicit_crs_respected(self):
        spec = CoordinateSpec(lat=52.0, lon=13.0, size_m=320, resolution=10, timespan="2020-01-01", crs=UTM)
        assert spec.to_anchor().crs == UTM

    def test_none_crs_auto_picks_local_utm(self):
        spec = CoordinateSpec(lat=52.0, lon=13.0, size_m=320, resolution=10, timespan="2020-01-01")
        assert spec.to_anchor().crs != "EPSG:4326"

    def test_rectangular_size(self):
        spec = CoordinateSpec(lat=52.0, lon=13.0, size_m=(320, 160), resolution=10, timespan="2020-01-01")
        anchor = spec.to_anchor()
        assert anchor.width == 32
        assert anchor.height == 16


class TestGeometrySpec:
    def test_explicit_crs_respected(self):
        point = {"type": "Point", "coordinates": [13.0, 52.0]}
        spec = GeometrySpec(geometry=point, resolution=10, timespan="2020-01-01", crs=UTM)
        assert spec.to_anchor().crs == UTM


class TestSpecFromDict:
    def test_dispatches_by_type(self):
        data = {"type": "coordinate", "lat": 52.0, "lon": 13.0, "timespan": "2020-01-01", "size_m": 320}
        spec = spec_from_dict(data)
        assert isinstance(spec, CoordinateSpec)

    def test_unknown_type_raises(self):
        with pytest.raises(ValidationError):
            spec_from_dict({"type": "circle", "timespan": "2020-01-01"})

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            spec_from_dict({"type": "bbox", "timespan": "2020-01-01"})

    def test_malformed_date_raises_here_not_inside_to_anchor(self):
        """_coerce_timespan reuses parse_daterange — a bad date string fails at spec_from_dict, not later."""
        with pytest.raises(ValidationError):
            spec_from_dict({"type": "bbox", "bbox": list(BBOX), "crs": UTM, "timespan": "not-a-date"})

    def test_timespan_range_as_json_list_coerces_to_tuple(self):
        data = {
            "type": "bbox",
            "bbox": list(BBOX),
            "crs": UTM,
            "resolution": 10,
            "timespan": ["2020-01-01", "2020-01-31"],
        }
        anchor = spec_from_dict(data).to_anchor()
        assert anchor.start.isoformat().startswith("2020-01-01")
        assert anchor.end.isoformat().startswith("2020-01-31")

