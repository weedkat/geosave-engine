"""Unit tests for calculate_crs's UTM/UPS zone selection.

No network — pyproj's UTM zone database is bundled locally. calculate_crs
had zero test coverage before this.
"""
from __future__ import annotations

from geosave_engine.geodata.utils.spatial.crs import UPS_NORTH_EPSG, UPS_SOUTH_EPSG, calculate_crs


class TestCalculateCrsUtmZones:
    def test_known_northern_point_returns_expected_zone(self):
        assert calculate_crs(52.0, 13.0).to_epsg() == 32633  # Berlin, UTM zone 33N

    def test_known_southern_point_returns_expected_zone(self):
        assert calculate_crs(-6.2, 106.8).to_epsg() == 32748  # Jakarta, UTM zone 48S


class TestCalculateCrsPolarFallback:
    def test_beyond_utm_north_falls_back_to_ups_north(self):
        assert calculate_crs(85.0, 10.0).to_epsg() == UPS_NORTH_EPSG

    def test_beyond_utm_south_falls_back_to_ups_south(self):
        assert calculate_crs(-85.0, 10.0).to_epsg() == UPS_SOUTH_EPSG
