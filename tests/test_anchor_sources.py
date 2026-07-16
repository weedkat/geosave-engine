"""Unit tests for anchor source configuration contracts."""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest
from odc.geo.geobox import GeoBox
from pydantic import ValidationError

from geosave_engine.geodata.pipeline import (
    CoordinateSource,
    GeoJSONSource,
    GeotiffSource,
    PolygonSource,
    ZarrSource,
)
from geosave_engine.geodata.tile import GeoAnchor

UTM = "EPSG:32633"
BBOX = (500000, 5000000, 500320, 5000320)  # 32 x 32 px at 10 m


def _tile(dt: datetime) -> GeoAnchor:
    gb = GeoBox.from_bbox(BBOX, crs=UTM, resolution=10, anchor="edge")
    arr = np.zeros((1, gb.height, gb.width), dtype="uint16")
    return GeoAnchor(geobox=gb, datetime=dt).with_np(arr, ["b1"])


def test_geotiff_source_uses_fixed_filename_dates(tmp_path):
    source = GeotiffSource(src=tmp_path)

    assert source.src == tmp_path
    assert set(GeotiffSource.model_fields) == {"type", "src", "tile_size_m"}
    with pytest.raises(ValidationError, match="date_format"):
        GeotiffSource(src=tmp_path, date_format="%Y%m%d")


class TestToAnchorsIsLazy:
    """to_anchors() yields lazily, one anchor at a time — not a pre-built list.

    Every subclass shares this contract (AnchorSource.to_anchors -> Iterator),
    so each is checked for both "really an iterator" and, where loading a real
    file is the expensive part (Geotiff/Zarr), that limit= genuinely stops
    before touching anything past it — not just after building everything
    then slicing.
    """

    def test_geotiff_source_returns_iterator_not_list(self, tmp_path):
        _tile(datetime(2023, 2, 1)).to_geotiff(tmp_path / "a-20230201.tif")
        result = GeotiffSource(src=tmp_path).to_anchors()
        assert not isinstance(result, list)
        assert iter(result) is result

    def test_geotiff_source_limit_never_touches_files_beyond_it(self, tmp_path):
        """A corrupt file sorted after the limit must never actually be opened."""
        _tile(datetime(2023, 2, 1)).to_geotiff(tmp_path / "a-20230201.tif")
        _tile(datetime(2023, 2, 2)).to_geotiff(tmp_path / "b-20230202.tif")
        (tmp_path / "c-20230203.tif").write_bytes(b"not a real geotiff")

        anchors = list(GeotiffSource(src=tmp_path).to_anchors(limit=2))
        assert len(anchors) == 2  # would raise on the corrupt 3rd file if it were touched

    def test_zarr_source_returns_iterator_not_list(self, tmp_path):
        _tile(datetime(2023, 2, 1)).to_zarr(tmp_path / "a.zarr")
        result = ZarrSource(src=tmp_path).to_anchors()
        assert not isinstance(result, list)
        assert iter(result) is result

    def test_zarr_source_limit_never_touches_stores_beyond_it(self, tmp_path):
        _tile(datetime(2023, 2, 1)).to_zarr(tmp_path / "a.zarr")
        _tile(datetime(2023, 2, 2)).to_zarr(tmp_path / "b.zarr")
        (tmp_path / "c.zarr").mkdir()
        (tmp_path / "c.zarr" / "not_zarr.txt").write_text("garbage")

        anchors = list(ZarrSource(src=tmp_path).to_anchors(limit=2))
        assert len(anchors) == 2  # would raise on the corrupt 3rd store if it were touched

    def test_coordinate_source_returns_iterator_yielding_one(self):
        source = CoordinateSource(lat=52.0, lon=13.0, datetime=datetime(2023, 2, 1), size_m=320)
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
        source = GeoJSONSource(src=tmp_path / "aoi.geojson", datetime=datetime(2023, 2, 1))
        result = source.to_anchors()
        assert not isinstance(result, list)
        assert iter(result) is result
        assert len(list(result)) == 2

    def test_polygon_source_limit_truncates_chunked_anchors(self):
        big_square = {
            "type": "Polygon",
            "coordinates": [[[13.0, 52.0], [13.0, 52.02], [13.02, 52.02], [13.02, 52.0], [13.0, 52.0]]],
        }
        source = PolygonSource(geom=big_square, datetime=datetime(2023, 2, 1), tile_size_m=500)
        result = source.to_anchors(limit=2)
        assert iter(result) is result
        assert len(list(result)) == 2
