"""Primitive runtime settings become native anchor and STAC objects."""

from copy import deepcopy
import json

import pytest
from pystac_client import Client

from geosave_engine.geodata.core.anchor import GeoAnchor
from geosave_engine.geodata.stac.client import StacClient
from geosave_engine.geodata.stac.source import StacSource, StacSourceConfig
from geosave_engine.geodata.utils import io
from geosave_engine.workflow.runtime import open_anchor, open_sources


@pytest.fixture
def coordinates():
    return {
        "type": "coordinates",
        "latitude": 45.0,
        "longitude": 12.0,
        "shape": [4, 6],
        "resolution": 10,
        "crs": "EPSG:32633",
        "timespan": "2025-01",
    }


@pytest.fixture
def geojson(tmp_path):
    path = tmp_path / "area.geojson"
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [12.0, 45.0],
                                    [12.002, 45.0],
                                    [12.002, 45.002],
                                    [12.0, 45.002],
                                    [12.0, 45.0],
                                ]
                            ],
                        },
                    }
                ],
            }
        )
    )
    return path


def test_coordinates_decode_named_latitude_longitude_and_grid(coordinates):
    settings = deepcopy(coordinates)
    result = open_anchor(settings)
    expected = GeoAnchor.from_coordinates(
        45, 12, (4, 6), 10, crs="EPSG:32633", timespan="2025-01"
    )
    assert result.geobox == expected.geobox
    assert result.geobox.shape.yx == (4, 6)
    assert result.crs.to_epsg() == 32633
    assert result.timespan == expected.timespan
    assert settings == coordinates


def test_coordinate_defaults_use_native_local_grid_and_square_shape(coordinates):
    settings = {**coordinates, "shape": 4}
    settings.pop("crs")
    settings["timespan"] = ["2025-01-01", "2025-01-03"]
    result = open_anchor(settings)
    expected = GeoAnchor.from_coordinates(
        45, 12, 4, 10, timespan=("2025-01-01", "2025-01-03")
    )
    assert result.geobox == expected.geobox
    assert result.timespan == expected.timespan


@pytest.mark.parametrize(
    "grid",
    [
        {"resolution": 10, "crs": "EPSG:32633", "pad": 10},
        {"resolution": 0.001, "crs": "EPSG:4326"},
        {"shape": [4, 6], "crs": "EPSG:4326"},
        {"shape": 4},
    ],
)
def test_geojson_uses_native_footprint_and_grid_rules(geojson, grid):
    settings = {"type": "geojson", "path": str(geojson), "timespan": "2025-01", **grid}
    result = open_anchor(settings)
    native = {**grid, "timespan": "2025-01"}
    if isinstance(native.get("shape"), list):
        native["shape"] = tuple(native["shape"])
    expected = GeoAnchor.from_geometry(io.read_vector(geojson).footprint, **native)
    assert result.geobox == expected.geobox
    assert result.timespan == expected.timespan
    longitude, latitude = result.geographic_centroid
    assert abs(longitude - 12.001) < 0.002
    assert abs(latitude - 45.001) < 0.002


@pytest.mark.parametrize(
    "change",
    [
        {"type": "point"},
        {"type": None},
        {"path": "unexpected.geojson"},
        {"pad": 1},
        {"extra": True},
        {"shape": [4]},
        {"shape": [4, 0]},
        {"shape": True},
        {"latitude": 91},
        {"resolution": 0},
        {"timespan": {"not": "a range"}},
    ],
)
def test_coordinate_invalid_or_wrong_variant_fields_fail(coordinates, change):
    with pytest.raises((TypeError, ValueError)):
        open_anchor({**coordinates, **change})


@pytest.mark.parametrize(
    "field", ["type", "latitude", "longitude", "shape", "resolution"]
)
def test_coordinate_required_fields_are_explicit(coordinates, field):
    coordinates.pop(field)
    with pytest.raises(ValueError, match=field):
        open_anchor(coordinates)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"shape": [4, 4], "resolution": 10},
        {"resolution": 10, "latitude": 45},
        {"resolution": 10, "extra": True},
    ],
)
def test_geojson_invalid_combinations_fail_before_reading(change):
    with pytest.raises(ValueError):
        open_anchor({"type": "geojson", "path": "/does-not-exist.geojson", **change})


@pytest.fixture
def opened_catalog(monkeypatch):
    opened = []
    catalog = Client(id="local", description="Offline STAC client")
    monkeypatch.setattr(Client, "open", lambda url: opened.append(url) or catalog)
    return opened, catalog


def test_sources_use_native_client_query_and_load_settings(opened_catalog):
    opened, _catalog = opened_catalog
    settings = {
        "sentinel-2.l2a": {
            "url": "https://example.test/stac",
            "collection": "sentinel-2-l2a",
            "query": {
                "datetime": "2025-01",
                "max_items": 5,
                "filter": {"op": "<=", "args": [{"property": "eo:cloud_cover"}, 10]},
            },
            "load": {
                "bands": ["B04", "B08"],
                "groupby": "time",
                "chunks": {"x": 32, "y": 16},
            },
        },
        "dem": {"url": "https://example.test/stac", "collection": "dem"},
    }
    original = deepcopy(settings)
    sources = open_sources(settings)
    assert opened == ["https://example.test/stac", "https://example.test/stac"]
    assert isinstance(sources["sentinel-2.l2a"], StacSource)
    assert isinstance(sources["sentinel-2.l2a"].client, StacClient)
    assert sources["sentinel-2.l2a"].collection == "sentinel-2-l2a"
    assert sources["sentinel-2.l2a"].query.to_search_params() == {
        "collections": ["sentinel-2-l2a"],
        "datetime": "2025-01",
        "max_items": 5,
        "filter": settings["sentinel-2.l2a"]["query"]["filter"],
        "filter_lang": "cql2-json",
    }
    assert sources["sentinel-2.l2a"].config.bands == ["B04", "B08"]
    assert sources["sentinel-2.l2a"].config.chunks == {"x": 32, "y": 16}
    assert sources["dem"].config == StacSourceConfig()
    assert sources["dem"].query.collections == ["dem"]
    assert settings == original


@pytest.mark.parametrize(
    "settings",
    [
        {"collection": "example"},
        {"url": "https://example.test"},
        {"url": "https://example.test", "collection": "example", "extra": True},
        {
            "url": "https://example.test",
            "collection": "example",
            "load": {"typo": True},
        },
        {
            "url": "https://example.test",
            "collection": "example",
            "query": {"typo": True},
        },
        {
            "url": "https://example.test",
            "collection": "example",
            "query": {"collections": ["other"]},
        },
    ],
)
def test_invalid_source_settings_fail_before_opening_catalog(opened_catalog, settings):
    opened, _catalog = opened_catalog
    with pytest.raises((TypeError, ValueError)):
        open_sources({"source": settings})
    assert opened == []
