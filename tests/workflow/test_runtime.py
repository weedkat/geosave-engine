"""Primitive runtime settings become native anchor and STAC objects."""

from copy import deepcopy
import json

import pytest
from pystac_client import Client
from pystac_client.exceptions import APIError
import requests
from rasterio.errors import RasterioIOError

from geosave_engine.geodata.core.anchor import GeoAnchor
from geosave_engine.geodata.errors import AnchorFetchError
from geosave_engine.geodata.stac.client import StacClient
from geosave_engine.geodata.stac.source import StacSource, StacSourceConfig
from geosave_engine.geodata.utils import io
from geosave_engine.workflow.runtime import open_anchor, open_sources
from geosave_engine.workflow.ingestion import acquire
from geosave_engine.workflow.spec import RasterRequirement


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
def catalog_http(monkeypatch):
    """Keep native STAC parsing and collection lookup; replace only HTTP."""
    visited = []
    responses = {}

    def send(session, request, **kwargs):
        url = request.url
        visited.append(url)
        root = url.split("/collections")[0]
        if "/collections/" in url:
            document = {
                "type": "Collection",
                "stac_version": "1.0.0",
                "id": url.rsplit("/", 1)[1],
                "description": "Offline collection",
                "license": "CC-BY-4.0",
                "extent": {
                    "spatial": {"bbox": [[-180, -90, 180, 90]]},
                    "temporal": {"interval": [["2025-01-01T00:00:00Z", None]]},
                },
                "links": [],
            }
        else:
            document = {
                "type": "Catalog",
                "stac_version": "1.0.0",
                "id": "offline",
                "description": "Offline catalog",
                "conformsTo": [
                    f"https://api.stacspec.org/v1.0.0/{kind}"
                    for kind in ("core", "collections", "item-search")
                ],
                "links": [
                    {"rel": "self", "href": root},
                    {"rel": "root", "href": root},
                    {"rel": "data", "href": root + "/collections"},
                    {
                        "rel": "search",
                        "href": root + "/search",
                        "method": "POST",
                        "type": "application/geo+json",
                    },
                ],
            }
        result = responses.get(url, document)
        if isinstance(result, Exception):
            raise result
        response = requests.Response()
        response.url = url
        response.status_code = result if isinstance(result, int) else 200
        response._content = (
            result.encode() if isinstance(result, str) else json.dumps(result).encode()
        )
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    return visited, responses


def requirement(collection="sentinel-2-l2a"):
    return RasterRequirement(
        channels=3,
        collection=collection,
        endpoints=("https://primary.test/stac", "https://backup.test/stac"),
    )


def test_sources_use_native_client_query_and_load_settings(catalog_http):
    visited, _ = catalog_http
    settings = {
        "sentinel-2.l2a": {
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
        "dem": {},
    }
    original = deepcopy(settings)
    sources = open_sources(
        settings, {"sentinel-2.l2a": requirement(), "dem": requirement("dem")}
    )
    assert visited == [
        "https://primary.test/stac",
        "https://primary.test/stac/collections/sentinel-2-l2a",
        "https://primary.test/stac",
        "https://primary.test/stac/collections/dem",
    ]
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
        {"extra": True},
        {"load": {"typo": True}},
        {"query": {"typo": True}},
        {"query": {"collections": ["other"]}},
        {"query": {"max_items": 0}},
        {"query": {"limit": "wrong"}},
        {"query": {"ids": 7}},
        {"query": []},
        {"load": []},
        {"load": {"patch_url": lambda url: url}},
        {"load": {"nodata": float("nan")}},
    ],
)
def test_invalid_source_settings_fail_before_opening_catalog(catalog_http, settings):
    visited, _ = catalog_http
    with pytest.raises((TypeError, ValueError)):
        open_sources(
            {"valid": {}, "invalid": settings},
            {"valid": requirement(), "invalid": requirement()},
        )
    assert visited == []


@pytest.mark.parametrize("failure", [requests.ConnectionError("offline"), 404, 503])
@pytest.mark.parametrize("stage", ["", "/collections/sentinel-2-l2a"])
def test_unavailable_endpoint_falls_back_in_declared_order(
    catalog_http, failure, stage
):
    visited, responses = catalog_http
    responses["https://primary.test/stac" + stage] = failure
    sources = open_sources({"imagery": {}}, {"imagery": requirement()})
    assert sources["imagery"].collection == "sentinel-2-l2a"
    expected = ["https://primary.test/stac"]
    if stage:
        expected.append("https://primary.test/stac" + stage)
    assert visited == expected + [
        "https://backup.test/stac",
        "https://backup.test/stac/collections/sentinel-2-l2a",
    ]


@pytest.mark.parametrize(
    "failure",
    [
        400,
        401,
        403,
        429,
        "not json",
        {"broken": True},
        APIError("status-less document error"),
        requests.exceptions.JSONDecodeError("malformed JSON", "broken", 0),
    ],
)
@pytest.mark.parametrize("stage", ["", "/collections/sentinel-2-l2a"])
def test_bad_credentials_and_malformed_documents_do_not_fail_over(
    catalog_http, failure, stage
):
    visited, responses = catalog_http
    responses["https://primary.test/stac" + stage] = failure
    with pytest.raises(Exception) as caught:
        open_sources({"imagery": {}}, {"imagery": requirement()})
    assert not isinstance(caught.value, ConnectionError)
    assert visited[-1] == "https://primary.test/stac" + stage
    assert all("backup" not in url for url in visited)


def test_endpoint_exhaustion_reports_each_cause_and_chains_last_error(catalog_http):
    visited, responses = catalog_http
    responses["https://primary.test/stac"] = requests.ConnectionError("offline")
    responses["https://backup.test/stac/collections/sentinel-2-l2a"] = 404
    with pytest.raises(ConnectionError) as caught:
        open_sources({"imagery": {}}, {"imagery": requirement()})
    message = str(caught.value)
    assert "https://primary.test/stac" in message
    assert "offline" in message
    assert "https://backup.test/stac" in message
    assert "404" in message
    assert isinstance(caught.value.__cause__, APIError)
    assert caught.value.__cause__.status_code == 404
    assert len(visited) == 3


def test_persisted_only_requirement_fails_before_other_sources_open(catalog_http):
    visited, _ = catalog_http
    with pytest.raises(ValueError, match="collection.*endpoints"):
        open_sources(
            {"valid": {}, "persisted": {}},
            {"valid": requirement(), "persisted": RasterRequirement(channels=3)},
        )
    assert visited == []


def test_null_collection_uses_next_endpoint(catalog_http, monkeypatch):
    visited, _ = catalog_http
    get_collection = Client.get_collection

    def collection(client, name):
        if client.get_self_href() == "https://primary.test/stac":
            return None
        return get_collection(client, name)

    monkeypatch.setattr(Client, "get_collection", collection)
    sources = open_sources({"imagery": {}}, {"imagery": requirement()})
    assert sources["imagery"].collection == "sentinel-2-l2a"
    assert visited == [
        "https://primary.test/stac",
        "https://backup.test/stac",
        "https://backup.test/stac/collections/sentinel-2-l2a",
    ]


@pytest.mark.parametrize("children", [[], ["other"]], ids=["empty", "nonempty"])
def test_static_catalog_missing_collection_uses_next_endpoint(catalog_http, children):
    visited, responses = catalog_http
    responses["https://primary.test/stac"] = {
        "type": "Catalog",
        "stac_version": "1.0.0",
        "id": "static",
        "description": "Static catalog without the requested collection",
        "links": [
            {
                "rel": "child",
                "href": f"https://primary.test/stac/collections/{name}",
                "type": "application/json",
            }
            for name in children
        ],
    }
    with pytest.warns(UserWarning):
        sources = open_sources({"imagery": {}}, {"imagery": requirement()})
    assert sources["imagery"].collection == "sentinel-2-l2a"
    assert visited == [
        "https://primary.test/stac",
        *(f"https://primary.test/stac/collections/{name}" for name in children),
        "https://backup.test/stac",
        "https://backup.test/stac/collections/sentinel-2-l2a",
    ]


def test_malformed_collection_key_error_does_not_fail_over(catalog_http):
    visited, responses = catalog_http
    responses["https://primary.test/stac/collections/sentinel-2-l2a"] = {
        "type": "Collection",
        "stac_version": "1.0.0",
        "id": "sentinel-2-l2a",
        "description": "Missing required license",
        "extent": {
            "spatial": {"bbox": [[-180, -90, 180, 90]]},
            "temporal": {"interval": [["2025-01-01T00:00:00Z", None]]},
        },
        "links": [],
    }
    with pytest.raises(KeyError, match="license"):
        open_sources({"imagery": {}}, {"imagery": requirement()})
    assert visited == [
        "https://primary.test/stac",
        "https://primary.test/stac/collections/sentinel-2-l2a",
    ]


def test_empty_search_does_not_try_another_endpoint(catalog_http, local_stac):
    visited, responses = catalog_http
    _, anchor = local_stac
    responses["https://primary.test/stac/search"] = {
        "type": "FeatureCollection",
        "features": [],
        "links": [],
    }
    sources = open_sources({"optical": {}}, {"optical": requirement("optical")})
    with pytest.raises(AnchorFetchError):
        acquire(sources, anchor)
    assert visited[-1] == "https://primary.test/stac/search"
    assert all("backup" not in url for url in visited)


def test_lazy_asset_failure_does_not_try_another_endpoint(catalog_http, local_stac):
    visited, responses = catalog_http
    local_sources, anchor = local_stac
    item = local_sources["optical"].client.items[0].to_dict()
    item["assets"]["red"]["href"] += ".missing"
    responses["https://primary.test/stac/search"] = {
        "type": "FeatureCollection",
        "features": [item],
        "links": [],
    }
    sources = open_sources(
        {"optical": {"load": {"bands": ["red"], "groupby": "time"}}},
        {"optical": requirement("optical")},
    )
    raw = acquire(sources, anchor)
    with pytest.raises(RasterioIOError):
        raw["optical"].red.compute()
    assert all("backup" not in url for url in visited)
