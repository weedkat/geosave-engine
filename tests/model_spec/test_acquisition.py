import pytest
from pystac_client.exceptions import APIError
import requests
from rasterio.errors import RasterioIOError
from types import SimpleNamespace

import geosave_engine.model_spec.stac as stac_module
from geosave_engine.geodata.core import GeoAnchor
from geosave_engine.geodata.errors import AnchorFetchError
from geosave_engine.model_spec import ModelSpec, RasterRequirement, StacRecipe


@pytest.fixture(autouse=True)
def clear_client_cache():
    stac_module._open_client.cache_clear()


def coordinate_anchor():
    return GeoAnchor.from_coordinates(
        45,
        12,
        4,
        10,
        crs="EPSG:32633",
        timespan="2025-01",
    )


def requirement(collection="sentinel-2-l2a"):
    return RasterRequirement.model_validate(
        {
            "channels": 3,
            "stac": {
                "collection": collection,
                "endpoints": [
                    "https://primary.test/stac",
                    "https://backup.test/stac",
                ],
            },
        }
    )


def test_load_rasters_reads_and_validates_one_local_source(stac_server):
    url, requests, expected_anchor = stac_server

    model = ModelSpec(
        schema_version=2,
        rasters={
            "image": RasterRequirement(
                variables=("red", "nir"),
                stac=StacRecipe.model_validate(
                    {
                        "collection": "optical",
                        "endpoints": (url,),
                        "load": {"bands": ["red", "nir"]},
                    }
                ),
                require_crs=True,
            )
        },
    )
    result = model.load_rasters(expected_anchor)["image"]

    assert list(result.data_vars) == ["red", "nir"]
    assert result.odc.geobox == expected_anchor.geobox
    assert result.red.chunks is not None
    assert requests == [["optical"]]


def test_load_stac_raster_reuses_open_client(catalog_http):
    visited, responses = catalog_http
    responses["https://primary.test/stac/search"] = {
        "type": "FeatureCollection",
        "features": [],
        "links": [],
    }

    for _ in range(2):
        with pytest.raises(AnchorFetchError):
            requirement().stac.load_raster(coordinate_anchor())

    assert visited == [
        "https://primary.test/stac",
        "https://primary.test/stac/collections/sentinel-2-l2a",
        "https://primary.test/stac/search",
        "https://primary.test/stac/search",
    ]


def test_open_client_dispatches_a_named_provider(monkeypatch):
    expected = SimpleNamespace(
        collection=lambda collection: SimpleNamespace(id=collection)
    )
    opened = []

    def open_endpoint(endpoint):
        opened.append(endpoint)
        return expected

    monkeypatch.setattr(stac_module.StacClient, "open", open_endpoint, raising=False)

    assert stac_module._open_client("sentinel-2-l2a", ("planetary_computer",)) is expected
    assert opened == ["planetary_computer"]


def test_open_client_falls_back_when_native_client_reports_missing_collection(
    monkeypatch,
):
    collection = "sentinel-2-l2a"

    def missing(_collection):
        raise ValueError(
            f"collection {collection!r} not found on this STAC endpoint; "
            "call collections() to see what is available"
        )

    primary = SimpleNamespace(collection=missing)
    backup = SimpleNamespace(
        collection=lambda name: SimpleNamespace(id=name)
    )
    clients = iter((primary, backup))
    monkeypatch.setattr(
        stac_module.StacClient,
        "open",
        lambda _endpoint: next(clients),
    )

    assert stac_module._open_client(
        collection,
        ("https://primary.test/stac", "https://backup.test/stac"),
    ) is backup


def test_stac_module_has_no_concurrency_configuration() -> None:
    assert not hasattr(stac_module, "source_concurrency")


@pytest.mark.parametrize("failure", [requests.ConnectionError("offline"), 404, 503])
@pytest.mark.parametrize("stage", ["", "/collections/sentinel-2-l2a"])
def test_unavailable_endpoint_falls_back_in_declared_order(
    catalog_http, failure, stage
):
    visited, responses = catalog_http
    responses["https://primary.test/stac" + stage] = failure
    responses["https://backup.test/stac/search"] = {
        "type": "FeatureCollection",
        "features": [],
        "links": [],
    }

    with pytest.raises(AnchorFetchError):
        requirement().stac.load_raster(coordinate_anchor())

    expected = ["https://primary.test/stac"]
    if stage:
        expected.append("https://primary.test/stac" + stage)
    assert visited == expected + [
        "https://backup.test/stac",
        "https://backup.test/stac/collections/sentinel-2-l2a",
        "https://backup.test/stac/search",
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
    ],
)
def test_credentials_and_malformed_documents_do_not_fall_back(catalog_http, failure):
    visited, responses = catalog_http
    responses["https://primary.test/stac"] = failure

    with pytest.raises(Exception) as caught:
        requirement().stac.load_raster(coordinate_anchor())

    assert not isinstance(caught.value, ConnectionError)
    assert visited == ["https://primary.test/stac"]


def test_malformed_collection_does_not_fall_back(catalog_http):
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
        requirement().stac.load_raster(coordinate_anchor())

    assert visited == [
        "https://primary.test/stac",
        "https://primary.test/stac/collections/sentinel-2-l2a",
    ]


def test_endpoint_exhaustion_reports_each_cause(catalog_http):
    _, responses = catalog_http
    responses["https://primary.test/stac"] = requests.ConnectionError("offline")
    responses["https://backup.test/stac/collections/sentinel-2-l2a"] = 404

    with pytest.raises(ConnectionError) as caught:
        requirement().stac.load_raster(coordinate_anchor())

    assert "https://primary.test/stac" in str(caught.value)
    assert "offline" in str(caught.value)
    assert "https://backup.test/stac" in str(caught.value)
    assert "404" in str(caught.value)
    assert isinstance(caught.value.__cause__, APIError)


def test_load_rasters_reports_all_missing_recipes_before_http(catalog_http):
    visited, _ = catalog_http
    model = ModelSpec(
        schema_version=2,
        rasters={
            "optical": RasterRequirement(channels=3),
            "elevation": RasterRequirement(channels=1),
        },
    )

    with pytest.raises(ValueError, match="optical.*elevation"):
        model.load_rasters(coordinate_anchor())

    assert visited == []


def test_require_recipes_lists_every_raster_without_a_recipe():
    model = ModelSpec(
        schema_version=2,
        rasters={
            "optical": RasterRequirement(channels=3),
            "elevation": RasterRequirement(channels=1),
        },
    )

    with pytest.raises(ValueError, match=r"\['optical', 'elevation'\]"):
        model.require_recipes()


def test_empty_search_does_not_try_another_endpoint(catalog_http):
    visited, responses = catalog_http
    responses["https://primary.test/stac/search"] = {
        "type": "FeatureCollection",
        "features": [],
        "links": [],
    }

    with pytest.raises(AnchorFetchError):
        requirement("optical").stac.load_raster(coordinate_anchor())

    assert visited[-1] == "https://primary.test/stac/search"
    assert all("backup" not in url for url in visited)


def test_lazy_asset_failure_does_not_try_another_endpoint(catalog_http, local_stac):
    visited, responses = catalog_http
    local_sources, _ = local_stac
    item = local_sources["optical"].client.items[0].to_dict()
    item["assets"]["red"]["href"] += ".missing"
    responses["https://primary.test/stac/search"] = {
        "type": "FeatureCollection",
        "features": [item],
        "links": [],
    }
    requirement = RasterRequirement.model_validate(
        {
            "variables": ["red"],
            "stac": {
                "collection": "optical",
                "endpoints": [
                    "https://primary.test/stac",
                    "https://backup.test/stac",
                ],
                "load": {"bands": ["red"], "groupby": "time"},
            },
        }
    )

    assert requirement.stac is not None
    raster = requirement.stac.load_raster(coordinate_anchor())

    with pytest.raises(RasterioIOError):
        raster.red.compute()
    assert all("backup" not in url for url in visited)
