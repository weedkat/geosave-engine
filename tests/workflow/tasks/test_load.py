import pytest
from pystac_client.exceptions import APIError
import requests
from rasterio.errors import RasterioIOError

import geosave_engine.workflow.tasks.load as load_module
from geosave_engine.geodata.core import GeoAnchor
from geosave_engine.geodata.errors import AnchorFetchError
from geosave_engine.workflow.configs import SourceConfig
from geosave_engine.workflow.specs import RasterRequirement
from geosave_engine.workflow.tasks import load_raster


@pytest.fixture(autouse=True)
def clear_client_cache():
    load_module._open_client.cache_clear()


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
            "collection": collection,
            "endpoints": [
                "https://primary.test/stac",
                "https://backup.test/stac",
            ],
        }
    )


def test_load_raster_reads_and_validates_one_local_source(stac_server):
    url, requests, expected_anchor = stac_server

    result = load_raster(
        expected_anchor,
        SourceConfig(),
        RasterRequirement(
            variables=("red", "nir"),
            collection="optical",
            endpoints=(url,),
            require_crs=True,
        ),
    )

    assert list(result.data_vars) == ["red", "nir"]
    assert result.odc.geobox == expected_anchor.geobox
    assert result.red.chunks is not None
    assert requests == [["optical"]]


def test_load_raster_reuses_open_client(catalog_http):
    visited, responses = catalog_http
    responses["https://primary.test/stac/search"] = {
        "type": "FeatureCollection",
        "features": [],
        "links": [],
    }

    for _ in range(2):
        with pytest.raises(AnchorFetchError):
            load_raster(coordinate_anchor(), SourceConfig(), requirement())

    assert visited == [
        "https://primary.test/stac",
        "https://primary.test/stac/collections/sentinel-2-l2a",
        "https://primary.test/stac/search",
        "https://primary.test/stac/search",
    ]


def test_load_module_has_no_concurrency_configuration() -> None:
    assert not hasattr(load_module, "source_concurrency")


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
        load_raster(coordinate_anchor(), SourceConfig(), requirement())

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
        load_raster(coordinate_anchor(), SourceConfig(), requirement())

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
        load_raster(coordinate_anchor(), SourceConfig(), requirement())

    assert visited == [
        "https://primary.test/stac",
        "https://primary.test/stac/collections/sentinel-2-l2a",
    ]


def test_endpoint_exhaustion_reports_each_cause(catalog_http):
    _, responses = catalog_http
    responses["https://primary.test/stac"] = requests.ConnectionError("offline")
    responses["https://backup.test/stac/collections/sentinel-2-l2a"] = 404

    with pytest.raises(ConnectionError) as caught:
        load_raster(coordinate_anchor(), SourceConfig(), requirement())

    assert "https://primary.test/stac" in str(caught.value)
    assert "offline" in str(caught.value)
    assert "https://backup.test/stac" in str(caught.value)
    assert "404" in str(caught.value)
    assert isinstance(caught.value.__cause__, APIError)


def test_model_source_without_catalog_identity_fails_before_http(catalog_http):
    visited, _ = catalog_http

    with pytest.raises(ValueError, match="collection and endpoints"):
        load_raster(
            coordinate_anchor(), SourceConfig(), RasterRequirement(channels=3)
        )

    assert visited == []


def test_empty_search_does_not_try_another_endpoint(catalog_http):
    visited, responses = catalog_http
    responses["https://primary.test/stac/search"] = {
        "type": "FeatureCollection",
        "features": [],
        "links": [],
    }

    with pytest.raises(AnchorFetchError):
        load_raster(
            coordinate_anchor(), SourceConfig(), requirement("optical")
        )

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
            "collection": "optical",
            "endpoints": [
                "https://primary.test/stac",
                "https://backup.test/stac",
            ],
        }
    )

    raster = load_raster(
        coordinate_anchor(),
        SourceConfig.model_validate({"load": {"groupby": "time"}}),
        requirement,
    )

    with pytest.raises(RasterioIOError):
        raster.red.compute()
    assert all("backup" not in url for url in visited)
