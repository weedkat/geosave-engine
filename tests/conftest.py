from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import dask.array as da
import numpy as np
from odc.geo.geobox import GeoBox
from prefect.settings import temporary_settings
from prefect.testing.utilities import prefect_test_harness
import pytest
import pystac
from pystac.extensions.projection import ProjectionExtension
import rasterio
import requests
from dotenv import load_dotenv

from geosave_engine.geodata.core.anchor import GeoAnchor
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.stac.query import StacQuery
from geosave_engine.geodata.stac.source import StacSource
from geosave_engine.model.spec import ModelSpec, RasterRequirement


def pytest_configure(config):
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)


@pytest.fixture
def dw_tif_path() -> Path:
    """Path to the DynamicWorld anchor used as a real test fixture."""
    return Path(__file__).parent / "data/dw_-22.7491991582_15.9791703445-20190223.tif"


@pytest.fixture(scope="module")
def prefect_server():
    with (
        temporary_settings({"server.analytics_enabled": False}),
        prefect_test_harness(),
    ):
        yield


@pytest.fixture
def spec():
    return ModelSpec(
        schema_version=2,
        rasters={
            "optical": RasterRequirement(variables=("nir", "red"), require_crs=True)
        },
    )


@pytest.fixture
def raw():
    box = GeoBox.from_bbox((0, 0, 40, 40), crs="EPSG:32633", resolution=10)
    return {
        "optical": raster(
            {
                name: da.full((4, 4), value, chunks=(2, 2))
                for name, value in (("red", 2.0), ("nir", 6.0), ("unused", 0.0))
            },
            box,
        )
    }


@pytest.fixture
def anchor(raw):
    return GeoAnchor(geobox=raw["optical"].gs.geobox)


@pytest.fixture
def source():
    # These unit tests replace loading; constructing a source performs no I/O.
    class EmptyCatalog:
        def search(self, query: object) -> list[pystac.Item]:
            return []

        def collection(self, collection: str) -> pystac.Collection:
            raise KeyError(collection)

    return StacSource(client=EmptyCatalog(), collection="example").set_config(
        bands=("unused",), chunks={"x": 2, "y": 2}, item_properties=("platform",)
    )


@dataclass
class LocalCatalog:
    """Replace only catalog search; assets and the raster loading path are real."""

    metadata: pystac.Collection
    items: list[pystac.Item]
    requests: list[StacQuery] = field(default_factory=list)

    def search(self, query):
        self.requests.append(query)
        return self.items if query.collections == [self.metadata.id] else []

    def collection(self, collection):
        assert collection == self.metadata.id
        return self.metadata


@pytest.fixture
def local_stac(tmp_path):
    anchor = GeoAnchor.from_coordinates(
        45.0,
        12.0,
        shape=4,
        crs="EPSG:32633",
        resolution=10,
        timespan="2025-01",
    )
    box = anchor.geobox
    bounds = list(box.geographic_extent.boundingbox)
    timestamp = datetime(2025, 1, 15, 12, tzinfo=timezone.utc)
    sources = {}
    for name, bands in (
        ("optical", {"red": 2000, "nir": 6000, "unused": 1}),
        ("terrain", {"height": 30}),
    ):
        collection = pystac.Collection(
            id=name,
            description="Local raster smoke test",
            license="CC-BY-4.0",
            extent=pystac.Extent(
                pystac.SpatialExtent([bounds]),
                pystac.TemporalExtent([[timestamp, timestamp]]),
            ),
        )
        item = pystac.Item(
            id=f"{name}-scene",
            geometry=box.geographic_extent.json,
            bbox=bounds,
            datetime=timestamp,
            properties={"platform": "local-sample"},
            collection=name,
        )
        ProjectionExtension.ext(item, add_if_missing=True).apply(
            code="EPSG:32633",
            shape=list(box.shape),
            transform=list(box.transform),
        )
        for band, value in bands.items():
            path = tmp_path / f"{name}-{band}.tif"
            pixels = np.full(tuple(box.shape), value, dtype="uint16")
            pixels[0, 0] = 0
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=box.width,
                height=box.height,
                count=1,
                dtype="uint16",
                crs="EPSG:32633",
                transform=box.transform,
                nodata=0,
            ) as destination:
                destination.write(pixels, 1)
            band_metadata = {"data_type": "uint16", "nodata": 0, "unit": "m"}
            if name == "optical":
                band_metadata.update(scale=0.0001, offset=0, unit="1")
            item.add_asset(
                band,
                pystac.Asset(
                    href=str(path),
                    media_type=pystac.MediaType.GEOTIFF,
                    roles=["data"],
                    extra_fields={"raster:bands": [band_metadata]},
                ),
            )
        sources[name] = (
            StacSource(LocalCatalog(collection, [item]), collection=name)
            .set_config(
                bands=("unused",) if name == "optical" else ("height",),
                groupby="time",
                chunks={"x": 2, "y": 2},
                item_properties=("platform",),
            )
            .set_query(
                datetime="2025-01-15", max_items=1, filter="eo:cloud_cover <= 10"
            )
        )
    return sources, anchor


@pytest.fixture
def stac_server(local_stac):
    """Serve fixture STAC documents while retaining real local raster assets."""
    sources, anchor = local_stac
    catalogues = {name: source.client for name, source in sources.items()}
    queries = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def reply(self, data):
            payload = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def search(self, params):
            requested = params.get("collections", list(catalogues))
            if isinstance(requested, str):
                requested = requested.split(",")
            queries.append(requested)
            self.reply(
                {
                    "type": "FeatureCollection",
                    "features": [
                        item.to_dict()
                        for name in requested
                        for item in catalogues[name].items
                    ],
                    "links": [],
                }
            )

        def do_POST(self):
            self.search(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/search":
                self.search(
                    {key: values[0] for key, values in parse_qs(parsed.query).items()}
                )
            elif parsed.path.startswith("/collections/"):
                self.reply(catalogues[parsed.path.rsplit("/", 1)[1]].metadata.to_dict())
            elif parsed.path == "/collections":
                self.reply(
                    {
                        "collections": [
                            entry.metadata.to_dict() for entry in catalogues.values()
                        ],
                        "links": [],
                    }
                )
            else:
                server = cast(ThreadingHTTPServer, self.server)
                root = f"http://127.0.0.1:{server.server_port}"
                self.reply(
                    {
                        "type": "Catalog",
                        "stac_version": "1.0.0",
                        "id": "local",
                        "description": "Local test catalog",
                        "conformsTo": [
                            f"https://api.stacspec.org/v1.0.0/{kind}"
                            for kind in ("core", "item-search", "collections")
                        ],
                        "links": [
                            {"rel": "self", "href": root},
                            {"rel": "root", "href": root},
                            {
                                "rel": "search",
                                "href": root + "/search",
                                "method": "POST",
                                "type": "application/geo+json",
                            },
                            {"rel": "data", "href": root + "/collections"},
                        ],
                    }
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", queries, anchor
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def catalog_http(monkeypatch):
    """Keep native STAC parsing and replace only HTTP transport."""
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
