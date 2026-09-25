"""Exercise primitive public flow inputs with real local artifacts and STAC."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from urllib.parse import parse_qs, urlparse

import numpy as np
from prefect import Flow
from prefect.settings import temporary_settings
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.testing.utilities import prefect_test_harness
import pytest

from geosave_engine.geodata.utils.io import zarr
from geosave_engine.workflow.flows import ingest
from geosave_engine.workflow.spec import ModelSpec, RasterRequirement


@pytest.fixture(scope="module", autouse=True)
def prefect_server():
    with (
        temporary_settings({"server.analytics_enabled": False}),
        prefect_test_harness(),
    ):
        yield


@contextmanager
def stac_server(sources):
    """Serve real fixture STAC documents through the native HTTP client."""
    catalogues = {name: source.client for name, source in sources.items()}
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
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
            requests.append(requested)
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
                root = f"http://127.0.0.1:{self.server.server_port}"
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
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_ingest_flow_uses_primitive_settings_with_real_stac(tmp_path, spec, local_stac):
    sources, expected_anchor = local_stac
    anchor = {
        "type": "coordinates",
        "latitude": 45.0,
        "longitude": 12.0,
        "shape": 4,
        "resolution": 10,
        "crs": "EPSG:32633",
        "timespan": "2025-01",
    }
    with stac_server(sources) as (url, requests):
        configuration = {
            name: {
                "query": {},
                "load": {"groupby": "time", "chunks": {"x": 2, "y": 2}},
            }
            for name in spec.sources
        }
        spec = ModelSpec(
            schema_version=2,
            sources={
                name: RasterRequirement(
                    **{
                        **requirement.model_dump(),
                        "collection": name,
                        "endpoints": [url],
                    }
                )
                for name, requirement in spec.sources.items()
            },
        )
        configured = ingest.with_options(
            task_runner=ThreadPoolTaskRunner(max_workers=1)
        )
        path = configured(
            configuration,
            anchor,
            spec=str(spec.save(tmp_path / "model")),
            output=str(tmp_path / "raw.zarr"),
        )
    assert type(configured) is Flow
    assert requests == [["optical"]]
    with zarr.read_stack(path) as result:
        assert result.gs.groups == ("optical",)
        assert result.gs.geobox == expected_anchor.geobox
        assert result["optical"].red.dims == ("time", "y", "x")
        np.testing.assert_allclose(result["optical"].nir[0, 1:, :], 6000)


def test_ingest_requires_model_spec(tmp_path):
    with pytest.raises(TypeError, match="spec"):
        ingest.fn({}, {}, output=str(tmp_path / "raw.zarr"))


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        ({"optical": {}, "unknown": {}}, "unknown"),
        ({"other": {}}, "optical"),
        ({}, "source|Source"),
    ],
)
def test_ingest_requires_exact_model_source_bindings(tmp_path, spec, sources, message):
    with pytest.raises(ValueError, match=message):
        ingest.fn(
            sources,
            {},
            output=str(tmp_path / "raw.zarr"),
            spec=str(spec.save(tmp_path / "model")),
        )
