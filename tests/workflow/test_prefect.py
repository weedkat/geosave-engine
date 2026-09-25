"""Exercise primitive public flow inputs with real local artifacts and STAC."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from urllib.parse import parse_qs, urlparse

import dask.array as da
import numpy as np
from prefect import Flow
from prefect.settings import temporary_settings
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.testing.utilities import prefect_test_harness
import pytest
import torch
from torch import nn

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.utils.io import zarr
from geosave_engine.ml.models.contract import ModelChain, chain_step
from geosave_engine.workflow import ingest, predict
from geosave_engine.workflow.io import write_stack
from geosave_engine.workflow.spec import (
    InferenceSpec,
    ModelSpec,
    RasterRequirement,
    TensorInputSpec,
    TilingSpec,
    TimeWindowSpec,
)


@pytest.fixture(scope="module", autouse=True)
def prefect_server():
    with (
        temporary_settings({"server.analytics_enabled": False}),
        prefect_test_harness(),
    ):
        yield


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))

    @chain_step(head=True)
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.gain


class TemporalMean(Identity):
    @chain_step(head=True)
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image.mean(dim=1) * self.gain


class Broken(Identity):
    @chain_step(head=True)
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("inference failure")


def save_model(path, spec, model_class=Identity):
    model = ModelChain(
        stages={"model": {"class_path": f"{__name__}.{model_class.__name__}"}}
    )
    model.save_pretrained(path)
    spec.save(path)
    return str(path)


def test_predict_flow_loads_artifacts_and_returns_completed_paths(tmp_path, spec, raw):
    source = write_stack(stack(raw), tmp_path / "raw.zarr")
    model = save_model(tmp_path / "model", spec)
    paths = predict(
        source, model=model, output=str(tmp_path / "predictions"), batch_size=2
    )
    assert type(predict) is Flow
    assert len(paths) == 1
    with zarr.read(paths[0]) as result:
        assert result.gs.geobox == raw["optical"].gs.geobox
        np.testing.assert_allclose(result.logits.isel(band=0), 6)
        np.testing.assert_allclose(result.logits.isel(band=1), 2)


def test_predict_accepts_named_paths_and_persists_preparation(tmp_path, spec, raw):
    source = zarr.write(raw["optical"], tmp_path / "image.zarr")
    model = save_model(tmp_path / "model", spec)
    paths = predict(
        {"optical": str(source)},
        model=model,
        output=str(tmp_path / "predictions"),
        prepared_output=str(tmp_path / "prepared.zarr"),
    )
    with zarr.read_stack(tmp_path / "prepared.zarr") as prepared:
        assert prepared.gs.groups == ("optical",)
        np.testing.assert_allclose(prepared["optical"].nir.compute(), 6)
    with zarr.read(paths[0]) as restored:
        np.testing.assert_allclose(restored.logits.isel(band=0), 6)


def test_saved_yaml_runs_temporal_model_and_persists_each_window(tmp_path, raw):
    series = raster(
        {
            "signal": da.from_array(
                np.broadcast_to(np.array([1.0, 3.0, 5.0])[:, None, None], (3, 4, 4)),
                chunks=(1, 2, 2),
            )
        },
        raw["optical"].gs.geobox,
        time=np.array(
            ["2025-01-01", "2025-01-02", "2025-01-03"], dtype="datetime64[ns]"
        ),
    )
    settings = ModelSpec(
        sources={
            "series": RasterRequirement(variables=("signal",), dims=("time", "y", "x"))
        },
        inference=InferenceSpec(
            inputs={"image": TensorInputSpec(raster="series", layout="TCHW")},
            tiling=TilingSpec(raster="series", tile_shape=(4, 4)),
            time_window=TimeWindowSpec(size=2, stride=1, tolerance="1D"),
        ),
    )
    model = save_model(tmp_path / "model", settings, TemporalMean)
    source = write_stack(stack({"series": series}), tmp_path / "series.zarr")
    paths = predict(source, model=model, output=str(tmp_path / "predictions"))
    assert len(paths) == 2
    for path, expected, day in zip(paths, (2, 4), ("01", "02"), strict=True):
        with zarr.read(path) as result:
            np.testing.assert_allclose(result.logits, expected)
            assert result.gs.geobox == series.gs.geobox
            assert result.attrs["time_coverage_start"].startswith(f"2025-01-{day}")


def test_flow_failures_propagate_model_error_and_leave_no_output(tmp_path, spec, raw):
    model = save_model(tmp_path / "model", spec, Broken)
    source = write_stack(stack(raw), tmp_path / "raw.zarr")
    with pytest.raises(RuntimeError, match="inference failure"):
        predict(source, model=model, output=str(tmp_path / "predictions"))
    assert not (tmp_path / "predictions").exists()


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
                "url": url,
                "collection": name,
                "load": {"groupby": "time", "chunks": {"x": 2, "y": 2}},
            }
            for name in sources
        }
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


def test_invalid_output_is_rejected_before_model_loading(tmp_path, spec):
    model = str(tmp_path / "model")
    spec.save(model)
    with pytest.raises(ValueError, match="local"):
        predict("missing.zarr", model=model, output="s3://bucket/result")
