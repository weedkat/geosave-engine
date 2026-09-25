from prefect import Flow
import pytest

from geosave_engine.geodata.utils import io
from geosave_engine.workflow.flows import ingest
from geosave_engine.workflow.specs import ModelSpec, RasterRequirement
from geosave_engine.workflow.tasks import load_raster


@pytest.mark.slow
def test_ingest_validates_primitives_and_writes_completed_stack(
    tmp_path, stac_server, prefect_server
):
    url, requests, expected_anchor = stac_server
    model_spec = ModelSpec(
        schema_version=2,
        sources={
            "optical": RasterRequirement(
                variables=("red", "nir"),
                collection="optical",
                endpoints=(url,),
                require_crs=True,
            )
        },
    ).save(tmp_path / "model_spec.yaml")
    output = tmp_path / "raw.zarr"

    result = ingest(
        sources={"optical": {"query": {}, "load": {}}},
        anchor={
            "kind": "coordinates",
            "latitude": 45,
            "longitude": 12,
            "shape": 4,
            "resolution": 10,
            "crs": "EPSG:32633",
            "timespan": "2025-01",
        },
        output=str(output),
        spec=str(model_spec),
    )

    assert isinstance(ingest, Flow)
    assert result == str(output)
    assert requests == [["optical"]]
    with io.read_stack(output, chunks="auto") as tree:
        assert list(tree.gs.rasters) == ["optical"]
        assert tree.gs.geobox == expected_anchor.geobox


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        ({"extra": {}}, "Unknown source bindings"),
        ({}, "Source bindings are missing"),
    ],
)
def test_ingest_rejects_source_names_before_task_submission(
    tmp_path, monkeypatch, sources, message
):
    def unexpected_submission(*args, **kwargs):
        raise AssertionError("load task was submitted before source validation")

    monkeypatch.setattr(load_raster, "submit", unexpected_submission)
    model_spec = ModelSpec(
        schema_version=2,
        sources={"optical": RasterRequirement(variables=("red", "nir"))},
    ).save(tmp_path / "model_spec.yaml")

    with pytest.raises(ValueError, match=message):
        ingest.fn(
            sources=sources,
            anchor={
                "kind": "coordinates",
                "latitude": 45,
                "longitude": 12,
                "shape": 4,
                "resolution": 10,
            },
            output=str(tmp_path / "raw.zarr"),
            spec=str(model_spec),
        )


def test_ingest_validates_nested_source_parameters_before_loading_spec(tmp_path):
    with pytest.raises(ValueError, match="unexpected"):
        ingest.fn(
            sources={"optical": {"query": {"unexpected": True}}},
            anchor={
                "kind": "coordinates",
                "latitude": 45,
                "longitude": 12,
                "shape": 4,
                "resolution": 10,
            },
            output=str(tmp_path / "raw.zarr"),
            spec=str(tmp_path / "missing.yaml"),
        )


@pytest.mark.slow
def test_ingest_propagates_source_task_failures(tmp_path, prefect_server):
    model_spec = ModelSpec(
        schema_version=2,
        sources={"optical": RasterRequirement(variables=("red", "nir"))},
    ).save(tmp_path / "model_spec.yaml")

    with pytest.raises(ValueError, match="collection and endpoints"):
        ingest(
            sources={"optical": {}},
            anchor={
                "kind": "coordinates",
                "latitude": 45,
                "longitude": 12,
                "shape": 4,
                "resolution": 10,
            },
            output=str(tmp_path / "raw.zarr"),
            spec=str(model_spec),
        )
