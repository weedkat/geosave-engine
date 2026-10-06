import inspect
from importlib import import_module

import numpy as np
from prefect import Flow
import pytest

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata import io
from geosave_engine.model.spec import ModelSpec, RasterRequirement, StacRecipe
from geosave_engine.workflow.configs import CoordinateAnchorConfig, RasterAnchorConfig
from geosave_engine.workflow.flows import ingest

ingest_module = import_module("geosave_engine.workflow.flows.ingest")


def model_spec(tmp_path, url):
    return ModelSpec(
        schema_version=2,
        rasters={
            "optical": RasterRequirement(
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
    ).save(tmp_path / "model_spec.yaml")


def test_ingest_writes_one_sample_folder_from_a_raster_anchor(tmp_path, stac_server):
    url, requests, anchor = stac_server
    label = raster(
        {"class": np.ones((4, 4), dtype="uint8")}, anchor.geobox
    ).assign_coords(time=np.datetime64("2025-01-15"))
    label_path = io.geotiff.write_cog(label, tmp_path / "label.tif")
    output = tmp_path / "raw"

    result = ingest.fn(
        anchor=RasterAnchorConfig(kind="raster", path=str(label_path)),
        output=str(output),
        spec=str(model_spec(tmp_path, url)),
    )

    assert isinstance(ingest, Flow)
    assert result == str(output)
    assert requests == [["optical"]]
    assert sorted(path.name for path in output.iterdir()) == ["optical.zarr"]
    with io.read_stack(output, chunks="auto") as stack:
        assert stack.gs.groups == ("optical",)
        assert stack.gs.geobox == anchor.geobox


def test_ingest_loads_every_model_raster_recipe(tmp_path, monkeypatch) -> None:
    spec = ModelSpec(
        schema_version=2,
        rasters={
            "optical": RasterRequirement(
                variables=("red",),
                stac=StacRecipe.model_validate(
                    {"collection": "optical", "endpoints": ("https://stac.test",)}
                ),
            ),
            "elevation": RasterRequirement(
                variables=("dem",),
                stac=StacRecipe.model_validate(
                    {"collection": "dem", "endpoints": ("https://stac.test",)}
                ),
            ),
        },
    ).save(tmp_path / "model_spec.yaml")
    captured = {}

    def load(model, anchor):
        for requirement in model.rasters.values():
            captured[requirement.variables[0]] = requirement.stac.collection
        return {"optical": "red", "elevation": "dem"}

    def write(rasters, output, **options):
        assert rasters == {"optical": "red", "elevation": "dem"}
        return str(output)

    monkeypatch.setattr(ModelSpec, "load_rasters", load)
    monkeypatch.setattr(ingest_module, "write_sample", write)

    result = ingest.fn(
        anchor=CoordinateAnchorConfig(
            kind="coordinates", latitude=45, longitude=12, shape=4, resolution=10
        ),
        output=str(tmp_path / "raw"),
        spec=str(spec),
    )

    assert result == str(tmp_path / "raw")
    assert captured == {"red": "optical", "dem": "dem"}


def test_ingest_refuses_rasters_without_a_stac_block(tmp_path):
    spec = ModelSpec(
        schema_version=2,
        rasters={
            "optical": RasterRequirement(variables=("red",)),
            "elevation": RasterRequirement(variables=("dem",)),
        },
    ).save(tmp_path / "model_spec.yaml")

    with pytest.raises(
        ValueError, match=r"no `stac` block: \['optical', 'elevation'\]"
    ):
        ingest.fn(
            anchor=CoordinateAnchorConfig(
                kind="coordinates", latitude=45, longitude=12, shape=4, resolution=10
            ),
            output=str(tmp_path / "raw"),
            spec=str(spec),
        )

    assert not (tmp_path / "raw").exists()


@pytest.mark.slow
def test_ingest_run_opens_a_json_anchor(tmp_path, prefect_server, monkeypatch) -> None:
    spec = ModelSpec(
        schema_version=2,
        rasters={
            "optical": RasterRequirement(
                variables=("red",),
                stac=StacRecipe.model_validate(
                    {"collection": "optical", "endpoints": ("https://stac.test",)}
                ),
            )
        },
    ).save(tmp_path / "model_spec.yaml")
    shapes = []

    def load(model, anchor):
        shapes.append(tuple(anchor.geobox.shape))
        return {"optical": "red"}

    monkeypatch.setattr(ModelSpec, "load_rasters", load)
    monkeypatch.setattr(
        ingest_module, "write_sample", lambda rasters, output, **options: output
    )

    result = ingest(
        anchor={
            "kind": "coordinates",
            "latitude": 45,
            "longitude": 12,
            "shape": 4,
            "resolution": 10,
        },
        output=str(tmp_path / "raw"),
        spec=str(spec),
    )

    assert result == str(tmp_path / "raw")
    assert shapes == [(4, 4)]


def test_ingest_refuses_an_existing_output(tmp_path, stac_server):
    url, requests, anchor = stac_server
    label = raster(
        {"class": np.ones((4, 4), dtype="uint8")}, anchor.geobox
    ).assign_coords(time=np.datetime64("2025-01-15"))
    label_path = io.geotiff.write_cog(label, tmp_path / "label.tif")
    (tmp_path / "raw").mkdir()

    with pytest.raises(FileExistsError):
        ingest.fn(
            anchor=RasterAnchorConfig(kind="raster", path=str(label_path)),
            output=str(tmp_path / "raw"),
            spec=str(model_spec(tmp_path, url)),
        )


def test_ingest_takes_no_catalog_argument() -> None:
    assert "catalog" not in inspect.signature(ingest.fn).parameters
