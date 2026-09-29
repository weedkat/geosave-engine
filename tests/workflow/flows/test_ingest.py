from importlib import import_module

import numpy as np
from prefect import Flow
import pytest

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.model_spec import ModelSpec, RasterRequirement, StacRecipe
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


def test_ingest_writes_one_raster_stack_from_a_raster_anchor(tmp_path, stac_server):
    url, requests, anchor = stac_server
    label = raster(
        {"class": np.ones((4, 4), dtype="uint8")}, anchor.geobox
    ).assign_coords(time=np.datetime64("2025-01-15"))
    label_path = io.geotiff.write_cog(label, tmp_path / "label.tif")
    output = tmp_path / "raw.zarr"

    result = ingest.fn(
        anchor={"kind": "raster", "path": str(label_path)},
        output=str(output),
        spec=str(model_spec(tmp_path, url)),
    )

    assert isinstance(ingest, Flow)
    assert result == str(output)
    assert requests == [["optical"]]
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

    def write(rasters, output):
        assert rasters == {"optical": "red", "elevation": "dem"}
        return str(output)

    monkeypatch.setattr(ModelSpec, "load_rasters", load)
    monkeypatch.setattr(ingest_module, "write_stack", write)

    result = ingest.fn(
        anchor={
            "kind": "coordinates",
            "latitude": 45,
            "longitude": 12,
            "shape": 4,
            "resolution": 10,
        },
        output=str(tmp_path / "raw.zarr"),
        spec=str(spec),
    )

    assert result == str(tmp_path / "raw.zarr")
    assert captured == {"red": "optical", "dem": "dem"}


def test_ingest_requires_all_stac_recipes_before_opening_anchor(tmp_path):
    spec = ModelSpec(
        schema_version=2,
        rasters={
            "optical": RasterRequirement(variables=("red",)),
            "elevation": RasterRequirement(variables=("dem",)),
        },
    ).save(tmp_path / "model_spec.yaml")

    with pytest.raises(ValueError, match="optical.*elevation"):
        ingest.fn(
            anchor={"kind": "raster", "path": tmp_path / "missing.tif"},
            output=str(tmp_path / "raw.zarr"),
            spec=str(spec),
        )
