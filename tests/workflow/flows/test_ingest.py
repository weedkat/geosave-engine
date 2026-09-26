from uuid import uuid4

import numpy as np
from prefect import Flow
from prefect.concurrency.asyncio import ConcurrencySlotAcquisitionError
import pytest

from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.utils import io
from geosave_engine.workflow.flows import ingest
from geosave_engine.workflow.specs import ModelSpec, RasterRequirement


def model_spec(tmp_path, url):
    return ModelSpec(
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


def test_ingest_writes_one_source_stack_from_a_raster_anchor(
    tmp_path, stac_server
):
    url, requests, anchor = stac_server
    label = raster(
        {"class": np.ones((4, 4), dtype="uint8")}, anchor.geobox
    ).assign_coords(time=np.datetime64("2025-01-15"))
    label_path = io.geotiff.write_cog(label, tmp_path / "label.tif")
    output = tmp_path / "raw.zarr"

    result = ingest.fn(
        sources={"optical": {"query": {}, "load": {}}},
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


@pytest.mark.slow
def test_ingest_requires_configured_source_limit(
    tmp_path, stac_server, prefect_server
):
    url, requests, anchor = stac_server
    label = raster(
        {"class": np.ones((4, 4), dtype="uint8")}, anchor.geobox
    ).assign_coords(time=np.datetime64("2025-01-15"))
    label_path = io.geotiff.write_cog(label, tmp_path / "label.tif")
    output = tmp_path / "raw.zarr"

    with pytest.raises(ConcurrencySlotAcquisitionError):
        ingest.fn(
            sources={
                "optical": {
                    "concurrency": f"missing-{uuid4()}",
                    "query": {},
                    "load": {},
                }
            },
            anchor={"kind": "raster", "path": str(label_path)},
            output=str(output),
            spec=str(model_spec(tmp_path, url)),
        )

    assert requests == []
    assert not output.exists()


@pytest.mark.parametrize("sources", [{}, {"optical": {}, "extra": {}}])
def test_ingest_requires_exact_source_bindings(tmp_path, sources):
    spec = ModelSpec(
        schema_version=2,
        sources={"optical": RasterRequirement(variables=("red",))},
    ).save(tmp_path / "model_spec.yaml")

    with pytest.raises(ValueError, match="must match model sources"):
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
            spec=str(spec),
        )


def test_ingest_validates_source_settings_before_opening_other_inputs(tmp_path):
    with pytest.raises(ValueError, match="unexpected"):
        ingest.fn(
            sources={"optical": {"query": {"unexpected": True}}},
            anchor={"kind": "raster", "path": tmp_path / "missing.tif"},
            output=str(tmp_path / "raw.zarr"),
            spec=str(tmp_path / "missing.yaml"),
        )
