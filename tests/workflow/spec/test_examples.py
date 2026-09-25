from pathlib import Path

import dask.array as da
import numpy as np
from dask.callbacks import Callback

from geosave_engine.workflow.processing import Processor
from geosave_engine.workflow.spec import ModelSpec


def test_documented_example_round_trips_and_stays_lazy(tmp_path, raw):
    for name in ("red", "nir"):
        raw["optical"][name] = (raw["optical"][name] * 1000).astype("uint16")
        raw["optical"][name].attrs.update(scale_factor=0.0001, add_offset=0.0)
    path = (
        Path(__file__).parents[3]
        / "src/geosave_engine/workflow/examples/model_spec.yaml"
    )
    spec = ModelSpec.load(path)
    restored = ModelSpec.load(spec.save(tmp_path))
    assert restored == spec
    computed = []
    with Callback(posttask=lambda *args: computed.append(True)):
        prepared = Processor(stage="preprocessing", spec=restored)(raw)
    assert computed == []
    assert list(prepared["optical"].data_vars) == ["red", "nir"]
    assert prepared["optical"].red.data is raw["optical"].red.data
    assert prepared["optical"].nir.data is raw["optical"].nir.data
    assert list(raw["optical"].data_vars) == ["red", "nir", "unused"]
    assert list(prepared["reflectance"].data_vars) == ["red", "nir"]
    assert isinstance(prepared["reflectance"].red.data, da.Array)
    assert prepared["reflectance"].gs.geobox == raw["optical"].gs.geobox
    np.testing.assert_allclose(prepared["reflectance"].nir.compute(), 0.6)
    np.testing.assert_allclose(prepared["reflectance"].red.compute(), 0.2)
    np.testing.assert_allclose(raw["optical"].nir.compute(), 6000)
    np.testing.assert_allclose(prepared["features"].ndvi.compute(), 0.5, rtol=1e-6)
    finished = Processor(spec, stage="postprocessing")(
        {"prediction": prepared["features"].ndvi}
    )
    assert list(finished["result"].data_vars) == ["prediction"]
    assert finished["result"].gs.geobox == raw["optical"].gs.geobox
