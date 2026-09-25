import dask.array as da
from dask.callbacks import Callback
import numpy as np

from geosave_engine.workflow.preprocessing import preprocess
from geosave_engine.workflow.spec import ModelSpec
from geosave_engine.workflow.examples import reflectance


def test_documented_example_round_trips_and_stays_lazy(tmp_path):
    spec = reflectance.make_spec()
    restored = ModelSpec.load(spec.save(tmp_path))
    assert restored == spec
    raw = reflectance.sample_rasters()
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        prepared = preprocess(raw, spec=restored)
    assert tasks == []
    assert set(prepared.children) == {"optical", "reflectance", "terrain"}
    result = prepared["reflectance"].to_dataset()
    assert list(result.data_vars) == ["nir", "red"]
    assert isinstance(result.nir.data, da.Array)
    assert result.gs.geobox == raw["optical"].gs.geobox
    np.testing.assert_allclose(result.nir.compute(), 0.6)
    np.testing.assert_allclose(result.red.compute(), 0.2)
    assert "scale_factor" not in result.nir.attrs
    assert raw["optical"].nir.attrs["scale_factor"] == 0.0001
    np.testing.assert_allclose(prepared["terrain"].height.compute(), 30)
