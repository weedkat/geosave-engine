import dask.array as da
import numpy as np
from tests.geodata.conftest import build_raster
import pytest
from dask.callbacks import Callback

from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.transform.chip import crop
from tests.geodata.transform.test_vector import _band


@pytest.mark.parametrize("kind", ["array", "dataset", "tree"])
def test_pixel_window_preserves_native_type_time_and_grid_lazily(kind):
    band = _band(times=2).chunk().assign_attrs(description="reflectance")
    source = band if kind == "array" else band.to_dataset()
    if kind == "tree":
        source = stack({"image": source})

    with Callback(pretask=lambda *_: pytest.fail("window computed pixels")):
        result = crop(source, (2, 2), (3, 3), mode="edge")

    assert type(result) is type(source)
    if kind == "tree":
        output = result.gs.rasters["image"].red
    elif kind == "dataset":
        output = result.red
    else:
        output = result
    assert output.shape == (2, 3, 3)
    assert isinstance(output.data, da.Array)
    assert output.attrs["description"] == "reflectance"
    np.testing.assert_array_equal(output.time, band.time)
    assert output.gs.geobox == band.gs.geobox.translate_pix(2, 2).crop((3, 3))
    np.testing.assert_array_equal(output.values, np.ones((2, 3, 3)))


def test_a_window_past_the_edge_is_filled_as_asked():
    data = build_raster(times=1).chunk()

    actual = crop(data, (0, 1), (1, 1))

    assert actual.sizes["y"] == actual.sizes["x"] == 1
    assert actual.gs.geobox == data.gs.geobox.translate_pix(1, 0).crop((1, 1))
    assert actual.red.chunks is not None
