import numpy as np
import xarray as xr

from geosave_engine.geodata.features import ndvi


def test_feature_returns_a_named_raster_on_the_input_grid() -> None:
    source = xr.Dataset(
        {
            "nir": (("y", "x"), [[3.0]]),
            "red": (("y", "x"), [[1.0]]),
        }
    )

    result = ndvi(source, name="vegetation", nir="nir", red="red")

    assert isinstance(result, xr.Dataset)
    assert list(result.data_vars) == ["vegetation"]
    np.testing.assert_allclose(result.vegetation.values, [[0.5]], rtol=1e-5)
