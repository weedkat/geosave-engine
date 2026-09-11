import numpy as np
import xarray as xr

from geosave_engine.geodata.features import compute_ndvi


def test_feature_returns_a_dataarray_on_the_input_grid() -> None:
    nir = xr.DataArray([[3.0]], dims=("y", "x"))
    red = xr.DataArray([[1.0]], dims=("y", "x"))

    result = compute_ndvi(nir=nir, red=red)

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("y", "x")
    np.testing.assert_allclose(result.values, [[0.5]], rtol=1e-5)
