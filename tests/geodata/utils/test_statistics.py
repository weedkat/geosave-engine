import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import geosave_engine.geodata  # noqa: F401


@pytest.mark.parametrize("as_array", [False, True])
def test_statistics_compute_shared_band_sources_once(as_array):
    reads = []

    @delayed
    def pixels():
        reads.append(1)
        return np.array([[0.0, 1.0, 3.0], [-9999.0, 10.0, 20.0]])

    values = da.from_delayed(pixels(), shape=(2, 3), dtype="float64")
    source = xr.Dataset(
        {
            "red": xr.DataArray(values[0], dims="x", attrs={"nodata": 0}),
            "nir": xr.DataArray(values[1], dims="x", attrs={"nodata": -9999}),
        }
    )
    if as_array:
        # Conversion requires a spatial pair; a singleton y axis adds no pixels.
        source = source.expand_dims(y=[0]).gs.to_array()

    table = source.gs.statistics()

    assert reads == [1]
    assert list(table.index) == ["red", "nir"]
    assert table.loc["red", "mean"] == 2.0
    assert table.loc["nir", "mean"] == 15.0
    assert table.loc["nir", "valid_percent"] == pytest.approx(200 / 3)


def test_empty_dataset_statistics_keep_the_table_columns():
    table = xr.Dataset().gs.statistics()

    assert isinstance(table, pd.DataFrame)
    assert table.empty
    assert list(table.columns) == [
        "minimum",
        "maximum",
        "mean",
        "stddev",
        "valid_percent",
    ]
