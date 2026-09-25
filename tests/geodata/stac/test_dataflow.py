"""A reusable geospatial observation flow through real local assets and storage."""

import dask.array as da
import numpy as np
import pytest

from geosave_engine.geodata import read_raster
from geosave_engine.geodata.features import ndvi


def test_observations_to_prepared_index_to_reusable_dataset(local_source, tmp_path):
    source, anchor = local_source
    raw = source.load(anchor)
    reflectance = raw.gs.to_nan().gs.unpack()
    derived = ndvi(reflectance, name="ndvi", red="red", nir="nir")
    composite = derived.median("time", keep_attrs=True)
    assert isinstance(composite.ndvi.data, da.Array)
    assert composite.gs.geobox == anchor.geobox
    path = composite.gs.to_zarr(tmp_path / "ndvi.zarr")

    with read_raster(path, chunks={}) as reopened:
        assert isinstance(reopened.ndvi.data, da.Array)
        assert reopened.gs.geobox == anchor.geobox
        assert reopened.attrs["stac_items"][0]["id"] == "scene"
        assert np.isnan(reopened.ndvi.values[0, 0])
        assert reopened.ndvi.values[1, 1] == pytest.approx((1.1 - 0.2) / (1.1 + 0.2))
        assert "scale_factor" not in reopened.ndvi.attrs
    # Preparation leaves the reusable source configuration and packed input intact.
    assert raw.red.attrs["scale_factor"] == 0.0001
    assert raw.red.values[0, 1, 1] == 2000
