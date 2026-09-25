import numpy as np
import pytest
import xarray as xr

from geosave_engine.geodata.transform.merge import merge_bands

from .conftest import build, geobox


def test_merge_preserves_band_order_values_and_metadata():
    optical = build(geobox(), labelled=True)
    radar = build(geobox()).rename_vars({"red": "VV"})
    radar["VV"].values[:] = 300
    result = merge_bands([radar, optical])
    assert tuple(result.data_vars) == ("VV", "red", "cls")
    np.testing.assert_array_equal(result.red, optical.red)
    np.testing.assert_array_equal(result.VV, radar.VV)
    assert result.cls.attrs == optical.cls.attrs
    assert result.red.attrs == optical.red.attrs
    assert result.gs.geobox == optical.gs.geobox


@pytest.mark.parametrize(
    "kind", ["grid", "crs", "time", "scalar_time", "timeless", "collision", "dims"]
)
def test_merge_rejects_implicit_alignment_or_broadcasting(kind):
    first = build(geobox())
    second = first.rename_vars({"red": "nir"})
    if kind == "grid":
        second = second.assign_coords(x=second.x + 5)
    elif kind == "crs":
        second = build(geobox(crs="EPSG:32632")).rename_vars({"red": "nir"})
    elif kind == "time":
        second = build(geobox(), start="2025-01-01").rename_vars({"red": "nir"})
    elif kind == "scalar_time":
        first, second = first.isel(time=0), second.isel(time=1)
    elif kind == "timeless":
        second = second.isel(time=0, drop=True)
    elif kind == "collision":
        second = first
    else:
        second = second.transpose("y", "x", "time")
    with pytest.raises(ValueError):
        merge_bands([first, second])


@pytest.mark.parametrize("rasters", [[], [xr.Dataset()]])
def test_merge_requires_nonempty_rasters(rasters):
    with pytest.raises(ValueError, match="non-empty"):
        merge_bands(rasters)
