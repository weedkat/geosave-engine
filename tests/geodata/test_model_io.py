from __future__ import annotations

import numpy as np
import pytest
import torch
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.core.raster import raster

UTM = "EPSG:32633"


def geobox(shape: tuple[int, int] = (8, 8)) -> GeoBox:
    """Build one grid for a model-input test.

    Args:
        shape: Grid height and width in pixels.

    Returns:
        Grid of `shape` at ten-metre pixels.
    """
    left, bottom = 300_000.0, 5_000_000.0
    return GeoBox.from_bbox(
        (left, bottom, left + shape[1] * 10, bottom + shape[0] * 10),
        crs=UTM,
        shape=shape,
        tight=True,
    )


def optical(shape: tuple[int, int] = (8, 8), *, times: int = 2) -> xr.Dataset:
    """Build one placed raster carrying two bands over a time axis.

    Args:
        shape: Grid height and width in pixels.
        times: Length of the time axis.

    Returns:
        Raster holding `B04` and `B08` as uint16, each variable filled with its
        own constant so band order is visible in the stacked array.
    """
    box = geobox(shape)
    labels = np.array(
        [f"2024-0{month + 1}-01" for month in range(times)], dtype="datetime64[ns]"
    )
    return raster(
        {
            "B04": np.full((times, *shape), 4, "uint16"),
            "B08": np.full((times, *shape), 8, "uint16"),
        },
        box,
        time=labels,
    ).gs.write_nodata(0)


def test_to_numpy_stacks_variables_ahead_of_the_other_axes() -> None:
    stacked = optical().gs.to_numpy()

    assert stacked.shape == (2, 2, 8, 8)
    assert stacked.dtype == np.dtype("uint16")


def test_to_numpy_stacks_in_the_order_xarray_holds_the_variables() -> None:
    stacked = optical()[["B08", "B04"]].gs.to_numpy()

    assert stacked[0, :, 0, 0].tolist() == [8, 4]


def test_to_numpy_reads_an_unplaced_raster() -> None:
    unplaced = raster({"pixels": np.zeros((4, 4), "uint8")})

    assert unplaced.gs.to_numpy().shape == (1, 4, 4)


def test_to_numpy_refuses_a_raster_carrying_no_variables() -> None:
    with pytest.raises(ValueError, match="no variables to stack"):
        optical()[[]].gs.to_numpy()


def test_to_numpy_refuses_variables_on_different_axes() -> None:
    source = optical()
    timeless = raster({"dem": np.zeros((8, 8), "uint16")}, geobox())
    mixed = source.assign(dem=timeless.dem)

    with pytest.raises(ValueError, match="different axes"):
        mixed.gs.to_numpy()


def test_to_numpy_refuses_variables_on_different_dtypes() -> None:
    source = optical()
    mixed = source.assign(ndvi=source.B04.astype("float32"))

    with pytest.raises(ValueError, match="different dtypes"):
        mixed.gs.to_numpy()


def test_to_numpy_casts_variables_onto_one_dtype() -> None:
    source = optical()
    mixed = source.assign(ndvi=source.B04.astype("float32"))

    assert mixed.gs.to_numpy(dtype="float32").dtype == np.dtype("float32")


def test_to_numpy_refuses_a_raster_already_carrying_the_stacking_axis() -> None:
    conflicting = raster({"pixels": np.zeros((3, 4, 4), "uint8")}, None, band=None)

    with pytest.raises(ValueError, match="already carries a 'band' dimension"):
        conflicting.gs.to_numpy()


def test_to_tensor_casts_to_float32_by_default() -> None:
    tensor = optical().gs.to_tensor()

    assert tensor.dtype is torch.float32
    assert tuple(tensor.shape) == (2, 2, 8, 8)


def test_to_tensor_honours_a_requested_dtype() -> None:
    assert optical().gs.to_tensor(dtype=torch.int16).dtype is torch.int16


def test_a_band_reads_with_the_grid_trailing() -> None:
    scrambled = xr.DataArray(
        np.zeros((8, 8, 3), "uint16"),
        dims=("y", "x", "band"),
        coords={"band": ["r", "g", "b"]},
    ).odc.assign_crs(UTM)

    assert scrambled.gs.to_numpy().shape == (3, 8, 8)
    assert scrambled.gs.to_tensor().shape == (3, 8, 8)


def test_a_band_keeps_its_other_axes_ahead_of_the_bands() -> None:
    cube = xr.DataArray(
        np.zeros((3, 2, 8, 8), "uint16"),
        dims=("band", "time", "y", "x"),
        coords={"band": ["r", "g", "b"], "time": [0, 1]},
    ).odc.assign_crs(UTM)

    assert cube.gs.to_numpy().shape == (2, 3, 8, 8)


def test_a_band_carrying_no_band_axis_reads_without_one() -> None:
    band = optical()["B04"]

    assert band.gs.to_numpy().shape == (2, 8, 8)
    assert optical()[["B04"]].gs.to_numpy().shape == (2, 1, 8, 8)


def test_a_band_casts_the_way_a_raster_does() -> None:
    band = optical()["B04"]

    assert band.gs.to_numpy(dtype="float32").dtype == np.float32
    assert band.gs.to_tensor().dtype is torch.float32
    assert band.gs.to_tensor(dtype=torch.bfloat16).dtype is torch.bfloat16
