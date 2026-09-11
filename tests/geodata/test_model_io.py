from __future__ import annotations

import numpy as np
import pytest
import torch
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.attrs import Tiling
from geosave_engine.geodata.core.raster import raster
from geosave_engine.geodata.core.stack import stack
from geosave_engine.geodata.core.stitcher import GeoStitcher

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


def test_to_numpy_follows_the_requested_variable_order() -> None:
    stacked = optical().gs.to_numpy(("B08", "B04"))

    assert stacked[0, :, 0, 0].tolist() == [8, 4]


def test_to_numpy_reads_an_unplaced_raster() -> None:
    unplaced = raster({"pixels": np.zeros((4, 4), "uint8")})

    assert unplaced.gs.to_numpy().shape == (1, 4, 4)


def test_to_numpy_refuses_an_absent_variable() -> None:
    with pytest.raises(KeyError, match="B99"):
        optical().gs.to_numpy(("B04", "B99"))


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


def test_stack_to_numpy_keys_arrays_by_group() -> None:
    scene = stack(
        {
            "optical": optical(),
            "dem": raster({"elevation": np.zeros((8, 8), "int16")}, geobox()),
        }
    )

    arrays = scene.gs.to_numpy()

    assert list(arrays) == ["optical", "dem"]
    assert arrays["optical"].shape == (2, 2, 8, 8)
    assert arrays["dem"].shape == (1, 8, 8)


def test_stack_to_numpy_drops_the_groups_left_out() -> None:
    scene = stack(
        {
            "optical": optical(),
            "dem": raster({"elevation": np.zeros((8, 8), "int16")}, geobox()),
        }
    )

    arrays = scene.gs.to_numpy({"optical": ("B04",)})

    assert list(arrays) == ["optical"]
    assert arrays["optical"].shape == (2, 1, 8, 8)


def test_stack_to_numpy_refuses_an_absent_group() -> None:
    scene = stack({"optical": optical()})

    with pytest.raises(KeyError, match="radar"):
        scene.gs.to_numpy({"radar": None})


def test_stack_to_tensor_casts_each_group_on_its_own_dtype() -> None:
    scene = stack(
        {
            "optical": optical(),
            "label": raster(
                {"class": np.ones((8, 8), "uint8")}, geobox()
            ).gs.write_nodata(255),
        }
    )

    batch = scene.gs.to_tensor(dtype={"label": torch.int64})

    assert batch["optical"].dtype is torch.float32
    assert batch["label"].dtype is torch.int64


def test_a_prediction_stitches_back_onto_the_grid_it_was_read_from() -> None:
    source = optical((16, 16))
    tiles = source.gs.tile((8, 8), group_id="scene-001")
    stitcher = GeoStitcher()

    for tile in tiles:
        model_input = tile.gs.to_tensor(("B04", "B08"))
        predicted = model_input.mean(dim=(0, 1)).numpy().astype("uint8")
        prediction = (
            raster({"class": predicted}, tile.gs.geobox)
            .gs.write_nodata(255)
            .gs.rebase(tile.gs.attrs.root.get(Tiling))
        )
        stitcher.add(prediction)

    (rebuilt,) = stitcher.flush()

    assert rebuilt.gs.geobox == source.gs.geobox
    assert rebuilt.gs.variables == ("class",)
    assert rebuilt["class"].dtype == np.dtype("uint8")
    assert np.array_equal(rebuilt["class"].values, np.full((16, 16), 6, "uint8"))
