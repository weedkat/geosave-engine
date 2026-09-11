from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.attrs import Packing
from geosave_engine.geodata.core.raster import raster

UTM = "EPSG:32633"


def geobox(*, crs: str = UTM, shape: tuple[int, int] = (8, 8)) -> GeoBox:
    """Build one grid for a raster-construction test.

    Args:
        crs: CRS the grid sits in.
        shape: Grid height and width in pixels.

    Returns:
        Grid of `shape` at ten-unit pixels.
    """
    left, bottom = 300_000.0, 5_000_000.0
    return GeoBox.from_bbox(
        (left, bottom, left + shape[1] * 10, bottom + shape[0] * 10),
        crs=crs,
        shape=shape,
        tight=True,
    )


def test_raster_places_arrays_and_conforms_them() -> None:
    box = geobox()

    built = raster({"red": np.zeros(box.shape, "uint16")}, box)

    assert built.gs.geobox == box
    assert built.gs.variables == ("red",)
    assert built.y.attrs["standard_name"] == "projection_y_coordinate"


def test_raster_names_leading_dims_and_labels_them() -> None:
    box = geobox()
    times = pd.date_range("2024-01-01", periods=2, freq="MS")

    built = raster({"red": np.zeros((2, *box.shape), "uint16")}, box, time=times)

    assert built.red.dims == ("time", *box.dimensions)
    assert built.gs.timespan is not None


def test_raster_leaves_an_unlabelled_leading_dim_alone() -> None:
    box = geobox()

    built = raster(
        {"probability": np.zeros((3, *box.shape), "float32")}, box, **{"class": None}
    )

    assert built.probability.dims == ("class", *box.dimensions)
    assert "class" not in built.coords


def test_raster_takes_spatial_names_from_the_grid() -> None:
    built = raster(
        {"red": np.zeros((4, 4), "uint16")}, geobox(crs="EPSG:4326", shape=(4, 4))
    )

    assert built.gs.grid_dims == ("latitude", "longitude")


def test_raster_without_a_geobox_stays_unplaced() -> None:
    built = raster({"band": np.zeros((4, 4), "uint8")})

    assert built.band.dims == ("y", "x")
    assert not built.coords
    with pytest.raises(ValueError, match="no locatable grid"):
        built.gs.geobox


def test_write_nodata_declares_it_under_both_names() -> None:
    box = geobox()

    built = raster({"red": np.zeros(box.shape, "uint16")}, box, nodata=0)

    assert built.gs.attrs.data_vars["red"].get(Packing).fill_value == 0
    assert built.red.odc.nodata == 0


def test_raster_states_no_fill_when_none_is_given() -> None:
    box = geobox()

    built = raster({"red": np.zeros(box.shape, "uint16")}, box)

    assert "nodata" not in built.red.attrs
    assert "_FillValue" not in built.red.attrs


def test_raster_keeps_pixels_untouched() -> None:
    box = geobox()
    values = np.arange(64, dtype="float32").reshape(box.shape)

    built = raster({"red": values}, box)

    assert np.array_equal(built.red.values, values)
    assert built.red.dtype == np.dtype("float32")


def test_raster_refuses_no_arrays() -> None:
    with pytest.raises(ValueError, match="at least one named array"):
        raster({})


def test_raster_refuses_arrays_on_different_shapes() -> None:
    with pytest.raises(ValueError, match="one axis is one length"):
        raster({"a": np.zeros((8, 8)), "b": np.zeros((4, 4))})


def test_raster_refuses_a_rank_that_does_not_match_dims() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="dimensional"):
        raster({"a": np.zeros((3, *box.shape))}, box)


def test_raster_refuses_trailing_axes_off_the_grid() -> None:
    with pytest.raises(ValueError, match="trailing axes"):
        raster({"a": np.zeros((4, 4))}, geobox())


def test_raster_refuses_a_coord_of_the_wrong_length() -> None:
    box = geobox()

    with pytest.raises(ValueError, match="2 labels for an axis of 3"):
        raster({"a": np.zeros((3, *box.shape))}, box, **{"class": [1, 2]})
