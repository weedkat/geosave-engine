"""Tests for baking class colours into display channels."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.attrs import Legend, rebase
from geosave_engine.geodata.core.array import array
from geosave_engine.geodata.transform.color import colorize

GRID = GeoBox((2, 3), Affine(1, 0, 10, 0, -1, 20), "EPSG:4326")


def landcover(legend: Legend | None) -> xr.DataArray:
    """Build a class band, one pixel of it naming no class."""
    codes = np.array([[1, 1, 1], [1, 1, 9]], dtype="uint8")
    band = array(codes, GRID, dims=("y", "x"))
    return band if legend is None else rebase(band, legend)


def test_colorize_bakes_each_class_colour_onto_the_grid() -> None:
    band = landcover(Legend(class_map={1: "water"}, color_map={1: "#0000ff"}))

    colored = colorize(band)

    assert colored.dims == ("band", "y", "x")
    assert list(colored.band.values) == ["red", "green", "blue"]
    assert colored.gs.geobox == GRID
    assert colored.y.attrs["standard_name"] == "latitude"
    np.testing.assert_array_equal(colored.sel(band="blue")[0], np.ones(3))
    assert np.isnan(colored.values[:, 1, 2]).all()
    xr.testing.assert_identical(band.gs.colorize(), colored)


def test_colorize_refuses_a_band_without_classes_or_colours() -> None:
    with pytest.raises(ValueError, match="lists no classes"):
        colorize(landcover(None))
    with pytest.raises(ValueError, match="carry no colour"):
        colorize(landcover(Legend(class_map={1: "water"})))
