from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from geosave_engine.geodata.attrs import GDALVariable, rebase


def build_bands(*colours: str | None) -> xr.Dataset:
    """Build a raster whose bands measure the named colours, in order.

    Args:
        *colours: What each band measures, None where it measures nothing.

    Returns:
        Dataset holding one `(y, x)` variable per colour, named `b0`, `b1`, and
        so on so the name never hints at what the band measures.
    """
    pixels = xr.DataArray(np.ones((2, 2), "uint8"), dims=("y", "x"))
    raster = xr.Dataset({f"b{band}": pixels for band, _ in enumerate(colours)})
    for band, colour in enumerate(colours):
        if colour is not None:
            raster = rebase(raster, GDALVariable(colorinterp=colour), target=f"b{band}")
    return raster


def test_rgb_indices_finds_the_three_colour_bands() -> None:
    assert GDALVariable.rgb_indices(build_bands("red", "green", "blue")) == (0, 1, 2)


def test_rgb_indices_follows_what_bands_measure_not_their_order() -> None:
    assert GDALVariable.rgb_indices(build_bands("blue", "red", "green")) == (1, 2, 0)


def test_rgb_indices_passes_over_a_band_measuring_nothing() -> None:
    assert GDALVariable.rgb_indices(build_bands("red", None, "green", "blue")) == (
        0,
        2,
        3,
    )


def test_rgb_indices_takes_the_first_of_two_bands_measuring_one_colour() -> None:
    assert GDALVariable.rgb_indices(build_bands("red", "red", "green", "blue")) == (
        0,
        2,
        3,
    )


def test_rgb_indices_refuses_a_raster_measuring_no_true_colour() -> None:
    # A false-colour composite draws NIR as red, so no band measures blue.
    with pytest.raises(ValueError, match=r"measures no \['blue'\]"):
        GDALVariable.rgb_indices(build_bands("nir", "red", "green"))


def test_colorinterp_takes_gdals_own_names() -> None:
    assert GDALVariable(colorinterp="rededge").colorinterp == "rededge"
    assert GDALVariable(colorinterp="undefined").colorinterp is None


def test_colorinterp_refuses_a_name_gdal_does_not_know() -> None:
    with pytest.raises(ValidationError, match="names no GDAL colour interpretation"):
        GDALVariable(colorinterp="chartreuse")
