from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import geosave_engine.geodata.attrs as attrs


def band(**band_attrs: object) -> xr.DataArray:
    return xr.DataArray(np.zeros((2, 2), "uint8"), dims=("y", "x"), attrs=band_attrs)


def test_a_legend_marks_class_codes() -> None:
    coded = attrs.rebase(band(), attrs.Legend(class_map={0: "bg", 1: "palm"}))

    assert attrs.is_flag(coded)


def test_a_palette_band_marks_class_codes_without_a_legend() -> None:
    assert attrs.is_flag(band(colorinterp="palette"))


@pytest.mark.parametrize("colorinterp", ["gray", "red"])
def test_other_interpretations_say_nothing(colorinterp: str) -> None:
    assert not attrs.is_flag(band(colorinterp=colorinterp))


def test_a_plain_integer_band_is_a_measurement() -> None:
    assert not attrs.is_flag(band(units="1"))


def test_a_half_spelled_listing_still_marks_class_codes() -> None:
    # A guard answers yes or no; it must not raise on attrs it cannot parse.
    assert attrs.is_flag(band(flag_masks=[1, 2]))
