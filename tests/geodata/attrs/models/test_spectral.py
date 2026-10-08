"""Spectral facts ride on the variable they describe."""

from __future__ import annotations

from geosave_engine.geodata.attrs import Spectral, model_scope


def test_spectral_round_trips_through_attrs() -> None:
    spectral = Spectral(common_name="red", center_wavelength=0.665)

    assert spectral.to_attrs() == {"common_name": "red", "center_wavelength": 0.665}
    assert Spectral.from_attrs({**spectral.to_attrs(), "units": "1"}) == spectral


def test_spectral_describes_a_variable() -> None:
    assert model_scope(Spectral) == "variable"
    assert Spectral.from_attrs({"units": "1"}) is None
