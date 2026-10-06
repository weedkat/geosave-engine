import pytest

from geosave_engine.geodata.sensors import band_wavelengths, sensor_bands


def test_unknown_sensor_raises_a_key_error() -> None:
    with pytest.raises(KeyError, match="unknown"):
        sensor_bands("unknown")


def test_unknown_band_raises_a_key_error() -> None:
    with pytest.raises(KeyError, match="missing"):
        band_wavelengths("sentinel-2-l2a", ["missing"])
