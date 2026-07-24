"""Per-sensor band metadata: order, RGB indices, ground sample distance, wavelength, mean/std.

Physical sensor knowledge (what a STAC collection's bands mean), not tied
to any one model — data lives in `sensors.yaml` next to this file, loaded
once at import.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_YAML_PATH = Path(__file__).parent / "sensors.yaml"

SENSOR_METADATA: dict[str, dict] = yaml.safe_load(_YAML_PATH.read_text())


def _get_sensor(sensor: str) -> dict:
    """Look up one sensor's metadata dict.

    Args:
        sensor: key of `SENSOR_METADATA`, e.g. `"sentinel-2-l2a"`.

    Returns:
        That sensor's metadata dict.

    Raises:
        ValueError: `sensor` not in `SENSOR_METADATA`.
    """
    if sensor not in SENSOR_METADATA:
        raise ValueError(f"{sensor!r} not in SENSOR_METADATA; must be one of {list(SENSOR_METADATA)}")
    return SENSOR_METADATA[sensor]


def _band_values(sensor: str, bands: list[str], field: str) -> list[float]:
    """Shared lookup for `band_wavelengths`/`band_mean`/`band_std` — required per-band fields.

    Args:
        sensor: key of `SENSOR_METADATA`.
        bands: band names to look up, in order.
        field: one of `"wavelength"`, `"mean"`, `"std"`.

    Returns:
        One value per band, in `bands` order.

    Raises:
        ValueError: `sensor` unknown, or a band in `bands` has no `field` entry.
    """
    values = _get_sensor(sensor)["bands"][field]
    missing = [b for b in bands if b not in values]
    if missing:
        raise ValueError(f"{field} missing for band(s) {missing} in sensor {sensor!r}")
    return [values[b] for b in bands]


def sensor_bands(sensor: str) -> list[str]:
    """This sensor's own full band order.

    Args:
        sensor: key of `SENSOR_METADATA`.

    Returns:
        Band names, in the sensor's own canonical order.

    Raises:
        ValueError: `sensor` not in `SENSOR_METADATA`.
    """
    return list(_get_sensor(sensor)["band_order"])


def sensor_gsd(sensor: str) -> float:
    """This sensor's own default/working ground sample distance (m).

    The resolution most (or all) of its bands share, or get resampled to
    before reaching a model — not necessarily every band's real native
    resolution (see `band_gsd` for that).

    Args:
        sensor: key of `SENSOR_METADATA`.

    Returns:
        Ground sample distance in meters.

    Raises:
        ValueError: `sensor` not in `SENSOR_METADATA`.
    """
    return float(_get_sensor(sensor)["gsd"])


def band_wavelengths(sensor: str, bands: list[str]) -> list[float]:
    """Per-band wavelength (µm), in order.

    Args:
        sensor: key of `SENSOR_METADATA`.
        bands: band names to look up, in order.

    Returns:
        One wavelength per band, in `bands` order.

    Raises:
        ValueError: `sensor` unknown, or a band in `bands` has no wavelength entry.
    """
    return _band_values(sensor, bands, "wavelength")


def band_mean(sensor: str, bands: list[str]) -> list[float]:
    """Per-band mean, native DN/reflectance scale, in order.

    Args:
        sensor: key of `SENSOR_METADATA`.
        bands: band names to look up, in order.

    Returns:
        One mean per band, in `bands` order.

    Raises:
        ValueError: `sensor` unknown, or a band in `bands` has no mean entry.
    """
    return _band_values(sensor, bands, "mean")


def band_std(sensor: str, bands: list[str]) -> list[float]:
    """Per-band std, native DN/reflectance scale, in order.

    Args:
        sensor: key of `SENSOR_METADATA`.
        bands: band names to look up, in order.

    Returns:
        One std per band, in `bands` order.

    Raises:
        ValueError: `sensor` unknown, or a band in `bands` has no std entry.
    """
    return _band_values(sensor, bands, "std")


def band_gsd(sensor: str, bands: list[str]) -> list[float]:
    """Per-band real ground sample distance (m), in order.

    Real per-band override where a sensor's bands have genuinely different
    native resolution (e.g. Sentinel-2's 10m/20m bands, MODIS's 250m/500m
    bands) — falls back to the sensor's own default (`sensor_gsd`) for any
    band without one.

    Args:
        sensor: key of `SENSOR_METADATA`.
        bands: band names to look up, in order.

    Returns:
        One ground sample distance per band, in `bands` order.

    Raises:
        ValueError: `sensor` not in `SENSOR_METADATA`.
    """
    metadata = _get_sensor(sensor)
    overrides = metadata["bands"].get("gsd", {})
    default = float(metadata["gsd"])
    return [float(overrides.get(b, default)) for b in bands]
