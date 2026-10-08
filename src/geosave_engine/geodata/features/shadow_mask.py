"""Project cloud pixels onto the ground their shadow is estimated to fall on."""

from __future__ import annotations

import numpy as np
import xarray as xr
from odc.geo.geobox import GeoBox

from geosave_engine.geodata.conventions import TIME_COORDINATE

from geosave_engine.geodata.utils.dask_mapping import map_spatial_overlap


def shadow_mask(
    scene: xr.Dataset,
    *,
    cloud: str,
    sun_azimuth: str | float,
    shadow_distance_m: float = 500,
) -> xr.DataArray:
    """Flag the ground cloud shadow is estimated to fall on.

    Shifts cloud pixels opposite the sun azimuth for up to
    `shadow_distance_m`, one pixel per step. Steps are counted along `x`, so
    pixels are taken as square.

    Args:
        scene: Dataset containing a boolean cloud mask on a metric grid.
        cloud: Cloud mask variable name.
        sun_azimuth: Coordinate or variable name holding degrees clockwise
            from north, scalar or along `time`, or a constant angle.
        shadow_distance_m: Furthest a shadow is projected, in metres.

    Returns:
        Unnamed bool band, True where shadow is estimated, on the mask's
        coordinates and lazy when it is.

    Raises:
        KeyError: A selected variable or coordinate is absent.
        ValueError: The mask is not boolean or sits on no metric grid, or the
            azimuth varies along anything but the mask's own dates.

    Examples:
        >>> shadow = shadow_mask(scene, cloud="cloud", sun_azimuth="sun_azimuth")
        >>> scene.assign(shadow=shadow).shadow.dtype
        dtype('bool')
    """
    mask = scene[cloud]
    if mask.dtype != bool:
        raise ValueError(f"cloud mask is {mask.dtype}, not boolean")
    geobox = mask.gs.geobox
    if (
        not isinstance(geobox, GeoBox)
        or geobox.crs is None
        or geobox.crs.units != ("metre", "metre")
    ):
        raise ValueError(
            "cloud mask sits on no grid measured in metres, so a shadow distance "
            "in metres spans no known pixels; reproject it to a projected CRS first"
        )

    azimuth = (
        scene[sun_azimuth]
        if isinstance(sun_azimuth, str)
        else xr.DataArray(sun_azimuth)
    )
    if set(azimuth.dims) - {TIME_COORDINATE}:
        raise ValueError(
            f"sun azimuth varies along {list(azimuth.dims)}; expected a scalar "
            f"or along {TIME_COORDINATE!r}"
        )
    steps = round(shadow_distance_m / abs(geobox.resolution.x))
    if not azimuth.dims:
        return _project(mask, float(azimuth), steps)

    if TIME_COORDINATE not in mask.dims:
        raise ValueError("sun azimuth varies with time but the cloud mask does not")
    dates = mask.sizes[TIME_COORDINATE]
    fields = [
        _project(mask.isel(time=index), float(azimuth.isel(time=index)), steps)
        for index in range(dates)
    ]
    # Each date's slice carries scalar copies of the time-indexed coordinates.
    return xr.concat(
        fields, dim=mask.coords[TIME_COORDINATE], coords="minimal", compat="override"
    ).assign_coords(mask.coords)


def _project(
    cloud_mask: xr.DataArray, sun_azimuth_deg: float, steps: int
) -> xr.DataArray:
    """Project one observation eagerly or through its lazy spatial chunks."""
    return map_spatial_overlap(
        _shadow_block,
        cloud_mask,
        depth=steps,
        dtype="bool",
        boundary=False,
        sun_azimuth_deg=sun_azimuth_deg,
        steps=steps,
    )


def _shadow_block(
    cloud_mask: np.ndarray, *, sun_azimuth_deg: float, steps: int
) -> np.ndarray:
    """Project one block's clouds along the shadow direction.

    Args:
        cloud_mask: Cloud block.
        sun_azimuth_deg: Sun azimuth in degrees, clockwise from north.
        steps: Pixels to project.

    Returns:
        Bool shadow block.
    """
    az_rad = np.radians(sun_azimuth_deg)
    col_step = -np.sin(az_rad)
    row_step = np.cos(az_rad)

    offsets = {(round(row_step * s), round(col_step * s)) for s in range(1, steps + 1)}
    offsets.discard((0, 0))

    shadow = np.zeros_like(cloud_mask, dtype=bool)
    for row_off, col_off in offsets:
        shifted = np.roll(cloud_mask, shift=(row_off, col_off), axis=(-2, -1))
        if row_off > 0:
            shifted[..., :row_off, :] = False
        elif row_off < 0:
            shifted[..., row_off:, :] = False
        if col_off > 0:
            shifted[..., :col_off] = False
        elif col_off < 0:
            shifted[..., col_off:] = False
        shadow |= shifted
    return shadow
