"""Project cloud pixels onto the ground their shadow is estimated to fall on."""

from __future__ import annotations

import numpy as np
import xarray as xr

from geosave_engine.geodata.utils.xarray import map_spatial_overlap

from ._raster import feature_raster


def shadow_mask(
    raster: xr.Dataset,
    *,
    name: str,
    cloud_mask: str,
    sun_azimuth: str,
    resolution: float = 10,
    shadow_distance_m: float = 500,
) -> xr.Dataset:
    """Derive a named shadow mask from clouds and sun-azimuth coordinates.

    Shifts cloud pixels opposite the sun azimuth for up to
    `shadow_distance_m`, one pixel per step. A sun-azimuth coordinate may be
    scalar or vary along time; each observation uses its own value.

    Args:
        raster: Raster carrying cloud pixels and sun-azimuth coordinates.
        name: Output variable name.
        cloud_mask: Boolean cloud-mask variable name.
        sun_azimuth: Scalar or time-varying coordinate name.
        resolution: Pixel size in meters.
        shadow_distance_m: Maximum shadow projection distance in meters.

    Returns:
        One-variable bool raster, True where shadow is estimated, lazy when
        the cloud mask is.

    Raises:
        ValueError: The mask is not boolean, resolution is not positive, or
            sun azimuth is absent or varies along unsupported dimensions.
    """
    cloud = raster[cloud_mask]
    if cloud.dtype != bool:
        raise ValueError(f"Cloud mask {cloud_mask!r} must be boolean")
    if resolution <= 0:
        raise ValueError(f"Resolution must be positive, got {resolution}")
    if sun_azimuth not in raster.coords:
        raise ValueError(f"Sun azimuth coordinate {sun_azimuth!r} is absent")
    azimuth = raster.coords[sun_azimuth]
    if any(dim != "time" for dim in azimuth.dims):
        raise ValueError(
            f"Sun azimuth coordinate {sun_azimuth!r} varies along "
            f"{list(azimuth.dims)}; expected a scalar or time coordinate"
        )
    if azimuth.dims and "time" not in cloud.dims:
        raise ValueError(
            f"Sun azimuth coordinate {sun_azimuth!r} varies with time but "
            f"cloud mask {cloud_mask!r} does not"
        )

    depth = round(shadow_distance_m / resolution)
    if not azimuth.dims:
        field = _project(cloud, float(azimuth.values), depth)
    else:
        fields = [
            _project(
                cloud.isel(time=index),
                float(azimuth.isel(time=index).values),
                depth,
            )
            for index in range(cloud.sizes["time"])
        ]
        field = xr.concat(
            fields,
            dim=cloud.coords["time"],
            coords="minimal",
            compat="override",
        ).assign_coords({sun_azimuth: azimuth})
    return feature_raster(raster, field, name=name, reference=cloud)


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
