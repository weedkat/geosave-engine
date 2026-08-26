"""build_shadow_mask: project cloud pixels onto their shadow footprint. See build_shadow_mask."""
from __future__ import annotations

import numpy as np
import xarray as xr

from geosave_engine.geodata.utils.array import map_overlap


def build_shadow_mask(
    cloud_mask: xr.DataArray,
    sun_azimuth_deg: float,
    *,
    resolution: int = 10,
    shadow_distance_m: float = 500,
) -> xr.DataArray:
    """Project cloud pixels downwind to approximate the shadow footprint.

    Shifts `cloud_mask` opposite the sun azimuth for up to
    `shadow_distance_m`, one pixel per step.

    Args:
        cloud_mask: (y, x) array of detected cloud pixels.
        sun_azimuth_deg: Sun azimuth in degrees, clockwise from north.
        resolution: Pixel size in meters.
        shadow_distance_m: Maximum shadow projection distance in meters.

    Returns:
        (y, x) bool mask, True where shadow is estimated, lazy when
        `cloud_mask` is.
    """
    # a cloud up to this many pixels away casts into this chunk, so the halo has to reach that far
    depth = round(shadow_distance_m / resolution)
    return map_overlap(
        _shadow_block,
        cloud_mask,
        depth=depth,
        dtype="bool",
        boundary=False,
        sun_azimuth_deg=sun_azimuth_deg,
        steps=depth,
    )


def _shadow_block(cloud_mask: np.ndarray, *, sun_azimuth_deg: float, steps: int) -> np.ndarray:
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

    shadow = np.zeros(cloud_mask.shape, dtype=bool)
    for row_off, col_off in offsets:
        shifted = np.roll(cloud_mask, shift=(row_off, col_off), axis=(0, 1))
        if row_off > 0:
            shifted[:row_off, :] = False
        elif row_off < 0:
            shifted[row_off:, :] = False
        if col_off > 0:
            shifted[:, :col_off] = False
        elif col_off < 0:
            shifted[:, col_off:] = False
        shadow |= shifted
    return shadow
