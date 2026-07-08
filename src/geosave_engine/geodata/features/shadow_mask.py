import numpy as np


def build_shadow_mask(
    cloud_mask: np.ndarray,
    sun_azimuth_deg: float,
    *,
    resolution: int = 10,
    shadow_distance_m: float = 500,
) -> np.ndarray:
    """Project cloud pixels downwind to approximate the shadow footprint.

    Shifts cloud_mask in the direction opposite to the sun azimuth for up to
    shadow_distance_m meters. Each offset step is one pixel at the given resolution.

    Args:
        cloud_mask: (H, W) bool array of detected cloud pixels.
        sun_azimuth_deg: Sun azimuth in degrees (clockwise from north).
        resolution: Pixel size in meters (default 10 m for Sentinel-2).
        shadow_distance_m: Maximum shadow projection distance in meters.

    Returns:
        (H, W) bool mask — True where shadow is estimated.
    """
    az_rad = np.radians(sun_azimuth_deg)
    col_step = -np.sin(az_rad)
    row_step = np.cos(az_rad)

    n_steps = round(shadow_distance_m / resolution)
    offsets = {(round(row_step * s), round(col_step * s)) for s in range(1, n_steps + 1)}
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
