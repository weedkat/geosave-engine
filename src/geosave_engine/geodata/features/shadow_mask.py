import numpy as np
import pystac

from geosave_engine.geodata.core import derive_step


@derive_step("shadow_mask")
def compute_shadow_mask(
    *bands: np.ndarray,
    stac_item: pystac.Item | None = None,
    resolution: float | None = None,
    shadow_distance_m: float = 500,
) -> np.ndarray:
    """Project cloud pixels downwind to approximate the shadow footprint.

    Reduce: yaml `bands:` order, exactly [cloud_mask_band].
    Shifts `cloud` in the direction opposite to the scene's sun azimuth for
    up to shadow_distance_m meters. Each offset step is one pixel at the
    given resolution.

    Args:
        *bands: (cloud,) — single detected-cloud mask array.
        stac_item: Scene's STAC item — auto-injected from the tile. Sun
            azimuth comes from its ``view:sun_azimuth`` property (0.0 if absent).
        resolution: Pixel size in meters — auto-injected from the tile; required.
        shadow_distance_m: Maximum shadow projection distance in meters.

    Returns:
        (H, W) bool mask — True where shadow is estimated.

    Raises:
        ValueError: If resolution is not provided.
    """
    if resolution is None:
        raise ValueError("shadow_mask requires resolution (injected from tile automatically)")
    (cloud,) = bands
    sun_azimuth_deg = stac_item.properties.get("view:sun_azimuth", 0.0) if stac_item is not None else 0.0
    az_rad = np.radians(sun_azimuth_deg)
    col_step = -np.sin(az_rad)
    row_step = np.cos(az_rad)

    n_steps = round(shadow_distance_m / resolution)
    offsets = {(round(row_step * s), round(col_step * s)) for s in range(1, n_steps + 1)}
    offsets.discard((0, 0))

    shadow = np.zeros(cloud.shape, dtype=bool)
    for row_off, col_off in offsets:
        shifted = np.roll(cloud, shift=(row_off, col_off), axis=(0, 1))
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
