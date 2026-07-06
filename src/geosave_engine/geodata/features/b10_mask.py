import numpy as np
import pystac

from geosave_engine.geodata.core import derive_step


@derive_step("b10_mask")
def compute_b10_mask(
    *bands: np.ndarray,
    stac_item: pystac.Item | None = None,
    resolution: float | None = None,
    b10_threshold: float = 0.01,
) -> np.ndarray:
    """Cirrus cloud mask via Sentinel-2 Band B10 reflectance.

    Reduce: yaml `bands:` order, exactly [B10].
    Sentinel-2 L1C: B10 is the cirrus band at 1375 nm.
    B10 reflectance > threshold indicates cirrus cloud presence.

    Args:
        *bands: (b10,) — single Band 10 TOA reflectance array.
        stac_item: unused, injected for signature consistency.
        resolution: unused, injected for signature consistency.
        b10_threshold: Reflectance threshold above which cirrus is flagged.

    Returns:
        (H, W) bool mask — True where cirrus detected.
    """
    (b10,) = bands
    return b10.astype(np.float32) > b10_threshold
