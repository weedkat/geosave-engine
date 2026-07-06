import numpy as np
import pystac
from scipy.ndimage import binary_opening

from geosave_engine.geodata.core import derive_step


@derive_step("binary_open")
def binary_open(
    *bands: np.ndarray,
    stac_item: pystac.Item | None = None,
    resolution: float | None = None,
    structure_size: int = 3,
) -> np.ndarray:
    """Morphological opening — removes small isolated noise from a binary mask.

    Reduce: yaml `bands:` order, exactly [mask_band].

    Args:
        *bands: (mask,) — single boolean/uint8 mask.
        stac_item: unused, injected for signature consistency.
        resolution: unused, injected for signature consistency.
        structure_size: Square structuring element side length.

    Returns:
        (H, W) uint8 mask, opened.
    """
    (mask,) = bands
    opened = binary_opening(mask.astype(bool), structure=np.ones((structure_size, structure_size)))
    return opened.astype(np.uint8)
