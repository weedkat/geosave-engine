import numpy as np
import pystac

from geosave_engine.geodata.core import derive_step


@derive_step("ndvi")
def compute_ndvi(
    *bands: np.ndarray,
    stac_item: pystac.Item | None = None,
    resolution: float | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalized Difference Vegetation Index.

    Reduce: yaml `bands:` order, exactly [nir, red] (Sentinel-2: B08, B04).
    Output range [-1, 1].

    Args:
        *bands: (nir, red) in that order.
        stac_item: unused, injected for signature consistency.
        resolution: unused, injected for signature consistency.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 NDVI values in [-1, 1].
    """
    nir, red = bands
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return (nir - red) / (nir + red + eps)
