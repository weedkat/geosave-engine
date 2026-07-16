import numpy as np


def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalized Difference Vegetation Index.

    Sentinel-2: nir=B08, red=B04. Output range [-1, 1].

    Args:
        nir: (H, W) float32 near-infrared reflectance.
        red: (H, W) float32 red reflectance.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) float32 NDVI values in [-1, 1].
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return (nir - red) / (nir + red + eps)
