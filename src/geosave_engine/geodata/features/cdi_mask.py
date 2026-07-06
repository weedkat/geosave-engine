import numpy as np
import pystac
from scipy.ndimage import gaussian_filter, uniform_filter

from geosave_engine.geodata.core import derive_step


def _local_var(arr: np.ndarray, size: int = 7) -> np.ndarray:
    """Local spatial variance over a (size x size) window."""
    mean = uniform_filter(arr, size=size)
    mean_sq = uniform_filter(arr**2, size=size)
    return mean_sq - mean**2


@derive_step("cdi_mask")
def compute_cdi_mask(
    *bands: np.ndarray,
    stac_item: pystac.Item | None = None,
    resolution: float | None = None,
    cdi_threshold: float = -0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Cloud Displacement Index mask (Frantz / Zupanc formulation).

    Reduce: yaml `bands:` order, exactly [B07, B08, B8A].
    CDI = (V(B07/B8A) − V(B08/B8A)) / (V(B07/B8A) + V(B08/B8A))

    Sentinel-2: B07 and B8A native 20 m; B08 native 10 m (bilinear resampled to 10 m).
    B08 pre-smoothed with sigma-1 Gaussian to match effective ~20 m resolution of B07/B8A.
    Pixels with CDI < cdi_threshold are flagged as cloud/non-vegetation.

    Args:
        *bands: (b07, b08, b8a) in that order.
        stac_item: unused, injected for signature consistency.
        resolution: unused, injected for signature consistency.
        cdi_threshold: CDI below this value is flagged as cloud.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) bool mask — True where cloud detected.
    """
    b07, b08, b8a = bands
    b07 = b07.astype(np.float32)
    b8a = b8a.astype(np.float32)
    b08 = gaussian_filter(b08.astype(np.float32), sigma=1.0)

    v8a7 = _local_var(b07 / (b8a + eps))
    v8a8 = _local_var(b08 / (b8a + eps))
    cdi = (v8a7 - v8a8) / (v8a7 + v8a8 + eps)
    return cdi < cdi_threshold
