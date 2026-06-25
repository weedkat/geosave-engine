import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from s2cloudless import S2PixelCloudDetector


def _local_var(arr: np.ndarray, size: int = 7) -> np.ndarray:
    """Local spatial variance over a (size x size) window."""
    mean = uniform_filter(arr, size=size)
    mean_sq = uniform_filter(arr**2, size=size)
    return mean_sq - mean**2


def compute_s2c_mask(
    b01: np.ndarray,
    b02: np.ndarray,
    b04: np.ndarray,
    b05: np.ndarray,
    b08: np.ndarray,
    b8a: np.ndarray,
    b09: np.ndarray,
    b10: np.ndarray,
    b11: np.ndarray,
    b12: np.ndarray,
    *,
    prob_threshold: float = 0.4,
) -> np.ndarray:
    """Cloud probability mask via s2cloudless.

    Sentinel-2 L1C TOA reflectance. Band order: B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12.
    Each band: (H, W) float32 in range [0, 1].

    Args:
        b01..b12: Individual Sentinel-2 bands as (H, W) float32 arrays.
        prob_threshold: Probability threshold above which a pixel is flagged as cloud.

    Returns:
        (H, W) int mask — True where cloud detected.
    """
    bands = np.stack([b01, b02, b04, b05, b08, b8a, b09, b10, b11, b12], axis=-1).astype(np.float32)
    detector = S2PixelCloudDetector(threshold=prob_threshold, all_bands=False)
    return detector.get_cloud_masks(bands[np.newaxis])[0]  # (1, H, W)


def compute_cdi_mask(
    b07: np.ndarray,
    b08: np.ndarray,
    b8a: np.ndarray,
    *,
    cdi_threshold: float = -0.5,
    eps: float = 1e-6,
) -> np.ndarray:
    """Cloud Displacement Index mask (Frantz / Zupanc formulation).

    CDI = (V(B07/B8A) − V(B08/B8A)) / (V(B07/B8A) + V(B08/B8A))

    Sentinel-2: B07 and B8A native 20 m; B08 native 10 m (bilinear resampled to 10 m).
    B08 pre-smoothed with sigma-1 Gaussian to match effective ~20 m resolution of B07/B8A.
    Pixels with CDI < cdi_threshold are flagged as cloud/non-vegetation.

    Args:
        b07: (H, W) float32 Band 7 reflectance.
        b08: (H, W) float32 Band 8 reflectance.
        b8a: (H, W) float32 Band 8A reflectance.
        cdi_threshold: CDI below this value is flagged as cloud.
        eps: Small value to avoid division by zero.

    Returns:
        (H, W) bool mask — True where cloud detected.
    """
    b07 = b07.astype(np.float32)
    b8a = b8a.astype(np.float32)
    b08 = gaussian_filter(b08.astype(np.float32), sigma=1.0)

    v8a7 = _local_var(b07 / (b8a + eps))
    v8a8 = _local_var(b08 / (b8a + eps))
    cdi = (v8a7 - v8a8) / (v8a7 + v8a8 + eps)
    return cdi < cdi_threshold


def compute_b10_mask(b10: np.ndarray, *, b10_threshold: float = 0.01) -> np.ndarray:
    """Cirrus cloud mask via Sentinel-2 Band B10 reflectance.

    Sentinel-2 L1C: B10 is the cirrus band at 1375 nm.
    B10 reflectance > threshold indicates cirrus cloud presence.

    Args:
        b10: (H, W) float32 Band 10 TOA reflectance.
        b10_threshold: Reflectance threshold above which cirrus is flagged.

    Returns:
        (H, W) bool mask — True where cirrus detected.
    """
    return b10.astype(np.float32) > b10_threshold
