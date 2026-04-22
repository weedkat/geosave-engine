"""Sentinel-2 cloud and shadow masking utilities."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from s2cloudless import S2PixelCloudDetector

CLOUD_THRESHOLD: float = 0.65
CDI_THRESHOLD:   float = -0.5
B10_THRESHOLD:   float = 0.01
SHADOW_DISTANCE_M: int = 1500


def _local_var(arr: np.ndarray, size: int = 7) -> np.ndarray:
    a      = arr.astype(np.float64)
    mean   = uniform_filter(a, size)
    mean_sq = uniform_filter(a ** 2, size)
    return (mean_sq - mean ** 2).astype(np.float32)


def compute_s2c_mask(bands: np.ndarray, threshold: float = CLOUD_THRESHOLD) -> np.ndarray:
    """Cloud probability mask via s2cloudless.

    Args:
        bands: (10, H, W) float32 TOA reflectance in band order
               [B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12].
        threshold: Probability cutoff for cloud classification.
    """
    data     = np.moveaxis(bands, 0, -1).astype(np.float32)
    detector = S2PixelCloudDetector(threshold=threshold, all_bands=False)
    prob     = detector.get_cloud_probability_maps(data[np.newaxis])
    return prob[0] >= threshold


def compute_cdi_mask(
    b07: np.ndarray,
    b08: np.ndarray,
    b8a: np.ndarray,
    *,
    threshold: float = CDI_THRESHOLD,
    eps: float = 1e-6,
) -> np.ndarray:
    """Cloud Displacement Index mask (Frantz / Zupanc formulation).

    CDI = (V(B07/B8A) − V(B08/B8A)) / (V(B07/B8A) + V(B08/B8A))

    where V is local spatial variance over a 7×7 window.  CDI < threshold
    flags pixels that are *not* spectrally vegetation-like (clouds, bare soil).

    B07 and B8A are native 20 m; B08 is native 10 m.  When all three bands
    are loaded at 10 m via bilinear resampling, B08 retains its native texture
    while B07/B8A are smoothed.  B08 is pre-smoothed with a sigma-1 Gaussian
    to match the effective ~20 m resolution of B07/B8A before the variance step.
    """
    b07 = b07.astype(np.float32)
    b8a = b8a.astype(np.float32)
    b08 = gaussian_filter(b08.astype(np.float32), sigma=1.0)

    v8a7 = _local_var(b07 / (b8a + eps))
    v8a8 = _local_var(b08 / (b8a + eps))
    cdi  = (v8a7 - v8a8) / (v8a7 + v8a8 + eps)
    return cdi < threshold


def compute_b10_mask(b10: np.ndarray, threshold: float = B10_THRESHOLD) -> np.ndarray:
    """Cirrus mask from B10 (1375 nm) in TOA reflectance units.

    Clear-sky B10 ≈ 0.001; threshold 0.01 gives a comfortable margin.
    """
    return b10.astype(np.float32) > threshold


def build_shadow_mask(
    cloud_mask: np.ndarray,
    sun_azimuth_deg: float,
    *,
    resolution: int = 10,
    distance_m: int = SHADOW_DISTANCE_M,
) -> np.ndarray:
    """Project cloud pixels downwind to approximate the shadow footprint."""
    az_rad   = np.radians(sun_azimuth_deg)
    col_step = -np.sin(az_rad)
    row_step =  np.cos(az_rad)

    n_steps = round(distance_m / resolution)
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
