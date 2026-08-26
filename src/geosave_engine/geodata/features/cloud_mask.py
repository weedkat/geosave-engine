"""Cloud masks for Sentinel-2, chunk-safe. See each compute_* function."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, uniform_filter
from s2cloudless.cloud_detector import S2PixelCloudDetector

from geosave_engine.geodata.utils.array import map_overlap

# s2cloudless post-filters its probabilities, so a chunk needs its neighbors' pixels too.
_S2C_DEPTH = 3
# gaussian sigma 1 (~3 px) plus two 7-px box filters.
_CDI_DEPTH = 8

S2C_BAND_ORDER = ("b01", "b02", "b04", "b05", "b08", "b8a", "b09", "b10", "b11", "b12")
SCL_CLOUD_CLASSES = (0, 1, 3, 8, 9, 10)  # no data, saturated/defective, shadow, cloud med/high, cirrus


@lru_cache(maxsize=4)
def _detector(prob_threshold: float) -> S2PixelCloudDetector:
    """Load the s2cloudless model once per threshold, per process.

    Args:
        prob_threshold: Probability above which a pixel counts as cloud.

    Returns:
        Detector reading the ten-band subset.
    """
    return S2PixelCloudDetector(threshold=prob_threshold, all_bands=False)


def _local_var(arr: np.ndarray, size: int = 7) -> np.ndarray:
    """Local spatial variance over a (size x size) window.

    Args:
        arr: Pixel block.
        size: Window side length.

    Returns:
        Variance per pixel, same shape.
    """
    mean = uniform_filter(arr, size=size)
    mean_sq = uniform_filter(arr**2, size=size)
    return mean_sq - mean**2


def compute_s2c_mask(
    b01: xr.DataArray,
    b02: xr.DataArray,
    b04: xr.DataArray,
    b05: xr.DataArray,
    b08: xr.DataArray,
    b8a: xr.DataArray,
    b09: xr.DataArray,
    b10: xr.DataArray,
    b11: xr.DataArray,
    b12: xr.DataArray,
    *,
    prob_threshold: float = 0.4,
) -> xr.DataArray:
    """Cloud mask via s2cloudless, one chunk at a time.

    Takes Sentinel-2 L1C TOA reflectance already scaled to [0, 1] — raw DN
    read straight from a provider is 10000x this and produces a meaningless
    mask.

    Args:
        b01: (y, x) Band 1 reflectance. b02..b12 likewise, native band order.
        b02: (y, x) Band 2 reflectance.
        b04: (y, x) Band 4 reflectance.
        b05: (y, x) Band 5 reflectance.
        b08: (y, x) Band 8 reflectance.
        b8a: (y, x) Band 8A reflectance.
        b09: (y, x) Band 9 reflectance.
        b10: (y, x) Band 10 reflectance.
        b11: (y, x) Band 11 reflectance.
        b12: (y, x) Band 12 reflectance.
        prob_threshold: Cloud probability above which a pixel is flagged.

    Returns:
        (y, x) uint8 mask, 1 where cloud, lazy when the inputs are.

    Examples:
        >>> scaled = raster.data / 10_000
        >>> mask = compute_s2c_mask(**{band: scaled.sel(band=name) for band, name in bands.items()})
    """
    return map_overlap(
        _s2c_block,
        b01, b02, b04, b05, b08, b8a, b09, b10, b11, b12,
        depth=_S2C_DEPTH,
        dtype="uint8",
        prob_threshold=prob_threshold,
    )


def _s2c_block(*bands: np.ndarray, prob_threshold: float) -> np.ndarray:
    """Run the detector over one block's ten bands.

    Args:
        *bands: One block per band, in `S2C_BAND_ORDER`.
        prob_threshold: Cloud probability above which a pixel is flagged.

    Returns:
        Mask block, 1 where cloud.
    """
    stacked = np.stack(bands, axis=-1).astype(np.float32)
    return _detector(prob_threshold).get_cloud_masks(stacked[np.newaxis])[0]


def compute_cdi_mask(
    b07: xr.DataArray,
    b08: xr.DataArray,
    b8a: xr.DataArray,
    *,
    cdi_threshold: float = -0.5,
    eps: float = 1e-6,
) -> xr.DataArray:
    """Cloud Displacement Index mask (Frantz / Zupanc formulation).

    CDI = (V(B07/B8A) - V(B08/B8A)) / (V(B07/B8A) + V(B08/B8A)), where V is
    local variance. B08 is pre-smoothed to match B07/B8A's coarser native
    resolution.

    Args:
        b07: (y, x) Band 7 reflectance.
        b08: (y, x) Band 8 reflectance.
        b8a: (y, x) Band 8A reflectance.
        cdi_threshold: CDI below this is flagged as cloud.
        eps: Guards division by zero.

    Returns:
        (y, x) bool mask, True where cloud, lazy when the inputs are.
    """
    return map_overlap(
        _cdi_block,
        b07, b08, b8a,
        depth=_CDI_DEPTH,
        dtype="bool",
        cdi_threshold=cdi_threshold,
        eps=eps,
    )


def _cdi_block(
    b07: np.ndarray,
    b08: np.ndarray,
    b8a: np.ndarray,
    *,
    cdi_threshold: float,
    eps: float,
) -> np.ndarray:
    """Compute CDI over one block.

    Args:
        b07: Band 7 block.
        b08: Band 8 block.
        b8a: Band 8A block.
        cdi_threshold: CDI below this is flagged as cloud.
        eps: Guards division by zero.

    Returns:
        Bool mask block.
    """
    b07 = b07.astype(np.float32)
    b8a = b8a.astype(np.float32)
    b08 = gaussian_filter(b08.astype(np.float32), sigma=1.0)

    v8a7 = _local_var(b07 / (b8a + eps))
    v8a8 = _local_var(b08 / (b8a + eps))
    return ((v8a7 - v8a8) / (v8a7 + v8a8 + eps)) < cdi_threshold


def compute_b10_mask(b10: xr.DataArray, *, b10_threshold: float = 0.01) -> xr.DataArray:
    """Cirrus mask from Sentinel-2 Band B10 reflectance.

    Args:
        b10: (y, x) Band 10 TOA reflectance, the 1375 nm cirrus band.
        b10_threshold: Reflectance above which cirrus is flagged.

    Returns:
        (y, x) bool mask, True where cirrus. Elementwise, so lazy when
        `b10` is.
    """
    return b10.astype(np.float32) > b10_threshold


def compute_scl_mask(
    scl: xr.DataArray,
    *,
    invalid_classes: tuple[int, ...] = SCL_CLOUD_CLASSES,
) -> xr.DataArray:
    """Cloud/shadow/invalid mask from Sentinel-2 L2A's Scene Classification Layer.

    Sen2Cor's own per-pixel classification: 0=no data, 1=saturated, 2=dark,
    3=cloud shadow, 4=vegetation, 5=bare soil, 6=water, 7=unclassified,
    8/9=cloud med/high prob, 10=cirrus, 11=snow/ice.

    Args:
        scl: (y, x) SCL classification values.
        invalid_classes: SCL values flagged as cloud/shadow/invalid. Default
            excludes snow (11) — pass it explicitly if snow should count too.

    Returns:
        (y, x) bool mask, True where flagged. Elementwise, so lazy when
        `scl` is.
    """
    return scl.isin(invalid_classes)
