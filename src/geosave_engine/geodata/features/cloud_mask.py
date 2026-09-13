"""Cloud masks for Sentinel-2, each safe to run chunk by chunk."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, uniform_filter
from s2cloudless.cloud_detector import S2PixelCloudDetector

from geosave_engine.geodata.utils.dask import map_blocks_with_halo


S2C_BAND_ORDER = ("b01", "b02", "b04", "b05", "b08", "b8a", "b09", "b10", "b11", "b12")
SCL_CLOUD_CLASSES = (
    0,  # no data,
    1,  # saturated/defective,
    3,  # shadow,
    8,  # cloud med/high,
    9,  # cirrus
    10,
)


@lru_cache(maxsize=4)
def _detector(prob_threshold: float) -> S2PixelCloudDetector:
    """Load the s2cloudless model once per threshold, per process.

    Args:
        prob_threshold: Probability above which a pixel counts as cloud.

    Returns:
        Detector reading the ten-band subset.
    """
    return S2PixelCloudDetector(threshold=prob_threshold, all_bands=False)


def _local_var(arr: np.ndarray, size: int | tuple[int, int, int] = 7) -> np.ndarray:
    """Local spatial variance over a (size x size) window.

    Args:
        arr: Pixel block.
        size: Spatial window size, with a leading time size when present.

    Returns:
        Variance per pixel, same shape.
    """
    # scipy accepts int or a per-axis sequence (see its own docstring); its stub is narrower.
    mean = uniform_filter(arr, size=size)  # pyright: ignore[reportArgumentType]
    mean_sq = uniform_filter(arr**2, size=size)  # pyright: ignore[reportArgumentType]
    return mean_sq - mean**2


def compute_s2c_mask(
    *,
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
    prob_threshold: float = 0.4,
) -> xr.DataArray:
    """Cloud mask via s2cloudless, one chunk at a time.

    Needs Sentinel-2 L1C TOA reflectance in [0, 1]; a band carrying `Packing`
    is unpacked automatically, one carrying raw DN and no `Packing` still
    produces a meaningless mask. Bands are ordered into `S2C_BAND_ORDER` here.

    Args:
        b01: Coastal aerosol reflectance.
        b02: Blue reflectance.
        b04: Red reflectance.
        b05: Red-edge 1 reflectance.
        b08: Near-infrared reflectance.
        b8a: Narrow near-infrared reflectance.
        b09: Water vapour reflectance.
        b10: Cirrus reflectance.
        b11: Shortwave-infrared 1 reflectance.
        b12: Shortwave-infrared 2 reflectance.
        prob_threshold: Cloud probability above which a pixel is flagged.

    Returns:
        (y, x) uint8 mask, 1 where cloud, lazy when the inputs are.

    Examples:
        >>> mask = compute_s2c_mask(b01=ds.gs["B01"], b02=ds.gs["B02"], ...)
    """
    b01, b02, b04, b05, b08, b8a, b09, b10, b11, b12 = (
        band.gs.unpack() for band in (b01, b02, b04, b05, b08, b8a, b09, b10, b11, b12)
    )
    return map_blocks_with_halo(
        _s2c_block,
        b01,
        b02,
        b04,
        b05,
        b08,
        b8a,
        b09,
        b10,
        b11,
        b12,
        depth=3,
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
    batch = stacked[np.newaxis] if stacked.ndim == 3 else stacked
    masks = _detector(prob_threshold).get_cloud_masks(batch)
    return masks[0] if stacked.ndim == 3 else masks


def compute_cdi_mask(
    *,
    b07: xr.DataArray,
    b08: xr.DataArray,
    b8a: xr.DataArray,
    cdi_threshold: float = -0.5,
    eps: float = 1e-6,
) -> xr.DataArray:
    """Cloud Displacement Index mask (Frantz / Zupanc formulation).

    CDI = (V(B07/B8A) - V(B08/B8A)) / (V(B07/B8A) + V(B08/B8A)), where V is
    local variance. B08 is pre-smoothed to match B07/B8A's coarser native
    resolution.

    Args:
        b07: Band 7 reflectance.
        b08: Band 8 reflectance.
        b8a: Band 8A reflectance.
        cdi_threshold: CDI below this is flagged as cloud.
        eps: Guards division by zero.

    Returns:
        (y, x) bool mask, True where cloud, lazy when the inputs are.
    """
    return map_blocks_with_halo(
        _cdi_block,
        b07,
        b08,
        b8a,
        depth=8,
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
    sigma = (0.0, 1.0, 1.0) if b08.ndim == 3 else 1.0
    window = (1, 7, 7) if b08.ndim == 3 else 7
    b08 = gaussian_filter(b08.astype(np.float32), sigma=sigma)

    v8a7 = _local_var(b07 / (b8a + eps), size=window)
    v8a8 = _local_var(b08 / (b8a + eps), size=window)
    return ((v8a7 - v8a8) / (v8a7 + v8a8 + eps)) < cdi_threshold


def compute_b10_mask(b10: xr.DataArray, *, b10_threshold: float = 0.01) -> xr.DataArray:
    """Cirrus mask from Sentinel-2 Band B10 reflectance.

    Args:
        b10: Band 10 TOA reflectance.
        b10_threshold: Reflectance above which cirrus is flagged.

    Returns:
        (y, x) bool mask, True where cirrus. Elementwise, so lazy when
        input is.
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
        scl: Scene Classification Layer values.
        invalid_classes: SCL values flagged as cloud/shadow/invalid. Default
            excludes snow (11) — pass it explicitly if snow should count too.

    Returns:
        (y, x) bool mask, True where flagged. Elementwise, so lazy when
        input is.
    """
    return scl.isin(invalid_classes)
